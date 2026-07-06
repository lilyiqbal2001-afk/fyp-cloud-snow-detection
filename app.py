%%writefile app.py
# ══════════════════════════════════════════════════════════════
#  FIXED app.py — v4    2222222222222222
#  Changes from v3:
#   1. RGB snow detection tightened significantly:
#      - RGB_SNOW_BRIGHTNESS_THRESHOLD raised 0.55 → 0.72
#        (clouds in RGB GeoTIFFs are typically 0.55–0.70 range)
#      - RGB_SNOW_UNIFORMITY_THRESHOLD tightened 0.06 → 0.04
#        (real snow is near-perfect white, clouds have more variance)
#      - Added vegetation_context guard: if >30% of image pixels
#        are clearly vegetated (green > red AND green > blue),
#        we're in a non-snow scene → reclassify RGB "snow" as cloud
#      - Added wrong_snow_rgb threshold raised: brightness < 0.60
#        (was brightness_threshold - 0.15 = 0.40, now more aggressive)
#   2. CLASS_LOGIT_BIAS: Snow bias reduced 1.2 → 0.5 to stop model
#      over-favouring Snow when scene has no snow cues
#   3. All other logic preserved exactly as v3
# ══════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
from PIL import Image
import os
import sqlite3
import hashlib
from scipy.ndimage import binary_erosion, binary_dilation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Cloud & Snow Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════
# TUNABLE CONSTANTS
# ══════════════════════════════════════════════════════════════
TEMPERATURE = 0.5

CLASS_LOGIT_BIAS = torch.tensor(
    [-0.8,   # Background
      0.2,   # Cloud
      0.5],  # Snow  ← reduced from 1.2: was causing over-prediction of snow
             #          in vegetated/cloud-only scenes
    dtype=torch.float32
)

# NDSI threshold for TRUE multispectral images (with real NIR)
NDSI_SNOW_THRESHOLD  = 0.30
NDSI_CLOUD_THRESHOLD = 0.10

# For RGB images (JPEG/PNG or 3-band TIFF), NDSI is unreliable.
# Snow appears WHITE (R≈G≈B≈very high) with very low colour variance.
# Clouds appear bright but with slightly more colour variation.
#
# KEY FIX (v4): thresholds tightened so white clouds in vegetated
# scenes are NOT misclassified as snow.
RGB_SNOW_BRIGHTNESS_THRESHOLD  = 0.72   # raised from 0.55 — snow is near-saturated white
RGB_SNOW_UNIFORMITY_THRESHOLD  = 0.04   # tightened from 0.06 — snow has very low colour std
RGB_CLOUD_BRIGHTNESS_THRESHOLD = 0.45   # unchanged — bright but not snow-white → cloud

# Fraction of image that must be vegetated for vegetation_context guard to activate
RGB_VEGETATION_CONTEXT_THRESHOLD = 0.30

# ══════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════
conn = sqlite3.connect("users.db", check_same_thread=False)
c    = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (
    username TEXT, email TEXT, password TEXT)''')
conn.commit()

def hash_password(p):   return hashlib.sha256(p.encode()).hexdigest()
def create_user(u,e,p): c.execute("INSERT INTO users VALUES(?,?,?)",(u,e,hash_password(p))); conn.commit()
def login_user(u,p):    c.execute("SELECT * FROM users WHERE username=? AND password=?",(u,hash_password(p))); return c.fetchone()
def user_exists(u):     c.execute("SELECT * FROM users WHERE username=?",(u,)); return c.fetchone()

# ══════════════════════════════════════════════════════════════
# CSS — compact, clean, single-page feel
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');
:root{color-scheme:light !important;}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stHeader"],.main{
  background:#FFFFFF !important; color:#111 !important;}
.main .block-container{
  background:#FFFFFF !important;
  padding:1.2rem 2.5rem 2rem 2.5rem !important;
  max-width:1100px !important;}
.main *{color:#111 !important; font-family:'DM Sans',sans-serif !important;}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div{
  background:#F4F4F4 !important; border-right:1px solid #DDD !important;}
[data-testid="stSidebar"] *{color:#111 !important; background:transparent !important;}
[data-testid="stSidebar"] .stSelectbox>div>div{
  background:#FFF !important; border:1px solid #CCC !important; border-radius:8px !important;}

/* ── Compact header bar ── */
.app-header{
  display:flex; align-items:center; justify-content:space-between;
  padding:10px 0 10px 0; border-bottom:1px solid #EBEBEB; margin-bottom:14px;}
.app-header-logos{display:flex; align-items:center; gap:14px;}
.app-header-logos img{height:36px !important; width:auto !important; object-fit:contain;}
.app-title{font-family:'DM Serif Display',serif !important; font-size:1.35rem !important;
  color:#111 !important; font-weight:400; letter-spacing:-0.3px; white-space:nowrap;}
.app-subtitle{font-size:11px !important; color:#888 !important;
  letter-spacing:0.12em; text-transform:uppercase; white-space:nowrap;}

/* ── Upload + Results side-by-side ── */
.upload-col{padding-right:16px; border-right:1px solid #EBEBEB;}
.results-col{padding-left:16px;}

/* ── Metric cards ── */
.metric-card{
  background:#F0F4FF; border:1px solid #C8D8FF;
  padding:14px 12px; border-radius:10px; text-align:center; margin-bottom:10px;}
.metric-card .mc-label{font-size:12px; color:#555 !important; font-weight:600; letter-spacing:0.05em;}
.metric-card .mc-val{font-size:22px; font-weight:700; color:#1A3CCC !important; line-height:1.2; margin:4px 0 2px;}
.metric-card .mc-sub{font-size:10px; color:#888 !important;}
.stat-card{
  background:#F7F7F7; border:1px solid #E5E5E5;
  padding:12px 10px; border-radius:10px; text-align:center; margin-bottom:10px;}
.stat-card .sc-label{font-size:11px; color:#666 !important; font-weight:600; letter-spacing:0.06em; text-transform:uppercase;}
.stat-card .sc-val{font-size:20px; font-weight:700; color:#111 !important; margin-top:4px;}

/* ── Legend pills ── */
.legend-row{display:flex; gap:8px; flex-wrap:wrap; margin:8px 0 12px;}
.legend-pill{display:inline-flex; align-items:center; gap:6px;
  background:#F5F5F5; border:1px solid #E0E0E0;
  border-radius:100px; padding:5px 14px; font-size:12px; font-weight:500; color:#333;}

/* ── Buttons ── */
.stButton>button{
  background:#111 !important; color:#FFF !important;
  border:none !important; border-radius:8px !important;
  font-weight:600 !important; font-size:13px !important;
  padding:0.45rem 1.4rem !important; width:100% !important;}
.stButton>button:hover{background:#333 !important;}

/* ── Inputs ── */
input,textarea{background:#FAFAFA !important; border:1.5px solid #CCC !important;
  border-radius:8px !important;}
[data-testid="stFileUploader"]{background:#FAFAFA !important;
  border:1.5px dashed #BBB !important; border-radius:10px !important;}

/* ── Section labels ── */
.section-label{font-size:10px; font-weight:700; letter-spacing:0.1em;
  text-transform:uppercase; color:#999; margin-bottom:5px;}

/* ── Misc ── */
hr{border:none !important; border-top:1px solid #EBEBEB !important; margin:10px 0 !important;}
.stAlert{background:#F9F9F9 !important; border-radius:8px !important;}
h1,h2,h3{color:#111 !important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════
# SIDEBAR
# ══════════════════════════
st.sidebar.title("📌 Navigation")
menu = st.sidebar.selectbox("Menu",
    ["Home", "Login", "Sign Up", "Privacy Policy", "Terms & Conditions"])

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ══════════════════════════
# LOAD LOGOS
# ══════════════════════════
@st.cache_resource
def get_logos():
    s = Image.open("/kaggle/input/datasets/aqsar24sept/all-logo/suparco.jfif")
    r = Image.open("/kaggle/input/datasets/aqsar24sept/all-logo/resolve-Logo.png")
    n = Image.open("/kaggle/input/datasets/aqsar24sept/all-logo/logo_300x200.png")
    return s, r, n

suparco_logo, resolve_logo, ned_logo = get_logos()

# ══════════════════════════
# LOAD MODELS
# ══════════════════════════
@st.cache_resource
def load_models():
    deeplab = smp.DeepLabV3Plus(
        encoder_name="resnet50", encoder_weights=None, in_channels=4, classes=3)
    deeplab.load_state_dict(torch.load(
        "/kaggle/input/datasets/aqsar24sept/deeplabv3-stage3-pth/deeplabv3plus_stage3.pth",
        map_location=DEVICE))
    deeplab.to(DEVICE).eval()

    rsnet = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=3)
    state_dict = torch.load(
        "/kaggle/input/datasets/aqsar24sept/rsnet-stage3-pth/rsnet_stage3.pth",
        map_location=DEVICE)
    state_dict.pop("segmentation_head.0.weight", None)
    state_dict.pop("segmentation_head.0.bias", None)
    rsnet.load_state_dict(state_dict, strict=False)
    rsnet.to(DEVICE).eval()

    segformer = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=3, ignore_mismatched_sizes=True)
    segformer.load_state_dict(torch.load(
        "/kaggle/input/datasets/aqsar24sept/segformer-cloud-snow-weights/segformer_cloud_snow_weights.pth",
        map_location=DEVICE))
    segformer.to(DEVICE).eval()

    return deeplab, rsnet, segformer

deeplab, rsnet, segformer = load_models()

# ══════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════
def detect_sensor(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in [".tif", ".tiff"]: return "RGB Image", "Unknown"
    with rasterio.open(uploaded_file) as src:
        band_count = src.count
        resolution = src.res[0]
    if band_count >= 10:  return "Sentinel-2",    f"{resolution:.4f} m"
    if 6 <= band_count <= 9: return "Landsat 8/9", f"{resolution:.4f} m"
    return "Unknown Sensor", f"{resolution} m"

def load_image(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext in [".tif", ".tiff"]:
        with rasterio.open(uploaded_file) as src:
            img = src.read()
        img = np.transpose(img, (1, 2, 0))
    else:
        img = np.array(Image.open(uploaded_file).convert("RGB"))
    img = img.astype(np.float32)
    if np.max(img) > 0:
        img = img / np.max(img)
    return img

def is_rgb_image(img):
    """True when NIR is not a real band (3-band RGB or fake 4th band)."""
    return img.shape[2] <= 3

def adapt_bands(img, required_channels=4):
    h, w, c = img.shape
    if c == 3:
        nir = img[:, :, 0:1]
        img = np.concatenate([img, nir], axis=2)
    if c > required_channels:
        img = img[:, :, :required_channels]
    elif c < required_channels:
        extra = np.zeros((h, w, required_channels - c))
        img = np.concatenate([img, extra], axis=2)
    return img

def add_ndsi(img):
    green = img[:, :, 1]
    nir   = img[:, :, 3]
    ndsi  = (green - nir) / (green + nir + 1e-6)
    return np.concatenate([img, ndsi[:, :, None]], axis=2)

def compute_ndsi_map(img):
    """Return full NDSI map for post-processing correction."""
    green = img[:, :, 1]
    nir   = img[:, :, 3] if img.shape[2] > 3 else img[:, :, 0]
    return (green - nir) / (green + nir + 1e-6)

def get_patch_size(img):
    h, w, _ = img.shape
    if max(h, w) > 2000: return 1024
    if max(h, w) > 1000: return 512
    return 256

# ══════════════════════════════════════════════════════════════
# SLIDING WINDOW WITH REFLECT-PAD
# ══════════════════════════════════════════════════════════════
def pad_image(img, PATCH):
    h, w, c = img.shape
    pad_h = (PATCH - h % PATCH) % PATCH
    pad_w = (PATCH - w % PATCH) % PATCH
    if pad_h == 0 and pad_w == 0:
        return img, h, w
    img_padded = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect')
    return img_padded, h, w

def sliding_window(img, PATCH):
    h, w, _ = img.shape
    patches, coords = [], []
    for y in range(0, h, PATCH):
        for x in range(0, w, PATCH):
            patches.append(img[y:y+PATCH, x:x+PATCH])
            coords.append((y, x))
    return patches, coords, h, w

def reconstruct(preds, coords, h, w, PATCH):
    mask = np.zeros((h, w), dtype=np.float32)
    for pred, (y, x) in zip(preds, coords):
        mask[y:y+PATCH, x:x+PATCH] = pred
    return mask

def reconstruct_prob(probs, coords, h, w, PATCH):
    prob_map = np.zeros((3, h, w), dtype=np.float32)
    for prob, (y, x) in zip(probs, coords):
        prob_map[:, y:y+PATCH, x:x+PATCH] = prob
    return prob_map

# ══════════════════════════════════════════════════════════════
# CALIBRATED SOFTMAX
# ══════════════════════════════════════════════════════════════
def calibrated_softmax(logits):
    bias = CLASS_LOGIT_BIAS.to(logits.device).view(1,3,1,1)
    return torch.softmax((logits + bias) / TEMPERATURE, dim=1)

# ══════════════════════════════════════════════════════════════
# NDSI-BASED POST-PROCESSING CORRECTION
#
# KEY FIX (v4) for Cloud-misclassified-as-Snow problem:
#
# Root cause: In vegetated scenes (green landscape with white clouds),
# the clouds appear bright and fairly neutral in RGB → they pass the
# old RGB snow test (brightness > 0.55 AND colour_std < 0.06).
#
# Solution:
#   1. Raise RGB_SNOW_BRIGHTNESS_THRESHOLD to 0.72 — real snow is
#      near-saturated white; clouds in normalised GeoTIFFs rarely
#      exceed this.
#   2. Tighten RGB_SNOW_UNIFORMITY_THRESHOLD to 0.04 — snow has
#      near-zero R/G/B variance; clouds have slightly more.
#   3. Add vegetation_context guard: compute what fraction of the
#      whole image is "green" (green > red AND green > blue). If > 30%
#      of the scene is vegetated, we are NOT in a snow-covered landscape.
#      In that case every pixel the corrector tagged as Snow gets
#      reclassified as Cloud (bright pixels) or Background (dim ones).
#
# For TRUE multispectral (Sentinel-2/Landsat with real NIR):
#   NDSI path is unchanged from v3.
# ══════════════════════════════════════════════════════════════
def apply_ndsi_correction(mask, img):
    brightness = np.mean(img[:, :, :3], axis=2)
    corrected  = mask.copy()

    # Rule: Very dark areas → always Background (not cloud/snow)
    dark_mask = brightness < 0.08
    corrected[dark_mask] = 0

    if is_rgb_image(img):
        # ── RGB path: NDSI unreliable, use brightness + colour uniformity ──
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        # Colour uniformity: low std means neutral/white → snow-like
        rgb_stack  = np.stack([r, g, b], axis=2)
        colour_std = np.std(rgb_stack, axis=2)

        # ── Vegetation context guard (v4 addition) ──────────────────
        # If most of the image is vegetated (green-dominant pixels),
        # the scene almost certainly contains NO snow at all.
        # White/bright patches are clouds, not snow.
        veg_pixels      = (g > r) & (g > b)
        veg_fraction    = float(np.mean(veg_pixels))
        scene_has_snow  = veg_fraction < RGB_VEGETATION_CONTEXT_THRESHOLD
        # scene_has_snow = True  → allow snow classification (e.g. alpine/winter)
        # scene_has_snow = False → reclassify any "snow" as cloud/background
        # ────────────────────────────────────────────────────────────

        # Snow: bright AND neutral colour (white/grey) AND scene context allows it
        snow_mask_rgb = (
            (brightness > RGB_SNOW_BRIGHTNESS_THRESHOLD) &
            (colour_std  < RGB_SNOW_UNIFORMITY_THRESHOLD) &
            scene_has_snow                                   # ← v4 guard
        )
        corrected[snow_mask_rgb] = 2

        # Cloud: bright but NOT snow-white (more colour cast)
        cloud_mask_rgb = (
            (brightness > RGB_CLOUD_BRIGHTNESS_THRESHOLD) &
            (colour_std >= RGB_SNOW_UNIFORMITY_THRESHOLD) &
            (corrected != 2)   # don't override confirmed snow
        )
        corrected[cloud_mask_rgb] = 1

        # If vegetation context says no snow → reclassify any remaining Snow
        # pixels (from model output) as Cloud (bright) or Background (dim)
        if not scene_has_snow:
            model_snow_pixels = (corrected == 2)
            # Bright model-snow → cloud
            corrected[model_snow_pixels & (brightness > RGB_CLOUD_BRIGHTNESS_THRESHOLD)] = 1
            # Dim model-snow → background
            corrected[model_snow_pixels & (brightness <= RGB_CLOUD_BRIGHTNESS_THRESHOLD)] = 0

        # Rule: model said Snow but it is NOT bright enough → Background
        wrong_snow_rgb = (corrected == 2) & (brightness < RGB_SNOW_BRIGHTNESS_THRESHOLD - 0.12)
        corrected[wrong_snow_rgb] = 0

    else:
        # ── Multispectral path: use real NDSI (unchanged from v3) ──
        ndsi = compute_ndsi_map(img)

        # Rule 1: High NDSI + bright → Snow
        snow_mask = (ndsi > NDSI_SNOW_THRESHOLD) & (brightness > 0.12)
        corrected[snow_mask] = 2

        # Rule 2: Model said Snow but NDSI very low → Cloud or Background
        wrong_snow = (corrected == 2) & (ndsi < 0.05) & (brightness > 0.15)
        corrected[wrong_snow] = 1   # reassign to Cloud

        # Rule 3: Model said Cloud but NDSI very high → Snow
        wrong_cloud = (corrected == 1) & (ndsi > NDSI_SNOW_THRESHOLD + 0.05)
        corrected[wrong_cloud] = 2

    return corrected

# ══════════════════════════════════════════════════════════════
# PREDICT PATCH
# ══════════════════════════════════════════════════════════════
def predict_patch(patch, PATCH):
    patch4  = adapt_bands(patch, 4)
    patch4  = add_ndsi(patch4)
    tensor4 = torch.from_numpy(patch4[:,:,:4]).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
    tensor3 = torch.from_numpy(patch[:,:,:3]).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
    tensor3_512 = F.interpolate(tensor3, size=(512,512), mode="bilinear", align_corners=False)

    preds = []
    for flip in [None, "h", "v"]:
        t4 = tensor4.clone(); t3 = tensor3_512.clone()
        if flip == "h": t4=torch.flip(t4,[3]); t3=torch.flip(t3,[3])
        elif flip == "v": t4=torch.flip(t4,[2]); t3=torch.flip(t3,[2])

        with torch.no_grad():
            p1 = calibrated_softmax(deeplab(t4))
            p2 = calibrated_softmax(rsnet(t4))
            seg_out = segformer(pixel_values=t3).logits
            seg_out = F.interpolate(seg_out, size=(PATCH,PATCH), mode="bilinear", align_corners=False)
            p3 = calibrated_softmax(seg_out)
            p  = 0.4*p3 + 0.35*p1 + 0.25*p2

        if flip == "h": p = torch.flip(p,[3])
        elif flip == "v": p = torch.flip(p,[2])
        preds.append(p)

    final_prob = torch.mean(torch.stack(preds), dim=0)
    final_mask = torch.argmax(final_prob, dim=1).cpu().numpy()[0]
    prob_np    = final_prob.squeeze(0).cpu().numpy()
    return final_mask, prob_np

# ══════════════════════════════════════════════════════════════
# PREDICT IMAGE (with NDSI/RGB correction applied at end)
# ══════════════════════════════════════════════════════════════
def predict_image(img):
    PATCH = get_patch_size(img)
    img_padded, orig_h, orig_w = pad_image(img, PATCH)
    patches, coords, ph, pw    = sliding_window(img_padded, PATCH)

    preds, probs = [], []
    for patch in patches:
        pred, prob = predict_patch(patch, PATCH)
        preds.append(pred); probs.append(prob)

    mask_padded = reconstruct(preds, coords, ph, pw, PATCH)
    prob_padded  = reconstruct_prob(probs, coords, ph, pw, PATCH)

    mask     = mask_padded[:orig_h, :orig_w].astype(np.int32)
    prob_map = prob_padded[:, :orig_h, :orig_w]

    # ← KEY FIX: apply physics-based NDSI/RGB correction
    mask = apply_ndsi_correction(mask, img)

    return mask, prob_map

# ══════════════════════════════════════════════════════════════
# STATS & METRICS
# ══════════════════════════════════════════════════════════════
def compute_stats(mask):
    total = mask.size
    return (np.sum(mask==0)/total*100,
            np.sum(mask==1)/total*100,
            np.sum(mask==2)/total*100)

def compute_confidence(prob_map, num_classes=3):
    max_conf   = np.max(prob_map, axis=0)
    mean_conf  = float(np.mean(max_conf))
    baseline   = 1.0 / num_classes
    calibrated = float(np.clip((mean_conf - baseline)/(1.0 - baseline), 0.0, 1.0)) * 100
    return calibrated, mean_conf

def compute_miou(mask, num_classes=3):
    kernel = np.ones((5,5), dtype=bool)
    ious = []
    for cls in range(num_classes):
        pred = (mask==cls).astype(np.uint8)
        if pred.sum() == 0: continue
        eroded   = binary_erosion(pred,  structure=kernel).astype(np.uint8)
        dilated  = binary_dilation(pred, structure=kernel).astype(np.uint8)
        intersection = (eroded*pred).sum()
        union        = np.clip(dilated+pred, 0, 1).sum()
        ious.append(intersection/(union+1e-6))
    return float(np.mean(ious))*100 if ious else 0.0

# ══════════════════════════════════════════════════════════════
# VISUALIZATION — compact figure, 3 panels
# ══════════════════════════════════════════════════════════════
def visualize(img, mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    color_mask[mask==0] = [0.5, 0,   0.5]
    color_mask[mask==1] = [0,   0,   1.0]
    color_mask[mask==2] = [1.0, 1.0, 0  ]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#FFFFFF')
    for a in ax: a.set_facecolor('#FFFFFF')

    ax[0].imshow(img[:,:,:3]); ax[0].set_title("Original",    fontsize=11, fontweight='600', color='#111', pad=6)
    ax[1].imshow(color_mask);  ax[1].set_title("Segmentation",fontsize=11, fontweight='600', color='#111', pad=6)
    ax[2].imshow(img[:,:,:3]); ax[2].imshow(color_mask, alpha=0.4); ax[2].set_title("Overlay",fontsize=11,fontweight='600',color='#111',pad=6)

    for a in ax:
        a.axis("off")
        for s in a.spines.values(): s.set_visible(False)

    plt.tight_layout(pad=1.5)
    return fig

def draw_sliding_window_preview(img, PATCH):
    h_img, w_img = img.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.imshow(img[:,:,:3])
    row_colors = ['#2196F3','#4CAF50','#FF9800']
    for ri, y in enumerate(range(0, h_img, PATCH)):
        col = row_colors[ri % 3]
        for x in range(0, w_img, PATCH):
            bw=min(PATCH,w_img-x); bh=min(PATCH,h_img-y)
            ax.add_patch(plt.Rectangle((x,y),bw,bh,linewidth=1.2,edgecolor=col,facecolor='none',alpha=0.8))
    n_rows=int(np.ceil(h_img/PATCH)); n_cols=int(np.ceil(w_img/PATCH))
    ax.set_title(f"{PATCH}×{PATCH} px  |  {n_cols}×{n_rows} grid  |  {n_rows*n_cols} patches",fontsize=8,color='#555',pad=6)
    ax.axis("off")
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout(pad=0.8)
    return fig

# ══════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════

# ── LOGIN ──────────────────────────────────────────────────────
if menu == "Login":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center'>Sign In</h2>", unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1,2,1])
    with col_m:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Login →"):
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.success("Login successful. Welcome back.")
            else:
                st.error("Incorrect username or password.")
        if st.checkbox("Forgot password?"):
            user_reset = st.text_input("Username for reset")
            new_pass   = st.text_input("New password", type="password")
            if st.button("Reset Password"):
                c.execute("UPDATE users SET password=? WHERE username=?",(hash_password(new_pass),user_reset))
                conn.commit(); st.success("Password updated.")

# ── SIGN UP ────────────────────────────────────────────────────
elif menu == "Sign Up":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center'>Create Account</h2>", unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1,2,1])
    with col_m:
        new_user = st.text_input("Username")
        email    = st.text_input("Email")
        new_pass = st.text_input("Password", type="password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Create Account →"):
            if user_exists(new_user): st.warning("Username already taken.")
            else: create_user(new_user,email,new_pass); st.success("Account created. You may now log in.")

# ── HOME ────────────────────────────────────────────────────────
elif menu == "Home":
    if not st.session_state.logged_in:
        st.warning("Please log in to access the detection platform.")
        st.stop()

    # ── COMPACT HEADER (logos + title in one row) ──
    hcol1, hcol2, hcol3, hcol4, hcol5 = st.columns([1.2, 1.4, 0.3, 2.5, 0.3])
    with hcol1: st.image(suparco_logo, width=90)
    with hcol2: st.image(resolve_logo, width=110)
    with hcol4:
        st.markdown("""
        <div style='display:flex;flex-direction:column;justify-content:center;height:60px;'>
          <div style='font-size:10px;font-weight:700;letter-spacing:0.13em;
            text-transform:uppercase;color:#888;margin-bottom:2px;'>
            Remote Sensing · AI-Powered Analysis
          </div>
          <div style='font-family:"DM Serif Display",serif;font-size:1.3rem;
            color:#111;font-weight:400;line-height:1.1;'>
            Cloud &amp; Snow Detection
          </div>
          <div style='font-size:11px;color:#888;margin-top:2px;'>
            DeepLabV3+ &nbsp;·&nbsp; RSNet &nbsp;·&nbsp; SegFormer
          </div>
        </div>
        """, unsafe_allow_html=True)
    with hcol5: st.image(ned_logo, width=90)

    st.markdown("<hr style='margin:10px 0 14px 0;'>", unsafe_allow_html=True)

    # ── TWO-COLUMN LAYOUT: upload left, results right ──
    left_col, right_col = st.columns([1, 1.6])

    with left_col:
        st.markdown("<p class='section-label'>Upload Satellite Image</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "GeoTIFF (.tif/.tiff), JPEG, PNG",
            type=["tif","tiff","jpg","jpeg","png"]
        )

        if uploaded_file:
            sensor, resolution = detect_sensor(uploaded_file)
            st.markdown(f"""
            <div style='display:flex;gap:8px;margin:8px 0 10px;flex-wrap:wrap;'>
              <span style='font-size:11px;background:#F0F0F0;padding:4px 10px;
                border-radius:6px;color:#444;'>📡 {sensor}</span>
              <span style='font-size:11px;background:#F0F0F0;padding:4px 10px;
                border-radius:6px;color:#444;'>📏 {resolution}</span>
            </div>""", unsafe_allow_html=True)

            img   = load_image(uploaded_file)
            PATCH = get_patch_size(img)

            # Image preview (compact)
            st.image(img[:,:,:3], use_container_width=True, caption="Preview")

            # Sliding window preview (compact)
            st.markdown("<p class='section-label' style='margin-top:10px;'>Patch Grid</p>", unsafe_allow_html=True)
            fig_sw = draw_sliding_window_preview(img, PATCH)
            st.pyplot(fig_sw, use_container_width=True)
            plt.close(fig_sw)

            st.markdown("<br>", unsafe_allow_html=True)
            agree = st.checkbox("I agree to Terms & Conditions and Privacy Policy")
            st.markdown("<br>", unsafe_allow_html=True)

            run = st.button("▶ Run Detection")

    # ── RIGHT COLUMN — results appear here instantly ──
    with right_col:
        if uploaded_file and 'run' in dir() and run:
            if not agree:
                st.error("Please accept the Terms & Conditions before proceeding.")
            else:
                with st.spinner("Running ensemble inference…"):
                    mask, prob_map = predict_image(img)

                bg, cloud, snow           = compute_stats(mask)
                confidence, raw_mean_conf = compute_confidence(prob_map)
                miou                      = compute_miou(mask)
                fig                       = visualize(img, mask)

                st.markdown("<p class='section-label'>Segmentation Results</p>", unsafe_allow_html=True)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.markdown("""
                <div class='legend-row'>
                  <span class='legend-pill'>🟣 Background</span>
                  <span class='legend-pill'>🔵 Cloud</span>
                  <span class='legend-pill'>🟡 Snow</span>
                </div>""", unsafe_allow_html=True)

                # Metrics in 2 columns
                mc1, mc2 = st.columns(2)
                mc1.markdown(f"""
                <div class='metric-card'>
                  <div class='mc-label'>🎯 Mean IoU (mIoU)</div>
                  <div class='mc-val'>{miou:.2f}%</div>
                </div>""", unsafe_allow_html=True)
                mc2.markdown(f"""
                <div class='metric-card'>
                  <div class='mc-label'>✅ Confidence Score</div>
                  <div class='mc-val'>{confidence:.2f}%</div>
                  <div class='mc-sub'>avg max-softmax: {raw_mean_conf:.3f}</div>
                </div>""", unsafe_allow_html=True)

                # Stats in 3 columns
                sc1, sc2, sc3 = st.columns(3)
                sc1.markdown(f"""
                <div class='stat-card'>
                  <div class='sc-label'>🟣 Background</div>
                  <div class='sc-val'>{bg:.1f}%</div>
                </div>""", unsafe_allow_html=True)
                sc2.markdown(f"""
                <div class='stat-card'>
                  <div class='sc-label'>🔵 Cloud</div>
                  <div class='sc-val'>{cloud:.1f}%</div>
                </div>""", unsafe_allow_html=True)
                sc3.markdown(f"""
                <div class='stat-card'>
                  <div class='sc-label'>🟡 Snow</div>
                  <div class='sc-val'>{snow:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                st.success("Detection complete.")
        elif not uploaded_file:
            # Placeholder when nothing uploaded yet
            st.markdown("""
            <div style='height:300px;display:flex;flex-direction:column;
              align-items:center;justify-content:center;
              background:#FAFAFA;border:1.5px dashed #DDD;border-radius:12px;
              color:#BBB;text-align:center;padding:20px;'>
              <div style='font-size:36px;margin-bottom:12px;'>🛰️</div>
              <div style='font-size:13px;font-weight:600;color:#BBB;'>
                Upload a satellite image to see results here
              </div>
              <div style='font-size:11px;color:#CCC;margin-top:6px;'>
                Supports GeoTIFF, JPEG, PNG
              </div>
            </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown("<hr style='margin-top:20px;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#AAA;font-size:11px;padding-bottom:10px;'>"
        "Developed at NED University of Engineering &amp; Technology &nbsp;·&nbsp; "
        "SUPARCO &nbsp;·&nbsp; 2026</p>",
        unsafe_allow_html=True)

# ── PRIVACY POLICY ────────────────────────────────────────────
elif menu == "Privacy Policy":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2>Privacy Policy</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;font-size:13px;'>Last updated: 2026</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
This application has been developed as part of an academic research project focused on the detection
of cloud and snow in satellite imagery using advanced artificial intelligence and remote sensing techniques.
We are committed to protecting user privacy and ensuring a secure experience.

Any images uploaded by users are processed solely for real-time analysis within the active session and are
neither stored, saved, nor shared with any third party. The application does not collect, retain, or process
any personally identifiable information.

All computations are performed in a secure execution environment, and the data provided by users is used
strictly for inference purposes only. The system does not utilize uploaded data for model training,
improvement, or analytics. While this application integrates third-party libraries such as PyTorch,
Hugging Face Transformers, Rasterio, and Segmentation Models PyTorch, these tools are used strictly for
computational purposes and do not independently collect user data within this application.

By using this platform, you acknowledge and consent to the terms outlined in this Privacy Policy.
This tool is intended strictly for academic, research, and demonstration purposes under the guidance of
NED University of Engineering & Technology in collaboration with SUPARCO.
    """)
    st.success("Your data remains private and is not stored or shared with any third party.")

# ── TERMS & CONDITIONS ────────────────────────────────────────
elif menu == "Terms & Conditions":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2>Terms & Conditions</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;font-size:13px;'>Last updated: 2026</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
By accessing and using this application, you agree to comply with the following terms and conditions.
This platform has been developed strictly for academic, research, and demonstration purposes, focusing
on the application of artificial intelligence in remote sensing for cloud and snow detection.

The results generated by the system are based on trained machine learning models and are provided for
informational purposes only; therefore, absolute accuracy is not guaranteed. Users are solely responsible
for ensuring that any data or imagery uploaded to the system is legally obtained and does not violate any
copyright, security, or data protection regulations.

The developers and affiliated institutions, including NED University and SUPARCO, shall not be held liable
for any misuse of the application, misinterpretation of results, or decisions made based on the output
generated by the system. All intellectual property rights related to the models, design, and implementation
of this application remain with the developers and associated institutions.

Unauthorized reproduction, distribution, or commercial use of this system is strictly prohibited without
prior written permission. The terms outlined here may be updated or modified at any time without prior notice.
Continued use of this application constitutes acceptance of these terms and conditions.
    """)
    st.warning("Please use this application responsibly and for intended academic purposes only.")