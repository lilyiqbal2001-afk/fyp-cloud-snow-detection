# ☁️ Cloud & Snow Detection in Satellite Imagery

## 🚀 Objective
This Final Year Project (FYP) aims to provide a robust, ensemble-based deep learning solution for accurate cloud and snow coverage detection in satellite imagery. The project focuses on mitigating the common challenge of misclassifying bright clouds as snow by integrating physics-based post-processing with multi-model inference.

## 🛠 Tech Stack
* **Language:** Python 3
* **Frameworks:** Streamlit (UI/Dashboard), PyTorch (Deep Learning)
* **Libraries:** `segmentation-models-pytorch`, `transformers` (SegFormer), `rasterio` (Geospatial data), `numpy`
* **Backend:** SQLite (User Authentication)
* **Environment:** Developed and trained entirely on Kaggle Kernels.

## 🧠 Key Challenges & Solutions
* **Ensemble Inference:** Combined DeepLabV3+, RSNet, and SegFormer models to improve detection reliability beyond a single model's capability.
* **Physics-Based Correction:** Implemented NDSI (Normalized Difference Snow Index) calculation and RGB uniformity analysis to effectively distinguish between snow and bright, cloud-covered regions.
* **Vegetation Guard:** Added a vegetation context guard to reclassify "snow" detections in heavily vegetated scenes, significantly reducing false positives.
* **Flexible Data Support:** Enabled processing for both multispectral GeoTIFF files and standard RGB imagery (JPEG/PNG).

## 🧪 Evaluation & Performance
* **Metrics:** The application provides real-time performance analytics, including Mean IoU (Intersection over Union) and confidence scores based on softmax outputs to validate segmentation accuracy.

## ⚠️ Important Note
* **Environment Dependency:** This application is configured to run specifically within the Kaggle environment. Please note that `app.py` is dependent on specific file paths and configurations provided by the Kaggle platform; it may not function correctly if executed outside of this environment.
