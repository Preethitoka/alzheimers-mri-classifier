# 🧠 Alzheimer's MRI Stage Classifier

A deep learning project that classifies Alzheimer's disease progression stages from brain MRI scans using Transfer Learning (EfficientNetB0) with Grad-CAM explainability.

## 📌 Problem Statement
Alzheimer's disease affects millions worldwide, and early detection is critical. This project builds an automated MRI classification system to identify the stage of cognitive decline from brain scans — reducing diagnostic time and supporting clinicians.

## 🏷️ Classes
| Class | Description |
|---|---|
| Non Demented | No signs of dementia |
| Very Mild Dementia | Very early cognitive decline |
| Mild Dementia | Noticeable memory and cognitive issues |
| Moderate Dementia | Significant impairment |

## 📁 Dataset
**Preprocessed Alzheimer's Disease MRI Dataset** — sourced from Kaggle  
~6,400 MRI brain scan images, pre-split into train / val / test folders.

## 🧪 Model Architecture
- **Base:** EfficientNetB0 pretrained on ImageNet
- **Head:** GlobalAveragePooling → BatchNorm → Dense(256) → Dropout → Dense(128) → Dropout → Softmax(4)
- **Training:** Two-phase — frozen base first, then fine-tuning top 30 layers

## ⭐ Unique Feature — Grad-CAM Explainability
Unlike typical classifiers, this project uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to highlight which regions of the brain MRI drove each prediction — making the model interpretable and clinically relevant.

## 📊 Outputs Generated
- `class_distribution.png` — dataset class balance across splits
- `sample_mri_images.png` — sample brain scans for all 4 classes
- `training_curves.png` — accuracy and loss over epochs
- `confusion_matrix.png` — prediction breakdown by class
- `per_class_accuracy.png` — per-class accuracy bar chart
- `gradcam_visualizations.png` — Grad-CAM heatmaps overlaid on MRI scans

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/alzheimers-mri-classifier.git
cd alzheimers-mri-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download the dataset from Kaggle and place it as:
```
alzheimers-mri-classifier/
└── alziemer_dataset/
    ├── train/
    ├── val/
    └── test/
```

### 4. Run the notebook
```bash
jupyter notebook alzheimers_mri_classifier.ipynb
```

## 🛠️ Tech Stack
- Python 3.10
- TensorFlow / Keras
- EfficientNetB0 (Transfer Learning)
- Grad-CAM (Explainability)
- Scikit-learn, Matplotlib, Seaborn

## 📚 References
- EfficientNet: Tan & Le, 2019
- Grad-CAM: Selvaraju et al., 2017
- Dataset: Kaggle — Preprocessed Alzheimer's Disease MRI Dataset
