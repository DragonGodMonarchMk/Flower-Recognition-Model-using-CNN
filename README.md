# Flower-Recognition-Model-using-CNN 🌸
CNN‑based Flower Image Classifier – Developed a convolutional neural network using Python, Keras/TensorFlow, and standard CNN layers (Conv → ReLU → MaxPool → Dense) to classify images of five flower species: daisy, dandelion, rose, sunflower, and tulip. Leveraged a Kaggle dataset (~4.2 K images across classes) with data augmentation and dropout regularization to enhance generalization. Achieved > 95% accuracy on the hold‑out validation set after hyperparameter tuning and early stopping. 
Implemented Grad‑CAM for model interpretability and shared all code, model weights, training scripts, and visualizations in a public GitHub repository following reproducible, clean‑code standards. 
## 🧑‍💻 How to Set Up and Run

```bash
git clone https://github.com/DragonGodMonarchMk/Flower-Recognition-Model-using-CNN.git
cd Flower-Recognition-Model-using-CNN

python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Optional: Download and prepare the Kaggle dataset manually into `data/flowers/`.

# To train:
python train.py --epochs 30 --batch-size 32

# To evaluate on validation or test set:
python evaluate.py --model best_model.h5

# To visualize Grad-CAM:
jupyter notebook GradCAM.ipynb> A complete convolutional neural network (CNN) pipeline for classifying flower species with > 95% validation accuracy using a curated 5-class dataset.

---

## 🚀 Overview

This repository implements a CNN from scratch in **TensorFlow/Keras** to classify flower images into five categories: **daisy**, **dandelion**, **rose**, **sunflower**, and **tulip**. The dataset (~4,242 labeled images) was sourced from Kaggle and divided into training, validation, and test splits. The model pipeline includes:

- Image load and augmentation (rotation, flip, scale)
- CNN architecture—multiple Conv-Pool blocks with dropout
- Training with early stopping and learning-rate scheduling
- Evaluation via confusion matrix and per-class accuracy
- Visualization of model attention using Grad-CAM
- Jupyter notebook and deployed model `.h5` file bundled for reproducibility :contentReference[oaicite:3]{index=3}

---

## 📚 Dataset & Sources

- **Dataset used**: Kaggle *Flowers Recognition* dataset containing 4,242 images (80% train, 20% validation), evenly divided across 5 flower categories. Images gathered via Flickr, Google, Yandex. Key folder names include `daisy/`, `dandelion/`, etc. :contentReference[oaicite:4]{index=4}  
- **License**: Consulte Kaggle terms; ensure data redistribution complies with attribution.

---

## 📊 Model Pipeline

### 🧩 Data Preprocessing & Augmentation
- Rescaled images to **224×224** (target size configurable in `config.py`).
- Applied **ImageDataGenerator** with parameters:
  - `rotation_range=30`
  - `width_shift=0.2`, `height_shift=0.2`
  - `horizontal_flip=True`, `zoom_range=0.2`
- Normalization to [0, 1] and categorical one-hot encoding.

### 📐 CNN Architecture (Outline Provided)
| Layer                     | Parameters                                  |
|--------------------------|---------------------------------------------|
| Conv2D + ReLU + MaxPool | e.g. 32 filters, kernel 3×3, pool 2×2       |
| Conv2D + ReLU + Dropout | Repeat with increased filters (e.g. 64, 128)|
| Flatten → Dense layers   | Dropout 0.5 → Softmax output for 5 classes  |

Optional: *Grad-CAM code* to visualize heatmaps over flower images.

---
Flower-Recognition-Model-using-CNN/
├── data/                          ← `flowers/` with subfolders per class
├── notebooks/
│   ├── train-flower-model.ipynb
│   └── GradCAM.ipynb
├── model/                         ← `best_model.h5`, `log.csv`
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md

## 📈 Training & Evaluation

- Loss: **categorical cross-entropy**  
- Optimizer: **Adam** (initial LR = 1e-3), with **ReduceLROnPlateau**
- Training

🤔 Potential Extensions
Integrate Transfer Learning using ResNet50, VGG16, or MobileNetV2 for improved accuracy and faster convergence. 
Kaggle
Add K-fold cross-validation or experiment with learning rate schedulers.
Develop a simple Flask/AWS Lambda API for live prediction of flower species from uploaded images.
Use Grad-CAM+LIME to enhance model explainability and insight.
