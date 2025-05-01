# Diabetic Retinopathy Classifier with XAI (Streamlit App)

This project is a web-based application for multi-class classification of diabetic retinopathy (DR) using **EfficientNetB1** model with integrated **explainable AI** techniques: **Grad-CAM++** and **LIME**.

---

## 🚀 Features
- Upload a retina image and classify DR severity with EfficientNet B1
- View Grad-CAM++ heatmap of model focus
- View LIME explanation of prediction
- Built with Streamlit for fast and clean UI

---

## 🧠 DR Classification Levels
- **0** — No DR
- **1** — Mild
- **2** — Moderate
- **3** — Severe
- **4** — Proliferative DR

---

## 🗂️ Project Structure
```
├── app.py                      # Streamlit app code
├── requirements.txt           # Python dependencies
├── EfficientNetB0_best.keras     # Trained model weight
```

---

## ⚙️ Installation


# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the App
```bash
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Link your GitHub repo and deploy 🎉


## 🧑‍💻 Author
**Khadijah Badmos**  
Master's in Data Science  
[GitHub](https://github.com/Khadee86) | [LinkedIn](www.linkedin.com/in/khadijah-badmos)

---

# Multi-Class Diabetic Retinopathy Grading with EfficientNet & XAI

A deep learning pipeline for diabetic retinopathy (DR) severity grading using EfficientNetB0 to B3, enhanced with explainable AI (Grad-CAM++ and LIME), and deployed on Streamlit for real-time predictions.



## Live Demo

Check out the GitHub repo of the deployed model on Streamlit:
https://github.com/Khadee86/efficientnetb1_dr_deployment

---

## Project Structure

```
├── EfficientNet_EDA.ipynb       # Exploratory Data Analysis
├── EfficientNet_B0_B3.ipynb     # Full training & XAI pipeline for EfficientNetB0-B3
├── logs/             # EfficientNetB0-B3 Training logs
├── efficientnet_results.csv     # Model leaderboard
├── saved_models/               # Saved models in '.keras' format
└── README.md                    # ReadMe.md
└── XAI_imgs/  				#output explainable AI images
└── aptos2019-blindness-detection.zip  #dataset

## 📄 Project Summary

This project targets early detection and severity classification (0-4) of diabetic retinopathy using lightweight CNNs (EfficientNetB0–B3). The APTOS 2019 dataset is used with heavy preprocessing and data augmentation to ensure robustness. EfficientNetB3 had the best accuracy, while EfficientNetB1 was selected for deployment due to its performance-efficiency balance.

Explainability was addressed through Grad-CAM++ (feature heatmaps) and LIME (local surrogate interpretation), and the best model was deployed via Streamlit.


## How to Run This Project (Google Colab)

### 1. Dataset Setup
- Upload APTOS 2019 dataset to your Google Drive:

/content/drive/MyDrive/aptos/
├── train.csv
├── aptos_2019/
└── test_images

### 2. Open Colab and run the following:
1. `EfficientNet_EDA.ipynb` — EDA and class distribution
2. `EfficientNet_B0_B3.ipynb` — model training, evaluation, and XAI

> Make sure to adjust all Drive paths to match your file structure if necessary.

---

##  Key Features

###  EDA Highlights
- Visualizes class imbalance (heavily skewed to Class 0)
- Shows sample images per class
- Brightness, contrast, and resolution analysis

###  Model Training (`EfficientNet_B0_B3.ipynb`)
- **EfficientNetB0 to B3** using transfer learning
- Custom classification head
- Dynamic input resizing for each model (e.g. 224x224 for B0, 300x300 for B3)
- Categorical Crossentropy loss + Class weighting
- `Adam` optimizer with early stopping & ReduceLROnPlateau

###  Data Pipeline
- Stratified split (80% train, 10% val, 10% test)
- Image normalization and augmentation using `tf.data` pipeline
- One-hot encoded labels for 5-class classification

###  Evaluation Metrics
- Accuracy, AUROC, F1-score
- Sensitivity, Specificity
- Confusion Matrix & ROC curves

###  XAI Integration
- **Grad-CAM++**: Heatmaps to highlight influential regions
- **LIME**: Perturbation-based visual explanations for model decisions
- Explanation visualizations saved for correct and incorrect predictions

###  Streamlit Deployment (EfficientNetB1)
- Upload fundus image and get prediction
- Visual overlay of Grad-CAM++ and LIME
- Real-time, lightweight web app interface

---

## 📊 Results Snapshot

| Model         | Accuracy | F1 Score | AUROC | Best Epoch |
|---------------|----------|----------|--------|-------------|
| EfficientNetB0| ~77%     | ~0.76    | High   | ~15         |
| EfficientNetB1| ~78%     | ~0.77    | High   | ~11         |
| EfficientNetB2| ~78%     | ~0.76    | High   | ~10         |
| **EfficientNetB3**| **79%** | **0.78** | **Highest** | ~10         |

EfficientNetB1 was chosen for deployment due to its optimal tradeoff.

---

## 📌 Requirements
- Python 3.9+
- TensorFlow >= 2.8
- OpenCV
- tf-keras-vis
- LIME
- Streamlit


## 📚 Datasets
- APTOS 2019 Blindness Detection: https://www.kaggle.com/competitions/aptos2019-blindness-detection


-The code was run on Google Colab.

