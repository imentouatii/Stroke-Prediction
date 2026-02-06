# 🩺 Stroke Prediction Project 🩺

## 💡 Overview

This project focuses on predicting **heart disease risk** (a proxy for stroke) using clinical data from patients. The goal is to build a reliable machine learning tool that can assist healthcare professionals in early detection.

The dataset contains vital clinical features like:

- **Age**  
- **Sex**  
- **Chest Pain Type**  
- **Resting Blood Pressure**  
- **Cholesterol**  
- **Exercise Angina**  
- And more...

We experimented with various classical machine learning models **and** a deep learning neural network — with the deep learning model showing the best results! 🚀

---
<!-- 

## 📂 Project Structure

heart-stroke/

├── data/

│ └── heart.csv # Raw dataset with patient info

├── notebooks/

│ └── stroke_prediction.ipynb # EDA, modeling, evaluation

├── models/

│ └── saved_model.h5 # Trained deep learning model checkpoint

├── src/

│ ├── preprocessing.py # Data cleaning & feature engineering

│ ├── train_models.py # Scripts to train classical & DL models

│ └── utils.py # Helper functions for metrics & viz

├── requirements.txt # Python dependencies

└── README.md # This file
-->

---

## ⚙️ Installation

1. **Clone the repo**

``` bash
git clone https://github.com/yourusername/heart-stroke.git
cd heart-stroke
```

2. **Create and activate a virtual environment**
``` bash
python -m venv heart-stroke-env
source heart-stroke-env/bin/activate  # Linux/Mac
heart-stroke-env\Scripts\activate     # Windows
```

3. **Install dependencies**
``` bash
pip install -r requirements.txt
```



## 🚀 Usage

### Explore & Preprocess Data  
Run the notebook `notebooks/stroke_prediction.ipynb` for data analysis, cleaning, and feature engineering.

### Train & Evaluate Models  
Use the notebook or `src/train_models.py` to train classical ML models and the deep learning model. Evaluate performance using metrics like **accuracy**, **precision**, **recall**, and **F1-score**.

### Blend Models for Better Performance  
Combine predictions from multiple models (e.g., Random Forest + SVM + XGBoost) to boost accuracy.

### Deep Learning  
The neural network model gave the best results — ~90% accuracy with balanced precision and recall. 💪

---

## 📊 Results

- Deep Learning model achieved **~90% accuracy** ❤️  
- Classical models like Random Forest, SVM, and XGBoost scored around **88% accuracy**  
- Model blending improved classical models but still slightly below deep learning  

---

## 🛠 Dependencies

- Python 3.8+  
- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  
- xgboost  
- lightgbm  
- tensorflow (for deep learning)  

---

## 🔮 Next Steps

- Enhance deep learning architecture & tune hyperparameters further  
- Add model explainability tools (e.g., SHAP, LIME) for clinical trust  
- Deploy the model via a web app for real-time stroke risk prediction ❤️‍🔥  
- Expand dataset with additional patient info if available  

---

