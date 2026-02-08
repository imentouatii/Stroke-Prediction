# 🩺 Heart Disease Risk Prediction 🩺

## 💡 Overview

This project focuses on predicting **heart disease risk** (a proxy for stroke) using clinical data from patients. The goal is to build a reliable machine learning tool that can assist healthcare professionals in early detection.

> ⚠️ This project is **for educational and research purposes only** and is **not intended for clinical diagnosis**.


## 📊 Dataset Description
The dataset contains vital clinical features like:

- **Age**  
- **Sex**  
- **Chest Pain Type**  
- **Resting Blood Pressure**  
- **Cholesterol**  
- **Exercise Angina**  
- And more...

We experimented with various classical machine learning models **and** a deep learning neural network.

---

## ⚙️ Installation

1. **Clone the repo**

``` bash
git clone https://github.com/imentouatii/Heart-Disease-Detection.git
cd heart_disease_prediction
```

2. **Create and activate a virtual environment**
``` bash
python -m venv heart_disease_prediction-env
source heart_disease_prediction-env/bin/activate  # Linux/Mac
heart_disease_prediction-env\Scripts\activate     # Windows
```

3. **Install dependencies**
``` bash
pip install -r requirements.txt
```



## 🚀 Usage

### Explore & Preprocess Data  
Run the notebook `notebooks/heart_disease_prediction.ipynb` for data analysis, cleaning, and feature engineering.

### Train & Evaluate Models  
Use the notebook or `src/train_model.py` to train classical ML models and the deep learning model. Evaluate performance using metrics like **accuracy**, **precision**, **recall**, and **F1-score**.

### Blend Models for Better Performance  
Combine predictions from multiple models (e.g., Random Forest + SVM + XGBoost) to boost accuracy.

### Deep Learning  
The neural network model gave the results of ~90% accuracy with balanced precision and recall. 

---

## 🖥️💻 Application Interface (Web & Desktop)

In addition to the machine learning pipeline, this project includes an **interactive application** available as:

- 🌐 **Web interface** for easy access via a browser  
- 🖥️ **Desktop application** for offline or local use  

The interface allows users to:
- Input patient clinical data  
- Get real-time **heart disease risk predictions**  
- Visualize prediction results in a user-friendly manner  
---

## 📊 Results
- Random Forest achieved **~94% accuracy**
- Deep Learning model achieved **~90% accuracy** 
- Classical models like Random Forest, SVM, and XGBoost scored around **88% accuracy**  
- Model blending improved classical models but still slightly below Deep Learning  & Random Forest

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
- Expand dataset with additional patient info if available  

---





