
# 🧠 Healthcare Stroke Prediction System

## 🩺 Overview
This project predicts the **risk of stroke** using various **Machine Learning** and **Deep Learning** algorithms.  
It is designed with a **Streamlit frontend** and a **Python backend**, making it easy to deploy as a web app for medical research or educational purposes.

---

## 🚀 Features
- Handles **missing data** and performs **label & one-hot encoding** automatically  
- Applies **SMOTE** to fix dataset imbalance  
- Trains **9 ML models** + **1 ANN** (Artificial Neural Network)  
- Automatically picks and saves the **best model based on Recall** (most important in healthcare use cases)  
- Clean **Streamlit interface** for real-time predictions  
- Modular design: separate backend (model training) & frontend (app)  
- Supports both `.pkl` (Sklearn) and `.h5` (ANN) model formats  

---

## 🧩 Tech Stack
| Component | Technology Used |
|------------|----------------|
| Programming Language | Python |
| Frameworks | Streamlit, TensorFlow / Keras |
| ML Libraries | Scikit-learn, XGBoost, LightGBM, imbalanced-learn |
| Visualization | Pandas, Numpy |
| Model Saving | Pickle (.pkl), Keras (.h5) |

---

## 🧠 Models Trained
1. Logistic Regression  
2. K-Nearest Neighbors (KNN)  
3. Support Vector Machine (SVM)  
4. Decision Tree  
5. Random Forest  
6. Gradient Boosting  
7. XGBoost  
8. LightGBM  
9. Naive Bayes  
10. Artificial Neural Network (ANN)

> The best model is selected automatically based on **Recall**, since minimizing false negatives is critical in stroke prediction.

---

## 🧹 Data Preprocessing
- Removed entries with invalid gender (“Other”)  
- Filled missing BMI values with mean  
- Encoded categorical columns using `LabelEncoder` and `OneHotEncoder`  
- Standardized features using `StandardScaler`  
- Balanced the dataset using **SMOTE (Synthetic Minority Oversampling Technique)**

---

## ⚙️ How to Run the Project

### 1️⃣ Backend (Model Training)
```bash
python backend_training.py
````

This script:

* Loads and preprocesses data
* Trains all models
* Selects the best model based on **Recall**
* Saves trained model (`best_model.pkl` or `best_model.h5`)
* Saves preprocessing transformers (`column_transformer.pkl`, `scaler.pkl`)

### 2️⃣ Frontend (Streamlit App)

```bash
streamlit run app.py
```

Then open the local URL (e.g. [http://localhost:8501](http://localhost:8501)) to interact with the model.

---

## 🧾 Input Features

| Feature               | Description                            |
| --------------------- | -------------------------------------- |
| Gender                | Male / Female                          |
| Age                   | 0–120                                  |
| Hypertension          | 0 = No, 1 = Yes                        |
| Heart Disease         | 0 = No, 1 = Yes                        |
| Ever Married          | 0 = No, 1 = Yes                        |
| Work Type             | Private, Self-employed, Govt_job, etc. |
| Residence Type        | Urban / Rural                          |
| Average Glucose Level | Numeric                                |
| BMI                   | Numeric                                |
| Smoking Status        | never/former/smokes/unknown            |

---

## 📊 Evaluation Metrics

| Metric                   | Description                                           |
| ------------------------ | ----------------------------------------------------- |
| **Accuracy**             | Overall correctness                                   |
| **Precision**            | Correctness of positive predictions                   |
| **Recall (Sensitivity)** | Ability to catch true stroke cases *(most important)* |
| **F1 Score**             | Balance between precision and recall                  |

---

## 🧠 Why Focus on Recall?

In healthcare, **false negatives are dangerous** — missing a stroke-prone patient could have serious consequences.
Hence, **Recall** was chosen as the key metric for model selection.

---

## 🧰 Files in Repository

| File                               | Description                              |
| ---------------------------------- | ---------------------------------------- |
| `backend_training.py`              | Trains and saves all models              |
| `app.py`                           | Streamlit frontend for predictions       |
| `column_transformer.pkl`           | One-hot encoder for categorical features |
| `scaler.pkl`                       | StandardScaler for input normalization   |
| `best_model.pkl` / `best_model.h5` | Saved best model                         |
| `PROJECT_DOCUMENTATION.md`         | Detailed explanation of every step       |
| `code.ipynb`                       | EDA                                      |
| `README.md`                        | Overview and usage instructions          |

---


