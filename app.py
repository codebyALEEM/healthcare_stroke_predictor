import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

try:
    model = tf.keras.models.load_model("best_model.h5")
    model_type = "ANN"
except:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    model_type = "Sklearn"

# Load preprocessors
with open("column_transformer.pkl", "rb") as f:
    ct = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    sc = pickle.load(f)
    
st.title("🩺 Stroke Prediction App")
st.markdown("""
Welcome to the **Stroke Risk Prediction System**.
Please provide the patient details below, and the model will predict the likelihood of stroke.
""")

with st.form("stroke_form"):
    st.header("Patient Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    submit = st.form_submit_button("🔍 Predict Stroke")
    
def preprocess_input():
    """Convert user input into the same format as training data"""
    input_dict = {
        "gender": [1 if gender == "Male" else 0],
        "age": [age],
        "hypertension": [1 if hypertension == "Yes" else 0],
        "heart_disease": [1 if heart_disease == "Yes" else 0],
        "ever_married": [1 if ever_married == "Yes" else 0],
        "work_type": [work_type],
        "Residence_type": [1 if residence == "Urban" else 0],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status]
    }

    df = pd.DataFrame(input_dict)
    X = ct.transform(df)
    X_scaled = sc.transform(X)
    return X_scaled   

if submit:
    X_scaled = preprocess_input()

    if model_type == "ANN":
        prediction = (model.predict(X_scaled) > 0.5).astype("int32")[0][0]
    else:
        prediction = model.predict(X_scaled)[0]

    if prediction == 1:
        st.error("🚨 High Risk of Stroke Detected!")
    else:
        st.success("✅ Low Risk of Stroke. Stay Healthy!")

    st.caption(f"Model Used: **{model_type}**") 