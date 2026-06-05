import streamlit as st
import joblib
import numpy as np
import os

# Load saved model and scaler
 BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, "model", "diabetes_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient information to predict diabetes.")

# User Inputs
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    format="%.3f"
)
age = st.number_input("Age", min_value=1)

if st.button("Predict"):

    data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    if prediction == 1:
        st.error(
            f"⚠️ Diabetic\n\nProbability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Non-Diabetic\n\nProbability: {(1-probability):.2%}"
        )
