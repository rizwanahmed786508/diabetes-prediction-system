import streamlit as st
import joblib
import numpy as np
import os

# Modern page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# Model load with safe path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model", "diabetes_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

st.title("🩺 Diabetes Prediction System")
st.write("Patient ki details daalo, AI risk predict karega")

# Step 1: Gender select
gender = st.radio("**Gender select karo:**", ["Female", "Male"], horizontal=True)

# Step 2: Features - 2 column me modern look
col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input("Glucose Level", 0, 200, 120)
    bp = st.number_input("Blood Pressure", 0, 140, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 25)
    insulin = st.number_input("Insulin", 0, 900, 0)

with col2:
    bmi = st.number_input("BMI", 0.0, 70.0, 28.0, 0.1)
    dpf = st.number_input("Diabetes Pedigree", 0.0, 2.5, 0.5, 0.01)
    age = st.number_input("Age", 1, 120, 30)

    # Pregnancies logic
    if gender == "Male":
        pregnancies = 0
        st.info("Male select hai → Pregnancies = 0 auto set")
    else:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)

# Predict button
if st.button("🔍 Predict Now", use_container_width=True, type="primary"):
    features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Diabetic Risk: **{prob*100:.1f}%** chance")
        st.write("Doctor se consult karna behtar rahega")
    else:
        st.success(f"✅ No Diabetes Risk: **{(1-prob)*100:.1f}%** safe")
        st.balloons()