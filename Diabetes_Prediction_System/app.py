import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Diabetes Predictor AI", page_icon="🩺", layout="centered")

# Model load
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, "model", "diabetes_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

st.title("🩺 Diabetes Prediction System")

gender = st.radio("**Gender:**", ["Female", "Male"], horizontal=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = 0 if gender == "Male" else st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 25)
    insulin = st.number_input("Insulin", 0, 1000, 0)
    bmi = st.number_input("BMI", 0.0, 70.0, 28.0, 0.1)

with col2:
    dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5, 0.01)
    age = st.number_input("Age", 1, 120, 30)

  
if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):
    try:
        # 13 features ka array - ORDER IMPORTANT HAI
        features = np.array([[
            pregnancies, glucose, bp, skin, insulin, bmi, dpf, age,
         
        ]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ Diabetic Risk: {prob*100:.1f}%")
        else:
            st.success(f"✅ No Risk: {(1-prob)*100:.1f}% safe")
            st.balloons()

    except Exception as e:
        st.error(f"Error: {e}")
        st.code(f"Features shape: {features.shape}, Values: {features}")
