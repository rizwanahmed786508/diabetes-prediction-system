import streamlit as st
import joblib
import numpy as np
import os

# Page setup - modern look
st.set_page_config(page_title="Diabetes Predictor AI", page_icon="🩺", layout="centered")

# Model + Scaler load with safe path
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model", "diabetes_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

st.title("🩺 Diabetes Prediction System")
st.markdown("**Patient ki details daalo, AI 2 second me risk bata dega**")

# Step 1: Gender
gender = st.radio("**Gender select karo:**", ["Female", "Male"], horizontal=True, key="gender")

# Step 2: Input fields - 2 columns
col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120, step=1)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1)
    skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=25, step=1)
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=0, step=1)

with col2:
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0, step=0.1, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.2f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

    # Pregnancies logic
    if gender == "Male":
        pregnancies = 0
        st.info("👨 Male selected → Pregnancies auto = 0")
    else:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)

# Predict button
if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):

    try:
        # Features ko float me convert + correct order
        features = np.array([[float(pregnancies), float(glucose), float(bp),
                              float(skin), float(insulin), float(bmi),
                              float(dpf), float(age)]])

        # Scaling
        features_scaled = scaler.transform(features)

        # Prediction
        prediction = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]

        # Result show
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ **Diabetic Risk Detected**")
            st.metric(label="Risk Probability", value=f"{prob*100:.1f}%")
            st.warning("Doctor se consult karna recommended hai")
        else:
            st.success(f"✅ **No Diabetes Risk**")
            st.metric(label="Safe Probability", value=f"{(1-prob)*100:.1f}%")
            st.balloons()

    except Exception as e:
        st.error(f"Prediction me error: {e}")
        st.code(f"Features used: {features}")