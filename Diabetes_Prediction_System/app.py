import streamlit as st
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Load Model and Scaler
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model", "diabetes_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

# Header
st.title("🩺 Diabetes Prediction System")
st.markdown("### Predict Diabetes Risk Using Machine Learning")

st.divider()

# Gender Selection
gender = st.radio(
    "Select Patient Gender",
    ["Male", "Female"],
    horizontal=True
)

st.subheader("Patient Information")

# Two-column layout
col1, col2 = st.columns(2)

# Pregnancy logic
if gender == "Female":
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0
    )
else:
    pregnancies = 0
    st.info("Pregnancies automatically set to 0 for male patients.")

with col1:
    glucose = st.number_input(
        "Glucose Level",
        min_value=0,
        max_value=300,
        value=120
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

with col2:
    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=85
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.35
    )

st.divider()

# Prediction Button
if st.button("🔍 Predict Diabetes", use_container_width=True):

    input_data = np.array([
        [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]
    ])

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Probability
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    st.write(f"**Risk Probability:** {probability:.2%}")

    st.progress(float(probability))

st.divider()

st.caption(
    "This prediction is generated using a Machine Learning model trained on the Pima Indians Diabetes Dataset."
)