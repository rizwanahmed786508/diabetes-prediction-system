from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated
import joblib
import numpy as np

app = FastAPI()


# ─── Patient Input Schema ────────────────────────────────────────────────
class Patient(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: Annotated[float, Field(gt=18, lt=67)]
    DiabetesPedigreeFunction: float
    Age: Annotated[float, Field(gt=21, lt=81)]


# ─── Load Model and Scaler ───────────────────────────────────────────────
model = joblib.load("Diabetes_Model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")


# ─── Prediction Endpoint ─────────────────────────────────────────────────
@app.post("/predict")
def predict(patient: Patient):

    input_data = np.array([[
        patient.Pregnancies,
        patient.Glucose,
        patient.BloodPressure,
        patient.SkinThickness,
        patient.Insulin,
        patient.BMI,
        patient.DiabetesPedigreeFunction,
        patient.Age
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probabilities = model.predict_proba(input_scaled)[0]

    diabetic_probability = round(float(probabilities[1]) * 100, 1)
    non_diabetic_probability = round(float(probabilities[0]) * 100, 1)

    if prediction == 1:
        result = "Diabetic"
    else:
        result = "Non-Diabetic"

    return {
        "prediction": int(prediction),
        "result": result,
        "diabetic_probability": diabetic_probability,
        "non_diabetic_probability": non_diabetic_probability
    }