
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated
import mlflow
import pandas as pd

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


# ─── MLflow Configuration ────────────────────────────────────────────────

MLFLOW_TRACKING_URI = "http://localhost:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ─── Load Champion Model ─────────────────────────────────────────────────

MODEL_URI = "models:/DiabetesPredictionModel@champion"

model = mlflow.sklearn.load_model(MODEL_URI)

print("=" * 60)
print("Champion model loaded successfully!")
print(f"Model URI: {MODEL_URI}")
print(f"Model Type: {type(model)}")
print(f"Has predict_proba: {hasattr(model, 'predict_proba')}")
print("=" * 60)


# ─── Prediction Endpoint ─────────────────────────────────────────────────

@app.post("/predict")
def predict(patient: Patient):

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame([{
        "Pregnancies": patient.Pregnancies,
        "Glucose": patient.Glucose,
        "BloodPressure": patient.BloodPressure,
        "SkinThickness": patient.SkinThickness,
        "Insulin": patient.Insulin,
        "BMI": patient.BMI,
        "DiabetesPedigreeFunction": patient.DiabetesPedigreeFunction,
        "Age": patient.Age
    }])

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = int(
        model.predict(input_data)[0]
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        input_data
    )[0]

    # Debug information
    print("=" * 60)
    print("PREDICTION REQUEST")
    print("=" * 60)
    print("Input:")
    print(input_data)
    print(f"Prediction: {prediction}")
    print(f"Probabilities: {probabilities}")
    print(f"Probability Type: {type(probabilities)}")
    print("=" * 60)

    # --------------------------------------------------------
    # Calculate probabilities
    # --------------------------------------------------------

    non_diabetic_probability = round(
        float(probabilities[0]) * 100,
        1
    )

    diabetic_probability = round(
        float(probabilities[1]) * 100,
        1
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if prediction == 1:
        result = "Diabetic"
    else:
        result = "Non-Diabetic"

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "prediction": prediction,
        "result": result,
        "diabetic_probability": diabetic_probability,
        "non_diabetic_probability": non_diabetic_probability
    }
