FROM python:3.14-slim

WORKDIR /app

COPY main.py .
COPY Diabetes_Model.pkl .
COPY diabetes_scaler.pkl .

RUN pip install --no-cache-dir fastapi uvicorn pydantic joblib scikit-learn numpy

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]