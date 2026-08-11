You are an expert technical writer and MLOps engineer. I have built an end-to-end project called **“MLOps Diabetes Prediction System”** and I want you to create a **high-level, professional, production-style GitHub README.md** for this repository.

I will provide my current README and/or project files/details below. Analyze them carefully and rewrite/update the README based strictly on the actual implementation. **Do not invent technologies, features, commands, architecture components, metrics, or deployment steps that are not present in the project.**

## Project Overview

The project is an automated end-to-end MLOps pipeline for diabetes prediction.

The current architecture/workflow is approximately:

**Apache Airflow → train.py → MLflow Tracking → Model Evaluation → Best Model Selection → MLflow Model Registry → Champion Model → FastAPI → Streamlit Frontend**

The project uses Docker/Docker Compose for containerized execution.

### Current functionality

* Apache Airflow orchestrates the ML pipeline.
* `train.py` automatically trains multiple machine learning models.
* Models are evaluated using appropriate classification metrics.
* MLflow tracks experiments, parameters, metrics, and model artifacts.
* The pipeline automatically compares trained models and identifies the best-performing model.
* The best model is registered in the MLflow Model Registry.
* A `champion` model alias is maintained for the selected production model.
* FastAPI loads the Champion model from MLflow Model Registry.
* FastAPI exposes a `/predict` REST API.
* Streamlit provides the user-facing prediction interface.
* Streamlit communicates with FastAPI through the Docker Compose service name.
* Docker Compose runs the application components in containers.
* MLflow, Airflow, PostgreSQL, Redis, FastAPI backend, and Streamlit frontend are part of the overall MLOps environment where applicable in the actual project.
* The system supports automated model training, model selection, registration, and model serving.

## README Goals

The README should make the repository look like a **serious university/final-year/project portfolio MLOps project**, suitable for:

* GitHub portfolio
* ML/MLOps internship applications
* Software/ML engineering interviews
* University project evaluation
* Technical presentations

The README should be **professional, technically accurate, clean, and visually organized**.

## Required README Structure

Use a structure similar to:

1. Project Title
2. Short professional tagline
3. Badges
4. Project Overview
5. Key Features
6. Problem Statement
7. MLOps Architecture
8. End-to-End Workflow
9. Technology Stack
10. Project Structure
11. Machine Learning Pipeline
12. Model Training & Evaluation
13. MLflow Tracking & Model Registry
14. Champion Model Strategy
15. FastAPI Backend
16. Streamlit Frontend
17. Docker / Docker Compose
18. Apache Airflow Automation
19. How to Run the Project
20. API Usage / Prediction Example
21. Example Output
22. MLOps Workflow
23. Future Improvements
24. Project Highlights
25. Author
26. License

You may adjust the structure if the actual repository contains better or more relevant sections.

## Architecture Diagram

Create a clean **Mermaid architecture diagram** showing the actual flow:

Dataset
↓
Apache Airflow
↓
train.py
↓
Multiple ML Models
↓
MLflow Tracking
↓
Model Evaluation
↓
Best Model Selection
↓
MLflow Model Registry
↓
Champion Model
↓
FastAPI Backend
↓
Streamlit Frontend

Also show Docker/Docker Compose around the relevant services.

Do not add components that do not actually exist.

## Airflow Section

Clearly explain:

* What Airflow is doing in this project.
* How the DAG triggers the training pipeline.
* How `train.py` is executed.
* How automation eliminates manual model training.
* What happens after training.
* How the pipeline connects training with MLflow.

If the actual DAG contains specific tasks, explain those tasks based on the provided code.

## MLflow Section

Explain professionally:

* Experiment tracking
* Parameters
* Metrics
* Model artifacts
* Model registration
* Model versions
* Champion alias
* How the Champion model is consumed by FastAPI

Mention the actual model name if present in the repository.

## Model Selection

Explain that multiple models are trained and evaluated.

Document the actual models and metrics found in the project.

For example, if the repository confirms these models, document them:

* Logistic Regression
* Random Forest
* KNN

Explain how the best model is selected based on the project's actual selection metric.

Do not claim that one metric is used if the code does not confirm it.

## FastAPI Section

Explain:

* API architecture
* `/predict` endpoint
* Request schema
* Model loading from MLflow
* Champion model usage
* Prediction response
* Probability output if implemented

Include a real JSON request example based on the actual Pydantic schema.

## Streamlit Section

Explain:

* User input interface
* Communication with FastAPI
* Prediction display
* Probability/results display
* Docker networking

If the frontend uses:

`http://backend:8000`

explain why the Docker Compose service name is used instead of `localhost`.

## Docker Section

Explain the containerized architecture.

Include actual Docker commands based on the repository configuration.

For example, only if supported by the repository:

```bash
docker compose up -d
docker compose ps
docker compose logs -f
docker compose down
```

Explain the relevant ports only after verifying them from the project files.

## Project Structure

Create a clean tree such as:

```text
project-root/
├── airflow/
├── backend/
├── frontend/
├── src/
├── dags/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

BUT **replace this with the actual repository structure** after analyzing the files I provide.

Do not invent directories.

## Installation / Setup

Provide a complete step-by-step setup guide for Windows/Linux where appropriate.

Include:

1. Prerequisites
2. Clone repository
3. Environment setup if required
4. Docker Desktop requirements
5. Starting services
6. Accessing Airflow
7. Accessing MLflow
8. Accessing FastAPI Swagger UI
9. Accessing Streamlit
10. Running/testing a prediction
11. Verifying Champion model

Use actual ports and credentials only if they are present in the repository or provided information.

Never expose real passwords, tokens, API keys, or secrets.

Use placeholders for sensitive values.

## API Example

Provide a realistic `/predict` request based on the actual schema.

Example format:

```json
{
  "Pregnancies": 3,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 79,
  "BMI": 25.5,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 22
}
```

Only modify the fields if the actual API schema differs.

## Professional Presentation

Use:

* Clear headings
* Tables where useful
* Mermaid diagrams
* Code blocks
* Short technical explanations
* Consistent terminology
* Professional GitHub formatting
* Minimal unnecessary emojis

Do not make the README unnecessarily verbose.

The README should feel like it was written by an **MLOps/ML Engineer**, not generated as generic documentation.

## Important Accuracy Rules

Before writing the README:

1. Analyze all provided project files/details.
2. Identify the actual architecture.
3. Identify the actual technologies.
4. Identify the actual model names.
5. Identify actual metrics.
6. Identify actual ports.
7. Identify actual Docker services.
8. Identify actual Airflow DAG/tasks.
9. Identify actual MLflow model/registry configuration.
10. Identify the actual project structure.

If information is missing, use a clearly marked placeholder such as:

`<ADD_GITHUB_USERNAME>`

rather than inventing information.

Do not claim Kubernetes, Kubeflow, CI/CD, cloud deployment, monitoring, model drift detection, automated retraining, or production infrastructure unless the provided project actually implements them.

## Final README Style

The final README should immediately communicate:

> This is an end-to-end automated MLOps system where machine learning models are trained and evaluated through Apache Airflow, experiments are tracked with MLflow, the best model is registered and promoted as the Champion model, and the model is served through FastAPI with a Streamlit frontend, all running through Docker-based infrastructure.

Make the README **portfolio-ready, technically credible, and interview-ready**.

I will now provide my current README and/or repository files. Use them as the primary source of truth and produce the **complete final `README.md`**, not an explanation of what should be written.
