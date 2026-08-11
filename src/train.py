import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve
)


# ============================================================
# MLflow Configuration
# ============================================================

EXPERIMENT_NAME = "Diabetes Prediction V2"
REGISTERED_MODEL_NAME = "DiabetesPredictionModel"

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# Artifact Directory
# ============================================================

os.makedirs("../mlflow_artifacts", exist_ok=True)


# ============================================================
# Store Model Runs
# ============================================================

model_runs = []


# ============================================================
# 1. Load Dataset
# ============================================================
df = pd.read_csv("../dataset/diabetes.csv")

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. Features and Target
# ============================================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# ============================================================
# 3. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# Helper Function
# ============================================================

def log_classification_results(
    model_name,
    y_test,
    predictions,
    probabilities
):

    # --------------------------------------------------------
    # Calculate Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    # --------------------------------------------------------
    # Log Metrics
    # --------------------------------------------------------

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })


    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        predictions
    )

    report_path = (
        f"../mlflow_artifacts/"
        f"{model_name}_classification_report.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)

    mlflow.log_artifact(report_path)


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    cm_path = (
        f"../mlflow_artifacts/"
        f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        bbox_inches="tight"
    )

    plt.close()

    mlflow.log_artifact(cm_path)


    # --------------------------------------------------------
    # ROC Curve
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    plt.figure()

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"{model_name} - ROC Curve"
    )

    plt.legend()

    roc_path = (
        f"../mlflow_artifacts/"
        f"{model_name}_roc_curve.png"
    )

    plt.savefig(
        roc_path,
        bbox_inches="tight"
    )

    plt.close()

    mlflow.log_artifact(roc_path)


    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------

    print(f"\n## {model_name}")
    print("--------------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# 4. Logistic Regression
# ============================================================

with mlflow.start_run(
    run_name="Logistic Regression"
):

    lr_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                random_state=42
            )
        )
    ])

    lr_model.fit(
        X_train,
        y_train
    )

    lr_pred = lr_model.predict(
        X_test
    )

    lr_prob = lr_model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    mlflow.log_params({
        "model_type": "Logistic Regression",
        "test_size": 0.2,
        "random_state": 42
    })


    # --------------------------------------------------------
    # Metrics + Artifacts
    # --------------------------------------------------------

    lr_metrics = log_classification_results(
        "logistic_regression",
        y_test,
        lr_pred,
        lr_prob
    )


    # --------------------------------------------------------
    # Log Model
    # --------------------------------------------------------

    mlflow.sklearn.log_model(
        lr_model,
        name="model"
    )


    # --------------------------------------------------------
    # Save Run Information
    # --------------------------------------------------------

    lr_run_id = mlflow.active_run().info.run_id

    model_runs.append({
        "model": "Logistic Regression",
        "run_id": lr_run_id,
        "model_uri": f"runs:/{lr_run_id}/model",
        "f1_score": lr_metrics["f1_score"],
        "accuracy": lr_metrics["accuracy"],
        "roc_auc": lr_metrics["roc_auc"]
    })


# ============================================================
# 5. Random Forest
# ============================================================

with mlflow.start_run(
    run_name="Random Forest"
):

    rf_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ])

    rf_model.fit(
        X_train,
        y_train
    )

    rf_pred = rf_model.predict(
        X_test
    )

    rf_prob = rf_model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    mlflow.log_params({
        "model_type": "Random Forest",
        "n_estimators": 100,
        "random_state": 42,
        "test_size": 0.2
    })


    # --------------------------------------------------------
    # Metrics + Artifacts
    # --------------------------------------------------------

    rf_metrics = log_classification_results(
        "random_forest",
        y_test,
        rf_pred,
        rf_prob
    )


    # --------------------------------------------------------
    # Feature Importance
    # --------------------------------------------------------

    rf_classifier = rf_model.named_steps["classifier"]

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": rf_classifier.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )


    feature_importance_path = (
        "../mlflow_artifacts/"
        "random_forest_feature_importance.csv"
    )

    feature_importance.to_csv(
        feature_importance_path,
        index=False
    )

    mlflow.log_artifact(
        feature_importance_path
    )


    # --------------------------------------------------------
    # Log Model
    # --------------------------------------------------------

    mlflow.sklearn.log_model(
        rf_model,
        name="model"
    )


    # --------------------------------------------------------
    # Save Run Information
    # --------------------------------------------------------

    rf_run_id = mlflow.active_run().info.run_id

    model_runs.append({
        "model": "Random Forest",
        "run_id": rf_run_id,
        "model_uri": f"runs:/{rf_run_id}/model",
        "f1_score": rf_metrics["f1_score"],
        "accuracy": rf_metrics["accuracy"],
        "roc_auc": rf_metrics["roc_auc"]
    })


# ============================================================
# 6. KNN
# ============================================================

with mlflow.start_run(
    run_name="KNN"
):

    knn_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ])

    knn_model.fit(
        X_train,
        y_train
    )

    knn_pred = knn_model.predict(
        X_test
    )

    knn_prob = knn_model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    mlflow.log_params({
        "model_type": "KNN",
        "n_neighbors": 5,
        "test_size": 0.2,
        "random_state": 42
    })


    # --------------------------------------------------------
    # Metrics + Artifacts
    # --------------------------------------------------------

    knn_metrics = log_classification_results(
        "knn",
        y_test,
        knn_pred,
        knn_prob
    )


    # --------------------------------------------------------
    # Log Model
    # --------------------------------------------------------

    mlflow.sklearn.log_model(
    knn_model,
    name="model",
    skops_trusted_types=[
        "sklearn.metrics._dist_metrics.EuclideanDistance64",
        "sklearn.neighbors._kd_tree.KDTree"
    ]
)

    # --------------------------------------------------------
    # Save Run Information
    # --------------------------------------------------------

    knn_run_id = mlflow.active_run().info.run_id

    model_runs.append({
        "model": "KNN",
        "run_id": knn_run_id,
        "model_uri": f"runs:/{knn_run_id}/model",
        "f1_score": knn_metrics["f1_score"],
        "accuracy": knn_metrics["accuracy"],
        "roc_auc": knn_metrics["roc_auc"]
    })


# ============================================================
# 7. Automatic Best Model Selection
# ============================================================

best_model = max(
    model_runs,
    key=lambda x: x["f1_score"]
)


# ============================================================
# 8. Model Comparison
# ============================================================

print("\n")
print("=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

for model in model_runs:

    print(
        f"{model['model']:<20}"
        f" F1: {model['f1_score']:.4f}"
        f" | Accuracy: {model['accuracy']:.4f}"
        f" | ROC-AUC: {model['roc_auc']:.4f}"
    )


# ============================================================
# 9. Best Model
# ============================================================

print("\n")
print("=" * 65)
print("BEST MODEL")
print("=" * 65)

print(
    f"Model    : {best_model['model']}"
)

print(
    f"F1 Score : {best_model['f1_score']:.4f}"
)

print(
    f"Accuracy : {best_model['accuracy']:.4f}"
)

print(
    f"ROC-AUC  : {best_model['roc_auc']:.4f}"
)

print(
    f"Run ID   : {best_model['run_id']}"
)

print(
    f"Model URI: {best_model['model_uri']}"
)

print("=" * 65)


# ============================================================
# 10. Register Best Model
# ============================================================

print("\nRegistering best model in MLflow Model Registry...")

registered_model = mlflow.register_model(
    model_uri=best_model["model_uri"],
    name=REGISTERED_MODEL_NAME
)

print("\n" + "=" * 65)
print("MODEL REGISTERED SUCCESSFULLY")
print("=" * 65)

print(
    f"Model Name    : {REGISTERED_MODEL_NAME}"
)

print(
    f"Version       : {registered_model.version}"
)

print(
    f"Source Run ID : {best_model['run_id']}"
)

print(
    f"F1 Score      : {best_model['f1_score']:.4f}"
)

print("=" * 65)


print(
    "\nMLflow experiment tracking, "
    "best-model selection, and model registration "
    "completed successfully!"
)