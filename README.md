# diabetes-prediction-system

# 🩺 Diabetes Prediction System Using Machine Learning

A machine learning-based application designed to predict the likelihood of diabetes using patient medical data. This project applies data preprocessing, exploratory data analysis (EDA), machine learning classification techniques, and a graphical user interface (GUI) to provide predictions in a user-friendly manner.

---

## 📌 Project Overview

Diabetes is one of the most common chronic diseases worldwide. Early prediction can help patients seek timely medical attention and reduce health risks.

This project uses the **PIMA Indians Diabetes Dataset** and machine learning algorithms to predict whether a patient is diabetic based on several medical attributes.

---

## 🎯 Objectives

* Analyze diabetes-related medical data.
* Build and train machine learning classification models.
* Evaluate model performance using standard metrics.
* Provide an easy-to-use GUI for diabetes prediction.
* Demonstrate the practical application of machine learning in healthcare.

---

## 📊 Dataset Information

The project uses the **PIMA Indians Diabetes Dataset**, which contains the following features:

| Feature                  | Description                       |
| ------------------------ | --------------------------------- |
| Pregnancies              | Number of pregnancies             |
| Glucose                  | Plasma glucose concentration      |
| BloodPressure            | Diastolic blood pressure          |
| SkinThickness            | Triceps skin fold thickness       |
| Insulin                  | 2-Hour serum insulin              |
| BMI                      | Body Mass Index                   |
| DiabetesPedigreeFunction | Diabetes hereditary score         |
| Age                      | Age of patient                    |
| Outcome                  | Diabetes status (0 = No, 1 = Yes) |

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Tkinter (GUI)

---

## 🧠 Machine Learning Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Scaling
5. Model Training
6. Model Evaluation
7. GUI Development
8. Prediction & Deployment

---

## 📈 Exploratory Data Analysis

### Correlation Heatmap

![Correlation Heatmap](./Diabetes%20Prediction%20System/images/heatmap.png)

### Dataset Distribution

![Dataset Distribution](./Diabetes%20Prediction%20System/images/distribution.png)

---

## 🤖 Model Training

The dataset was divided into training and testing sets.

Algorithms used:

* Logistic Regression
* Random Forest Classifier
* KNN

Feature scaling was performed using **StandardScaler** to improve model performance.

---

## 📊 Model Evaluation

Evaluation metrics:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1-Score

### Confusion Matrix

![Confusion Matrix](./Diabetes%20Prediction%20System/images/confusion_matrix.png)

### Model Accuracy

**Logistic Regression Accuracy: 0.7532467532467533
Random Forest Accuracy: 0.7597402597402597
KNN Accuracy: 0.6948051948051948*



---

## 🖥️ Graphical User Interface (GUI)

The application includes a user-friendly GUI where users can enter medical information and receive a diabetes prediction instantly.

### Application Interface

![GUI Screenshot](./Diabetes%20Prediction%20System/images/gui.png)

---

## 🚀 Installation & Usage

### Clone Repository

```bash
git clone https://github.com/rizwanahmed786508/diabetes-prediction-system.git
```

### Navigate to Project Directory

```bash
cd diabetes-prediction-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

---

## 📂 Project Structure

## 📂 Project Structure

```text
diabetes-prediction-system/
│
├── data/
│   └── diabetes.csv
│
├── images/
│   ├── gui.png
│   ├── heatmap.png
│   └── confusion_matrix.png
│
├── Diabetes_Prediction.ipynb
│    
├── models/
│   └── diabetes_model.pkl
│
├── requirements.txt
├── README.md
└── app.py
```

---

## 🔮 Future Improvements

* Hyperparameter Tuning
* Streamlit Web Application
* Cloud Deployment
* Deep Learning-Based Prediction

---

## 👨‍💻 Author

**Rizwan Ahmed**

Software Engineering Student | Machine Learning & Data Science Enthusiast

GitHub: https://github.com/rizwanahmed786508

---


