<div align="center">

# 🩺 Diabetes Prediction System

### A Machine Learning–Powered Web Application for Early Diabetes Risk Assessment

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](#license)

[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-76%25-brightgreen?style=flat-square)]()
[![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-75.97%25-success?style=flat-square)]()
[![Open Issues](https://img.shields.io/github/issues/rizwanahmed786508/diabetes-prediction-system?style=flat-square)]()
[![Last Commit](https://img.shields.io/github/last-commit/rizwanahmed786508/diabetes-prediction-system?style=flat-square)]()

**[🚀 Live Demo](https://diabetes-prediction-system-zhnsdbyfenhgd5xgjq4ngt.streamlit.app/) &nbsp;•&nbsp; [📂 Repository](https://github.com/rizwanahmed786508/diabetes-prediction-system) 

</div>

---

## 📖 Overview

Diabetes is one of the most prevalent chronic diseases worldwide, and **early detection** plays a critical role in reducing long-term health complications. The **Diabetes Prediction System** is an end-to-end machine learning application that analyzes key clinical indicators — such as glucose level, BMI, and age — to assess a patient's risk of diabetes in real time.

This project demonstrates a complete **Data Science workflow**: data preprocessing, exploratory data analysis (EDA), multi-model training and evaluation, feature importance analysis, and deployment as an interactive **Streamlit web application** with downloadable PDF reports.

---

## 🎯 Objectives

- 📊 Analyze and preprocess diabetes-related clinical data
- 🤖 Build, train, and evaluate multiple ML classification models
- 📈 Compare model performance using standard evaluation metrics
- 🔍 Identify key risk factors using feature importance analysis
- 🖥️ Provide an interactive, easy-to-use prediction interface
- 📄 Generate downloadable clinical reports for end users
- 🏥 Demonstrate the real-world application of ML in healthcare

---

## 📊 Dataset Information

This project uses the well-known **PIMA Indians Diabetes Dataset**, containing diagnostic measurements for female patients of Pima Indian heritage.

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body Mass Index (kg/m²) |
| `DiabetesPedigreeFunction` | Diabetes hereditary risk score |
| `Age` | Age of the patient (years) |
| `Outcome` | Target variable — `0` = Non-Diabetic, `1` = Diabetic |

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Model Persistence** | Joblib |
| **Web Application** | Streamlit |
| **Report Generation** | ReportLab |
| **GUI (legacy)** | Tkinter |

---

## 🧠 Machine Learning Workflow

```
Data Collection → Data Cleaning & Preprocessing → Exploratory Data Analysis (EDA)
        → Feature Scaling → Model Training → Model Evaluation
        → Web App Development → Prediction & Deployment
```

1. **Data Collection** — Imported the PIMA Indians Diabetes Dataset
2. **Data Cleaning & Preprocessing** — Handled missing/zero values and outliers
3. **Exploratory Data Analysis (EDA)** — Visualized feature distributions and correlations
4. **Feature Scaling** — Standardized features using `StandardScaler`
5. **Model Training** — Trained multiple classification algorithms
6. **Model Evaluation** — Compared models using accuracy, precision, recall, and F1-score
7. **Web App Development** — Built an interactive Streamlit interface
8. **Deployment** — Hosted the application for public access

---

## 📈 Exploratory Data Analysis

<table>
<tr>
<td align="center" width="50%">

**Correlation Heatmap**
<br>
<img src="Diabetes_Prediction_System/images/heatmap.png" alt="Correlation Heatmap" width="100%">

</td>
<td align="center" width="50%">

**Dataset Distribution**
<br>
<img src="Diabetes_Prediction_System/images/distribution.png" alt="Dataset Distribution" width="100%">

</td>
</tr>
</table>

---

## 🤖 Model Training & Evaluation

The dataset was split into **training** and **testing** sets, with feature scaling applied via `StandardScaler` to improve model convergence and performance.

### Models Implemented

- 📐 **Logistic Regression**
- 🌲 **Random Forest Classifier**
- 📍 **K-Nearest Neighbors (KNN)**

### Evaluation Metrics

- ✅ Accuracy Score
- 🎯 Precision & Recall
- ⚖️ F1-Score
- 📊 Confusion Matrix

### 📊 Model Performance Comparison

| Model | Accuracy |
|---|---|
| 🌲 **Random Forest** | **75.97%** ⭐ |
| 📐 Logistic Regression | 75.32% |
| 📍 K-Nearest Neighbors (KNN) | 69.48% |

<div align="center">
<img src="Diabetes_Prediction_System/images/confusion_matrix.png" alt="Confusion Matrix" width="60%">
</div>

---

## 🖥️ Application Interface

The deployed application provides a clean, intuitive interface where users can input clinical parameters and receive an instant diabetes risk prediction — complete with confidence scores, feature importance visualization, and a downloadable PDF report.

<table>
<tr>
<td align="center" width="50%">
<img src="Diabetes_Prediction_System/images/gui.png" alt="Application Interface" width="100%">
</td>
<td align="center" width="50%">
<img src="Diabetes_Prediction_System/images/gui2.png" alt="Application Interface" width="100%">
</td>
</tr>
</table>

### ✨ Key Features

- 🔬 **Real-time prediction** across three ML algorithms
- 📊 **Feature importance analysis** to explain model decisions
- 📄 **One-click PDF report generation** for clinical record-keeping
- ↺ **Reset functionality** for quick re-testing
- 💬 **Built-in feedback system**
- 📱 **Responsive design** for desktop and mobile

---

## 🚀 Installation & Usage

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/rizwanahmed786508/diabetes-prediction-system.git
cd diabetes-prediction-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`.

### 🌐 Live Demo

> **[👉 Try the Live Application](https://diabetes-prediction-system-zhnsdbyfenhgd5xgjq4ngt.streamlit.app/)**

---

## 📂 Project Structure

```text
diabetes-prediction-system/
│
├── data/
│   └── diabetes.csv
│
├── images/
│   ├── gui.png
│   ├── gui2.png
│   ├── heatmap.png
│   ├── distribution.png
│   └── confusion_matrix.png
│
├── models/
│   ├── Diabetes_Model.pkl
│   └── diabetes_scaler.pkl
│
├── Diabetes_Prediction.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- [ ] Deep learning–based prediction model
- [ ] Database integration for patient history tracking
- [ ] Enhanced UI/UX with improved data visualizations
- [ ] SHAP-based model explainability
- [ ] Multi-language support

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

## 👨‍💻 Author

<div align="center">

**Rizwan Ahmed**

Software Engineering Student | Data Science & Machine Learning Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rizwanahmed786508)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](linkedin.com/in/rizwanahmed78)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:rizwanmb310@gmail.com)

</div>

---

<div align="center">

### ⭐ If you found this project helpful, consider giving it a star!

*Built with ❤️ using Python, Scikit-learn & Streamlit*

</div>
