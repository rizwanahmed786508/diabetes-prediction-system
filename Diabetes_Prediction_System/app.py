import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (dark futuristic) ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%);
        color: #e0e6ff;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: rgba(10, 20, 40, 0.95) !important;
        border-right: 1px solid rgba(0, 180, 255, 0.2);
    }

    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00b4ff, #7b2fff, #00b4ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 10px 0 4px;
        animation: shimmer 3s linear infinite;
    }

    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    .subtitle {
        text-align: center;
        color: rgba(160, 190, 255, 0.7);
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 30px;
    }

    .card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 180, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 100, 255, 0.05);
        backdrop-filter: blur(10px);
    }

    .section-label {
        font-family: 'Orbitron', monospace;
        font-size: 0.75rem;
        color: #00b4ff;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(0, 180, 255, 0.2);
    }

    .result-diabetic {
        background: linear-gradient(135deg, rgba(255,50,50,0.15), rgba(180,0,0,0.08));
        border: 1px solid rgba(255, 80, 80, 0.5);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        animation: pulse-red 2s ease-in-out infinite;
    }

    .result-safe {
        background: linear-gradient(135deg, rgba(0,255,120,0.12), rgba(0,180,80,0.06));
        border: 1px solid rgba(0, 255, 120, 0.4);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        animation: pulse-green 2s ease-in-out infinite;
    }

    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 20px rgba(255,50,50,0.2); }
        50% { box-shadow: 0 0 40px rgba(255,50,50,0.4); }
    }

    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 20px rgba(0,255,120,0.15); }
        50% { box-shadow: 0 0 40px rgba(0,255,120,0.3); }
    }

    .result-label {
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-sub {
        font-size: 0.85rem;
        opacity: 0.75;
    }

    .accuracy-badge {
        display: inline-block;
        background: rgba(0, 180, 255, 0.12);
        border: 1px solid rgba(0, 180, 255, 0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        color: #00b4ff;
        font-weight: 500;
    }

    div[data-testid="stSlider"] > div { color: #a0c0ff; }

    div[data-testid="stNumberInput"] input {
        background: rgba(0, 180, 255, 0.05) !important;
        border: 1px solid rgba(0, 180, 255, 0.25) !important;
        color: #e0e6ff !important;
        border-radius: 8px !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0050cc, #7b2fff);
        color: white;
        border: none;
        padding: 14px;
        border-radius: 12px;
        font-family: 'Orbitron', monospace;
        font-size: 0.85rem;
        letter-spacing: 2px;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 10px;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0070ff, #9b4fff);
        box-shadow: 0 0 25px rgba(123, 47, 255, 0.5);
        transform: translateY(-1px);
    }

    .stSelectbox > div > div {
        background: rgba(0, 180, 255, 0.05) !important;
        border: 1px solid rgba(0, 180, 255, 0.25) !important;
        color: #e0e6ff !important;
        border-radius: 8px !important;
    }

    .metric-row {
        display: flex;
        gap: 12px;
        margin-bottom: 12px;
    }

    .metric-box {
        flex: 1;
        background: rgba(0, 180, 255, 0.05);
        border: 1px solid rgba(0, 180, 255, 0.15);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }

    .metric-val {
        font-family: 'Orbitron', monospace;
        font-size: 1.3rem;
        color: #00b4ff;
        font-weight: 700;
    }

    .metric-name {
        font-size: 0.7rem;
        color: rgba(160, 190, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* Reset button — ghost style, separate from Predict */
    [data-testid="stButton"][key="reset_btn"] > button,
    button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid rgba(0, 180, 255, 0.35) !important;
        color: #00b4ff !important;
        padding: 6px 16px !important;
        font-size: 0.78rem !important;
        letter-spacing: 1px !important;
        border-radius: 8px !important;
        margin-top: 0 !important;
        width: auto !important;
    }
    button[kind="secondary"]:hover {
        background: rgba(0, 180, 255, 0.08) !important;
        box-shadow: 0 0 12px rgba(0, 180, 255, 0.2) !important;
        transform: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Training (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models():
    # Load the Pima Indians Diabetes dataset directly from URL
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    try:
        df = pd.read_csv(url, names=cols)
    except Exception:
        # Fallback: generate synthetic data if offline
        np.random.seed(42)
        n = 768
        df = pd.DataFrame({
            "Pregnancies": np.random.randint(0, 17, n),
            "Glucose": np.random.randint(70, 200, n),
            "BloodPressure": np.random.randint(40, 120, n),
            "SkinThickness": np.random.randint(0, 100, n),
            "Insulin": np.random.randint(0, 850, n),
            "BMI": np.round(np.random.uniform(18, 67, n), 1),
            "DiabetesPedigreeFunction": np.round(np.random.uniform(0.07, 2.4, n), 3),
            "Age": np.random.randint(21, 81, n),
            "Outcome": np.random.randint(0, 2, n),
        })

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }

    trained, accuracies = {}, {}
    for name, m in models.items():
        m.fit(X_train_sc, y_train)
        trained[name] = m
        accuracies[name] = round(accuracy_score(y_test, m.predict(X_test_sc)) * 100, 2)

    # Store feature names and correlation-based importance as KNN fallback
    feature_names = list(X.columns)
    corr_importance = np.abs(df[feature_names].corrwith(df["Outcome"])).values

    return trained, scaler, accuracies, feature_names, corr_importance


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">🤖 Model Selection</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Choose Algorithm",
        ["Logistic Regression", "Random Forest", "K-Nearest Neighbors"],
        label_visibility="collapsed"
    )

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem; color:rgba(160,190,255,0.7); line-height:1.7">
    This system uses the <b style="color:#00b4ff">Pima Indians Diabetes Dataset</b>
    to predict diabetes risk based on clinical measurements.<br><br>
    Three algorithms are trained and compared:<br>
    • Logistic Regression<br>
    • Random Forest<br>
    • K-Nearest Neighbors
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 Model Accuracy</div>', unsafe_allow_html=True)


# ─── MAIN ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🩺 DIABETES PREDICTION</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Clinical Risk Assessment</p>', unsafe_allow_html=True)

# Train models
with st.spinner("🔄 Initializing AI models..."):
    trained_models, scaler, accuracies, feature_names, corr_importance = train_models()

# Show accuracies in sidebar after training
with st.sidebar:
    for mname, acc in accuracies.items():
        icon = "🟢" if acc >= 78 else "🟡"
        active = "border: 1px solid #00b4ff;" if mname == selected_model else ""
        st.markdown(f"""
        <div style="background:rgba(0,180,255,0.05);{active}border-radius:8px;
             padding:8px 12px;margin-bottom:6px;font-size:0.8rem;">
            {icon} <b>{mname}</b><br>
            <span style="color:#00b4ff;font-weight:700">{acc}%</span>
            <span style="color:rgba(160,190,255,0.5)"> accuracy</span>
        </div>
        """, unsafe_allow_html=True)

# ─── Session State Defaults ──────────────────────────────────────────────────
DEFAULTS = {
    "pregnancies": 1, "glucose": 120, "blood_pressure": 70,
    "skin_thickness": 20, "insulin": 80, "bmi": 28.5,
    "dpf": 0.350, "age": 30,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_form():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# ─── Input Form ──────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)

# Header row: label left, Reset button right
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown('<div class="section-label" style="margin-bottom:0">📋 Patient Clinical Data</div>', unsafe_allow_html=True)
with hcol2:
    st.button("↺ Reset", on_click=reset_form, key="reset_btn",
              help="Reset all inputs to default values")

st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "🤱 Pregnancies", min_value=0, max_value=20, step=1,
        key="pregnancies", help="Number of times pregnant"
    )
    glucose = st.number_input(
        "🩸 Glucose (mg/dL)", min_value=0, max_value=300,
        key="glucose", help="Plasma glucose concentration (2-hour oral glucose tolerance test)"
    )
    blood_pressure = st.number_input(
        "💓 Blood Pressure (mmHg)", min_value=0, max_value=200,
        key="blood_pressure", help="Diastolic blood pressure"
    )
    skin_thickness = st.number_input(
        "📏 Skin Thickness (mm)", min_value=0, max_value=100,
        key="skin_thickness", help="Triceps skin fold thickness"
    )

with col2:
    insulin = st.number_input(
        "💉 Insulin (μU/mL)", min_value=0, max_value=900,
        key="insulin", help="2-Hour serum insulin"
    )
    bmi = st.number_input(
        "⚖️ BMI (kg/m²)", min_value=0.0, max_value=70.0, step=0.1,
        key="bmi", help="Body Mass Index"
    )
    dpf = st.number_input(
        "🧬 Diabetes Pedigree Function", min_value=0.0, max_value=3.0,
        step=0.001, format="%.3f",
        key="dpf", help="Diabetes heredity score based on family history"
    )
    age = st.number_input(
        "🎂 Age (years)", min_value=1, max_value=120,
        key="age", help="Patient age in years"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ─── Predict Button ───────────────────────────────────────────────────────────
predict_clicked = st.button("🔬 ANALYZE & PREDICT")

if predict_clicked:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    model = trained_models[selected_model]
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    diabetic_prob   = round(proba[1] * 100, 1)
    nondiabetic_prob = round(proba[0] * 100, 1)

    st.markdown("---")

    if prediction == 1:
        st.markdown(f"""
        <div class="result-diabetic">
            <div class="result-label" style="color:#ff5555">⚠️ DIABETIC RISK DETECTED</div>
            <div class="result-sub">The model indicates a high probability of diabetes.</div>
            <br>
            <div style="font-family:Orbitron,monospace;font-size:2rem;color:#ff5555;font-weight:700">
                {diabetic_prob}%
            </div>
            <div style="font-size:0.8rem;color:rgba(255,120,120,0.7)">Diabetes Probability</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-label" style="color:#00ff78">✅ NON-DIABETIC</div>
            <div class="result-sub">No significant diabetes risk detected.</div>
            <br>
            <div style="font-family:Orbitron,monospace;font-size:2rem;color:#00ff78;font-weight:700">
                {nondiabetic_prob}%
            </div>
            <div style="font-size:0.8rem;color:rgba(0,255,120,0.7)">Non-Diabetic Probability</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Breakdown Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 Prediction Breakdown</div>', unsafe_allow_html=True)

    bar_diabetic = int(diabetic_prob)
    bar_safe     = int(nondiabetic_prob)

    st.markdown(f"""
    <div style="margin-bottom:12px">
        <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px">
            <span>🔴 Diabetic</span><span style="color:#ff5555">{diabetic_prob}%</span>
        </div>
        <div style="background:rgba(255,255,255,0.06);border-radius:6px;height:10px;overflow:hidden">
            <div style="width:{bar_diabetic}%;background:linear-gradient(90deg,#cc0000,#ff5555);
                        height:100%;border-radius:6px;transition:all 0.5s"></div>
        </div>
    </div>
    <div>
        <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px">
            <span>🟢 Non-Diabetic</span><span style="color:#00ff78">{nondiabetic_prob}%</span>
        </div>
        <div style="background:rgba(255,255,255,0.06);border-radius:6px;height:10px;overflow:hidden">
            <div style="width:{bar_safe}%;background:linear-gradient(90deg,#009944,#00ff78);
                        height:100%;border-radius:6px;transition:all 0.5s"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <br>
    <div style="display:flex;gap:12px;font-size:0.8rem;color:rgba(160,190,255,0.6)">
        <span>🤖 Model: <b style="color:#00b4ff">{selected_model}</b></span>
        <span>|</span>
        <span>📈 Accuracy: <b style="color:#00b4ff">{accuracies[selected_model]}%</b></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ─── Risk Factors Chart ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">⚠️ Risk Factors — Feature Importance</div>', unsafe_allow_html=True)

    # Get importance scores based on model type
    if selected_model == "Random Forest":
        importances = trained_models[selected_model].feature_importances_
        method_note = "Source: Random Forest `feature_importances_` (Gini impurity reduction)"
    elif selected_model == "Logistic Regression":
        importances = np.abs(trained_models[selected_model].coef_[0])
        importances = importances / importances.sum()   # normalize to 0-1 range
        method_note = "Source: Logistic Regression `|coef_|` (absolute coefficient weight)"
    else:  # KNN — no native importance, use correlation
        importances = corr_importance / corr_importance.sum()
        method_note = "Source: Pearson correlation with Outcome (KNN has no native importance)"

    # Short display labels
    short_labels = ["Pregnancies", "Glucose", "Blood\nPressure", "Skin\nThickness",
                    "Insulin", "BMI", "Pedigree\nFunction", "Age"]

    # Sort descending
    sorted_idx = np.argsort(importances)[::-1]
    sorted_imp = importances[sorted_idx]
    sorted_labels = [short_labels[i] for i in sorted_idx]

    # Color: top 3 highlight, rest muted
    bar_colors = []
    for rank, _ in enumerate(sorted_imp):
        if rank == 0:
            bar_colors.append("#00b4ff")    # top feature — cyan
        elif rank == 1:
            bar_colors.append("#7b2fff")    # 2nd — purple
        elif rank == 2:
            bar_colors.append("#ff6b35")    # 3rd — orange
        else:
            bar_colors.append("#2a4a7a")

    # Build matplotlib figure (dark themed)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor("#0d1420")
    ax.set_facecolor("#0d1420")

    bars = ax.barh(
        sorted_labels[::-1],      # reverse so highest is at top
        sorted_imp[::-1],
        color=bar_colors[::-1],
        height=0.55,
        edgecolor="none",
    )

    # Subtle glow effect via twin bar (slightly wider, very transparent)
    ax.barh(
        sorted_labels[::-1],
        sorted_imp[::-1],
        color=bar_colors[::-1],
        height=0.72,
        edgecolor="none",
        alpha=0.12,
    )

    # Value labels on bars
    for bar, val in zip(bars, sorted_imp[::-1]):
        ax.text(
            val + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left",
            fontsize=8, color="#a0c8ff", fontweight="600"
        )

    # Styling
    ax.set_xlabel("Importance Score", color="#6080b0", fontsize=8, labelpad=8)
    ax.tick_params(axis="y", colors="#c0d8ff", labelsize=8.5)
    ax.tick_params(axis="x", colors="#506080", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1e3050")
    ax.spines["bottom"].set_color("#1e3050")
    ax.xaxis.grid(True, color="#1a2a40", linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)

    # Legend patches for top 3
    patches = [
        mpatches.Patch(color="#00b4ff", label=f"1st: {sorted_labels[0]}"),
        mpatches.Patch(color="#7b2fff", label=f"2nd: {sorted_labels[1]}"),
        mpatches.Patch(color="#ff6b35", label=f"3rd: {sorted_labels[2]}"),
    ]
    ax.legend(
        handles=patches, loc="lower right",
        fontsize=7.5, framealpha=0.15,
        labelcolor="#c0d8ff", edgecolor="#1e3050",
        facecolor="#0a1020"
    )

    plt.tight_layout(pad=1.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Method note + interview tip
    st.markdown(f"""
    <div style="font-size:0.72rem;color:rgba(120,160,220,0.55);margin-top:8px;line-height:1.6">
        📌 {method_note}
    </div>
    <div style="background:rgba(0,180,255,0.05);border:1px solid rgba(0,180,255,0.15);
         border-radius:10px;padding:12px 16px;margin-top:14px;font-size:0.8rem;
         color:rgba(160,200,255,0.8);line-height:1.7">
        🎓 <b style="color:#00b4ff">Interview Tip:</b>
        <b>Glucose</b> & <b>BMI</b> consistently rank highest across all three models —
        they have the strongest clinical correlation with diabetes onset.
        Random Forest importances are most reliable (model-native), while LR coefficients
        show linear contribution, and KNN uses correlation as a proxy since it has no
        built-in feature importance mechanism.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Medical disclaimer
    st.markdown("""
    <div style="background:rgba(255,200,0,0.05);border:1px solid rgba(255,200,0,0.2);
         border-radius:10px;padding:14px;margin-top:10px;font-size:0.78rem;
         color:rgba(255,220,100,0.75);text-align:center">
        ⚕️ <b>Disclaimer:</b> This tool is for educational purposes only.
        Always consult a qualified healthcare professional for medical advice.
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:0.72rem;color:rgba(100,130,180,0.4);
     letter-spacing:1px;padding-bottom:10px">
    DIABETES PREDICTION SYSTEM • BUILT WITH STREAMLIT + SCIKIT-LEARN
</div>
""", unsafe_allow_html=True)
