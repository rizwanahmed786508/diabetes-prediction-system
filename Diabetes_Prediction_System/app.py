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
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, html, body { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #F0F4F8 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #1a2332 !important;
}

/* Remove ALL Streamlit default spacing */
[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
[data-testid="stMainBlockContainer"] {
    padding: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
}

/* Kill ALL flex gaps Streamlit injects — including inline style overrides */
[data-testid="stVerticalBlock"] {
    gap: 0 !important;
    row-gap: 0 !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    padding: 0 !important;
    margin: 0 !important;
}
[data-testid="stHorizontalBlock"] {
    gap: 0 !important;
    padding: 0 !important;
    column-gap: 0 !important;
}
[data-testid="column"] {
    padding: 0 !important;
    min-width: 0 !important;
}
/* The actual flex children Streamlit wraps each widget in */
[data-testid="column"] > div { padding: 0 !important; gap: 0 !important; }
[data-testid="column"] > div > div { gap: 0 !important; row-gap: 0 !important; }

/* Streamlit wraps every widget in stElementContainer with margin */
[data-testid="stElementContainer"] {
    margin: 0 !important;
    padding: 0 !important;
}
/* stWidgetLabel gap */
[data-testid="stWidgetLabel"] { margin-bottom: 1px !important; }

/* Every direct child block gap = 0 */
.stMarkdown { margin: 0 !important; padding: 0 !important; }
.element-container { margin: 0 !important; padding: 0 !important; }

section[data-testid="stSidebar"] { display: none !important; }
footer, #MainMenu, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── TOP NAV ── */
.top-nav {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    width: 100%;
}
.nav-brand { display: flex; align-items: center; gap: 10px; }
.nav-icon { width: 34px; height: 34px; background: linear-gradient(135deg,#2563eb,#0ea5e9);
    border-radius: 9px; display:flex; align-items:center; justify-content:center; font-size:16px; flex-shrink:0; }
.nav-title { font-size: 1rem; font-weight: 700; color: #0f172a; line-height:1.2; }
.nav-sub { font-size: 0.68rem; color: #64748b; font-weight: 400; }
.nav-right { display:flex; align-items:center; gap:8px; }
.nav-badge { background:#f1f5f9; border:1px solid #e2e8f0; border-radius:20px;
    padding:4px 12px; font-size:0.72rem; color:#475569; font-weight:500; }
.nav-avatar { width:32px; height:32px; background:linear-gradient(135deg,#7c3aed,#a78bfa);
    border-radius:50%; display:flex; align-items:center; justify-content:center;
    font-size:13px; color:white; font-weight:700; flex-shrink:0; }

/* ── COLUMN PANELS ── */
.left-panel {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    padding: 16px 14px;
}
.mid-panel {
    background: #F0F4F8;
    padding: 16px 14px;
}
.right-panel {
    background: #ffffff;
    border-left: 1px solid #e2e8f0;
    padding: 16px 14px;
}
@media (max-width: 900px) {
    .left-panel, .mid-panel, .right-panel {
        border: none;
        border-bottom: 1px solid #e2e8f0;
        padding: 14px 12px;
    }
    .nav-badge { display: none; }
    .nav-title { font-size: 0.88rem; }
}

/* ── LAYOUT ── */
.dashboard { width:100%; }
.panel { padding: 18px 16px; }
.panel-left { background:#ffffff; border-right:1px solid #e2e8f0; }
.panel-mid  { background:#F0F4F8; }
.panel-right{ background:#ffffff; border-left:1px solid #e2e8f0; }

/* ── CARDS ── */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.card-header { font-size:0.82rem; font-weight:700; color:#0f172a; margin-bottom:2px; }
.card-sub { font-size:0.7rem; color:#94a3b8; margin-bottom:10px; }

/* ── PATIENT PANEL ── */
.patient-header { display:flex; align-items:center; gap:10px; margin-bottom:10px; padding-bottom:10px;
    border-bottom:1px solid #f1f5f9; }
.patient-icon { width:32px; height:32px; background:#eff6ff; border-radius:8px;
    display:flex; align-items:center; justify-content:center; font-size:15px; }
.patient-title { font-size:0.86rem; font-weight:700; color:#0f172a; }
.patient-sub { font-size:0.67rem; color:#94a3b8; }

/* ── Number inputs — zero all gaps ── */
div[data-testid="stNumberInput"] {
    margin: 0 !important;
    padding: 0 !important;
}
div[data-testid="stNumberInput"] > div {
    margin: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
}
div[data-testid="stNumberInput"] p {
    margin: 0 0 1px 0 !important;
    padding: 0 !important;
    font-size: 0.73rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
    line-height: 1.2 !important;
}
div[data-testid="stNumberInput"] label {
    margin: 0 !important;
    padding: 0 !important;
}
div[data-testid="stNumberInput"] label p {
    font-size: 0.73rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
    margin-bottom: 1px !important;
    line-height: 1.2 !important;
}
div[data-testid="stNumberInput"] input {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 7px !important;
    color: #0f172a !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 4px 8px !important;
    height: 30px !important;
    margin: 0 !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.1) !important;
    outline: none !important;
}
/* The wrapper div around the actual input+stepper */
div[data-testid="stNumberInput"] > div > div {
    margin: 0 !important; padding: 0 !important; gap: 0 !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    padding: 12px 20px !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
    margin-top: 8px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Reset button ── */
button[kind="secondary"] {
    background: #f1f5f9 !important;
    border: 1.5px solid #e2e8f0 !important;
    color: #475569 !important;
    padding: 7px 14px !important;
    font-size: 0.78rem !important;
    border-radius: 8px !important;
    margin-top: 0 !important;
    width: auto !important;
    box-shadow: none !important;
}
button[kind="secondary"]:hover {
    background: #e2e8f0 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    width: 100% !important;
    background: #f0fdf4 !important;
    color: #15803d !important;
    border: 1.5px solid #bbf7d0 !important;
    padding: 10px 16px !important;
    border-radius: 9px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    margin-top: 0 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #dcfce7 !important;
    transform: none !important;
}

/* ── Select box ── */
div[data-testid="stSelectbox"] label { font-size:0.78rem !important; color:#475569 !important; font-weight:500 !important; }
div[data-testid="stSelectbox"] > div > div {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    color: #0f172a !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}

/* ── Text area ── */
div[data-testid="stTextArea"] textarea {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    font-size: 0.82rem !important;
}
div[data-testid="stTextArea"] label { font-size:0.78rem !important; color:#475569 !important; }

/* ── Risk result cards ── */
.risk-high {
    background: linear-gradient(135deg,#fef2f2,#fff1f1);
    border: 1.5px solid #fca5a5;
    border-radius: 12px; padding: 16px 18px; text-align:center;
    animation: glow-red 2.5s ease-in-out infinite;
}
.risk-low {
    background: linear-gradient(135deg,#f0fdf4,#ecfdf5);
    border: 1.5px solid #86efac;
    border-radius: 12px; padding: 16px 18px; text-align:center;
    animation: glow-green 2.5s ease-in-out infinite;
}
@keyframes glow-red {
    0%,100% { box-shadow: 0 0 0 rgba(239,68,68,0.15); }
    50% { box-shadow: 0 0 18px rgba(239,68,68,0.2); }
}
@keyframes glow-green {
    0%,100% { box-shadow: 0 0 0 rgba(34,197,94,0.12); }
    50% { box-shadow: 0 0 18px rgba(34,197,94,0.18); }
}

/* ── SHAP rows ── */
.shap-row {
    display:flex; align-items:center; justify-content:space-between;
    padding: 8px 0; border-bottom: 1px solid #f1f5f9;
    font-size: 0.82rem;
}
.shap-bar-pos { height:6px; background:linear-gradient(90deg,#f97316,#ef4444); border-radius:3px; }
.shap-bar-neg { height:6px; background:linear-gradient(90deg,#22c55e,#10b981); border-radius:3px; }

/* ── Donut legend ── */
.legend-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:6px; }

/* ── Recent table ── */
.rec-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
.rec-table th { background:#f8fafc; color:#64748b; font-weight:600; font-size:0.72rem;
    text-transform:uppercase; letter-spacing:0.5px; padding:8px 12px; text-align:left;
    border-bottom:1px solid #e2e8f0; }
.rec-table td { padding:9px 12px; border-bottom:1px solid #f1f5f9; color:#334155; }
.rec-table tr:last-child td { border-bottom:none; }
.badge-high { background:#fef2f2; color:#dc2626; border:1px solid #fca5a5;
    border-radius:20px; padding:2px 10px; font-size:0.7rem; font-weight:600; white-space:nowrap; }
.badge-mod  { background:#fffbeb; color:#d97706; border:1px solid #fcd34d;
    border-radius:20px; padding:2px 10px; font-size:0.7rem; font-weight:600; white-space:nowrap; }
.badge-low  { background:#f0fdf4; color:#16a34a; border:1px solid #86efac;
    border-radius:20px; padding:2px 10px; font-size:0.7rem; font-weight:600; white-space:nowrap; }

/* ── Quick action buttons ── */
.qa-btn {
    display:flex; align-items:center; gap:10px;
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:9px;
    padding:10px 14px; margin-bottom:8px; cursor:pointer;
    font-size:0.82rem; color:#334155; font-weight:500;
    transition: all 0.15s;
}
.qa-btn:hover { background:#eff6ff; border-color:#bfdbfe; color:#1d4ed8; }
.qa-icon { font-size:15px; }

/* ── Note box ── */
.note-box {
    background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
    padding:12px 14px; display:flex; gap:10px; align-items:flex-start;
    font-size:0.78rem; color:#166534; margin-top:12px;
}

/* ── Model selector ── */
.model-pill {
    display:inline-block; background:#eff6ff; border:1px solid #bfdbfe;
    border-radius:20px; padding:3px 12px; font-size:0.72rem; color:#1d4ed8;
    font-weight:600; margin-bottom:10px;
}

/* ── Accuracy mini cards ── */
.acc-row { display:flex; gap:8px; margin-bottom:14px; }
.acc-card { flex:1; background:#f8fafc; border:1px solid #e2e8f0; border-radius:9px;
    padding:8px; text-align:center; }
.acc-val { font-size:0.95rem; font-weight:700; color:#0f172a; }
.acc-label { font-size:0.62rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px; }

/* Mobile: force Streamlit columns to stack vertically */
@media screen and (max-width: 768px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        flex-wrap: wrap !important;
    }
    [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    /* Hide right panel on mobile to save space */
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) .right-panel {
        display: none;
    }
}

/* spinner override */
[data-testid="stSpinner"] { display:none !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model Training ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    try:
        df = pd.read_csv(url, names=cols)
    except Exception:
        np.random.seed(42); n=768
        df = pd.DataFrame({
            "Pregnancies":np.random.randint(0,17,n), "Glucose":np.random.randint(70,200,n),
            "BloodPressure":np.random.randint(40,120,n), "SkinThickness":np.random.randint(0,100,n),
            "Insulin":np.random.randint(0,850,n), "BMI":np.round(np.random.uniform(18,67,n),1),
            "DiabetesPedigreeFunction":np.round(np.random.uniform(0.07,2.4,n),3),
            "Age":np.random.randint(21,81,n), "Outcome":np.random.randint(0,2,n),
        })
    X = df.drop("Outcome", axis=1); y = df["Outcome"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train); Xs = scaler.transform(X_test)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100,random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }
    trained, accuracies = {}, {}
    for name, m in models.items():
        m.fit(Xt, y_train)
        trained[name] = m
        accuracies[name] = round(accuracy_score(y_test, m.predict(Xs))*100, 2)
    feature_names = list(X.columns)
    corr_importance = np.abs(df[feature_names].corrwith(df["Outcome"])).values
    return trained, scaler, accuracies, feature_names, corr_importance

trained_models, scaler, accuracies, feature_names, corr_importance = train_models()

# ─── Session State ────────────────────────────────────────────────────────────
DEFAULTS = {"pregnancies":0,"glucose":0,"blood_pressure":0,"skin_thickness":0,
            "insulin":0,"bmi":0.0,"dpf":0.000,"age":0}
for k,v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

if "history" not in st.session_state: st.session_state.history = []
if "last_result" not in st.session_state: st.session_state.last_result = None

def reset_form():
    for k,v in DEFAULTS.items(): st.session_state[k] = v
    st.session_state.last_result = None
    st.rerun()

# ─── PDF Generator ────────────────────────────────────────────────────────────
def generate_pdf(patient_data, prediction, diabetic_prob, nondiabetic_prob,
                 selected_model, model_accuracy, importances, sorted_labels, sorted_imp):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.8*cm, bottomMargin=1.8*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    C_DARK=colors.HexColor("#0d1420"); C_BLUE=colors.HexColor("#2563eb")
    C_GREEN=colors.HexColor("#16a34a"); C_RED=colors.HexColor("#dc2626")
    C_TEXT=colors.HexColor("#1a2332"); C_MUTED=colors.HexColor("#64748b")
    C_BORDER=colors.HexColor("#e2e8f0"); C_LBLUE=colors.HexColor("#f0f9ff")
    def S(name,**kw):
        d=dict(fontName="Helvetica",fontSize=10,textColor=C_TEXT,leading=14); d.update(kw)
        return ParagraphStyle(name,**d)
    sTitle=S("T",fontName="Helvetica-Bold",fontSize=22,textColor=C_DARK,alignment=TA_CENTER,leading=28,spaceAfter=2)
    sSub=S("Su",fontSize=9,textColor=C_MUTED,alignment=TA_CENTER,leading=12)
    sSecHdr=S("SH",fontName="Helvetica-Bold",fontSize=9,textColor=C_BLUE,leading=14,spaceBefore=4,spaceAfter=2)
    W=A4[0]-4*cm; story=[]
    story.append(Paragraph("DIABETES RISK PREDICTION REPORT",sTitle))
    story.append(Paragraph("AI-Powered Clinical Risk Assessment",sSub))
    story.append(Spacer(1,6))
    now=datetime.now()
    story.append(Paragraph(f"Generated: {now.strftime('%B %d, %Y')} &nbsp;|&nbsp; Time: {now.strftime('%H:%M:%S')} &nbsp;|&nbsp; Model: {selected_model}",
        S("dt",fontSize=8,textColor=C_MUTED,alignment=TA_CENTER,leading=12)))
    story.append(Spacer(1,10))
    story.append(HRFlowable(width=W,thickness=2,color=C_BLUE,spaceAfter=14))
    story.append(Paragraph("PATIENT CLINICAL DATA",sSecHdr))
    labels=["Pregnancies","Glucose (mg/dL)","Blood Pressure (mmHg)","Skin Thickness (mm)",
            "Insulin (uU/mL)","BMI (kg/m2)","Diabetes Pedigree Function","Age (years)"]
    values=[str(v) for v in patient_data]
    td=[["Parameter","Value","Parameter","Value"]]
    for i in range(0,8,2): td.append([labels[i],values[i],labels[i+1],values[i+1]])
    tbl=Table(td,colWidths=[W*.30,W*.18,W*.30,W*.18],repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),C_DARK),("TEXTCOLOR",(0,0),(-1,0),C_BLUE),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),8),
        ("ALIGN",(0,0),(-1,0),"CENTER"),("TOPPADDING",(0,0),(-1,0),5),("BOTTOMPADDING",(0,0),(-1,0),5),
        ("FONTSIZE",(0,1),(-1,-1),8.5),("FONTNAME",(1,1),(1,-1),"Helvetica-Bold"),
        ("FONTNAME",(3,1),(3,-1),"Helvetica-Bold"),("ALIGN",(1,0),(1,-1),"CENTER"),
        ("ALIGN",(3,0),(3,-1),"CENTER"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,C_LBLUE]),
        ("GRID",(0,0),(-1,-1),0.5,C_BORDER),("TOPPADDING",(0,1),(-1,-1),5),
        ("BOTTOMPADDING",(0,1),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),8),
    ]))
    story.append(tbl); story.append(Spacer(1,14))
    story.append(HRFlowable(width=W,thickness=0.5,color=C_BORDER,spaceAfter=10))
    story.append(Paragraph("PREDICTION RESULT",sSecHdr))
    if prediction==1:
        res_label="DIABETIC RISK DETECTED"; res_conf=f"{diabetic_prob}%"
        res_sub=f"Non-Diabetic Probability: {nondiabetic_prob}%"
        bg_color=colors.HexColor("#fef2f2"); txt_color=C_RED; border_c=C_RED
    else:
    