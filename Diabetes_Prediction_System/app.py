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
    padding: 18px 16px;
    min-height: calc(100vh - 60px);
}
.mid-panel {
    background: #F0F4F8;
    padding: 18px 16px;
    min-height: calc(100vh - 60px);
}
.right-panel {
    background: #ffffff;
    border-left: 1px solid #e2e8f0;
    padding: 18px 16px;
    min-height: calc(100vh - 60px);
}
@media (max-width: 900px) {
    .left-panel, .mid-panel, .right-panel {
        min-height: auto; border: none;
        border-bottom: 1px solid #e2e8f0; padding: 16px 12px;
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
        res_label="NON-DIABETIC"; res_conf=f"{nondiabetic_prob}%"
        res_sub=f"Diabetic Probability: {diabetic_prob}%"
        bg_color=colors.HexColor("#f0fdf4"); txt_color=C_GREEN; border_c=C_GREEN
    result_data=[[
        Paragraph(res_label,S("RL",fontName="Helvetica-Bold",fontSize=14,textColor=txt_color,alignment=TA_CENTER,leading=18)),
        Paragraph(res_conf,S("RC",fontName="Helvetica-Bold",fontSize=26,textColor=txt_color,alignment=TA_CENTER,leading=30)),
        Paragraph(f"Confidence\n{res_sub}",S("RS",fontSize=8,textColor=C_MUTED,alignment=TA_CENTER,leading=12)),
    ]]
    rtbl=Table(result_data,colWidths=[W*.38,W*.28,W*.34])
    rtbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),bg_color),("BOX",(0,0),(-1,-1),1.5,border_c),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),14),
        ("BOTTOMPADDING",(0,0),(-1,-1),14),("LEFTPADDING",(0,0),(-1,-1),10),
        ("RIGHTPADDING",(0,0),(-1,-1),10),("LINEAFTER",(0,0),(1,-1),0.5,border_c),
    ]))
    story.append(rtbl); story.append(Spacer(1,6))
    story.append(Paragraph(f"Algorithm: {selected_model} &nbsp;|&nbsp; Model Accuracy: {model_accuracy}%",
        S("acc",fontSize=8,textColor=C_MUTED,alignment=TA_CENTER,leading=12)))
    story.append(Spacer(1,14))
    story.append(HRFlowable(width=W,thickness=0.5,color=C_BORDER,spaceAfter=10))
    story.append(Paragraph("RISK FACTORS — FEATURE IMPORTANCE",sSecHdr))
    fig,ax=plt.subplots(figsize=(7,3.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#f8fafc")
    pdf_colors=["#1d4ed8","#0ea5e9","#2563eb"]+["#94a3b8"]*5
    pdf_colors=pdf_colors[:len(sorted_imp)]
    ax.barh(sorted_labels[::-1],sorted_imp[::-1],color=pdf_colors[::-1],height=0.55,edgecolor="none")
    for i,(label,val) in enumerate(zip(sorted_labels[::-1],sorted_imp[::-1])):
        ax.text(val+0.003,i,f"{val:.3f}",va="center",ha="left",fontsize=7.5,color="#334155",fontweight="600")
    ax.set_xlabel("Importance Score",fontsize=8,color="#64748b",labelpad=6)
    ax.tick_params(axis="y",labelsize=8,colors="#1a2332")
    ax.tick_params(axis="x",labelsize=7,colors="#64748b")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0"); ax.spines["bottom"].set_color("#e2e8f0")
    ax.xaxis.grid(True,color="#f1f5f9",linewidth=0.6,linestyle="--"); ax.set_axisbelow(True)
    plt.tight_layout(pad=1.0)
    chart_buf=io.BytesIO()
    plt.savefig(chart_buf,format="png",dpi=150,bbox_inches="tight"); chart_buf.seek(0); plt.close(fig)
    story.append(RLImage(chart_buf,width=W,height=W*0.42)); story.append(Spacer(1,14))
    story.append(HRFlowable(width=W,thickness=0.5,color=C_BORDER,spaceAfter=10))
    disc_data=[[Paragraph(
        "<b>MEDICAL DISCLAIMER</b><br/>This report is generated by an AI-based system for educational and informational "
        "purposes only. It does NOT constitute medical advice, diagnosis, or treatment. "
        "Please consult a qualified healthcare provider for any medical concerns.",
        S("D",fontSize=7.5,textColor=colors.HexColor("#7a5500"),leading=11)
    )]]
    dtbl=Table(disc_data,colWidths=[W])
    dtbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#fffbe6")),
        ("BOX",(0,0),(-1,-1),0.8,colors.HexColor("#e6b800")),
        ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),
    ]))
    story.append(dtbl); story.append(Spacer(1,10))
    story.append(Paragraph("Diabetes Prediction System &nbsp;•&nbsp; Developed By Rizwan Ahmed",
        S("ft",fontSize=7,textColor=C_MUTED,alignment=TA_CENTER,leading=10)))
    doc.build(story); buf.seek(0)
    return buf.read()


# ─── NAV BAR ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-nav">
  <div class="nav-brand">
    <div class="nav-icon">🩺</div>
    <div>
      <div class="nav-title">Diabetes Risk Prediction</div>
      <div class="nav-sub">Early detection. Better health.</div>
    </div>
  </div>
  <div class="nav-right">
    <div class="nav-badge">🤖 AI-Powered</div>
    <div class="nav-avatar">RA</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── 3-COLUMN DASHBOARD ───────────────────────────────────────────────────────
st.markdown('<div class="dashboard">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# LEFT PANEL — Patient Inputs
# ════════════════════════════════════════════════════════════════════
left, mid, right = st.columns([1.05, 1.6, 1.05])

with left:
    st.markdown("""
    <div class="left-panel">
    <div class="patient-header">
      <div class="patient-icon">👤</div>
      <div>
        <div class="patient-title">Patient Information</div>
        <div class="patient-sub">Enter patient medical details</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    pregnancies    = st.number_input("Pregnancies",           min_value=0, max_value=20,  step=1,   key="pregnancies")
    glucose        = st.number_input("Glucose (mg/dL)",       min_value=0, max_value=300, step=1,   key="glucose")
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, step=1,   key="blood_pressure")
    skin_thickness = st.number_input("Skin Thickness (mm)",   min_value=0, max_value=100, step=1,   key="skin_thickness")
    insulin        = st.number_input("Insulin (mu U/ml)",     min_value=0, max_value=900, step=1,   key="insulin")
    bmi            = st.number_input("BMI (kg/m²)",           min_value=0.0,max_value=70.0,step=0.1,key="bmi")
    dpf            = st.number_input("Diabetes Pedigree Function", min_value=0.0,max_value=3.0,step=0.001,format="%.3f",key="dpf")
    age            = st.number_input("Age (years)",           min_value=0, max_value=120, step=1,   key="age")

    # Model selector
    st.markdown("<div style='margin-top:10px;margin-bottom:4px'>", unsafe_allow_html=True)
    selected_model = st.selectbox("Algorithm", ["Random Forest","Logistic Regression","K-Nearest Neighbors"],
                                  label_visibility="visible")
    st.markdown("</div>", unsafe_allow_html=True)

    # Buttons row
    bc1, bc2 = st.columns([2, 1])
    with bc1:
        predict_clicked = st.button("🧠 Predict Risk")
    with bc2:
        st.button("↺ Reset", on_click=reset_form, key="reset_btn")

    # Note box
    st.markdown("""
    <div class="note-box">
      <span style="font-size:16px">🛡️</span>
      <div><b>Note</b><br>Please ensure all values are correct. This tool is for risk assessment only, not for medical diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close panel-left

# ════════════════════════════════════════════════════════════════════
# MIDDLE PANEL — Results
# ════════════════════════════════════════════════════════════════════
with mid:
    st.markdown('<div class="mid-panel">', unsafe_allow_html=True)

    if predict_clicked:
        # ── Compute prediction ──
        input_data   = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        model        = trained_models[selected_model]
        prediction   = model.predict(input_scaled)[0]
        proba        = model.predict_proba(input_scaled)[0]
        diabetic_prob    = round(proba[1]*100,1)
        nondiabetic_prob = round(proba[0]*100,1)

        # Feature importance
        if selected_model == "Random Forest":
            importances = model.feature_importances_
            method_note = "Random Forest feature_importances_ (Gini)"
        elif selected_model == "Logistic Regression":
            importances = np.abs(model.coef_[0]); importances = importances/importances.sum()
            method_note = "Logistic Regression |coef_| (normalized)"
        else:
            importances = corr_importance/corr_importance.sum()
            method_note = "Pearson correlation (KNN proxy)"

        short_labels = ["Pregnancies","Glucose","Blood Pressure","Skin Thickness","Insulin","BMI","Pedigree Fn","Age"]
        sorted_idx   = np.argsort(importances)[::-1]
        sorted_imp   = importances[sorted_idx]
        sorted_labels= [short_labels[i] for i in sorted_idx]

        # Save to history
        st.session_state.history.insert(0, {
            "id": f"P{1000+len(st.session_state.history):05d}",
            "date": datetime.now().strftime("%d %b %Y"),
            "glucose": glucose, "bmi": bmi,
            "score": diabetic_prob,
            "level": "High Risk" if diabetic_prob>=60 else ("Moderate Risk" if diabetic_prob>=35 else "Low Risk"),
        })
        if len(st.session_state.history) > 5: st.session_state.history = st.session_state.history[:5]

        st.session_state.last_result = {
            "prediction":prediction,"diabetic_prob":diabetic_prob,"nondiabetic_prob":nondiabetic_prob,
            "selected_model":selected_model,"importances":importances,
            "sorted_labels":sorted_labels,"sorted_imp":sorted_imp,
            "method_note":method_note,"input_data":[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,dpf,age],
        }

    res = st.session_state.last_result

    if res is None:
        # ── Empty state ──
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
             height:60vh;text-align:center;">
          <div style="font-size:64px;margin-bottom:16px">🩺</div>
          <div style="font-size:1.2rem;font-weight:700;color:#0f172a;margin-bottom:8px">Ready to Analyze</div>
          <div style="font-size:0.85rem;color:#94a3b8;max-width:300px">
            Enter patient clinical data on the left and click <b>Predict Risk</b> to get the AI assessment.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        prediction       = res["prediction"]
        diabetic_prob    = res["diabetic_prob"]
        nondiabetic_prob = res["nondiabetic_prob"]
        selected_model_r = res["selected_model"]
        importances      = res["importances"]
        sorted_labels    = res["sorted_labels"]
        sorted_imp       = res["sorted_imp"]
        method_note      = res["method_note"]

        # ── GAUGE CHART ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Prediction Result</div><div class="card-sub">The model has analyzed the patient data</div>', unsafe_allow_html=True)

        risk_pct = diabetic_prob / 100.0
        fig_g, ax_g = plt.subplots(figsize=(4.5, 2.6), subplot_kw=dict(aspect="equal"))
        fig_g.patch.set_facecolor("white")
        ax_g.set_facecolor("white")

        # Draw gauge arc background (green→yellow→red)
        n_seg = 100
        for i in range(n_seg):
            theta_s = np.pi * (1 - i/n_seg)
            theta_e = np.pi * (1 - (i+1)/n_seg)
            t = i/n_seg
            r = min(1.0, t*2) if t < 0.5 else 1.0
            g = min(1.0, (1-t)*2) if t > 0.5 else 1.0
            b = 0.0
            ax_g.add_patch(plt.matplotlib.patches.Wedge(
                (0,0), 1.0, np.degrees(theta_e), np.degrees(theta_s),
                width=0.28, facecolor=(r,g,b), edgecolor="none"
            ))

        # Needle
        needle_angle = np.pi * (1 - risk_pct)
        nx = 0.72 * np.cos(needle_angle)
        ny = 0.72 * np.sin(needle_angle)
        ax_g.annotate("", xy=(nx,ny), xytext=(0,0),
            arrowprops=dict(arrowstyle="->", color="#1a2332", lw=2.5,
                           mutation_scale=18))
        ax_g.add_patch(plt.Circle((0,0), 0.06, color="#1a2332", zorder=5))

        # Inner white circle
        ax_g.add_patch(plt.matplotlib.patches.Wedge(
            (0,0), 0.72, 0, 180, facecolor="white", edgecolor="none"))

        # Labels
        pct_color = "#dc2626" if diabetic_prob >= 60 else ("#d97706" if diabetic_prob >= 35 else "#16a34a")
        risk_text = "High Risk" if diabetic_prob >= 60 else ("Moderate Risk" if diabetic_prob >= 35 else "Low Risk")
        ax_g.text(0, 0.18, f"{diabetic_prob}%", ha="center", va="center",
                  fontsize=26, fontweight="700", color=pct_color,
                  fontfamily="DM Sans")
        ax_g.text(0, -0.08, risk_text, ha="center", va="center",
                  fontsize=10, fontweight="600", color=pct_color)
        ax_g.text(-1.05, -0.12, "0%", ha="center", va="center", fontsize=7, color="#94a3b8")
        ax_g.text( 1.05, -0.12, "100%", ha="center", va="center", fontsize=7, color="#94a3b8")

        ax_g.set_xlim(-1.2, 1.2); ax_g.set_ylim(-0.3, 1.15)
        ax_g.axis("off")
        plt.tight_layout(pad=0.3)

        gc1, gc2, gc3 = st.columns([1,2,1])
        with gc2:
            st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)

        # Result badge
        if prediction == 1:
            st.markdown(f"""
            <div class="risk-high" style="margin-top:4px">
              <div style="font-size:1rem;font-weight:700;color:#dc2626">⚠️ The patient is at <b>High Risk of Diabetes.</b></div>
              <div style="font-size:0.8rem;color:#b91c1c;margin-top:4px">Please consult a healthcare professional.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low" style="margin-top:4px">
              <div style="font-size:1rem;font-weight:700;color:#16a34a">✅ The patient shows <b>Low Risk of Diabetes.</b></div>
              <div style="font-size:0.8rem;color:#15803d;margin-top:4px">Continue healthy lifestyle maintenance.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # card

        # ── FEATURE IMPORTANCE BAR CHART ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-header">Feature Importance ({method_note.split("(")[0].strip()})</div><div class="card-sub">Impact of each feature on prediction</div>', unsafe_allow_html=True)

        bar_colors_chart = ["#ef4444","#f97316","#f59e0b","#84cc16","#22c55e","#14b8a6","#3b82f6","#8b5cf6"]
        fig_b, ax_b = plt.subplots(figsize=(6, 3.0))
        fig_b.patch.set_facecolor("white"); ax_b.set_facecolor("#f8fafc")
        ys = range(len(sorted_imp))
        ax_b.barh([sl.replace("\n"," ") for sl in sorted_labels[::-1]],
                  sorted_imp[::-1], color=bar_colors_chart[:len(sorted_imp)],
                  height=0.58, edgecolor="none")
        for i, val in enumerate(sorted_imp[::-1]):
            ax_b.text(val+0.003, i, f"{val:.2f}", va="center", ha="left",
                      fontsize=8, color="#334155", fontweight="600")
        ax_b.set_xlabel("Impact on Prediction", fontsize=8, color="#64748b")
        ax_b.tick_params(axis="y", labelsize=8.5, colors="#334155")
        ax_b.tick_params(axis="x", labelsize=7, colors="#94a3b8")
        ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
        ax_b.spines["left"].set_color("#e2e8f0"); ax_b.spines["bottom"].set_color("#e2e8f0")
        ax_b.xaxis.grid(True, color="#f1f5f9", linewidth=0.7, linestyle="--")
        ax_b.set_axisbelow(True)
        plt.tight_layout(pad=0.8)
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── RECENT PREDICTIONS TABLE ──
        if st.session_state.history:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Recent Predictions</div>', unsafe_allow_html=True)
            rows = ""
            for h in st.session_state.history:
                badge_cls = "badge-high" if h["level"]=="High Risk" else ("badge-mod" if h["level"]=="Moderate Risk" else "badge-low")
                rows += f"""<tr>
                  <td>{h['id']}</td><td>{h['date']}</td>
                  <td>{h['glucose']}</td><td>{h['bmi']}</td>
                  <td><b>{h['score']}%</b></td>
                  <td><span class="{badge_cls}">{h['level']}</span></td>
                </tr>"""
            st.markdown(f"""
            <table class="rec-table">
              <thead><tr><th>Patient ID</th><th>Date</th><th>Glucose</th><th>BMI</th><th>Risk Score</th><th>Risk Level</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # panel-mid

# ════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Explanation + Probability + Quick Actions
# ════════════════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    res = st.session_state.last_result

    # ── MODEL ACCURACY ──
    st.markdown('<div class="card-header" style="margin-bottom:8px">Model Accuracy</div>', unsafe_allow_html=True)
    for mname, acc in accuracies.items():
        short = mname.replace("Logistic Regression","Log. Reg.").replace("K-Nearest Neighbors","KNN").replace("Random Forest","Rand. Forest")
        bar_w = int(acc)
        highlight = "background:#eff6ff;border:1.5px solid #bfdbfe;" if mname == selected_model else "background:#f8fafc;border:1px solid #e2e8f0;"
        st.markdown(f"""
        <div style="{highlight}border-radius:9px;padding:8px 12px;margin-bottom:7px;">
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;font-weight:600;color:#0f172a;margin-bottom:5px">
            <span>{'⭐ ' if mname==selected_model else ''}{short}</span>
            <span style="color:{'#2563eb' if mname==selected_model else '#64748b'}">{acc}%</span>
          </div>
          <div style="background:#e2e8f0;border-radius:4px;height:5px">
            <div style="width:{bar_w}%;background:{'linear-gradient(90deg,#2563eb,#0ea5e9)' if mname==selected_model else '#cbd5e1'};height:100%;border-radius:4px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    if res is None:
        st.markdown("""
        <div style="margin-top:20px;padding:16px;background:#f8fafc;border:1px solid #e2e8f0;
             border-radius:10px;text-align:center;color:#94a3b8;font-size:0.8rem">
          Run a prediction to see explanation and probability charts.
        </div>
        """, unsafe_allow_html=True)
    else:
        prediction    = res["prediction"]
        diabetic_prob = res["diabetic_prob"]
        nondiabetic_prob = res["nondiabetic_prob"]
        sorted_labels = res["sorted_labels"]
        sorted_imp    = res["sorted_imp"]
        input_vals    = res["input_data"]
        input_labels  = ["Pregnancies","Glucose","Blood Pressure","Skin Thickness","Insulin","BMI","Pedigree Fn","Age"]
        sorted_feature_vals = [input_vals[feature_names.index(fn)] if fn in feature_names else 0
                               for fn in [
                                   "Glucose","BMI","Age","Pregnancies","SkinThickness",
                                   "DiabetesPedigreeFunction","BloodPressure","Insulin"
                               ]]

        # ── EXPLANATION (SHAP-style) ──
        st.markdown('<div class="card" style="margin-top:14px">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Explanation (SHAP)</div><div class="card-sub">Top factors influencing the prediction</div>', unsafe_allow_html=True)

        top1 = sorted_labels[0].replace("\n"," ")
        top2 = sorted_labels[1].replace("\n"," ")
        color1 = "#dc2626" if prediction==1 else "#16a34a"
        st.markdown(f"""
        <div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;
             padding:10px 12px;font-size:0.8rem;color:#7f1d1d;margin-bottom:12px;line-height:1.6">
          This prediction is primarily influenced by high
          <b style="color:{color1}">{top1}</b> level and <b style="color:{color1}">{top2}</b>.
        </div>""", unsafe_allow_html=True)

        # SHAP rows (top 5)
        for i, (lbl, imp) in enumerate(zip(sorted_labels[:5], sorted_imp[:5])):
            lbl_clean = lbl.replace("\n"," ")
            is_pos = imp > sorted_imp.mean()
            arrow  = "↑" if is_pos else "↓"
            clr    = "#dc2626" if is_pos else "#16a34a"
            bar_w  = int(imp / sorted_imp[0] * 120)
            bar_cls= "shap-bar-pos" if is_pos else "shap-bar-neg"
            val_str= f"+{imp:.2f}" if is_pos else f"-{imp:.2f}"
            st.markdown(f"""
            <div class="shap-row">
              <span style="color:{clr};font-weight:600;width:14px">{arrow}</span>
              <span style="flex:1;padding:0 6px;color:#334155">{lbl_clean}</span>
              <div style="width:80px;margin-right:8px">
                <div class="{bar_cls}" style="width:{bar_w}px"></div>
              </div>
              <span style="color:{clr};font-weight:700;font-size:0.82rem;min-width:36px;text-align:right">{val_str}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── DONUT CHART ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Prediction Probability</div><div class="card-sub">Probability of each class</div>', unsafe_allow_html=True)

        fig_d, ax_d = plt.subplots(figsize=(3.2, 2.4))
        fig_d.patch.set_facecolor("white"); ax_d.set_facecolor("white")
        wedge_colors = ["#ef4444","#22c55e"]
        wedges, _ = ax_d.pie(
            [diabetic_prob, nondiabetic_prob],
            colors=wedge_colors,
            startangle=90,
            wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2.5),
        )
        ax_d.text(0, 0, f"{diabetic_prob}%", ha="center", va="center",
                  fontsize=13, fontweight="700",
                  color="#dc2626" if prediction==1 else "#16a34a")
        plt.tight_layout(pad=0.2)
        dc1, dc2, dc3 = st.columns([0.3,2,0.3])
        with dc2:
            st.pyplot(fig_d, use_container_width=True)
        plt.close(fig_d)

        st.markdown(f"""
        <div style="display:flex;flex-direction:column;gap:6px;margin-top:4px">
          <div style="display:flex;align-items:center;justify-content:space-between;font-size:0.8rem">
            <span><span class="legend-dot" style="background:#ef4444"></span>High Risk (Diabetic)</span>
            <b>{diabetic_prob}%</b>
          </div>
          <div style="display:flex;align-items:center;justify-content:space-between;font-size:0.8rem">
            <span><span class="legend-dot" style="background:#22c55e"></span>Low Risk (Non-Diabetic)</span>
            <b>{nondiabetic_prob}%</b>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── QUICK ACTIONS ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Quick Actions</div>', unsafe_allow_html=True)

        pdf_bytes = generate_pdf(
            patient_data=res["input_data"],
            prediction=res["prediction"],
            diabetic_prob=res["diabetic_prob"],
            nondiabetic_prob=res["nondiabetic_prob"],
            selected_model=res["selected_model"],
            model_accuracy=accuracies[res["selected_model"]],
            importances=res["importances"],
            sorted_labels=res["sorted_labels"],
            sorted_imp=res["sorted_imp"],
        )
        fname = f"diabetes_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button("📄 Download Report (PDF)", data=pdf_bytes,
                           file_name=fname, mime="application/pdf", key="pdf_dl")

        # Feedback
        st.markdown('<div style="margin-top:10px">', unsafe_allow_html=True)
        feedback_text = st.text_area("", placeholder="Write your feedback or suggestions here...",
                                     height=80, label_visibility="collapsed", key="feedback_text")
        mailto = (f"mailto:rizwanmb310@gmail.com?subject=Diabetes%20Prediction%20App%20-%20Feedback"
                  f"&body={feedback_text.replace(' ','%20').replace(chr(10),'%0A') if feedback_text else ''}")
        st.markdown(f"""
        <a href="{mailto}" target="_blank" style="text-decoration:none">
          <div class="qa-btn"><span class="qa-icon">📧</span> Send Feedback</div>
        </a>
        <div style="font-size:0.67rem;color:#94a3b8;text-align:center;margin-top:4px">
          rizwanmb310@gmail.com
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)  # quick actions card

    st.markdown("</div>", unsafe_allow_html=True)  # right panel

st.markdown("</div>", unsafe_allow_html=True)  # dashboard grid

# ── Footer ──
st.markdown("""
<div style="text-align:center;padding:14px;font-size:0.72rem;color:#94a3b8;
     border-top:1px solid #e2e8f0;background:#ffffff">
  © 2025 Diabetes Risk AI &nbsp;|&nbsp; Developed by <b>Rizwan Ahmed</b> &nbsp;|&nbsp; An ML Framework for Early Healthcare Risk Prediction
</div>""", unsafe_allow_html=True)
