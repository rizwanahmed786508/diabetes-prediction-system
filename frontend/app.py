import streamlit as st
import requests
import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# ─── Backend Config ───────────────────────────────────────────────────────────
# All ML processing (scaling, prediction) happens on the FastAPI backend.
# The frontend NEVER loads a model or scaler itself.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

# ─── Fixed Model Display Metadata ────────────────────────────────────────────
# The backend serves the current MLflow "Champion" model — it is selected
# automatically by the MLflow Model Registry based on evaluation metrics,
# not hardcoded here. This is display-only metadata for the UI/PDF; the
# frontend does not train, select, or compare models itself.
MODEL_NAME = "Random Forest"
MODEL_ACCURACY = "72.00%"
MODEL_SELECTION_NOTE = "Automatically selected via MLflow Model Registry as the current Champion Model."

# ─── Field Display Metadata (used for friendly validation messages) ─────────
# Maps the API/JSON field name -> (friendly label, icon)
FIELD_META = {
    "Pregnancies":              ("Pregnancies", "🤱"),
    "Glucose":                  ("Glucose", "🩸"),
    "BloodPressure":            ("Blood Pressure", "💓"),
    "SkinThickness":            ("Skin Thickness", "📏"),
    "Insulin":                  ("Insulin", "💉"),
    "BMI":                      ("BMI", "⚖️"),
    "DiabetesPedigreeFunction": ("Diabetes Pedigree Function", "🧬"),
    "Age":                      ("Age", "🎂"),
}

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (clean white + blue medical theme) ───────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: #f5f9fd;
        color: #0f2942;
        font-family: 'Inter', 'Segoe UI', Roboto, Arial, sans-serif;
    }

    [data-testid="stAppViewContainer"] > .main {
        overflow-x: hidden;
    }

    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2ecf7;
    }

    [data-testid="stSidebar"] * {
        color: #17324a;
    }

    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #123a66;
        text-align: center;
        padding: 10px 0 4px;
        letter-spacing: 0.5px;
    }

    .subtitle {
        text-align: center;
        color: #5b7a9a;
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 28px;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e2ecf7;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 14px rgba(15, 41, 66, 0.06);
    }

    .section-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: #1d6fd1;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #e2ecf7;
    }

    .result-diabetic {
        background: #fef4f4;
        border: 1px solid #f3b8b8;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .result-safe {
        background: #f2faf5;
        border: 1px solid #b9e3c6;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .result-label {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-sub { font-size: 0.88rem; color: #4d6478; opacity: 0.9; }

    /* Validation error card — consistent white/blue theme with amber accent */
    .validation-card {
        background: #fff9ec;
        border: 1px solid #f2ddab;
        border-left: 4px solid #e2a63b;
        border-radius: 14px;
        padding: 20px 22px;
        margin: 6px 0 4px;
        box-shadow: 0 2px 14px rgba(15, 41, 66, 0.05);
    }

    .validation-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #8a5a10;
        margin-bottom: 6px;
    }

    .validation-sub {
        font-size: 0.85rem;
        color: #7a6a45;
        margin-bottom: 14px;
        line-height: 1.5;
    }

    .validation-line {
        font-size: 0.88rem;
        font-weight: 600;
        color: #123a66;
        padding: 6px 0;
        line-height: 1.5;
    }

    div[data-testid="stNumberInput"] input {
        color: #0f2942 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        background: #ffffff !important;
        border: 1px solid #cfe0f3 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stNumberInput"] input::placeholder {
        color: #8fa4bb !important;
        font-weight: 400 !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border: 1px solid #1d6fd1 !important;
        box-shadow: 0 0 0 2px rgba(29, 111, 209, 0.15) !important;
        outline: none !important;
    }
    div[data-testid="stNumberInput"] label,
    div[data-testid="stNumberInput"] label p {
        color: #23415e !important;
        font-weight: 500 !important;
    }
    div[data-testid="stNumberInput"] button {
        background: #f3f8fd !important;
        border: 1px solid #cfe0f3 !important;
        color: #1d6fd1 !important;
    }

    @media (max-width: 768px) {
        div[data-testid="stNumberInput"] input { font-size: 16px !important; }
        .main-title { font-size: 1.6rem; }
        .card { padding: 16px; }
    }

    .stButton > button {
        width: 100%;
        background: #1d6fd1;
        color: #ffffff;
        border: none;
        padding: 14px;
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-top: 10px;
    }
    .stButton > button:hover {
        background: #1558a8;
        box-shadow: 0 4px 14px rgba(29, 111, 209, 0.25);
        transform: translateY(-1px);
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* Reset button */
    button[kind="secondary"] {
        background: #ffffff !important;
        border: 1px solid #cfe0f3 !important;
        color: #1d6fd1 !important;
        padding: 6px 16px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        border-radius: 8px !important;
        margin-top: 0 !important;
        width: auto !important;
    }
    button[kind="secondary"]:hover {
        background: #f3f8fd !important;
        box-shadow: none !important;
        transform: none !important;
    }

    /* PDF download button */
    [data-testid="stDownloadButton"] > button {
        width: 100%;
        background: #1a9d5c !important;
        color: #ffffff !important;
        border: none !important;
        padding: 13px !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.5px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: #158049 !important;
        box-shadow: 0 4px 14px rgba(26, 157, 92, 0.25) !important;
        transform: translateY(-1px) !important;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Validation Error Parsing Helpers ────────────────────────────────────────
def _humanize_constraint(err, label):
    """
    Turns a single Pydantic/FastAPI error dict into a short, simple
    sentence using the field's display label, without ever leaking raw
    error types, locs, or ctx blobs.
    """
    err_type = str(err.get("type", ""))
    ctx = err.get("ctx", {}) if isinstance(err.get("ctx"), dict) else {}

    limit = ctx.get("gt", ctx.get("ge", ctx.get("lt", ctx.get("le"))))
    if isinstance(limit, float) and limit == int(limit):
        limit = int(limit)

    if "greater_than_equal" in err_type:
        return f"{label} must be greater than or equal to {limit}." if limit is not None else f"{label} value is invalid."
    if "greater_than" in err_type:
        return f"{label} must be greater than {limit}." if limit is not None else f"{label} value is invalid."
    if "less_than_equal" in err_type:
        return f"{label} must be less than or equal to {limit}." if limit is not None else f"{label} value is invalid."
    if "less_than" in err_type:
        return f"{label} must be less than {limit}." if limit is not None else f"{label} value is invalid."
    if "missing" in err_type:
        return f"{label} is required."
    if "type_error" in err_type or "int_parsing" in err_type or "float_parsing" in err_type:
        return f"{label} must be a valid number."
    # Fallback: generic, non-technical message (never show raw msg/type/loc)
    return f"{label} value is invalid."


def parse_validation_errors(detail):
    """
    Safely converts a FastAPI/Pydantic `detail` payload (usually a list of
    error dicts, but could be a plain string or something unexpected) into
    a clean list of (field_label, icon, friendly_message) tuples.

    This function NEVER raises and NEVER returns raw technical content —
    if anything is unrecognized, it degrades to a generic message.
    """
    friendly_errors = []

    if isinstance(detail, list):
        for err in detail:
            if not isinstance(err, dict):
                continue
            loc = err.get("loc", [])
            # loc is typically like ["body", "BMI"] — take the last element
            field_key = loc[-1] if loc else None
            label, icon = FIELD_META.get(field_key, (str(field_key) if field_key else "Input", "📋"))
            message = _humanize_constraint(err, label)
            friendly_errors.append((label, icon, message))

    # De-duplicate while preserving order
    seen = set()
    unique_errors = []
    for item in friendly_errors:
        if item not in seen:
            seen.add(item)
            unique_errors.append(item)

    return unique_errors


def render_validation_error(friendly_errors):
    """Renders a clean, theme-consistent validation error card."""
    lines_html = ""
    if friendly_errors:
        for label, icon, message in friendly_errors:
            lines_html += f'<div class="validation-line">{icon} {message}</div>'
    else:
        lines_html = '<div class="validation-line">One or more entered values could not be processed. Please review your inputs and try again.</div>'

    st.markdown(f"""
    <div class="validation-card">
        <div class="validation-title">⚠️ Please check your input values</div>
        <div class="validation-sub">Some entered values are outside the allowed range. Please review the fields below.</div>
        {lines_html}
    </div>
    """, unsafe_allow_html=True)


# ─── Backend Helper ───────────────────────────────────────────────────────────
def call_predict_api(payload):
    """
    Sends patient data to the FastAPI backend and returns (result, error_info).

    error_info is either:
      - None (no error), or
      - a dict of the form:
            {"kind": "validation", "errors": [(label, icon, message), ...]}
            {"kind": "connection" | "timeout" | "invalid_response" | "other", "message": "..."}

    Raw backend/Pydantic error payloads (lists of dicts with 'type'/'loc'/'ctx',
    stack traces, etc.) are never surfaced to the caller — everything is
    translated into a safe, user-friendly structure here.
    """
    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=15)
        response.raise_for_status()
        return response.json(), None

    except requests.exceptions.ConnectionError:
        return None, {
            "kind": "connection",
            "message": f"🔌 **Cannot connect to the backend.** Make sure the FastAPI server is running at `{BACKEND_URL}`.",
        }
    except requests.exceptions.Timeout:
        return None, {
            "kind": "timeout",
            "message": "⏱️ **Backend timeout.** The server took too long to respond. Please try again.",
        }
    except requests.exceptions.HTTPError:
        # This is the case that previously leaked raw Pydantic validation
        # errors (422) straight to the user. Now we translate it safely.
        if response is not None and response.status_code == 422:
            try:
                detail = response.json().get("detail", [])
            except Exception:
                detail = []
            friendly_errors = parse_validation_errors(detail)
            return None, {"kind": "validation", "errors": friendly_errors}
        else:
            return None, {
                "kind": "other",
                "message": "⚠️ **The backend rejected the request.** Please review your inputs and try again.",
            }
    except ValueError:
        return None, {
            "kind": "invalid_response",
            "message": "⚠️ **Invalid response** received from the backend.",
        }
    except requests.exceptions.RequestException:
        return None, {
            "kind": "other",
            "message": "❌ **Unexpected error.** Please try again in a moment.",
        }


# ─── PDF Generator ────────────────────────────────────────────────────────────
def generate_pdf(patient_data, prediction, diabetic_prob, nondiabetic_prob,
                 model_name, model_accuracy):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.8*cm, bottomMargin=1.8*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    C_DARK=colors.HexColor("#123a66"); C_BLUE=colors.HexColor("#1d6fd1")
    C_GREEN=colors.HexColor("#1a9d5c"); C_RED=colors.HexColor("#d64545")
    C_TEXT=colors.HexColor("#1a2a3a"); C_MUTED=colors.HexColor("#5a7090")
    C_BORDER=colors.HexColor("#d0e4f0"); C_LBLUE=colors.HexColor("#eaf2fe")
    def S(name, **kw):
        d = dict(fontName="Helvetica", fontSize=10, textColor=C_TEXT, leading=14); d.update(kw)
        return ParagraphStyle(name, **d)
    sTitle  = S("T", fontName="Helvetica-Bold", fontSize=22, textColor=C_DARK, alignment=TA_CENTER, leading=28, spaceAfter=2)
    sSub    = S("Su", fontSize=9, textColor=C_MUTED, alignment=TA_CENTER, leading=12)
    sSecHdr = S("SH", fontName="Helvetica-Bold", fontSize=9, textColor=C_BLUE, leading=14, spaceBefore=4, spaceAfter=2)
    W = A4[0] - 4*cm; story = []
    story.append(Paragraph("DIABETES PREDICTION REPORT", sTitle))
    story.append(Paragraph("AI-Powered Clinical Risk Assessment", sSub))
    story.append(Spacer(1, 6))
    now = datetime.now()
    story.append(Paragraph(
        f"Generated: {now.strftime('%B %d, %Y')} &nbsp;|&nbsp; Time: {now.strftime('%H:%M:%S')} &nbsp;|&nbsp; Model: {model_name}",
        S("dt", fontSize=8, textColor=C_MUTED, alignment=TA_CENTER, leading=12)))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width=W, thickness=2, color=C_BLUE, spaceAfter=14))
    story.append(Paragraph("PATIENT CLINICAL DATA", sSecHdr))
    labels = ["Pregnancies","Glucose (mg/dL)","Blood Pressure (mmHg)","Skin Thickness (mm)",
              "Insulin (uU/mL)","BMI (kg/m2)","Diabetes Pedigree Function","Age (years)"]
    values = [str(v) for v in patient_data]
    td = [["Parameter","Value","Parameter","Value"]]
    for i in range(0, 8, 2): td.append([labels[i], values[i], labels[i+1], values[i+1]])
    tbl = Table(td, colWidths=[W*.30, W*.18, W*.30, W*.18], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),C_DARK),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),8),
        ("ALIGN",(0,0),(-1,0),"CENTER"),("TOPPADDING",(0,0),(-1,0),5),("BOTTOMPADDING",(0,0),(-1,0),5),
        ("FONTSIZE",(0,1),(-1,-1),8.5),("FONTNAME",(1,1),(1,-1),"Helvetica-Bold"),
        ("FONTNAME",(3,1),(3,-1),"Helvetica-Bold"),("ALIGN",(1,0),(1,-1),"CENTER"),
        ("ALIGN",(3,0),(3,-1),"CENTER"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,C_LBLUE]),
        ("GRID",(0,0),(-1,-1),0.5,C_BORDER),("TOPPADDING",(0,1),(-1,-1),5),
        ("BOTTOMPADDING",(0,1),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),8),
    ]))
    story.append(tbl); story.append(Spacer(1, 14))
    story.append(HRFlowable(width=W, thickness=0.5, color=C_BORDER, spaceAfter=10))
    story.append(Paragraph("PREDICTION RESULT", sSecHdr))
    if prediction == 1:
        res_label="DIABETIC RISK DETECTED"; res_conf=f"{diabetic_prob}%"
        res_sub=f"Non-Diabetic Probability: {nondiabetic_prob}%"
        bg_color=colors.HexColor("#fef4f4"); txt_color=C_RED; border_c=C_RED
    else:
        res_label="NON-DIABETIC"; res_conf=f"{nondiabetic_prob}%"
        res_sub=f"Diabetic Probability: {diabetic_prob}%"
        bg_color=colors.HexColor("#f2faf5"); txt_color=C_GREEN; border_c=C_GREEN
    result_data = [[
        Paragraph(res_label, S("RL", fontName="Helvetica-Bold", fontSize=14, textColor=txt_color, alignment=TA_CENTER, leading=18)),
        Paragraph(res_conf,  S("RC", fontName="Helvetica-Bold", fontSize=26, textColor=txt_color, alignment=TA_CENTER, leading=30)),
        Paragraph(f"Confidence\n{res_sub}", S("RS", fontSize=8, textColor=C_MUTED, alignment=TA_CENTER, leading=12)),
    ]]
    rtbl = Table(result_data, colWidths=[W*.38, W*.28, W*.34])
    rtbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),bg_color),("BOX",(0,0),(-1,-1),1.5,border_c),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),14),
        ("BOTTOMPADDING",(0,0),(-1,-1),14),("LEFTPADDING",(0,0),(-1,-1),10),
        ("RIGHTPADDING",(0,0),(-1,-1),10),("LINEAFTER",(0,0),(1,-1),0.5,border_c),
    ]))
    story.append(rtbl); story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Algorithm: {model_name} &nbsp;|&nbsp; Model Accuracy: {model_accuracy}",
        S("acc", fontSize=8, textColor=C_MUTED, alignment=TA_CENTER, leading=12)))
    story.append(Paragraph(
        "Automatically selected via MLflow Model Registry as the current Champion Model.",
        S("sel", fontSize=7.5, textColor=C_MUTED, alignment=TA_CENTER, leading=11)))
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width=W, thickness=0.5, color=C_BORDER, spaceAfter=10))
    disc_data = [[Paragraph(
        "<b>MEDICAL DISCLAIMER</b><br/>This report is generated by an AI-based system for educational and informational "
        "purposes only. It does NOT constitute medical advice, diagnosis, or treatment. "
        "Please consult a qualified healthcare provider for any medical concerns.",
        S("D", fontSize=7.5, textColor=colors.HexColor("#7a5500"), leading=11))]]
    dtbl = Table(disc_data, colWidths=[W])
    dtbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#fffbe6")),
        ("BOX",(0,0),(-1,-1),0.8,colors.HexColor("#e6b800")),
        ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),
    ]))
    story.append(dtbl); story.append(Spacer(1, 10))
    story.append(Paragraph("Diabetes Prediction System &nbsp;•&nbsp; Developed By Rizwan Ahmed",
        S("ft", fontSize=7, textColor=C_MUTED, alignment=TA_CENTER, leading=10)))
    doc.build(story); buf.seek(0)
    return buf.read()


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">📌 About Project</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.85rem; color:#3a5468; line-height:1.9">
        <b style="color:#123a66">Diabetes Prediction System</b><br>
        AI-based diabetes risk prediction<br><br>
        <b>Dataset:</b> Pima Indians Diabetes Dataset<br>
        <b>Backend:</b> FastAPI<br>
        <b>Frontend:</b> Streamlit<br>
        <b>ML Model:</b> {MODEL_NAME}<br>
        <b>Model Accuracy:</b> {MODEL_ACCURACY}
    </div>
    <div style="font-size:0.76rem; color:#5b7a9a; line-height:1.6; margin-top:10px;
         background:#f3f8fd; border:1px solid #dcebfa; border-radius:8px; padding:10px 12px;">
        🏆 {MODEL_SELECTION_NOTE}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">ℹ️ How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem; color:#3a5468; line-height:1.7">
    Enter the patient's clinical measurements, then click
    <b style="color:#1d6fd1">Analyze & Predict</b>. All scaling and
    inference are performed by the FastAPI backend — this interface
    only collects input and displays results.
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🩺 Diabetes Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Clinical Risk Assessment</p>', unsafe_allow_html=True)

# ─── Session State Defaults ──────────────────────────────────────────────────
DEFAULTS = {
    "pregnancies": 0, "glucose": 0, "blood_pressure": 0,
    "skin_thickness": 0, "insulin": 0, "bmi": 0.0,
    "dpf": 0.000, "age": 0,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_form():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

# ─── Input Form ──────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)

hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown('<div class="section-label" style="margin-bottom:0">📋 Patient Clinical Data</div>', unsafe_allow_html=True)
with hcol2:
    st.button("↺ Reset", on_click=reset_form, key="reset_btn", help="Reset all inputs to default values")

st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    pregnancies    = st.number_input("🤱 Pregnancies",           min_value=0, max_value=20,  step=1,    key="pregnancies",    help="Number of times pregnant")
    glucose        = st.number_input("🩸 Glucose (mg/dL)",       min_value=0, max_value=300, step=1,    key="glucose",        help="Plasma glucose concentration")
    blood_pressure = st.number_input("💓 Blood Pressure (mmHg)", min_value=0, max_value=200, step=1,    key="blood_pressure", help="Diastolic blood pressure")
    skin_thickness = st.number_input("📏 Skin Thickness (mm)",   min_value=0, max_value=100, step=1,    key="skin_thickness", help="Triceps skin fold thickness")
with col2:
    insulin        = st.number_input("💉 Insulin (μU/mL)",       min_value=0, max_value=900, step=1,    key="insulin",        help="2-Hour serum insulin")
    bmi            = st.number_input("⚖️ BMI (kg/m²)",           min_value=0.0,max_value=70.0,step=0.1, key="bmi",            help="Body Mass Index")
    dpf            = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0,max_value=3.0,step=0.001,format="%.3f", key="dpf", help="Diabetes heredity score")
    age            = st.number_input("🎂 Age (years)",           min_value=0, max_value=120, step=1,    key="age",            help="Patient age in years")

st.markdown('</div>', unsafe_allow_html=True)

# ─── Predict Button ──────────────────────────────────────────────────────────
predict_clicked = st.button("🔬 ANALYZE & PREDICT")

if predict_clicked:
    # Frontend only builds the payload — no scaling, no model, no training.
    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    with st.spinner("🔬 Analyzing patient data..."):
        result, error_info = call_predict_api(payload)

    st.markdown("---")

    if error_info:
        kind = error_info.get("kind")
        if kind == "validation":
            # Clean, friendly, theme-consistent validation card —
            # no raw Pydantic types/locs/ctx ever reach the user.
            render_validation_error(error_info.get("errors", []))
        else:
            st.error(error_info.get("message", "⚠️ Something went wrong. Please try again."))
    else:
        prediction       = result["prediction"]
        diabetic_prob    = result["diabetic_probability"]
        nondiabetic_prob = result["non_diabetic_probability"]

        # Model is fixed and already known — we don't rely on the backend
        # to describe it, so "Unknown Model" can never be shown.
        model_name = MODEL_NAME
        model_accuracy = MODEL_ACCURACY

        # ── Result card ──
        if prediction == 1:
            st.markdown(f"""
            <div class="result-diabetic">
                <div class="result-label" style="color:#c23b3b">⚠️ DIABETIC RISK DETECTED</div>
                <div class="result-sub">The model indicates a high probability of diabetes.</div>
                <br>
                <div style="font-family:'Inter',sans-serif;font-size:2rem;color:#c23b3b;font-weight:700">{diabetic_prob}%</div>
                <div style="font-size:0.8rem;color:#a35555">Diabetes Probability</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-label" style="color:#1a9d5c">✓ NON-DIABETIC</div>
                <div class="result-sub">No significant diabetes risk detected.</div>
                <br>
                <div style="font-family:'Inter',sans-serif;font-size:2rem;color:#1a9d5c;font-weight:700">{nondiabetic_prob}%</div>
                <div style="font-size:0.8rem;color:#4d9a70">Non-Diabetic Probability</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability breakdown ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📊 Prediction Breakdown</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-bottom:12px">
            <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:4px;color:#23415e">
                <span>🔴 Diabetic</span><span style="color:#c23b3b;font-weight:600">{diabetic_prob}%</span>
            </div>
            <div style="background:#eef3f9;border-radius:6px;height:10px;overflow:hidden">
                <div style="width:{int(diabetic_prob)}%;background:linear-gradient(90deg,#e26a6a,#c23b3b);height:100%;border-radius:6px"></div>
            </div>
        </div>
        <div>
            <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:4px;color:#23415e">
                <span>🟢 Non-Diabetic</span><span style="color:#1a9d5c;font-weight:600">{nondiabetic_prob}%</span>
            </div>
            <div style="background:#eef3f9;border-radius:6px;height:10px;overflow:hidden">
                <div style="width:{int(nondiabetic_prob)}%;background:linear-gradient(90deg,#4fbf85,#1a9d5c);height:100%;border-radius:6px"></div>
            </div>
        </div>
        <br>
        <div style="display:flex;flex-wrap:wrap;gap:12px;font-size:0.82rem;color:#5b7a9a;align-items:center">
            <span>🤖 Model: <b style="color:#1d6fd1">{model_name}</b></span>
            <span>|</span>
            <span>📈 Accuracy: <b style="color:#1d6fd1">{model_accuracy}</b></span>
            <span>|</span>
            <span>🏆 <span style="color:#7a8ba3">Champion Model (MLflow)</span></span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── PDF Download ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📄 Download Report</div>', unsafe_allow_html=True)
        with st.spinner("📝 Preparing PDF report..."):
            pdf_bytes = generate_pdf(
                patient_data=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age],
                prediction=prediction, diabetic_prob=diabetic_prob, nondiabetic_prob=nondiabetic_prob,
                model_name=model_name, model_accuracy=model_accuracy,
            )
        fname = f"diabetes_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button("📥 Download Patient Report (PDF)", data=pdf_bytes,
                           file_name=fname, mime="application/pdf", key="pdf_download")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Disclaimer ──
        st.markdown("""
        <div style="background:#fff9e6;border:1px solid #f0dfa0;
             border-radius:10px;padding:14px;margin-top:10px;font-size:0.78rem;
             color:#8a6d1f;text-align:center">
            ⚕️ <b>Disclaimer:</b> This tool is for educational purposes only.
            Always consult a qualified healthcare professional for medical advice.
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:0.72rem;color:#8fa4bb;
     letter-spacing:1px;padding-bottom:10px">
    DIABETES PREDICTION SYSTEM • DEVELOPED BY RIZWAN AHMED
</div>
""", unsafe_allow_html=True)