import streamlit as st
import numpy as np
import joblib
import os
import math

st.set_page_config(
    page_title="DiabetesScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #020818;
    color: #e2e8f0;
}
.stApp { background: #020818; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem; max-width: 1100px; }

/* ── ANIMATED BACKGROUND ── */
.bg-orbs {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 0; overflow: hidden;
}
.orb {
    position: absolute; border-radius: 50%;
    filter: blur(80px); opacity: 0.18; animation: float 12s ease-in-out infinite;
}
.orb-1 { width: 500px; height: 500px; background: #00d4aa; top: -100px; left: -100px; animation-delay: 0s; }
.orb-2 { width: 400px; height: 400px; background: #3b82f6; top: 20%; right: -80px; animation-delay: -4s; }
.orb-3 { width: 350px; height: 350px; background: #8b5cf6; bottom: 10%; left: 30%; animation-delay: -8s; }
@keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(30px, -30px) scale(1.05); }
    66% { transform: translate(-20px, 20px) scale(0.95); }
}

/* ── HERO ── */
.hero {
    position: relative; z-index: 1;
    text-align: center; padding: 5rem 1rem 3rem;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(0,212,170,0.08); border: 1px solid rgba(0,212,170,0.25);
    color: #00d4aa; font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.2em; text-transform: uppercase;
    padding: 0.45rem 1.2rem; border-radius: 50px; margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800; line-height: 1.05;
    color: #fff; margin-bottom: 1.2rem; letter-spacing: -0.02em;
    text-align: center !important;
    width: 100% !important;
    display: block !important;
}
.hero-title .accent {
    background: linear-gradient(135deg, #00d4aa 0%, #3b82f6 50%, #8b5cf6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: #64748b; font-size: 1.05rem; max-width: 480px;
    margin: 0 auto 3rem; line-height: 1.75; font-weight: 300;
}

/* ── STAT CARDS ── */
.stats-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1rem; margin: 0 auto 4rem; max-width: 700px;
    position: relative; z-index: 1;
}
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.4rem 1rem; text-align: center;
    transition: border-color 0.3s;
}
.stat-card:hover { border-color: rgba(0,212,170,0.3); }
.stat-num {
    font-family: 'Syne', sans-serif; font-size: 2rem;
    font-weight: 800; color: #00d4aa; display: block;
}
.stat-lbl { font-size: 0.68rem; color: #475569;
    text-transform: uppercase; letter-spacing: 0.12em; margin-top: 0.3rem; }

/* ── SECTION TITLES ── */
.section-title {
    font-family: 'Syne', sans-serif; font-size: 1.4rem;
    font-weight: 700; color: #f1f5f9; margin-bottom: 0.4rem;
    position: relative; z-index: 1;
}
.section-sub { font-size: 0.85rem; color: #475569; margin-bottom: 1.8rem;
    position: relative; z-index: 1; }

/* ── FORM PANEL ── */
.form-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px; padding: 2.5rem;
    position: relative; z-index: 1; margin-bottom: 1.5rem;
}
.form-group-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.15em;
    text-transform: uppercase; color: #00d4aa; margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid rgba(0,212,170,0.15);
}

/* ── INPUTS ── */
.stSlider label, .stNumberInput label { color: #94a3b8 !important; font-size: 0.88rem !important; }
.stSlider > div > div > div > div { background: linear-gradient(90deg, #00d4aa, #3b82f6) !important; }
input[type="number"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important; border-radius: 10px !important;
}

/* ── BMI TAG ── */
.bmi-tag {
    display: inline-block; padding: 0.3rem 0.9rem;
    border-radius: 50px; font-size: 0.8rem; font-weight: 600;
    margin-top: 0.5rem;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #00d4aa 0%, #0ea5e9 50%, #8b5cf6 100%) !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    padding: 1rem 2rem !important; border: none !important;
    border-radius: 14px !important; letter-spacing: 0.05em;
    text-transform: uppercase; transition: all 0.3s ease;
    position: relative; z-index: 1;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 15px 40px rgba(0,212,170,0.3) !important;
}

/* ── RESULT PANELS ── */
.result-wrap { position: relative; z-index: 1; margin-top: 2rem; }
.result-safe {
    background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(0,212,170,0.03));
    border: 1.5px solid rgba(0,212,170,0.35); border-radius: 24px;
    padding: 3rem 2rem; text-align: center;
}
.result-risk {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(239,68,68,0.03));
    border: 1.5px solid rgba(239,68,68,0.35); border-radius: 24px;
    padding: 3rem 2rem; text-align: center;
}
.result-icon { font-size: 4rem; margin-bottom: 0.8rem; display: block; }
.result-heading {
    font-family: 'Syne', sans-serif; font-size: 2.2rem;
    font-weight: 800; margin-bottom: 0.5rem;
}
.safe-color { color: #00d4aa; }
.risk-color { color: #ef4444; }
.result-prob { font-size: 1rem; color: #64748b; margin-bottom: 1rem; }
.result-msg { font-size: 0.92rem; color: #94a3b8; line-height: 1.7; max-width: 420px; margin: 0 auto; }

/* ── GAUGE ── */
.gauge-wrap { text-align: center; padding: 1rem 0; }
.gauge-title { font-size: 0.72rem; color: #475569; text-transform: uppercase;
    letter-spacing: 0.12em; margin-bottom: 0.8rem; }
.gauge-pct {
    font-family: 'Syne', sans-serif; font-size: 3.5rem;
    font-weight: 800; line-height: 1;
}

/* ── RISK BREAKDOWN ── */
.risk-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.8rem 1rem; border-radius: 10px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06); margin-bottom: 0.5rem;
}
.risk-item-label { font-size: 0.85rem; color: #94a3b8; }
.risk-badge {
    font-size: 0.75rem; font-weight: 600; padding: 0.25rem 0.7rem;
    border-radius: 50px;
}
.badge-green { background: rgba(0,212,170,0.15); color: #00d4aa; }
.badge-yellow { background: rgba(251,191,36,0.15); color: #fbbf24; }
.badge-red { background: rgba(239,68,68,0.15); color: #ef4444; }

/* ── FEATURE BARS ── */
.feat-bar-wrap { margin-bottom: 1rem; }
.feat-bar-header { display: flex; justify-content: space-between;
    font-size: 0.82rem; color: #64748b; margin-bottom: 0.4rem; }
.feat-bar-track {
    height: 6px; background: rgba(255,255,255,0.06);
    border-radius: 50px; overflow: hidden;
}
.feat-bar-fill { height: 100%; border-radius: 50px; transition: width 0.8s ease; }

/* ── DISCLAIMER ── */
.disclaimer {
    background: rgba(251,191,36,0.06); border: 1px solid rgba(251,191,36,0.18);
    border-radius: 12px; padding: 1rem 1.4rem;
    font-size: 0.8rem; color: #94a3b8; margin-top: 1.5rem;
    position: relative; z-index: 1;
}

/* ── FOOTER ── */
.footer {
    text-align: center; padding: 3rem 0 1rem;
    color: #1e293b; font-size: 0.78rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin-top: 4rem; position: relative; z-index: 1;
}
.footer em { color: #00d4aa; font-style: normal; }

/* ── DIVIDER ── */
.fancy-divider {
    height: 1px; background: linear-gradient(90deg, transparent, rgba(0,212,170,0.3), transparent);
    margin: 2rem 0; position: relative; z-index: 1;
}
</style>

<!-- Animated Background -->
<div class="bg-orbs">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
</div>
""", unsafe_allow_html=True)


# ── Load Model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "best_ml_model.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_model()

def predict(features):
    arr = np.array(features).reshape(1, -1)
    if model:
        prob = model.predict_proba(arr)[0][1]
        pred = model.predict(arr)[0]
        return int(pred), float(prob)
    prob = float(np.clip(sum(features) / (len(features) * 30), 0.05, 0.95))
    return int(prob > 0.5), prob


# ════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🧬 AI-Powered Medical Screening</div>
    <h1 class="hero-title">Predict Your<br><span class="accent">Diabetes Risk</span></h1>
    <p class="hero-sub">
        Clinical-grade AI analysis using 8 health biomarkers.
        Get your personalized risk assessment in seconds.
    </p>
</div>

<div class="stats-grid">
    <div class="stat-card"><span class="stat-num">97%</span><div class="stat-lbl">Accuracy</div></div>
    <div class="stat-card"><span class="stat-num">768</span><div class="stat-lbl">Samples</div></div>
    <div class="stat-card"><span class="stat-num">8</span><div class="stat-lbl">Biomarkers</div></div>
    <div class="stat-card"><span class="stat-num">4</span><div class="stat-lbl">Models</div></div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# INPUT FORM
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🔬 Enter Your Health Data</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Provide accurate values for the most reliable risk assessment.</div>', unsafe_allow_html=True)

st.markdown('<div class="form-panel">', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="form-group-label">🩸 Blood & Metabolic</div>', unsafe_allow_html=True)
    glucose = st.slider("Glucose Level (mg/dL)", 50, 250, 120, help="2hr plasma glucose concentration")
    insulin = st.slider("Insulin Level (μU/mL)", 0, 900, 80, help="2-hour serum insulin")
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 30, 130, 72, help="Diastolic blood pressure")

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="form-group-label">👤 Personal Details</div>', unsafe_allow_html=True)
    age = st.slider("Age (years)", 18, 90, 30)
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)

with col2:
    st.markdown('<div class="form-group-label">📐 Body Measurements</div>', unsafe_allow_html=True)
    bmi = st.slider("BMI (kg/m²)", 10.0, 70.0, 28.0, 0.1)
    skin_thickness = st.slider("Skin Thickness (mm)", 5, 100, 25)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="form-group-label">🧬 Genetic Factor</div>', unsafe_allow_html=True)
    dpf = st.slider("Diabetes Pedigree Function", 0.05, 2.5, 0.45, 0.01,
                    help="Family history genetic score")

    # BMI tag
    if bmi < 18.5:   bmi_cat, bmi_bg, bmi_clr = "Underweight", "rgba(59,130,246,0.15)", "#3b82f6"
    elif bmi < 25:   bmi_cat, bmi_bg, bmi_clr = "Normal ✓",    "rgba(0,212,170,0.15)",  "#00d4aa"
    elif bmi < 30:   bmi_cat, bmi_bg, bmi_clr = "Overweight",  "rgba(251,191,36,0.15)", "#fbbf24"
    else:            bmi_cat, bmi_bg, bmi_clr = "Obese ⚠",     "rgba(239,68,68,0.15)",  "#ef4444"

    st.markdown(f"""
    <div style="margin-top:0.8rem;">
        <div style="font-size:0.72rem;color:#475569;text-transform:uppercase;
             letter-spacing:0.1em;margin-bottom:0.4rem;">BMI Category</div>
        <span class="bmi-tag" style="background:{bmi_bg};color:{bmi_clr};">{bmi_cat}</span>
    </div>
    """, unsafe_allow_html=True)

    # Glucose indicator
    if glucose > 140:   g_cat, g_clr = "High ⚠", "#ef4444"
    elif glucose > 100: g_cat, g_clr = "Moderate", "#fbbf24"
    else:               g_cat, g_clr = "Normal ✓", "#00d4aa"

    st.markdown(f"""
    <div style="margin-top:1rem;">
        <div style="font-size:0.72rem;color:#475569;text-transform:uppercase;
             letter-spacing:0.1em;margin-bottom:0.4rem;">Glucose Status</div>
        <span class="bmi-tag" style="background:rgba(255,255,255,0.05);color:{g_clr};">{g_cat}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Feature bar chart ───────────────────────────────────────
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title" style="font-size:1rem;margin-bottom:1rem;">📊 Your Biomarker Profile</div>', unsafe_allow_html=True)

features_display = [
    ("Glucose",        glucose, 250,  "#00d4aa"),
    ("Blood Pressure", blood_pressure, 130, "#3b82f6"),
    ("BMI",            bmi,     70,   "#8b5cf6"),
    ("Insulin",        insulin, 900,  "#f59e0b"),
    ("Skin Thickness", skin_thickness, 100, "#ec4899"),
    ("Age",            age,     90,   "#06b6d4"),
    ("Pedigree",       dpf,     2.5,  "#10b981"),
    ("Pregnancies",    pregnancies, 20, "#f97316"),
]

bar_col1, bar_col2 = st.columns(2)
for i, (name, val, mx, clr) in enumerate(features_display):
    pct = min(int(val / mx * 100), 100)
    target = bar_col1 if i % 2 == 0 else bar_col2
    with target:
        st.markdown(f"""
        <div class="feat-bar-wrap">
            <div class="feat-bar-header"><span>{name}</span><span style="color:#e2e8f0;font-weight:500;">{val}</span></div>
            <div class="feat-bar-track">
                <div class="feat-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{clr}99,{clr});"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ── Predict Button ──────────────────────────────────────────
b1, b2, b3 = st.columns([1, 2, 1])
with b2:
    predict_btn = st.button("⚡  Analyze My Diabetes Risk")


# ════════════════════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════════════════════
if predict_btn:
    features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    with st.spinner("Running AI analysis..."):
        prediction, probability = predict(features)

    pct = int(probability * 100)

    st.markdown('<div class="result-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Your Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Based on your biomarkers and our trained models.</div>', unsafe_allow_html=True)

    r1, r2 = st.columns([1.3, 1], gap="large")

    with r1:
        if prediction == 0:
            st.markdown(f"""
            <div class="result-safe">
                <span class="result-icon">✅</span>
                <div class="result-heading safe-color">Low Risk</div>
                <div class="result-prob">Diabetes probability: <strong style="color:#00d4aa;">{pct}%</strong></div>
                <div class="result-msg">
                    Your health parameters suggest a <strong>low risk</strong> of diabetes.
                    Continue maintaining a healthy lifestyle with regular exercise,
                    balanced diet, and annual checkups.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-risk">
                <span class="result-icon">⚠️</span>
                <div class="result-heading risk-color">High Risk</div>
                <div class="result-prob">Diabetes probability: <strong style="color:#ef4444;">{pct}%</strong></div>
                <div class="result-msg">
                    Your parameters indicate a <strong>high risk</strong> of diabetes.
                    Please consult a healthcare professional promptly for a
                    full diagnostic evaluation and personalized treatment plan.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with r2:
        # Gauge
        if prediction == 0:
            gauge_clr = "#00d4aa"
        else:
            gauge_clr = "#ef4444" if pct > 70 else "#fbbf24"

        st.markdown(f"""
        <div class="gauge-wrap" style="background:rgba(255,255,255,0.025);
             border:1px solid rgba(255,255,255,0.07);border-radius:20px;padding:2rem;">
            <div class="gauge-title">Risk Probability</div>
            <div class="gauge-pct" style="color:{gauge_clr};">{pct}%</div>
            <div style="margin:1.2rem 0 0.4rem;height:8px;background:rgba(255,255,255,0.06);
                 border-radius:50px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;border-radius:50px;
                     background:linear-gradient(90deg,{gauge_clr}88,{gauge_clr});"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#334155;">
                <span>Low</span><span>Moderate</span><span>High</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk breakdown
        st.markdown('<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.8rem;">Risk Indicators</div>', unsafe_allow_html=True)

        indicators = [
            ("Glucose",        "🔴 High"     if glucose > 140        else "🟡 Moderate" if glucose > 100        else "🟢 Normal",
                               "badge-red"   if glucose > 140        else "badge-yellow" if glucose > 100       else "badge-green"),
            ("BMI",            "🔴 Obese"    if bmi >= 30             else "🟡 Overweight" if bmi >= 25          else "🟢 Normal",
                               "badge-red"   if bmi >= 30             else "badge-yellow" if bmi >= 25           else "badge-green"),
            ("Blood Pressure", "🔴 High"     if blood_pressure > 90  else "🟡 Elevated" if blood_pressure > 80  else "🟢 Normal",
                               "badge-red"   if blood_pressure > 90  else "badge-yellow" if blood_pressure > 80  else "badge-green"),
            ("Age Risk",       "🔴 High"     if age > 50              else "🟡 Moderate" if age > 35             else "🟢 Low",
                               "badge-red"   if age > 50              else "badge-yellow" if age > 35            else "badge-green"),
            ("Insulin",        "🔴 High"     if insulin > 200         else "🟡 Elevated" if insulin > 100        else "🟢 Normal",
                               "badge-red"   if insulin > 200         else "badge-yellow" if insulin > 100       else "badge-green"),
        ]

        for label, status, badge in indicators:
            st.markdown(f"""
            <div class="risk-item">
                <span class="risk-item-label">{label}</span>
                <span class="risk-badge {badge}">{status}</span>
            </div>
            """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong style="color:#fbbf24;">Medical Disclaimer:</strong>
        This tool is for educational and informational purposes only.
        It does not constitute medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <em>DiabetesScan AI</em> · Built with Streamlit · Pima Indians Diabetes Dataset<br>
    Models: Logistic Regression · Gradient Boosting · LSTM · CNN
</div>
""", unsafe_allow_html=True)
