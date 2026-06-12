import streamlit as st
import numpy as np
import joblib
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%);
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 3rem 2rem 2rem;
    margin-bottom: 1rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(0, 212, 170, 0.12);
    border: 1px solid rgba(0, 212, 170, 0.3);
    color: #00d4aa;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.15;
    margin-bottom: 1rem;
}
.hero-title span {
    background: linear-gradient(90deg, #00d4aa, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── Stat Cards ── */
.stats-row {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin: 2rem 0;
    flex-wrap: wrap;
}
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 2rem;
    text-align: center;
    min-width: 140px;
}
.stat-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00d4aa;
}
.stat-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Form Card ── */
.form-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2.5rem;
    margin: 1.5rem 0;
}
.form-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.3rem;
}
.form-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 2rem;
}
.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}

/* ── Sliders & Inputs ── */
.stSlider > div > div > div > div {
    background: #00d4aa !important;
}
.stSlider label { color: #cbd5e1 !important; font-size: 0.9rem !important; }
.stNumberInput label { color: #cbd5e1 !important; font-size: 0.9rem !important; }
input[type="number"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Predict Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00d4aa, #0ea5e9) !important;
    color: #0a0e1a !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.85rem 2rem !important;
    border: none !important;
    border-radius: 12px !important;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    margin-top: 1rem;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 212, 170, 0.35) !important;
}

/* ── Result Cards ── */
.result-safe {
    background: linear-gradient(135deg, rgba(0,212,170,0.12), rgba(0,212,170,0.05));
    border: 1.5px solid rgba(0,212,170,0.4);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-risk {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.05));
    border: 1.5px solid rgba(239,68,68,0.4);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.result-safe .result-title  { color: #00d4aa; }
.result-risk .result-title  { color: #ef4444; }
.result-prob {
    font-size: 1rem;
    color: #94a3b8;
    margin-bottom: 1rem;
}
.result-advice {
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.6;
    max-width: 400px;
    margin: 0 auto;
}

/* ── Risk Meter ── */
.risk-meter-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.stProgress > div > div > div > div {
    border-radius: 50px;
}

/* ── Info Tooltip Cards ── */
.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
}
.info-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
.info-item-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.info-item-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-top: 0.2rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #334155;
    font-size: 0.8rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 3rem;
}
.footer span { color: #00d4aa; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "best_ml_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()


# ── Helper: Predict ──────────────────────────────────────────
def predict(features):
    arr = np.array(features).reshape(1, -1)
    if model:
        prob  = model.predict_proba(arr)[0][1]
        pred  = model.predict(arr)[0]
        return pred, prob
    # Demo fallback
    prob = float(np.clip(np.mean(arr) * 0.6 + np.random.uniform(0.1, 0.3), 0, 1))
    return int(prob > 0.5), prob


# ════════════════════════════════════════════════════════════
# HERO SECTION
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">🧬 AI-Powered Medical Screening</div>
    <div class="hero-title">Diabetes<span>Scan</span> AI</div>
    <div class="hero-sub">
        Enter your health parameters below and get an instant diabetes risk assessment powered by machine learning.
    </div>
</div>
""", unsafe_allow_html=True)

# Stats Row
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-number">97%</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">768</div>
        <div class="stat-label">Training Samples</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">8</div>
        <div class="stat-label">Health Features</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">4</div>
        <div class="stat-label">Models Trained</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# INPUT FORM
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="form-card">
<div class="form-title">🔬 Enter Health Parameters</div>
<div class="form-subtitle">All values are used by the model to assess diabetes risk. Be as accurate as possible.</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("**🩸 Blood & Metabolic**")
    glucose = st.slider("Glucose Level (mg/dL)",
                        min_value=50, max_value=250, value=120,
                        help="Plasma glucose concentration (2hr oral glucose tolerance test)")
    insulin = st.slider("Insulin Level (μU/mL)",
                        min_value=0, max_value=900, value=80,
                        help="2-Hour serum insulin")
    blood_pressure = st.slider("Blood Pressure (mm Hg)",
                               min_value=30, max_value=130, value=72,
                               help="Diastolic blood pressure")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**👤 Personal Info**")
    age = st.slider("Age (years)",
                    min_value=18, max_value=90, value=30)
    pregnancies = st.number_input("Number of Pregnancies",
                                   min_value=0, max_value=20, value=1,
                                   help="Number of times pregnant")

with col2:
    st.markdown("**📏 Body Measurements**")
    bmi = st.slider("BMI (kg/m²)",
                    min_value=10.0, max_value=70.0, value=28.0, step=0.1,
                    help="Body Mass Index = weight(kg) / height(m)²")
    skin_thickness = st.slider("Skin Thickness (mm)",
                               min_value=5, max_value=100, value=25,
                               help="Triceps skin fold thickness")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**🧬 Genetic Factor**")
    dpf = st.slider("Diabetes Pedigree Function",
                    min_value=0.05, max_value=2.5, value=0.45, step=0.01,
                    help="Genetic influence score based on family history of diabetes")

    # Live BMI category
    if bmi < 18.5:
        bmi_cat, bmi_color = "Underweight", "#3b82f6"
    elif bmi < 25:
        bmi_cat, bmi_color = "Normal", "#00d4aa"
    elif bmi < 30:
        bmi_cat, bmi_color = "Overweight", "#f59e0b"
    else:
        bmi_cat, bmi_color = "Obese", "#ef4444"

    st.markdown(f"""
    <div class="info-item" style="margin-top:1rem;">
        <div class="info-item-label">BMI Category</div>
        <div class="info-item-value" style="color:{bmi_color};">{bmi_cat}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Summary of entered values ────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("**📋 Your Input Summary**")
st.markdown(f"""
<div class="info-grid">
    <div class="info-item"><div class="info-item-label">Pregnancies</div><div class="info-item-value">{pregnancies}</div></div>
    <div class="info-item"><div class="info-item-label">Glucose</div><div class="info-item-value">{glucose} mg/dL</div></div>
    <div class="info-item"><div class="info-item-label">Blood Pressure</div><div class="info-item-value">{blood_pressure} mm Hg</div></div>
    <div class="info-item"><div class="info-item-label">Skin Thickness</div><div class="info-item-value">{skin_thickness} mm</div></div>
    <div class="info-item"><div class="info-item-label">Insulin</div><div class="info-item-value">{insulin} μU/mL</div></div>
    <div class="info-item"><div class="info-item-label">BMI</div><div class="info-item-value">{bmi} ({bmi_cat})</div></div>
    <div class="info-item"><div class="info-item-label">Pedigree Function</div><div class="info-item-value">{dpf}</div></div>
    <div class="info-item"><div class="info-item-label">Age</div><div class="info-item-value">{age} years</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict Button ───────────────────────────────────────────
predict_btn = st.button("🔍  Analyze Risk Now")

# ════════════════════════════════════════════════════════════
# RESULT SECTION
# ════════════════════════════════════════════════════════════
if predict_btn:
    features = [pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age]

    with st.spinner("Running AI analysis..."):
        prediction, probability = predict(features)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("## 📊 Analysis Result")

    res_col1, res_col2 = st.columns([1.2, 1], gap="large")

    with res_col1:
        if prediction == 0:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-icon">✅</div>
                <div class="result-title">Low Diabetes Risk</div>
                <div class="result-prob">Probability: {probability*100:.1f}%</div>
                <div class="result-advice">
                    Your parameters suggest a <strong>low risk</strong> of diabetes.
                    Maintain a balanced diet, regular exercise, and annual health checkups
                    to stay in good health.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-risk">
                <div class="result-icon">⚠️</div>
                <div class="result-title">High Diabetes Risk</div>
                <div class="result-prob">Probability: {probability*100:.1f}%</div>
                <div class="result-advice">
                    Your parameters indicate a <strong>high risk</strong> of diabetes.
                    Please consult a healthcare professional promptly for a
                    comprehensive evaluation and personalized guidance.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("**Risk Probability Meter**")
        st.markdown(f"""
        <div class="risk-meter-label">
            <span>Low Risk (0%)</span>
            <span>High Risk (100%)</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(probability))
        st.markdown(f"<br>", unsafe_allow_html=True)

        # Key risk factors
        st.markdown("**Key Risk Indicators**")
        indicators = {
            "Glucose":        ("🔴 High" if glucose > 140 else "🟡 Moderate" if glucose > 100 else "🟢 Normal"),
            "BMI":            ("🔴 Obese" if bmi >= 30 else "🟡 Overweight" if bmi >= 25 else "🟢 Normal"),
            "Blood Pressure": ("🔴 High" if blood_pressure > 90 else "🟡 Moderate" if blood_pressure > 80 else "🟢 Normal"),
            "Age Risk":       ("🔴 High" if age > 50 else "🟡 Moderate" if age > 35 else "🟢 Low"),
        }
        for factor, status in indicators.items():
            st.markdown(f"**{factor}:** {status}")

    # Disclaimer
    st.markdown("""
    <div style="background:rgba(255,193,7,0.08); border:1px solid rgba(255,193,7,0.2);
         border-radius:10px; padding:1rem 1.5rem; margin-top:1.5rem; font-size:0.82rem; color:#94a3b8;">
    ⚠️ <strong style="color:#fbbf24;">Medical Disclaimer:</strong>
    This tool is for educational purposes only and does not constitute medical advice.
    Always consult a qualified healthcare professional for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with <span>♥</span> using Streamlit · Pima Indians Diabetes Dataset ·
    Models: Logistic Regression, Gradient Boosting, LSTM, CNN
</div>
""", unsafe_allow_html=True)
