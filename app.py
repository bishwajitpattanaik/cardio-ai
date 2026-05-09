import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI · Heart Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --card:      #161a22;
    --border:    #232832;
    --accent:    #e8365d;
    --accent2:   #ff6b6b;
    --safe:      #00d4a0;
    --text:      #eef0f4;
    --muted:     #6b7280;
    --glow:      rgba(232,54,93,0.18);
}

/* ── GLOBAL RESET ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 2rem 4rem 2rem !important; max-width: 1200px !important; margin: 0 auto; }

/* ── HERO SECTION ── */
.hero {
    background: linear-gradient(135deg, #0a0c10 0%, #12101a 50%, #0a0c10 100%);
    border-bottom: 1px solid var(--border);
    padding: 3rem 0 2.5rem 0;
    text-align: center;
    margin: 0 -2rem 2.5rem -2rem;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%; transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(232,54,93,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(232,54,93,0.12);
    border: 1px solid rgba(232,54,93,0.3);
    color: var(--accent2);
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    margin-bottom: 1.2rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.05;
    margin: 0 0 0.8rem 0;
    letter-spacing: -0.03em;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    max-width: 500px;
    margin: 0 auto !important;
    line-height: 1.7;
    font-weight: 300;
    text-align: center !important;
    display: block;
    width: 100%;
}
.hero-pulse {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: block;
    animation: hpulse 1.8s ease-in-out infinite;
}
@keyframes hpulse {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.12); }
}

/* ── SECTION HEADINGS ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.4rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 1.2rem;
}

/* ── CARDS ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(232,54,93,0.35); }

.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.4rem;
    flex-wrap: wrap;
}
.metric-chip {
    flex: 1;
    min-width: 110px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-chip .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-chip .lbl {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

/* ── RESULT BANNER ── */
.result-high {
    background: linear-gradient(135deg, rgba(232,54,93,0.15), rgba(232,54,93,0.05));
    border: 1.5px solid var(--accent);
    border-radius: 18px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(232,54,93,0.2);
    display: flex;
    flex-direction: column;
    align-items: center;
}
.result-low {
    background: linear-gradient(135deg, rgba(0,212,160,0.12), rgba(0,212,160,0.03));
    border: 1.5px solid var(--safe);
    border-radius: 18px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,212,160,0.15);
    display: flex;
    flex-direction: column;
    align-items: center;
}
.result-icon { font-size: 3.5rem; display: block; margin-bottom: 0.8rem; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
}
.result-high .result-title { color: var(--accent2); }
.result-low  .result-title { color: var(--safe); }
.result-desc {
    color: var(--muted);
    font-size: 0.9rem;
    max-width: 400px;
    margin: 0 auto !important;
    line-height: 1.7;
    text-align: center !important;
    display: block;
    width: 100%;
}

/* ── STREAMLIT WIDGET OVERRIDES ── */
[data-testid="stSlider"] > div > div > div {
    background: var(--accent) !important;
}
.stSlider [data-testid="stMarkdownContainer"] p {
    color: var(--muted) !important;
    font-size: 0.8rem !important;
}
div[data-baseweb="select"] {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] * { color: var(--text) !important; }
div[data-baseweb="input"] {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}
div[data-baseweb="input"] input { color: var(--text) !important; }

/* Number input buttons */
[data-testid="stNumberInput"] button {
    background: var(--border) !important;
    color: var(--text) !important;
}

label[data-testid="stWidgetLabel"] p {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
}

/* ── PREDICT BUTTON ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #e8365d, #c72048) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(232,54,93,0.35) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(232,54,93,0.5) !important;
}

/* ── DIVIDER ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
}

/* ── INFO CHIPS ── */
.info-chip {
    display: inline-block;
    background: rgba(107,114,128,0.12);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.72rem;
    color: var(--muted);
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}

/* ── DISCLAIMER ── */
.disclaimer {
    background: rgba(232,54,93,0.06);
    border-left: 3px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1.2rem;
    font-size: 0.78rem;
    color: var(--muted);
    line-height: 1.6;
    margin-top: 1.5rem;
}

/* ── FEATURE BARS ── */
.feat-bar-wrap { margin-bottom: 0.9rem; }
.feat-bar-header { display: flex; justify-content: space-between; margin-bottom: 0.3rem; }
.feat-bar-label { font-size: 0.78rem; color: var(--muted); }
.feat-bar-val   { font-size: 0.78rem; color: var(--text); font-weight: 600; }
.feat-bar-bg    { background: var(--border); border-radius: 4px; height: 6px; overflow: hidden; }
.feat-bar-fill  { height: 100%; border-radius: 4px; transition: width 0.8s ease; }

/* ── TABS ── */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("knn_heart_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    columns  = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, expected_columns = load_artifacts()

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-pulse">🫀</span>
    <div class="hero-badge">AI-Powered · KNN Classifier · Clinical Grade</div>
    <h1 class="hero-title">Cardio<span>AI</span></h1>
    <p class="hero-sub"><b>Advanced heart disease risk assessment tool by Bishwajit Pattanaik</b></p>
    <p class="hero-sub"><b>Enter your clinical measurements below</b></p>
</div>
""", unsafe_allow_html=True)

# ─── MODEL STATS ROW ──────────────────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
    <div class="metric-chip"><div class="val">89%</div><div class="lbl">Accuracy</div></div>
    <div class="metric-chip"><div class="val">KNN</div><div class="lbl">Algorithm</div></div>
    <div class="metric-chip"><div class="val">918</div><div class="lbl">Training Records</div></div>
    <div class="metric-chip"><div class="val">11</div><div class="lbl">Features</div></div>
    <div class="metric-chip"><div class="val">90%</div><div class="lbl">F1 Score</div></div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Risk Assessment", "📊  Data Insights"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # ── SECTION: PATIENT DEMOGRAPHICS ──
    st.markdown("<div class='section-label'>Step 01</div><div class='section-title'>Patient Demographics</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age (years)", 18, 100, 45, help="Patient's age in years")
    with col2:
        sex = st.selectbox("Biological Sex", ["M — Male", "F — Female"])
        sex = sex.split(" — ")[0]
    with col3:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No (0)", "Yes (1)"])
        fasting_bs = 1 if fasting_bs.startswith("Yes") else 0

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # ── SECTION: CARDIOVASCULAR VITALS ──
    st.markdown("<div class='section-label'>Step 02</div><div class='section-title'>Cardiovascular Vitals</div>", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    with col5:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    with col6:
        max_hr = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, 150)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # ── SECTION: CLINICAL INDICATORS ──
    st.markdown("<div class='section-label'>Step 03</div><div class='section-title'>Clinical Indicators</div>", unsafe_allow_html=True)

    col7, col8, col9 = st.columns(3)
    with col7:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA — Atypical Angina", "NAP — Non-Anginal Pain", "TA — Typical Angina", "ASY — Asymptomatic"],
            help="ATA: Atypical Angina · NAP: Non-Anginal Pain · TA: Typical Angina · ASY: Asymptomatic"
        )
        chest_pain = chest_pain.split(" — ")[0]

    with col8:
        resting_ecg = st.selectbox(
            "Resting ECG Result",
            ["Normal", "ST — ST-T Wave Abnormality", "LVH — Left Ventricular Hypertrophy"]
        )
        resting_ecg = resting_ecg.split(" — ")[0]

    with col9:
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["N — No", "Y — Yes"])
        exercise_angina = exercise_angina.split(" — ")[0]

    col10, col11 = st.columns(2)
    with col10:
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1,
                            help="ST depression induced by exercise relative to rest")
    with col11:
        st_slope = st.selectbox(
            "ST Slope",
            ["Up — Upsloping", "Flat — Flat", "Down — Downsloping"]
        )
        st_slope = st_slope.split(" — ")[0]

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # ── PREDICT BUTTON ──
    col_btn, col_empty = st.columns([1, 1])
    with col_btn:
        predict_clicked = st.button("🫀  Run Risk Assessment")

    # ── PREDICTION LOGIC ──
    if predict_clicked:
        with st.spinner("Analyzing cardiovascular biomarkers..."):
            time.sleep(0.8)

        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0]
        risk_pct = int(proba[1] * 100)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── RESULT BANNER ──
        if prediction == 1:
            st.markdown(f"""
            <div class="result-high">
                <span class="result-icon">⚠️</span>
                <p class="result-title">Elevated Cardiac Risk Detected</p>
                <p class="result-desc">The model estimates a <strong style="color:#ff6b6b">{risk_pct}%</strong> probability of heart disease based on the provided biomarkers. Immediate consultation with a cardiologist is strongly recommended.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                <span class="result-icon">✅</span>
                <p class="result-title">Low Cardiac Risk Profile</p>
                <p class="result-desc">The model estimates a <strong style="color:#00d4a0">{risk_pct}%</strong> probability of heart disease. Continue maintaining a healthy lifestyle and schedule regular check-ups.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── GAUGE CHART ──
        col_g, col_f = st.columns(2)

        with col_g:
            st.markdown("<div class='section-label'>Risk Score</div>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                number={'suffix': '%', 'font': {'size': 36, 'color': '#eef0f4', 'family': 'Syne'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#6b7280', 'tickfont': {'color': '#6b7280'}},
                    'bar': {'color': '#e8365d' if prediction == 1 else '#00d4a0'},
                    'bgcolor': '#161a22',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30],  'color': 'rgba(0,212,160,0.08)'},
                        {'range': [30, 70], 'color': 'rgba(255,200,50,0.08)'},
                        {'range': [70, 100],'color': 'rgba(232,54,93,0.08)'},
                    ],
                    'threshold': {'line': {'color': '#ffffff', 'width': 2}, 'thickness': 0.75, 'value': risk_pct}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='#161a22', plot_bgcolor='#161a22',
                font={'color': '#eef0f4', 'family': 'Syne'},
                margin=dict(l=20, r=20, t=30, b=20), height=250
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_f:
            st.markdown("<div class='section-label'>Feature Contribution (Normalized)</div>", unsafe_allow_html=True)
            feat_vals = input_df[expected_columns].values[0]
            feat_names_clean = [c.replace('_', ' ').replace('ChestPainType', 'ChestPain')
                                 .replace('ExerciseAngina', 'ExAngina') for c in expected_columns]
            feat_abs = np.abs(feat_vals)
            feat_max = feat_abs.max() if feat_abs.max() > 0 else 1

            # top 6 features
            top_idx = np.argsort(feat_abs)[::-1][:6]
            bars_html = ""
            colors = ["#e8365d", "#ff6b6b", "#ff9a9a", "#ffb3b3", "#ffd0d0", "#ffe8e8"]
            for i, idx in enumerate(top_idx):
                pct = int((feat_abs[idx] / feat_max) * 100)
                bars_html += f"""
                <div class="feat-bar-wrap">
                    <div class="feat-bar-header">
                        <span class="feat-bar-label">{feat_names_clean[idx]}</span>
                        <span class="feat-bar-val">{feat_vals[idx]:.1f}</span>
                    </div>
                    <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{pct}%;background:{colors[i]};"></div></div>
                </div>
                """
            st.markdown(bars_html, unsafe_allow_html=True)

        # ── DISCLAIMER ──
        st.markdown("""
        <div class="disclaimer">
            ⚕️ <strong>Medical Disclaimer</strong> — This tool is for informational purposes only and does not constitute medical advice.
            Results should not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("heart.csv")
        except:
            return None

    df = load_data()

    if df is not None:
        st.markdown("<div class='section-label'>Dataset Overview</div><div class='section-title'>Heart Disease Dataset Analytics</div>", unsafe_allow_html=True)

        # ── DATASET STATS ──
        total = len(df)
        diseased = df['HeartDisease'].sum()
        healthy = total - diseased

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-chip"><div class="val">{total}</div><div class="lbl">Total Patients</div></div>
            <div class="metric-chip"><div class="val" style="color:#e8365d">{diseased}</div><div class="lbl">Heart Disease</div></div>
            <div class="metric-chip"><div class="val" style="color:#00d4a0">{healthy}</div><div class="lbl">No Disease</div></div>
            <div class="metric-chip"><div class="val">{int(df['Age'].mean())}</div><div class="lbl">Avg Age</div></div>
            <div class="metric-chip"><div class="val">{int(df['MaxHR'].mean())}</div><div class="lbl">Avg Max HR</div></div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        # Age distribution
        with col_a:
            fig1 = px.histogram(df, x='Age', color='HeartDisease',
                                color_discrete_map={0: '#00d4a0', 1: '#e8365d'},
                                nbins=20, barmode='overlay', opacity=0.8,
                                labels={'HeartDisease': 'Heart Disease', 'Age': 'Age (years)'},
                                title='Age Distribution by Heart Disease Status')
            fig1.update_layout(
                paper_bgcolor='#161a22', plot_bgcolor='#111318',
                font={'color': '#eef0f4', 'family': 'DM Sans'},
                legend=dict(bgcolor='#111318', bordercolor='#232832'),
                title_font_family='Syne', title_font_size=14,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            fig1.update_xaxes(gridcolor='#232832', linecolor='#232832')
            fig1.update_yaxes(gridcolor='#232832', linecolor='#232832')
            st.plotly_chart(fig1, use_container_width=True)

        # Chest pain breakdown
        with col_b:
            cp_counts = df.groupby(['ChestPainType', 'HeartDisease']).size().reset_index(name='count')
            fig2 = px.bar(cp_counts, x='ChestPainType', y='count', color='HeartDisease',
                          color_discrete_map={0: '#00d4a0', 1: '#e8365d'},
                          barmode='group',
                          labels={'HeartDisease': 'Heart Disease', 'count': 'Count', 'ChestPainType': 'Chest Pain Type'},
                          title='Chest Pain Type vs. Heart Disease')
            fig2.update_layout(
                paper_bgcolor='#161a22', plot_bgcolor='#111318',
                font={'color': '#eef0f4', 'family': 'DM Sans'},
                legend=dict(bgcolor='#111318', bordercolor='#232832'),
                title_font_family='Syne', title_font_size=14,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            fig2.update_xaxes(gridcolor='#232832', linecolor='#232832')
            fig2.update_yaxes(gridcolor='#232832', linecolor='#232832')
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)

        # Cholesterol vs MaxHR scatter
        with col_c:
            fig3 = px.scatter(df, x='Cholesterol', y='MaxHR', color='HeartDisease',
                              color_discrete_map={0: '#00d4a0', 1: '#e8365d'},
                              opacity=0.65,
                              labels={'HeartDisease': 'Heart Disease', 'MaxHR': 'Max Heart Rate', 'Cholesterol': 'Cholesterol (mg/dL)'},
                              title='Cholesterol vs. Max Heart Rate')
            fig3.update_layout(
                paper_bgcolor='#161a22', plot_bgcolor='#111318',
                font={'color': '#eef0f4', 'family': 'DM Sans'},
                legend=dict(bgcolor='#111318', bordercolor='#232832'),
                title_font_family='Syne', title_font_size=14,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            fig3.update_xaxes(gridcolor='#232832', linecolor='#232832')
            fig3.update_yaxes(gridcolor='#232832', linecolor='#232832')
            st.plotly_chart(fig3, use_container_width=True)

        # ST Slope distribution
        with col_d:
            slope_counts = df.groupby(['ST_Slope', 'HeartDisease']).size().reset_index(name='count')
            fig4 = px.bar(slope_counts, x='ST_Slope', y='count', color='HeartDisease',
                          color_discrete_map={0: '#00d4a0', 1: '#e8365d'},
                          barmode='stack',
                          labels={'HeartDisease': 'Heart Disease', 'count': 'Count', 'ST_Slope': 'ST Slope'},
                          title='ST Slope Distribution')
            fig4.update_layout(
                paper_bgcolor='#161a22', plot_bgcolor='#111318',
                font={'color': '#eef0f4', 'family': 'DM Sans'},
                legend=dict(bgcolor='#111318', bordercolor='#232832'),
                title_font_family='Syne', title_font_size=14,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            fig4.update_xaxes(gridcolor='#232832', linecolor='#232832')
            fig4.update_yaxes(gridcolor='#232832', linecolor='#232832')
            st.plotly_chart(fig4, use_container_width=True)

        # ── RAW DATA PREVIEW ──
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Raw Data</div><div class='section-title'>Dataset Preview</div>", unsafe_allow_html=True)
        st.dataframe(
            df.head(20).style.applymap(
                lambda v: 'color: #e8365d; font-weight: bold' if v == 1 and isinstance(v, (int, float)) else '',
                subset=['HeartDisease']
            ),
            use_container_width=True, height=350
        )
    else:
        st.info("Place `heart.csv` in the same directory as app.py to view dataset insights.")

# ── FOOTER ──
st.markdown("""
<div style="text-align:center; margin-top: 3rem; color: #2d3748; font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; font-family: 'Syne', sans-serif;">
    CardioAI by Bishwajit Pattanaik· Heart Disease Predictor · For Educational Use Only
</div>
""", unsafe_allow_html=True)
