import streamlit as st
import joblib
import numpy as np
import pandas as pd
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Smartphone Addiction Predictor",
    page_icon="📱",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("Smartphone_Addiction_Prediction_System.pkl")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #020617, #0f172a, #1e293b);
    color: white;
}

.header {
    background: linear-gradient(135deg,#3b82f6,#8b5cf6);
    padding:25px;
    border-radius:20px;
    text-align:center;
    box-shadow:0 10px 30px rgba(0,0,0,0.4);
}

.card {
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:18px;
    backdrop-filter: blur(10px);
}

div[data-testid="stButton"] button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    color:white;
    border:none;
    border-radius:12px;
    font-size:18px;
    padding:12px;
    font-weight:600;
}

input[type="range"] {
    accent-color: #6366f1;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header">
    <h1 style="color:white;">📱 Smartphone Addiction Predictor</h1>
    <p style="color:#e0e7ff;">AI-powered behavioral analysis for addiction risk detection</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# KPI CARDS
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", "XGBoost")
c2.metric("Accuracy", "92%")
c3.metric("Type", "Classification")
c4.metric("Status", "Active")

st.write("")

# =========================
# INPUTS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Usage Behavior")
    screen_time = st.slider("Daily Screen Time", 0.0, 15.0, 6.0)
    social_media = st.slider("Social Media Hours", 0.0, 10.0, 3.0)
    gaming = st.slider("Gaming Hours", 0.0, 10.0, 2.0)
    notifications = st.slider("Notifications per Day", 0, 300, 80)
    app_opens = st.slider("App Opens per Day", 0, 200, 40)
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    age = st.slider("Age", 10, 60, 25)
    work_study_hours = st.slider("Work/Study Hours", 0.0, 12.0, 6.0)
    weekend_screen_time = st.slider("Weekend Screen Time", 0.0, 15.0, 7.0)

with col2:
    st.subheader("📈 Personal Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    academic = st.selectbox("Academic Impact", ["Low", "Medium", "High"])

st.write("")

# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict Addiction Risk"):

    gender_map = {"Male": 1, "Female": 0}
    stress_map = {"Low": 0, "Medium": 1, "High": 2}
    academic_map = {"Low": 0, "Medium": 1, "High": 2}

    gender_val = gender_map[gender]
    stress_val = stress_map[stress]
    academic_val = academic_map[academic]

    input_data = pd.DataFrame([{
    "age": age,
    "gender": gender_val,
    "daily_screen_time_hours": screen_time,
    "social_media_hours": social_media,
    "gaming_hours": gaming,
    "work_study_hours": work_study_hours,
    "sleep_hours": sleep,
    "notifications_per_day": notifications,
    "app_opens_per_day": app_opens,
    "weekend_screen_time": weekend_screen_time,
    "stress_level": stress_val,
    "academic_work_impact": academic_val
}])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("⚠ High Risk of Smartphone Addiction")
    else:
        st.success("✅ Low Risk of Smartphone Addiction")

    st.progress(float(prob))
    st.write(f"Probability: {prob*100:.2f}%")
# =========================
# FOOTER
# =========================
st.write("---")
st.markdown("<center>🚀 Developed by Vishal Borse</center>", unsafe_allow_html=True)
