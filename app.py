import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* Main App Background with better contrast */
.stApp {
    background: radial-gradient(circle at top right, #0F172A, #020617);
    color: #E5E7EB; /* Brightest white-blue for main text */
    font-family: 'Inter', sans-serif;
}

/* Section Card - Improved contrast and border */
.section-card {
    padding: 30px;
    border-radius: 20px;
    background: rgba(30, 41, 59, 0.7); /* Solid dark blue-grey for readability */
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    margin-bottom: 25px;
}

/* Headings - High Contrast Colors */
h1 {
    color: #22D3EE !important; /* Neon Cyan */
    font-weight: 700 !important;
    text-shadow: 0px 0px 10px rgba(34, 211, 238, 0.3);
}
h2 {
    color: #38BDF8 !important; /* Sky Blue */
    font-weight: 700 !important;
    margin-top: 20px !important;
}
h3 {
    color: #E2E8F0 !important; /* Off White */
    font-weight: 600 !important;
}

/* Labels for inputs - Bold and Clear */
.stWidgetLabel p {
    color: #FFFFFF !important; /* Bright Grey */
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 8px !important;
}
            
/* 🧾 Labels (VERY IMPORTANT FIX) */
label {
    color: #E5E7EB !important;
    font-weight: 500;
}

/* Inputs - Higher contrast text */
input, select, div[data-baseweb="select"] {
    background-color: #1E293B !important;
    color: #FFFFFF !important; /* Pure white for input text */
    border: 1px solid #475569 !important;
    border-radius: 10px !important;
}

/* Button - More Vibrant Gradient */
.stButton>button {
    background: linear-gradient(90deg, #06B6D4, #4F46E5) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    height: 3.5em !important;
    box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 20px rgba(6, 182, 212, 0.5) !important;
}

/* Custom Success/Error/Warning/Info Boxes */
.stAlert {
    border-radius: 15px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background-color: rgba(15, 23, 42, 0.8) !important;
}

/* Status Text Brightness */
.stAlert p {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

/* Divider Visibility */
hr {
    border: 0;
    height: 1px;
    background: linear-gradient(to right, transparent, #334155, transparent);
    margin: 30px 0 !important;
}

</style>
""", unsafe_allow_html=True)

def explain_risk(diabetes_input, heart_input, d_prob, h_prob):
    explanation = []

    if d_prob > 0.5:
        if diabetes_input['Glucose'] > 125:
            explanation.append("High glucose level increases diabetes risk.")
        if diabetes_input['BMI'] > 30:
            explanation.append("High BMI indicates obesity-related risk.")
        if diabetes_input['Age'] > 45:
            explanation.append("Age contributes to higher diabetes probability.")

    if h_prob > 0.5:
        if heart_input['chol'] > 240:
            explanation.append("High cholesterol increases heart disease risk.")
        if heart_input['trestbps'] > 140:
            explanation.append("High blood pressure is a major risk factor.")
        if heart_input['thalach'] < 100:
            explanation.append("Low max heart rate may indicate poor heart condition.")

    if not explanation:
        explanation.append("All health indicators are within normal range.")

    return explanation


# Load models
import os
BASE_DIR = os.path.dirname(__file__)

diabetes_model = joblib.load(os.path.join(BASE_DIR, "diabetes_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
heart_model = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))
heart_columns = joblib.load(os.path.join(BASE_DIR, "heart_columns.pkl"))

st.set_page_config(page_title="Health AI Predictor", layout="wide")

st.title("🧠 AI Health Predictor")
st.write("Predict Diabetes & Heart Disease Risk")

# ------------------- INPUTS -------------------

st.header("🩺 Diabetes Inputs")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 120)
    bp = st.number_input("Blood Pressure", 0, 150, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

st.header("❤️ Heart Inputs")

col3, col4 = st.columns(2)

with col3:
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)

with col4:
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    ca = st.number_input("Major Vessels", 0, 4, 0)

# ------------------- SUGGESTIONS -------------------

def get_suggestions(diabetes_result, heart_result):
    suggestions = []

    if diabetes_result == 1:
        suggestions.append("Reduce sugar intake")
        suggestions.append("Exercise regularly")

    if heart_result == 1:
        suggestions.append("Avoid oily food")
        suggestions.append("Control cholesterol")

    if diabetes_result == 0 and heart_result == 0:
        suggestions.append("Maintain healthy lifestyle")

    return suggestions

# ------------------- PREDICTION -------------------

if st.button("🔍 Predict"):

    with st.spinner("Analyzing your health..."):
        time.sleep(1.2)

    # Diabetes input
    diabetes_input = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age,
        'BMI_Age': bmi * age
    }

    df_d = pd.DataFrame([diabetes_input])

    # IMPORTANT: enforce column order
    df_d = df_d[diabetes_model.feature_names_in_]

    df_d = pd.DataFrame(
        scaler.transform(df_d),
        columns=diabetes_model.feature_names_in_
    )

    d_pred = diabetes_model.predict(df_d)[0]
    d_prob = diabetes_model.predict_proba(df_d)[0][1]

    # Heart input
    heart_input = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 0,
        'restecg': 'normal',
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': 'upsloping',
        'ca': ca,
        'thal': 'normal'
    }

    df_h = pd.DataFrame([heart_input])
    df_h = pd.get_dummies(df_h)
    df_h = df_h.reindex(columns=heart_columns, fill_value=0)

    h_pred = heart_model.predict(df_h)[0]
    h_prob = heart_model.predict_proba(df_h)[0][1]

    # AI Explanation
    explanations = explain_risk(diabetes_input, heart_input, d_prob, h_prob)

    st.subheader("🧠 AI Health Insights")

    for exp in explanations:
        st.warning(exp)

    # Suggestions
    suggestions = get_suggestions(d_pred, h_pred)

    # ------------------- OUTPUT -------------------

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Results")

    # ------------------ STATUS CARDS ------------------

    if d_pred:
        st.error(f"🩸 Diabetes Risk: {d_prob*100:.2f}%")
    else:
        st.success(f"🩸 Diabetes Safe: {d_prob*100:.2f}%")

    if h_pred:
        st.error(f"❤️ Heart Disease Risk: {h_prob*100:.2f}%")
    else:
        st.success(f"❤️ Heart Safe: {h_prob*100:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ RISK METERS ------------------

    st.write("### 🩸 Diabetes Risk Meter")
    st.progress(float(d_prob))

    st.write("### ❤️ Heart Risk Meter")
    st.progress(float(h_prob))

    # ------------------ RISK Gauge ------------------
    col1, col2 = st.columns(2)

    def create_gauge(value, title):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value * 100,
            title={'text': title},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00FFAA"},
                'steps': [
                    {'range': [0, 30], 'color': "#00FFAA"},
                    {'range': [30, 70], 'color': "#FFD700"},
                    {'range': [70, 100], 'color': "#FF4B4B"}
                ],
            }
        ))
        return fig

    with col1:
        st.plotly_chart(create_gauge(d_prob, "Diabetes Risk"), use_container_width=True)

    with col2:
        st.plotly_chart(create_gauge(h_prob, "Heart Risk"), use_container_width=True)

    # ------------------ CHART ------------------

    st.subheader("📊 Risk Comparison")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=["Diabetes", "Heart"],
        y=[d_prob * 100, h_prob * 100],
        text=[f"{d_prob*100:.1f}%", f"{h_prob*100:.1f}%"],
        textposition='auto'
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ------------------ SUGGESTIONS ------------------

    st.subheader("💡 Suggestions")

    for s in suggestions:
        st.info(s)
