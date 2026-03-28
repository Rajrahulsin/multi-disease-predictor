import streamlit as st
import joblib
import numpy as np
import pandas as pd
from database.db import init_db, save_prediction, get_history

st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2d2f3e;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2d2f3e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-card h2 { color: #7c83fd; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #8b8fa8; margin: 4px 0 0 0; font-size: 0.85rem; }
    .result-positive {
        background: linear-gradient(135deg, #2d1b1b, #3d2020);
        border: 1px solid #ff4b4b;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #1b2d1b, #203d20);
        border: 1px solid #00c853;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .result-positive h2 { color: #ff4b4b; font-size: 2rem; }
    .result-negative h2 { color: #00c853; font-size: 2rem; }
    .result-positive p  { color: #ff8a80; font-size: 1.1rem; }
    .result-negative p  { color: #69f0ae; font-size: 1.1rem; }
    .section-header {
        border-left: 4px solid #7c83fd;
        padding-left: 12px;
        margin: 20px 0 16px 0;
    }
    .section-header h3 { color: #e0e0e0; margin: 0; }
    .section-header p  { color: #8b8fa8; margin: 4px 0 0 0; font-size: 0.85rem; }
    .disease-badge {
        display: inline-block;
        background: #7c83fd22;
        border: 1px solid #7c83fd55;
        color: #7c83fd;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 16px;
    }
    .param-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 8px;
    }
    label { color: #c0c4d6 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #7c83fd, #5c63d8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 32px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    hr { border-color: #2d2f3e; }
    .footer {
        text-align: center;
        color: #8b8fa8;
        font-size: 0.8rem;
        padding: 20px 0;
        border-top: 1px solid #2d2f3e;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

init_db()

@st.cache_resource
def load_models():
    return {
        "Heart Disease" : joblib.load("saved_models/heart_svm.pkl"),
        "Diabetes"      : joblib.load("saved_models/diabetes_svm.pkl"),
        "Liver Disease" : joblib.load("saved_models/liver_svm.pkl"),
        "Heart Symptom" : joblib.load("saved_models/heart_attack_symptom_svm.pkl"),
        "Diabetes Symptom" : joblib.load("saved_models/diabetes_symptom_svm.pkl"),
        "Liver Symptom" : joblib.load("saved_models/liver_symptom_svm.pkl"),
    }
models = load_models()

with st.sidebar:
    st.markdown("## 🧬 MediPredict AI")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Navigation", [
        "General Predict",
        "Predict",
        "AI Health Assistant",
        "Parameter Guide",
        "Dataset Info",
        "History"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='color:#8b8fa8; font-size:0.85rem; line-height:1.8'>
    <b style='color:#7c83fd'>For Everyone:</b><br>
    General Predict · AI Assistant<br><br>
    <b style='color:#7c83fd'>For Clinicians:</b><br>
    Clinical Predict (blood test values)<br><br>
    <b style='color:#7c83fd'>Diseases:</b><br>
    Heart · Diabetes · Liver
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='color:#4a4d5e; font-size:0.75rem'>Final Year B.Tech Project<br>Developed by Rahul Raj Singh</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════
if page == "Predict":
    st.markdown("## Disease Risk Prediction")
    st.markdown("<p style='color:#8b8fa8'>Enter patient details below to predict disease risk</p>", unsafe_allow_html=True)
    st.markdown("---")

    rows = get_history()
    col1, col2, col3 = st.columns(3)
    pos = sum(1 for r in rows if r[3] == "Positive")
    neg = len(rows) - pos
    with col1:
        st.markdown(f"<div class='metric-card'><h2>{len(rows)}</h2><p>Total Predictions</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h2>{pos}</h2><p>Positive Cases</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h2>{neg}</h2><p>Negative Cases</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    disease = st.selectbox("Select Disease to Predict", [
        "Heart Disease", "Diabetes", "Liver Disease"
    ])
    st.markdown(f"<div class='disease-badge'>🧬 {disease}</div>", unsafe_allow_html=True)
    st.caption("Not sure what a parameter means? Check the **Parameter Guide** in the sidebar.")

    if disease == "Heart Disease":
        st.markdown("<div class='section-header'><h3>Patient Information</h3></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age      = st.slider("Age (years)", 20, 80, 50)
            sex      = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
            cp       = st.selectbox("Chest Pain Type", [0,1,2,3],
                                    format_func=lambda x: ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"][x])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol     = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
            fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1],
                                    format_func=lambda x: "No (Normal)" if x==0 else "Yes (Elevated)")
            restecg  = st.selectbox("Resting ECG Result", [0,1,2],
                                    format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])
        with col2:
            thalch   = st.number_input("Max Heart Rate Achieved (bpm)", 60, 220, 150)
            exang    = st.selectbox("Exercise Induced Angina", [0,1],
                                    format_func=lambda x: "No" if x==0 else "Yes")
            oldpeak  = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
            slope    = st.selectbox("Slope of ST Segment", [0,1,2],
                                    format_func=lambda x: ["Upsloping (Normal)","Flat (Borderline)","Downsloping (Concerning)"][x])
            ca       = st.selectbox("Major Vessels Colored (0=Best, 3=Worst)", [0,1,2,3])
            thal     = st.selectbox("Thalassemia", [0,1,2,3],
                                    format_func=lambda x: ["Normal","Fixed Defect","Reversible Defect","Unknown"][x])
        input_data    = [age,sex,cp,trestbps,chol,fbs,restecg,thalch,exang,oldpeak,slope,ca,thal]
        feature_names = ["age","sex","cp","trestbps","chol","fbs","restecg","thalch","exang","oldpeak","slope","ca","thal"]

        warnings = []
        if trestbps > 140: warnings.append("Blood Pressure is high (above 140 mm Hg)")
        if chol > 240:     warnings.append("Cholesterol is high (above 240 mg/dl)")
        if thalch < 100:   warnings.append("Max Heart Rate is low (below 100 bpm)")
        if oldpeak > 2:    warnings.append("ST Depression is significant (above 2)")
        for w in warnings: st.warning(w)

    elif disease == "Diabetes":
        st.markdown("<div class='section-header'><h3>Patient Information</h3></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.slider("Number of Pregnancies", 0, 17, 3)
            glucose     = st.number_input("Glucose Level (mg/dl)", 0, 200, 120)
            bp          = st.number_input("Blood Pressure (mm Hg)", 0, 130, 70)
            skin        = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        with col2:
            insulin     = st.number_input("Insulin Level (IU/ml)", 0, 900, 80)
            bmi         = st.number_input("BMI (Body Mass Index)", 0.0, 70.0, 25.0)
            dpf         = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age         = st.slider("Age (years)", 10, 100, 30)
        input_data    = [pregnancies,glucose,bp,skin,insulin,bmi,dpf,age]
        feature_names = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

        warnings = []
        if glucose > 126: warnings.append("Glucose is in diabetic range (above 126 mg/dl)")
        if glucose > 100 and glucose <= 126: warnings.append("Glucose is in prediabetic range (100-126 mg/dl)")
        if bmi > 30:      warnings.append("BMI indicates obesity (above 30)")
        if bmi > 25 and bmi <= 30: warnings.append("BMI indicates overweight (25-30)")
        if bp > 90:       warnings.append("Blood Pressure is high (above 90 mm Hg)")
        if insulin > 25:  warnings.append("Insulin level is elevated (above 25 IU/ml)")
        for w in warnings: st.warning(w)

    elif disease == "Liver Disease":
        st.markdown("<div class='section-header'><h3>Patient Information</h3></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age     = st.slider("Age (years)", 4, 90, 40)
            gender  = st.selectbox("Gender", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
            tb      = st.number_input("Total Bilirubin (mg/dl)", 0.0, 75.0, 1.0)
            db      = st.number_input("Direct Bilirubin (mg/dl)", 0.0, 20.0, 0.3)
            alkphos = st.number_input("Alkaline Phosphotase (IU/L)", 60, 2110, 200)
        with col2:
            sgpt    = st.number_input("SGPT / ALT (IU/L)", 10, 2000, 35)
            sgot    = st.number_input("SGOT / AST (IU/L)", 10, 5000, 40)
            tp      = st.number_input("Total Proteins (g/dl)", 2.0, 10.0, 6.5)
            alb     = st.number_input("Albumin (g/dl)", 0.0, 6.0, 3.5)
            agr     = st.number_input("Albumin/Globulin Ratio", 0.0, 3.0, 1.0)
        input_data    = [age,gender,tb,db,alkphos,sgpt,sgot,tp,alb,agr]
        feature_names = ["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase",
                         "Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens",
                         "Albumin","Albumin_and_Globulin_Ratio"]

        warnings = []
        if tb > 2.0:      warnings.append("Total Bilirubin is high (above 2.0 mg/dl)")
        if db > 0.3:      warnings.append("Direct Bilirubin is elevated (above 0.3 mg/dl)")
        if alkphos > 147: warnings.append("Alkaline Phosphotase is high (above 147 IU/L)")
        if sgpt > 56:     warnings.append("SGPT is elevated (above 56 IU/L)")
        if sgot > 40:     warnings.append("SGOT is elevated (above 40 IU/L)")
        if alb < 3.5:     warnings.append("Albumin is low (below 3.5 g/dl)")
        if agr < 1.0:     warnings.append("Albumin/Globulin Ratio is low (below 1.0)")
        for w in warnings: st.warning(w)

    st.markdown("---")
    if st.button("Run Prediction"):
        X_input    = pd.DataFrame([input_data], columns=feature_names)
        model      = models[disease]
        prediction = model.predict(X_input)[0]
        confidence = model.predict_proba(X_input)[0][int(round(prediction))] * 100
        prediction = int(round(prediction))
        result     = "Positive" if prediction == 1 else "Negative"

        st.markdown("---")
        st.markdown("### Prediction Result")

        if prediction == 1:
            st.markdown(f"""
            <div class='result-positive'>
                <h2>Risk Detected</h2>
                <p>Prediction: {result} &nbsp;|&nbsp; Confidence: {confidence:.1f}% &nbsp;|&nbsp; Model: SVM (RBF Kernel)</p>
                <p style='color:#ff8a80; font-size:0.85rem; margin-top:10px'>
                This is a screening tool only. Please consult a qualified medical professional for proper diagnosis.
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-negative'>
                <h2>No Significant Risk</h2>
                <p>Prediction: {result} &nbsp;|&nbsp; Confidence: {confidence:.1f}% &nbsp;|&nbsp; Model: SVM (RBF Kernel)</p>
                <p style='color:#69f0ae; font-size:0.85rem; margin-top:10px'>
                No significant risk detected. Maintain a healthy lifestyle and regular checkups.
                </p>
            </div>""", unsafe_allow_html=True)

        save_prediction(
            disease=disease,
            input_data=dict(zip(feature_names, input_data)),
            prediction=result,
            confidence=confidence,
            model_used="SVM"
        )
        st.toast("Prediction saved to history!", icon="✅")

    st.markdown("<div class='footer'>MediPredict AI · Multi Disease Prediction System · Developed by Rahul Raj Singh</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE — GENERAL PREDICT
# ══════════════════════════════════════════════════════
elif page == "General Predict":
    st.markdown("## 👤 General Health Risk Check")
    st.markdown("<p style='color:#8b8fa8'>Answer simple Yes/No questions about your symptoms — no medical knowledge needed!</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Info banner
    st.markdown("""
    <div style='background:#1e2130; border-left:4px solid #7c83fd;
    border-radius:0 10px 10px 0; padding:12px 16px; margin-bottom:16px;'>
        <p style='color:#7c83fd; margin:0; font-size:0.85rem; font-weight:500'>
        For Common Users</p>
        <p style='color:#8b8fa8; margin:4px 0 0 0; font-size:0.85rem'>
        This screening tool uses symptom-based questions. No blood test or medical report needed.
        Results are for awareness only — always consult a doctor for proper diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

    disease = st.selectbox("Select Disease to Check", [
        "Heart Disease", "Diabetes", "Liver Disease"
    ])
    st.markdown(f"<div class='disease-badge'>👤 {disease} · Symptom Check</div>",
                unsafe_allow_html=True)

    # ── Heart Disease Symptom ───────────────────────────
    if disease == "Heart Disease":
        st.markdown("<div class='section-header'><h3>Heart Disease Symptoms</h3><p>Answer based on what you have been experiencing recently</p></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            age         = st.slider("Age", 10, 90, 40)
            gender      = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
            chest_pain  = st.selectbox("Do you have chest pain?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            breath      = st.selectbox("Shortness of breath?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            fatigue     = st.selectbox("Do you feel fatigued/tired?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            palpitation = st.selectbox("Heart palpitations (racing heart)?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            dizziness   = st.selectbox("Do you feel dizzy?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            swelling    = st.selectbox("Swelling in legs or feet?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            pain_arms   = st.selectbox("Pain in arms, jaw, or back?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            cold_sweats = st.selectbox("Cold sweats or nausea?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        with col2:
            high_bp     = st.selectbox("Do you have high blood pressure?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            high_chol   = st.selectbox("Do you have high cholesterol?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            diabetes    = st.selectbox("Do you have diabetes?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            smoking     = st.selectbox("Do you smoke?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            obesity     = st.selectbox("Are you overweight/obese?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            sedentary   = st.selectbox("Sedentary lifestyle (no exercise)?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            family_hist = st.selectbox("Family history of heart disease?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            stress      = st.selectbox("Do you have chronic stress?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

        input_data = [chest_pain, breath, fatigue, palpitation, dizziness,
                      swelling, pain_arms, cold_sweats, high_bp, high_chol,
                      diabetes, smoking, obesity, sedentary, family_hist,
                      stress, gender, age]
        feature_names = ['Chest_Pain','Shortness_of_Breath','Fatigue','Palpitations',
                         'Dizziness','Swelling','Pain_Arms_Jaw_Back','Cold_Sweats_Nausea',
                         'High_BP','High_Cholesterol','Diabetes','Smoking','Obesity',
                         'Sedentary_Lifestyle','Family_History','Chronic_Stress',
                         'Gender','Age']
        model = models["Heart Symptom"]

        # Warnings
        warnings = []
        if chest_pain == 1 and breath == 1:
            warnings.append("Chest pain with shortness of breath — seek medical attention soon!")
        if cold_sweats == 1 and pain_arms == 1:
            warnings.append("Cold sweats with arm/jaw pain — possible heart attack symptom!")
        for w in warnings: st.warning(w)

    # ── Diabetes Symptom ────────────────────────────────
    elif disease == "Diabetes":
        st.markdown("<div class='section-header'><h3>Diabetes Symptoms</h3><p>Answer based on what you have been experiencing recently</p></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            age         = st.slider("Age", 10, 90, 35)
            gender      = st.selectbox("Gender", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
            freq_urine  = st.selectbox("Frequent urination?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            excess_thirst = st.selectbox("Excessive thirst?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            weight_loss = st.selectbox("Sudden weight loss?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            weakness    = st.selectbox("General weakness?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            excess_hunger = st.selectbox("Excessive hunger?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            visual_blur = st.selectbox("Visual blurring?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        with col2:
            itching     = st.selectbox("Itching?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            irritability = st.selectbox("Irritability?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            delayed_healing = st.selectbox("Delayed healing of wounds?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            muscle_weak = st.selectbox("Muscle weakness?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            muscle_stiff = st.selectbox("Muscle stiffness?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            hair_loss   = st.selectbox("Hair loss?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            overweight  = st.selectbox("Are you overweight?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

        input_data = [age, gender, freq_urine, excess_thirst, weight_loss,
                      weakness, excess_hunger, visual_blur, itching,
                      irritability, delayed_healing, muscle_weak,
                      muscle_stiff, hair_loss, overweight]
        feature_names = ['Age','Gender','Frequent urination','Excessive thirst',
                 'sudden weight loss','weakness','Excessive hunger',
                 'visual blurring','Itching','Irritability',
                 'delayed healing','Muscle weakness',
                 'muscle stiffness','Hair loss','Overweight']
        model = models["Diabetes Symptom"]

        # Warnings
        warnings = []
        if freq_urine == 1 and excess_thirst == 1:
            warnings.append("Frequent urination with excessive thirst — classic diabetes symptoms!")
        if weight_loss == 1 and weakness == 1:
            warnings.append("Sudden weight loss with weakness — consider getting a glucose test!")
        for w in warnings: st.warning(w)

    # ── Liver Disease Symptom ───────────────────────────
    elif disease == "Liver Disease":
        st.markdown("<div class='section-header'><h3>Liver Disease Symptoms</h3><p>Answer based on what you have been experiencing recently</p></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            jaundice    = st.selectbox("Yellowing of skin or eyes (jaundice)?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            nausea      = st.selectbox("Nausea?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            vomiting    = st.selectbox("Vomiting?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            abd_pain    = st.selectbox("Abdominal pain?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        with col2:
            fatigue     = st.selectbox("Fatigue or tiredness?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            loss_app    = st.selectbox("Loss of appetite?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            dark_urine  = st.selectbox("Dark colored urine?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            itching     = st.selectbox("Itching?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

        input_data = [jaundice, nausea, vomiting, abd_pain,
                      fatigue, loss_app, dark_urine, itching]
        feature_names = ['jaundice','nausea','vomiting','abdominal_pain',
                         'fatigue','loss_of_appetite','dark_urine','itching']
        model = models["Liver Symptom"]

        # Warnings
        warnings = []
        if jaundice == 1:
            warnings.append("Yellowing of skin/eyes detected — please see a doctor immediately!")
        if dark_urine == 1 and abd_pain == 1:
            warnings.append("Dark urine with abdominal pain — possible liver issue!")
        for w in warnings: st.warning(w)

    # ── Predict Button ──────────────────────────────────
    st.markdown("---")
    if st.button("Check My Risk", type="primary"):
        X_input    = pd.DataFrame([input_data], columns=feature_names)
        prediction = model.predict(X_input)[0]
        prediction = int(round(prediction))
        confidence = model.predict_proba(X_input)[0][prediction] * 100
        result     = "Positive" if prediction == 1 else "Negative"

        st.markdown("---")
        st.markdown("### Result")

        if prediction == 1:
            st.markdown(f"""
            <div class='result-positive'>
                <h2>⚠️ Risk Detected</h2>
                <p>Prediction: {result} &nbsp;|&nbsp; Confidence: {confidence:.1f}%</p>
                <p style='color:#ff8a80; font-size:0.85rem; margin-top:10px'>
                Based on your symptoms there is a potential risk. Please consult a doctor for proper diagnosis.
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-negative'>
                <h2>✅ No Significant Risk</h2>
                <p>Prediction: {result} &nbsp;|&nbsp; Confidence: {confidence:.1f}%</p>
                <p style='color:#69f0ae; font-size:0.85rem; margin-top:10px'>
                No significant risk detected based on your symptoms. Maintain a healthy lifestyle!
                </p>
            </div>""", unsafe_allow_html=True)

        save_prediction(
            disease=f"{disease} (Symptom)",
            input_data=dict(zip(feature_names, input_data)),
            prediction=result,
            confidence=confidence,
            model_used="SVM Symptom"
        )
        st.toast("Result saved to history!", icon="✅")

    st.markdown("<div class='footer'>MediPredict AI · Multi Disease Prediction System · Developed by Rahul Raj Singh</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 2 — AI HEALTH ASSISTANT (Connected to ML)
# ══════════════════════════════════════════════════════
elif page == "AI Health Assistant":
    st.markdown("## 🧬 AI Health Assistant")
    st.markdown("<p style='color:#8b8fa8'>Chat about your symptoms — AI will collect information and run our ML model to assess your risk</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <style>
    .chat-container { display:flex; flex-direction:column; gap:16px; padding:16px 0 20px 0; }
    .bubble-row-user { display:flex; justify-content:flex-end; align-items:flex-end; gap:8px; }
    .bubble-row-ai { display:flex; justify-content:flex-start; align-items:flex-end; gap:8px; }
    .bubble-user {
        background: linear-gradient(135deg, #7c83fd, #5c63d8);
        color: white; padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%; font-size: 0.95rem; line-height: 1.6; word-wrap: break-word;
    }
    .bubble-ai {
        background: #1e2130; color: #c0c4d6;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%; font-size: 0.95rem; line-height: 1.6;
        border: 1px solid #2d2f3e; word-wrap: break-word;
    }
    .bubble-result {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 2px solid #7c83fd;
        border-radius: 12px; padding: 16px;
        max-width: 80%; font-size: 0.9rem; line-height: 1.7;
        color: #c0c4d6;
    }
    .av-user {
        width:32px; height:32px; border-radius:50%;
        background: linear-gradient(135deg, #7c83fd, #5c63d8);
        display:flex; align-items:center; justify-content:center;
        font-size:14px; flex-shrink:0;
    }
    .av-ai {
        width:32px; height:32px; border-radius:50%;
        background:#1e2130; border:1px solid #2d2f3e;
        display:flex; align-items:center; justify-content:center;
        font-size:14px; flex-shrink:0;
    }
    </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "last_audio" not in st.session_state:
        st.session_state.last_audio = None
    if "input_value" not in st.session_state:
        st.session_state.input_value = ""
    if "prediction_done" not in st.session_state:
        st.session_state.prediction_done = False

    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        st.error("API key not configured.")
        st.stop()

    col_lang, _ = st.columns([2, 3])
    with col_lang:
        language = st.selectbox(
            "Language",
            ["English", "Hindi", "Gujarati", "Tamil", "Telugu", "Bengali", "Marathi"],
            label_visibility="collapsed"
        )

    lang_map = {
        "English":  {"code": "en-US", "tts": "en"},
        "Hindi":    {"code": "hi-IN", "tts": "hi"},
        "Gujarati": {"code": "gu-IN", "tts": "gu"},
        "Tamil":    {"code": "ta-IN", "tts": "ta"},
        "Telugu":   {"code": "te-IN", "tts": "te"},
        "Bengali":  {"code": "bn-IN", "tts": "bn"},
        "Marathi":  {"code": "mr-IN", "tts": "mr"},
    }
    lang_code = lang_map[language]["code"]
    lang_tts  = lang_map[language]["tts"]

    SYSTEM_PROMPT = """You are MediPredict AI — a medical screening chatbot connected to ML models.

Ask symptoms for ONE disease based on user concern:
- Heart Disease keys: Chest_Pain, Shortness_of_Breath, Fatigue, Palpitations, Dizziness, Swelling, Pain_Arms_Jaw_Back, Cold_Sweats_Nausea, High_BP, High_Cholesterol, Diabetes, Smoking, Obesity, Sedentary_Lifestyle, Family_History, Chronic_Stress, Gender, Age
- Diabetes keys: Age, Gender, Frequent urination, Excessive thirst, sudden weight loss, weakness, Excessive hunger, visual blurring, Itching, Irritability, delayed healing, Muscle weakness, muscle stiffness, Hair loss, Overweight
- Liver Disease keys: jaundice, nausea, vomiting, abdominal_pain, fatigue, loss_of_appetite, dark_urine, itching

RULES:
1. Ask 2-3 questions at a time
2. After EXACTLY 2 rounds of answers OUTPUT THE PREDICTION BLOCK — no exceptions
3. Use 1=Yes, 0=No, actual number for Age, 1=Male/0=Female for Gender
4. For emergency symptoms tell user to call emergency services immediately

AFTER 2 ROUNDS YOU MUST OUTPUT THIS AT THE END OF YOUR RESPONSE:
[PREDICTION]
{"ready_to_predict": true, "disease": "Diabetes", "symptoms": {"Age": 35, "Gender": 1, "Frequent urination": 1, "Excessive thirst": 1, "sudden weight loss": 0, "weakness": 1, "Excessive hunger": 1, "visual blurring": 1, "Itching": 0, "Irritability": 1, "delayed healing": 1, "Muscle weakness": 0, "muscle stiffness": 0, "Hair loss": 0, "Overweight": 0}}
[/PREDICTION]

Replace values with actual user answers. NEVER ask a 3rd round."""

    def run_ml_prediction(disease, symptoms):
        try:
            model_map = {
                "Heart Disease":  models["Heart Symptom"],
                "Diabetes":       models["Diabetes Symptom"],
                "Liver Disease":  models["Liver Symptom"],
            }
            model = model_map.get(disease)
            if not model:
                return None, None
            X_input = pd.DataFrame([symptoms])
            prediction = model.predict(X_input)[0]
            prediction = int(round(prediction))
            confidence = model.predict_proba(X_input)[0][prediction] * 100
            result = "Positive" if prediction == 1 else "Negative"
            return result, confidence
        except Exception as e:
            return None, str(e)

    if st.session_state.messages:
        chat_html = "<div class='chat-container'>"
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"""
                <div class='bubble-row-user'>
                    <div class='bubble-user'>{msg['content']}</div>
                    <div class='av-user'>👤</div>
                </div>"""
            elif msg["role"] == "assistant":
                chat_html += f"""
                <div class='bubble-row-ai'>
                    <div class='av-ai'>🧬</div>
                    <div class='bubble-ai'>{msg['content']}</div>
                </div>"""
            elif msg["role"] == "prediction":
                chat_html += f"""
                <div class='bubble-row-ai'>
                    <div class='av-ai'>📊</div>
                    <div class='bubble-result'>{msg['content']}</div>
                </div>"""
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding:40px; color:#4a4d5e;'>
            <div style='font-size:2.5rem'>🧬</div>
            <p style='margin-top:8px'>Tell me about your symptoms and I will assess your health risk using our ML model</p>
        </div>""", unsafe_allow_html=True)

    if st.session_state.last_audio:
        st.audio(st.session_state.last_audio, format="audio/mp3", autoplay=True)

    st.markdown("---")

    voice_component = f"""
    <div style="margin-bottom:12px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <button id="micBtn" onclick="startListening()" style="
                background: linear-gradient(135deg, #7c83fd, #5c63d8);
                color:white; border:none; border-radius:8px;
                padding:8px 16px; font-size:0.9rem; cursor:pointer;">
                🎤 Speak ({language})
            </button>
            <span id="status" style="color:#8b8fa8; font-size:0.85rem;"></span>
        </div>
    </div>
    <script>
    function startListening() {{
        const btn = document.getElementById('micBtn');
        const status = document.getElementById('status');
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
            status.innerHTML = '<span style="color:#ff4b4b">Not supported. Use Chrome.</span>';
            return;
        }}
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SR();
        recognition.lang = '{lang_code}';
        recognition.interimResults = false;
        btn.innerHTML = '🔴 Listening...';
        btn.style.background = '#ff4b4b';
        status.innerHTML = '<span style="color:#7c83fd">Listening...</span>';
        recognition.onresult = function(e) {{
            const text = e.results[0][0].transcript;
            status.innerHTML = '<span style="color:#00c853">Got: ' + text + '</span>';
            setTimeout(() => {{
                const inputs = window.parent.document.querySelectorAll('input[type="text"]');
                for (let inp of inputs) {{
                    if (inp.placeholder && inp.placeholder.includes('symptoms')) {{
                        const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                        setter.call(inp, text);
                        inp.dispatchEvent(new Event('input', {{bubbles:true}}));
                        inp.focus();
                        break;
                    }}
                }}
            }}, 300);
        }};
        recognition.onend = function() {{
            btn.innerHTML = '🎤 Speak ({language})';
            btn.style.background = 'linear-gradient(135deg, #7c83fd, #5c63d8)';
        }};
        recognition.onerror = function(e) {{
            btn.innerHTML = '🎤 Speak ({language})';
            btn.style.background = 'linear-gradient(135deg, #7c83fd, #5c63d8)';
            status.innerHTML = '<span style="color:#ff4b4b">Error: ' + e.error + '</span>';
        }};
        recognition.start();
    }}
    </script>
    """
    st.components.v1.html(voice_component, height=70)

    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Message",
            value=st.session_state.input_value,
            placeholder="Describe your symptoms or ask a health question...",
            label_visibility="collapsed",
            key="chat_text_input"
        )
    with col2:
        send_clicked = st.button("Send ➤", type="primary", use_container_width=True)

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.conversation = []
            st.session_state.last_audio = None
            st.session_state.input_value = ""
            st.session_state.prediction_done = False
            st.rerun()

    if send_clicked and user_input.strip():
        message_to_send = user_input.strip()
        st.session_state.input_value = ""

        st.session_state.messages.append({"role": "user", "content": message_to_send})
        st.session_state.conversation.append({"role": "user", "content": message_to_send})

        with st.spinner("Thinking..."):
            try:
                from groq import Groq
                import json
                import re

                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=2048,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.conversation
                )
                assistant_reply = response.choices[0].message.content

                # ── Check for prediction block ─────────────
                json_match = re.search(
                    r'\[PREDICTION\]\s*(\{.*?"ready_to_predict".*?\})\s*\[/PREDICTION\]',
                    assistant_reply, re.DOTALL
                )

                if json_match and not st.session_state.prediction_done:
                    try:
                        prediction_data = json.loads(json_match.group(1))
                        if prediction_data.get("ready_to_predict"):
                            disease  = prediction_data.get("disease")
                            symptoms = prediction_data.get("symptoms", {})
                            result, confidence = run_ml_prediction(disease, symptoms)

                            clean_reply = re.sub(
                                r'\[PREDICTION\].*?\[/PREDICTION\]',
                                '', assistant_reply, flags=re.DOTALL
                            ).strip()

                            if clean_reply:
                                st.session_state.messages.append({"role": "assistant", "content": clean_reply})
                                st.session_state.conversation.append({"role": "assistant", "content": clean_reply})

                            if result:
                                if result == "Positive":
                                    result_html = f"""
                                    <b style='color:#ff4b4b; font-size:1.1rem'>⚠️ Risk Detected — {disease}</b><br><br>
                                    <b>ML Model Prediction:</b> {result}<br>
                                    <b>Confidence:</b> {confidence:.1f}%<br>
                                    <b>Model:</b> SVM (Symptom-based)<br><br>
                                    <span style='color:#ff8a80; font-size:0.85rem'>
                                    This is a screening result. Please consult a qualified doctor for proper diagnosis.
                                    </span>"""
                                else:
                                    result_html = f"""
                                    <b style='color:#00c853; font-size:1.1rem'>✅ No Significant Risk — {disease}</b><br><br>
                                    <b>ML Model Prediction:</b> {result}<br>
                                    <b>Confidence:</b> {confidence:.1f}%<br>
                                    <b>Model:</b> SVM (Symptom-based)<br><br>
                                    <span style='color:#69f0ae; font-size:0.85rem'>
                                    No significant risk detected. Maintain a healthy lifestyle.
                                    </span>"""

                                st.session_state.messages.append({"role": "prediction", "content": result_html})
                                save_prediction(
                                    disease=f"{disease} (Chat)",
                                    input_data=symptoms,
                                    prediction=result,
                                    confidence=confidence,
                                    model_used="SVM Symptom (Chat)"
                                )
                                st.session_state.prediction_done = True

                            if clean_reply:
                                try:
                                    from gtts import gTTS
                                    import io
                                    tts = gTTS(text=clean_reply[:500], lang=lang_tts, slow=False)
                                    buf = io.BytesIO()
                                    tts.write_to_fp(buf)
                                    buf.seek(0)
                                    st.session_state.last_audio = buf.read()
                                except Exception:
                                    st.session_state.last_audio = None

                    except json.JSONDecodeError:
                        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                        st.session_state.conversation.append({"role": "assistant", "content": assistant_reply})

                else:
                    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                    st.session_state.conversation.append({"role": "assistant", "content": assistant_reply})
                    try:
                        from gtts import gTTS
                        import io
                        tts = gTTS(text=assistant_reply[:500], lang=lang_tts, slow=False)
                        buf = io.BytesIO()
                        tts.write_to_fp(buf)
                        buf.seek(0)
                        st.session_state.last_audio = buf.read()
                    except Exception:
                        st.session_state.last_audio = None

            except Exception as e:
                st.error(f"API Error: {str(e)}")

        st.rerun()

    st.markdown("<div class='footer'>MediPredict AI · Multi Disease Prediction System · Developed by Rahul Raj Singh</div>", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════
# PAGE 3 — PARAMETER GUIDE
# ══════════════════════════════════════════════════════
elif page == "Parameter Guide":
    st.markdown("## Parameter Guide")
    st.markdown("<p style='color:#8b8fa8'>Simple explanations for every medical term — no medical background needed!</p>", unsafe_allow_html=True)
    st.markdown("---")

    disease_guide = st.selectbox("Select Disease", ["Heart Disease", "Diabetes", "Liver Disease"])

    def show_params(params):
        for param in params:
            with st.expander(f"{param['name']} — {param['simple']}"):
                col1, col2 = st.columns([2,1])
                with col1:
                    st.markdown(f"<p style='color:#c0c4d6; line-height:1.8'>{param['detail']}</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class='param-card'>
                        <p style='color:#7c83fd; font-size:0.8rem; margin:0 0 6px 0'>Normal Range</p>
                        <p style='color:#c0c4d6; font-size:0.85rem; margin:0; white-space:pre-line'>{param['normal']}</p>
                        <p style='color:#4a4d5e; font-size:0.75rem; margin:8px 0 0 0'>Unit: {param['unit']}</p>
                    </div>""", unsafe_allow_html=True)

    if disease_guide == "Heart Disease":
        st.markdown("<div class='section-header'><h3>Heart Disease Parameters</h3></div>", unsafe_allow_html=True)
        show_params([
            {"name":"Age","simple":"How old are you?","detail":"Age is one of the biggest risk factors for heart disease. As we get older, the heart works harder and arteries can stiffen. Fatty deposits can also build up in arteries over decades.","normal":"Risk increases after:\n45 for men\n55 for women","unit":"Years"},
            {"name":"Sex","simple":"Your biological sex","detail":"Men generally have a higher risk of heart disease at a younger age. Women's risk increases significantly after menopause when protective estrogen levels drop.","normal":"0 = Female\n1 = Male","unit":"0 or 1"},
            {"name":"Chest Pain Type","simple":"What kind of chest pain do you feel?","detail":"Typical Angina is classic heart-related chest pain caused by reduced blood flow. Atypical Angina is chest discomfort not typical of the heart. Non-anginal Pain is chest pain unrelated to the heart. Asymptomatic means no chest pain at all.","normal":"0 = Typical Angina\n1 = Atypical Angina\n2 = Non-anginal Pain\n3 = Asymptomatic","unit":"0, 1, 2, or 3"},
            {"name":"Resting Blood Pressure","simple":"Your blood pressure when sitting quietly","detail":"Blood pressure is the force of blood pushing against artery walls. High blood pressure makes the heart work harder and damages artery walls over time, increasing the risk of heart attack and stroke.","normal":"Normal: below 120\nElevated: 120-129\nHigh Stage 1: 130-139\nHigh Stage 2: 140 or above","unit":"mm Hg"},
            {"name":"Cholesterol","simple":"Amount of fatty substance in your blood","detail":"Cholesterol is a waxy fat-like substance in your blood. Too much bad cholesterol builds up in artery walls forming plaques that can block blood flow and cause a heart attack.","normal":"Normal: below 200\nBorderline High: 200-239\nHigh: 240 or above","unit":"mg/dl"},
            {"name":"Fasting Blood Sugar","simple":"Is your blood sugar high after not eating for 8 hours?","detail":"High fasting blood sugar indicates diabetes or prediabetes. Diabetes is a major risk factor for heart disease because high blood sugar damages blood vessels and nerves that control the heart.","normal":"0 = No (sugar below 120 - normal)\n1 = Yes (sugar above 120 - elevated)","unit":"0 or 1"},
            {"name":"Resting ECG","simple":"Result of your heart electrical activity test at rest","detail":"An ECG records electrical signals in your heart. Abnormal results can indicate previous heart attacks, irregular heartbeats, or a thickened heart muscle.","normal":"0 = Normal\n1 = ST-T Wave Abnormality\n2 = Left Ventricular Hypertrophy","unit":"0, 1, or 2"},
            {"name":"Max Heart Rate","simple":"Highest heart rate reached during exercise","detail":"A healthy heart speeds up significantly during physical activity. If your maximum heart rate during exercise is lower than expected for your age, it may indicate the heart is not pumping blood efficiently.","normal":"Formula: 220 minus your age\nExample: Age 50 = max 170 bpm\nBelow 100 bpm is concerning","unit":"Beats per minute"},
            {"name":"Exercise Induced Angina","simple":"Do you feel chest pain during exercise?","detail":"Chest pain during physical activity is a warning sign that the heart may not be getting enough oxygen-rich blood when demand increases. This is a strong indicator of coronary artery disease.","normal":"0 = No (normal)\n1 = Yes (concerning)","unit":"0 or 1"},
            {"name":"ST Depression (Oldpeak)","simple":"A change in your ECG reading during exercise","detail":"During exercise an ECG can show a dip called ST depression. The larger the dip, the more stress the heart is under during exercise.","normal":"Normal: 0 to 1\nMild concern: 1 to 2\nSignificant: above 2","unit":"mm"},
            {"name":"Slope","simple":"Shape of the ECG curve at peak exercise","detail":"This describes whether the ST segment goes up, stays flat, or goes down at maximum exercise. Downsloping is the most concerning pattern.","normal":"0 = Upsloping (normal)\n1 = Flat (borderline)\n2 = Downsloping (concerning)","unit":"0, 1, or 2"},
            {"name":"Major Vessels","simple":"Number of major blood vessels visible in a scan","detail":"This comes from a scan where dye is injected to make blood vessels visible. More blocked vessels means blood supply to the heart is more restricted.","normal":"0 = No blocked vessels (best)\n1-2 = Some blockage\n3 = Most blockage (highest risk)","unit":"0, 1, 2, or 3"},
            {"name":"Thalassemia","simple":"Result of a nuclear stress test of heart blood flow","detail":"A fixed defect means permanently damaged tissue. A reversible defect means an area that shows stress during exercise but recovers at rest.","normal":"0 = Normal\n1 = Fixed defect\n2 = Reversible defect\n3 = Unknown","unit":"0, 1, 2, or 3"},
        ])

    elif disease_guide == "Diabetes":
        st.markdown("<div class='section-header'><h3>Diabetes Parameters</h3></div>", unsafe_allow_html=True)
        show_params([
            {"name":"Pregnancies","simple":"Number of times pregnant","detail":"Women who have been pregnant multiple times have a slightly higher risk of developing type 2 diabetes due to hormonal changes during pregnancy.","normal":"No specific threshold\nHigher numbers slightly increase risk","unit":"Count"},
            {"name":"Glucose","simple":"Amount of sugar in your blood","detail":"Glucose is the main sugar in your blood. In diabetes this system breaks down and glucose stays too high, damaging organs over time.","normal":"Normal: below 100\nPrediabetes: 100-125\nDiabetes: 126 or above","unit":"mg/dl"},
            {"name":"Blood Pressure","simple":"Force of blood against artery walls","detail":"High blood pressure and diabetes frequently occur together and each makes the other worse, significantly increasing risk of heart and kidney disease.","normal":"Normal diastolic: below 80\nHigh: 90 or above","unit":"mm Hg"},
            {"name":"Skin Thickness","simple":"Thickness of the skin fold at the back of your upper arm","detail":"This estimates body fat percentage. A thicker skin fold means more body fat which is strongly associated with insulin resistance and type 2 diabetes.","normal":"Normal: 10-25 mm\nAbove 35 mm may indicate excess body fat","unit":"mm"},
            {"name":"Insulin","simple":"A hormone made by your pancreas to control blood sugar","detail":"Insulin lets glucose enter your body's cells for energy. In type 2 diabetes the body stops making enough insulin or cells stop responding to it properly.","normal":"Normal fasting: 2-25 IU/ml\nAbove 25 may indicate insulin resistance","unit":"IU/ml"},
            {"name":"BMI","simple":"Body Mass Index — measure of body fat based on height and weight","detail":"BMI is weight divided by height squared. Being overweight or obese is the single biggest modifiable risk factor for type 2 diabetes.","normal":"Underweight: below 18.5\nNormal: 18.5-24.9\nOverweight: 25-29.9\nObese: 30 or above","unit":"kg/m2"},
            {"name":"Diabetes Pedigree Function","simple":"A score estimating your genetic risk based on family history","detail":"This calculates how likely you are to have inherited diabetes risk genes. A higher score means more family members with diabetes and higher genetic risk.","normal":"Low risk: below 0.5\nModerate: 0.5-1.0\nHigh: above 1.0","unit":"Score (0 to 2.5)"},
            {"name":"Age","simple":"Your age","detail":"Risk of type 2 diabetes increases with age especially after 45, partly because people tend to be less active and gain weight as they get older.","normal":"Risk increases significantly after age 45","unit":"Years"},
        ])

    elif disease_guide == "Liver Disease":
        st.markdown("<div class='section-header'><h3>Liver Disease Parameters</h3></div>", unsafe_allow_html=True)
        show_params([
            {"name":"Age","simple":"Your age","detail":"Liver disease risk increases with age due to longer exposure to risk factors such as alcohol, fatty foods, medications, and viral infections like hepatitis.","normal":"Risk gradually increases with age","unit":"Years"},
            {"name":"Gender","simple":"Your biological sex","detail":"Men are more likely to develop liver disease partly due to higher rates of alcohol consumption and certain genetic factors.","normal":"0 = Female\n1 = Male","unit":"0 or 1"},
            {"name":"Total Bilirubin","simple":"A yellow pigment produced when red blood cells break down","detail":"The liver processes bilirubin. If damaged, bilirubin builds up causing yellowing of the skin and eyes called jaundice.","normal":"Normal: 0.1-1.2 mg/dl\nMild: 1.2-2.0\nHigh (jaundice): above 2.0","unit":"mg/dl"},
            {"name":"Direct Bilirubin","simple":"The form of bilirubin processed by the liver","detail":"High direct bilirubin specifically points to a problem inside the liver or in the bile ducts.","normal":"Normal: 0.0-0.3 mg/dl\nAbove 0.3 suggests liver issue","unit":"mg/dl"},
            {"name":"Alkaline Phosphotase","simple":"An enzyme found in the liver and bones","detail":"When the liver is damaged or bile ducts are blocked, this enzyme leaks into the bloodstream. High levels warn of liver disease.","normal":"Normal: 44-147 IU/L\nAbove 147 may indicate liver damage","unit":"IU/L"},
            {"name":"SGPT (ALT)","simple":"An enzyme that leaks into blood when liver cells are damaged","detail":"SGPT is found mainly inside liver cells. When they are injured by alcohol, fatty liver, or hepatitis, SGPT is released into blood. It is the most specific test for liver cell damage.","normal":"Normal: 7-56 IU/L\nMildly elevated: 56-100\nHighly elevated: above 100","unit":"IU/L"},
            {"name":"SGOT (AST)","simple":"Another enzyme released when liver or heart cells are damaged","detail":"SGOT is found in liver, heart, and muscles. Elevated levels suggest liver damage but it is less specific than SGPT.","normal":"Normal: 10-40 IU/L\nElevated: above 40","unit":"IU/L"},
            {"name":"Total Proteins","simple":"Total amount of proteins in your blood","detail":"The liver makes most blood proteins. When severely damaged, it produces less protein causing total protein levels to drop.","normal":"Normal: 6.0-8.3 g/dl\nBelow 6.0 may indicate liver issue","unit":"g/dl"},
            {"name":"Albumin","simple":"The main protein made by the liver","detail":"Albumin keeps fluid from leaking out of blood vessels. Low albumin is one of the most reliable signs of chronic liver disease.","normal":"Normal: 3.5-5.0 g/dl\nBelow 3.5 suggests liver dysfunction","unit":"g/dl"},
            {"name":"Albumin/Globulin Ratio","simple":"Balance between two types of protein in your blood","detail":"In liver disease the liver makes less albumin so this ratio drops. A low ratio can also indicate kidney disease or immune disorders.","normal":"Normal: 1.0-2.5\nBelow 1.0 may indicate liver issue","unit":"Ratio"},
        ])

    st.markdown("<div class='footer'>MediPredict AI · Multi Disease Prediction System · Developed by Rahul Raj Singh</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE 4 — DATASET INFO
# ══════════════════════════════════════════════════════
elif page == "Dataset Info":
    st.markdown("## Dataset Information")
    st.markdown("<p style='color:#8b8fa8'>Details about the datasets used for training the models</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div class='section-header'><h3>Heart Disease Dataset</h3><p>UCI Heart Disease Dataset</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><h2>920</h2><p>Total Records</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h2>13</h2><p>Features</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h2>82.6%</h2><p>SVM Accuracy</p></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='param-card' style='margin-top:10px; line-height:1.9'>
    <p style='color:#c0c4d6; margin:0'><span style='color:#7c83fd'>Source:</span> UCI Machine Learning Repository</p>
    <p style='color:#c0c4d6; margin:0'><span style='color:#7c83fd'>Target:</span> Presence (1) or Absence (0) of heart disease</p>
    <p style='color:#c0c4d6; margin:0'><span style='color:#7c83fd'>Features:</span> Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, ECG, Heart Rate, and more</p>
    <p style='color:#c0c4d6; margin:0'><span style='color:#7c83fd'>Preprocessing:</span> Median imputation, OrdinalEncoder, StandardScaler for SVM</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'><h3>Diabetes Dataset</h3><p>Pima Indians Diabetes Database</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><h2>768</h2><p>Total Records</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h2>8</h2><p>Features</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h2>74.7%</h2><p>SVM Accuracy</p></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='param-card' style='margin-top:10px; line-height:1.9'>
    <p style='color:#c0c4d6; margin:0'><span style='color:#7c83fd'>Source:</span> National Institute of Diabetes and Digestive and Kidney Diseases</p>
    <p style='color:#c0c4d6; margin:0'><span style='color:#7c83fd'>Target:</span> Diabetic (1) or Non-Diabetic (0)</p>
    <p style='color:#c0c4d6; margin:0'><span style='color:#7c83fd'>Features:</span> Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Pedigree, Age</p>
    <p style='color:#c0c4d6; margin:0'><span style='color:#7c83fd'>Preprocessing:</span> Zero values replaced with median, StandardScaler for SVM</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'><h3>Liver Disease Dataset</h3><p>Indian Liver Patient Dataset (ILPD)</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><h2>583</h2><p>Total Records</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h2>10</h2><p>Features</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h2>74.0%</h2><p>SVM Accuracy</p></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='param-card' style='margin-top:10px; line-height:1.9'>
    <p style='color:#c0c4d6; margin:0'><span style='color:#7c83fd'>Source:</span> UCI Machine Learning Repository</p>
    <p style='color:#c0c4d6; margin:0'><span style='color:#7c83fd'>Target:</span> Liver Disease (1) or No Disease (0)</p>
    <p style='color:#c0c4d6; margin:0'><span style='color:#7c83fd'>Features:</span> Age, Gender, Bilirubin levels, Liver enzymes, Protein levels</p>
    <p style='color:#c0c4d6; margin:0'><span style='color:#7c83fd'>Preprocessing:</span> Label encoding for Gender, median imputation, StandardScaler for SVM</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='footer'>MediPredict AI · Multi Disease Prediction System · Developed by Rahul Raj Singh</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE 5 — HISTORY
# ══════════════════════════════════════════════════════
elif page == "History":
    st.markdown("## Prediction History")
    st.markdown("<p style='color:#8b8fa8'>All past predictions stored in the local database</p>", unsafe_allow_html=True)
    st.markdown("---")

    rows = get_history()

    if rows:
        pos = sum(1 for r in rows if r[3] == "Positive")
        neg = len(rows) - pos
        diseases = set(r[1] for r in rows)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><h2>{len(rows)}</h2><p>Total</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h2>{pos}</h2><p>Positive</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h2>{neg}</h2><p>Negative</p></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><h2>{len(diseases)}</h2><p>Disease Types</p></div>", unsafe_allow_html=True)

        st.markdown("---")

        filter_disease = st.selectbox("Filter by Disease", [
            "All", "Heart Disease", "Diabetes", "Liver Disease"
        ])

        df_hist = pd.DataFrame(rows, columns=[
            "ID","Disease","Inputs","Prediction","Confidence (%)","Model","Timestamp"
        ])
        df_hist = df_hist.drop(columns=["Inputs"])

        if filter_disease != "All":
            df_hist = df_hist[df_hist["Disease"] == filter_disease]

        st.dataframe(df_hist, use_container_width=True, hide_index=True)

    else:
        st.info("No predictions yet. Go to the Predict page to make your first prediction!")

    st.markdown("<div class='footer'>MediPredict AI · Multi Disease Prediction System · Developed by Rahul Raj Singh</div>", unsafe_allow_html=True)