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
    }

models = load_models()

with st.sidebar:
    st.markdown("## 🧬 MediPredict AI")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Navigation", [
        "Predict",
        "AI Health Assistant",
        "Parameter Guide",
        "Dataset Info",
        "History"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='color:#8b8fa8; font-size:0.85rem; line-height:1.8'>
    <b style='color:#7c83fd'>Models Used:</b><br>
    SVM (Primary) · Random Forest · XGBoost<br><br>
    <b style='color:#7c83fd'>Diseases:</b><br>
    Heart Disease · Diabetes · Liver Disease
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
        if trestbps > 140: warnings.append("Blood Pressure is high (above 140 mm Hg) — consider consulting a doctor")
        if chol > 240:     warnings.append("Cholesterol is high (above 240 mg/dl) — high risk range")
        if thalch < 100:   warnings.append("Max Heart Rate is low (below 100 bpm) — may indicate reduced heart function")
        if oldpeak > 2:    warnings.append("ST Depression is significant (above 2) — possible heart stress")
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
            dpf         = st.number_input("Diabetes Pedigree Function (Family History Score)", 0.0, 3.0, 0.5)
            age         = st.slider("Age (years)", 10, 100, 30)
        input_data    = [pregnancies,glucose,bp,skin,insulin,bmi,dpf,age]
        feature_names = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

        warnings = []
        if glucose > 126: warnings.append("Glucose is in diabetic range (above 126 mg/dl) — medical attention advised")
        if glucose > 100 and glucose <= 126: warnings.append("Glucose is in prediabetic range (100–126 mg/dl) — monitor carefully")
        if bmi > 30:      warnings.append("BMI indicates obesity (above 30) — major diabetes risk factor")
        if bmi > 25 and bmi <= 30: warnings.append("BMI indicates overweight (25–30) — moderate risk")
        if bp > 90:       warnings.append("Blood Pressure is high (above 90 mm Hg diastolic)")
        if insulin > 25:  warnings.append("Insulin level is elevated (above 25 IU/ml) — possible insulin resistance")
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
            sgpt    = st.number_input("SGPT / ALT — Liver Enzyme (IU/L)", 10, 2000, 35)
            sgot    = st.number_input("SGOT / AST — Liver Enzyme (IU/L)", 10, 5000, 40)
            tp      = st.number_input("Total Proteins (g/dl)", 2.0, 10.0, 6.5)
            alb     = st.number_input("Albumin — Liver Protein (g/dl)", 0.0, 6.0, 3.5)
            agr     = st.number_input("Albumin/Globulin Ratio", 0.0, 3.0, 1.0)
        input_data    = [age,gender,tb,db,alkphos,sgpt,sgot,tp,alb,agr]
        feature_names = ["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase",
                         "Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens",
                         "Albumin","Albumin_and_Globulin_Ratio"]

        warnings = []
        if tb > 2.0:      warnings.append("Total Bilirubin is high (above 2.0 mg/dl) — possible jaundice or liver issue")
        if db > 0.3:      warnings.append("Direct Bilirubin is elevated (above 0.3 mg/dl) — liver or bile duct concern")
        if alkphos > 147: warnings.append("Alkaline Phosphotase is high (above 147 IU/L) — possible liver damage")
        if sgpt > 56:     warnings.append("SGPT is elevated (above 56 IU/L) — liver cells may be damaged")
        if sgot > 40:     warnings.append("SGOT is elevated (above 40 IU/L) — liver or heart concern")
        if alb < 3.5:     warnings.append("Albumin is low (below 3.5 g/dl) — liver may not be producing enough protein")
        if agr < 1.0:     warnings.append("Albumin/Globulin Ratio is low (below 1.0) — possible liver or immune disorder")
        for w in warnings: st.warning(w)

    st.markdown("---")
    if st.button("Run Prediction"):
        X_input    = pd.DataFrame([input_data], columns=feature_names)
        model      = models[disease]
        prediction = model.predict(X_input)[0]
        confidence = model.predict_proba(X_input)[0][prediction] * 100
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
# PAGE 2 — AI HEALTH ASSISTANT
# ══════════════════════════════════════════════════════
elif page == "AI Health Assistant":
    st.markdown("## AI Health Assistant")
    st.markdown("<p style='color:#8b8fa8'>Describe your symptoms in plain English — our AI will assess your risk and explain everything simply</p>", unsafe_allow_html=True)
    st.markdown("---")

    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        st.error("API key not configured.")
        st.stop()
    
    if api_key:
        from groq import Groq

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "conversation" not in st.session_state:
            st.session_state.conversation = []

        SYSTEM_PROMPT = """You are MediPredict AI Health Assistant, a friendly medical screening chatbot.

Your job is to:
1. Have a friendly conversation about the user's symptoms, age, and health concerns
2. Ask relevant follow up questions naturally, one or two at a time
3. Assess their risk for Heart Disease, Diabetes, or Liver Disease based on what they share
4. Explain everything in very simple language anyone can understand
5. Always recommend consulting a real doctor for proper diagnosis
6. Be warm, empathetic and reassuring — never scary or alarmist

Important rules:
- Never diagnose — only assess risk level as Low, Moderate, or High
- Always recommend seeing a doctor if risk is moderate or high
- Use simple everyday language, avoid medical jargon
- If user mentions emergency symptoms like severe chest pain or difficulty breathing, tell them to call emergency services immediately
- You are a screening tool, not a replacement for medical care

Start by warmly greeting the user and asking what health concern brings them here today."""

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Describe your symptoms or ask a health question...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.conversation.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        from groq import Groq
                        client = Groq(api_key=api_key)
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            max_tokens=1024,
                            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.conversation
                        )
                        assistant_reply = response.choices[0].message.content
                        st.markdown(assistant_reply)

                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
                        st.session_state.conversation.append({"role": "assistant", "content": assistant_reply})

                    except Exception as e:
                       st.error(f"API Error: {str(e)}")

        if st.session_state.chat_history:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.conversation = []
                st.rerun()

    else:
        st.info("Please enter your Claude API key above to start chatting!")
        st.markdown("""
        <div class='param-card' style='margin-top:16px'>
            <p style='color:#7c83fd; font-size:0.9rem; margin:0 0 8px 0'>How to get your free API key:</p>
            <p style='color:#c0c4d6; font-size:0.85rem; margin:0; line-height:1.8'>
            1. Go to console.anthropic.com<br>
            2. Sign up for a free account<br>
            3. Go to API Keys section<br>
            4. Click Create Key<br>
            5. Copy and paste it above
            </p>
        </div>""", unsafe_allow_html=True)

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
            {"name":"Resting Blood Pressure","simple":"Your blood pressure when sitting quietly","detail":"Blood pressure is the force of blood pushing against artery walls. High blood pressure makes the heart work harder and damages artery walls over time, increasing the risk of heart attack and stroke.","normal":"Normal: below 120\nElevated: 120–129\nHigh Stage 1: 130–139\nHigh Stage 2: 140 or above","unit":"mm Hg"},
            {"name":"Cholesterol","simple":"Amount of fatty substance in your blood","detail":"Cholesterol is a waxy fat-like substance in your blood. Too much bad cholesterol builds up in artery walls forming plaques that can block blood flow and cause a heart attack.","normal":"Normal: below 200\nBorderline High: 200–239\nHigh: 240 or above","unit":"mg/dl"},
            {"name":"Fasting Blood Sugar","simple":"Is your blood sugar high after not eating for 8 hours?","detail":"High fasting blood sugar indicates diabetes or prediabetes. Diabetes is a major risk factor for heart disease because high blood sugar damages blood vessels and nerves that control the heart.","normal":"0 = No (sugar below 120 — normal)\n1 = Yes (sugar above 120 — elevated)","unit":"0 or 1"},
            {"name":"Resting ECG","simple":"Result of your heart electrical activity test at rest","detail":"An ECG records electrical signals in your heart. Abnormal results can indicate previous heart attacks, irregular heartbeats, or a thickened heart muscle.","normal":"0 = Normal\n1 = ST-T Wave Abnormality\n2 = Left Ventricular Hypertrophy","unit":"0, 1, or 2"},
            {"name":"Max Heart Rate","simple":"Highest heart rate reached during exercise","detail":"A healthy heart speeds up significantly during physical activity. If your maximum heart rate during exercise is lower than expected for your age, it may indicate the heart is not pumping blood efficiently.","normal":"Formula: 220 minus your age\nExample: Age 50 = max 170 bpm\nBelow 100 bpm is concerning","unit":"Beats per minute"},
            {"name":"Exercise Induced Angina","simple":"Do you feel chest pain during exercise?","detail":"Chest pain during physical activity is a warning sign that the heart may not be getting enough oxygen-rich blood when demand increases. This is a strong indicator of coronary artery disease.","normal":"0 = No (normal)\n1 = Yes (concerning)","unit":"0 or 1"},
            {"name":"ST Depression (Oldpeak)","simple":"A change in your ECG reading during exercise","detail":"During exercise an ECG can show a dip called ST depression. The larger the dip, the more stress the heart is under during exercise.","normal":"Normal: 0 to 1\nMild concern: 1 to 2\nSignificant: above 2","unit":"mm"},
            {"name":"Slope","simple":"Shape of the ECG curve at peak exercise","detail":"This describes whether the ST segment goes up, stays flat, or goes down at maximum exercise. Downsloping is the most concerning pattern.","normal":"0 = Upsloping (normal)\n1 = Flat (borderline)\n2 = Downsloping (concerning)","unit":"0, 1, or 2"},
            {"name":"Major Vessels","simple":"Number of major blood vessels visible in a scan","detail":"This comes from a scan where dye is injected to make blood vessels visible. More blocked vessels means blood supply to the heart is more restricted.","normal":"0 = No blocked vessels (best)\n1–2 = Some blockage\n3 = Most blockage (highest risk)","unit":"0, 1, 2, or 3"},
            {"name":"Thalassemia","simple":"Result of a nuclear stress test of heart blood flow","detail":"A fixed defect means permanently damaged tissue. A reversible defect means an area that shows stress during exercise but recovers at rest.","normal":"0 = Normal\n1 = Fixed defect\n2 = Reversible defect\n3 = Unknown","unit":"0, 1, 2, or 3"},
        ])

    elif disease_guide == "Diabetes":
        st.markdown("<div class='section-header'><h3>Diabetes Parameters</h3></div>", unsafe_allow_html=True)
        show_params([
            {"name":"Pregnancies","simple":"Number of times pregnant","detail":"Women who have been pregnant multiple times have a slightly higher risk of developing type 2 diabetes due to hormonal changes during pregnancy.","normal":"No specific threshold\nHigher numbers slightly increase risk","unit":"Count"},
            {"name":"Glucose","simple":"Amount of sugar in your blood","detail":"Glucose is the main sugar in your blood. In diabetes this system breaks down and glucose stays too high, damaging organs over time.","normal":"Normal: below 100\nPrediabetes: 100–125\nDiabetes: 126 or above","unit":"mg/dl"},
            {"name":"Blood Pressure","simple":"Force of blood against artery walls","detail":"High blood pressure and diabetes frequently occur together and each makes the other worse, significantly increasing risk of heart and kidney disease.","normal":"Normal diastolic: below 80\nHigh: 90 or above","unit":"mm Hg"},
            {"name":"Skin Thickness","simple":"Thickness of the skin fold at the back of your upper arm","detail":"This estimates body fat percentage. A thicker skin fold means more body fat which is strongly associated with insulin resistance and type 2 diabetes.","normal":"Normal: 10–25 mm\nAbove 35 mm may indicate excess body fat","unit":"mm"},
            {"name":"Insulin","simple":"A hormone made by your pancreas to control blood sugar","detail":"Insulin lets glucose enter your body's cells for energy. In type 2 diabetes the body stops making enough insulin or cells stop responding to it properly.","normal":"Normal fasting: 2–25 IU/ml\nAbove 25 may indicate insulin resistance","unit":"IU/ml"},
            {"name":"BMI","simple":"Body Mass Index — measure of body fat based on height and weight","detail":"BMI is weight divided by height squared. Being overweight or obese is the single biggest modifiable risk factor for type 2 diabetes.","normal":"Underweight: below 18.5\nNormal: 18.5–24.9\nOverweight: 25–29.9\nObese: 30 or above","unit":"kg/m2"},
            {"name":"Diabetes Pedigree Function","simple":"A score estimating your genetic risk based on family history","detail":"This calculates how likely you are to have inherited diabetes risk genes. A higher score means more family members with diabetes and higher genetic risk.","normal":"Low risk: below 0.5\nModerate: 0.5–1.0\nHigh: above 1.0","unit":"Score (0 to 2.5)"},
            {"name":"Age","simple":"Your age","detail":"Risk of type 2 diabetes increases with age especially after 45, partly because people tend to be less active and gain weight as they get older.","normal":"Risk increases significantly after age 45","unit":"Years"},
        ])

    elif disease_guide == "Liver Disease":
        st.markdown("<div class='section-header'><h3>Liver Disease Parameters</h3></div>", unsafe_allow_html=True)
        show_params([
            {"name":"Age","simple":"Your age","detail":"Liver disease risk increases with age due to longer exposure to risk factors such as alcohol, fatty foods, medications, and viral infections like hepatitis.","normal":"Risk gradually increases with age","unit":"Years"},
            {"name":"Gender","simple":"Your biological sex","detail":"Men are more likely to develop liver disease partly due to higher rates of alcohol consumption and certain genetic factors.","normal":"0 = Female\n1 = Male","unit":"0 or 1"},
            {"name":"Total Bilirubin","simple":"A yellow pigment produced when red blood cells break down","detail":"The liver processes bilirubin. If damaged, bilirubin builds up causing yellowing of the skin and eyes called jaundice.","normal":"Normal: 0.1–1.2 mg/dl\nMild: 1.2–2.0\nHigh (jaundice): above 2.0","unit":"mg/dl"},
            {"name":"Direct Bilirubin","simple":"The form of bilirubin processed by the liver","detail":"High direct bilirubin specifically points to a problem inside the liver or in the bile ducts.","normal":"Normal: 0.0–0.3 mg/dl\nAbove 0.3 suggests liver issue","unit":"mg/dl"},
            {"name":"Alkaline Phosphotase","simple":"An enzyme found in the liver and bones","detail":"When the liver is damaged or bile ducts are blocked, this enzyme leaks into the bloodstream. High levels warn of liver disease.","normal":"Normal: 44–147 IU/L\nAbove 147 may indicate liver damage","unit":"IU/L"},
            {"name":"SGPT (ALT)","simple":"An enzyme that leaks into blood when liver cells are damaged","detail":"SGPT is found mainly inside liver cells. When they are injured by alcohol, fatty liver, or hepatitis, SGPT is released into blood. It is the most specific test for liver cell damage.","normal":"Normal: 7–56 IU/L\nMildly elevated: 56–100\nHighly elevated: above 100","unit":"IU/L"},
            {"name":"SGOT (AST)","simple":"Another enzyme released when liver or heart cells are damaged","detail":"SGOT is found in liver, heart, and muscles. Elevated levels suggest liver damage but it is less specific than SGPT.","normal":"Normal: 10–40 IU/L\nElevated: above 40","unit":"IU/L"},
            {"name":"Total Proteins","simple":"Total amount of proteins in your blood","detail":"The liver makes most blood proteins. When severely damaged, it produces less protein causing total protein levels to drop.","normal":"Normal: 6.0–8.3 g/dl\nBelow 6.0 may indicate liver issue","unit":"g/dl"},
            {"name":"Albumin","simple":"The main protein made by the liver","detail":"Albumin keeps fluid from leaking out of blood vessels. Low albumin is one of the most reliable signs of chronic liver disease.","normal":"Normal: 3.5–5.0 g/dl\nBelow 3.5 suggests liver dysfunction","unit":"g/dl"},
            {"name":"Albumin/Globulin Ratio","simple":"Balance between two types of protein in your blood","detail":"In liver disease the liver makes less albumin so this ratio drops. A low ratio can also indicate kidney disease or immune disorders.","normal":"Normal: 1.0–2.5\nBelow 1.0 may indicate liver issue","unit":"Ratio"},
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