
import pickle
 
import pandas as pd
import streamlit as st
 
# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
)
 
# If your dataset used the opposite coding, flip this to 0.
# (In the common UCI/Kaggle heart.csv, target = 1 means heart disease present.)
DISEASE_LABEL = 1
 
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
 
 
# ----------------------------------------------------------------------------
# Load model + scaler (cached so they load only once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("model_knn_class.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler_knn_class.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler
 
 
try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Could not find `model_knn_class.pkl` / `scaler_knn_class.pkl`. "
        "Place them in the same folder as this app and restart."
    )
    st.stop()
 
# ----------------------------------------------------------------------------
# Custom styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #b71c1c 0%, #e53935 60%, #ff7043 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2rem;}
    .hero p {margin: .3rem 0 0 0; opacity: .92;}
    .result-card {
        padding: 1.4rem; border-radius: 14px; text-align: center;
        font-size: 1.3rem; font-weight: 600; margin-top: .5rem;
    }
    .risk-high {background:#fdecea; color:#b71c1c; border:2px solid #e53935;}
    .risk-low  {background:#e8f5e9; color:#1b5e20; border:2px solid #43a047;}
    </style>
    <div class="hero">
        <h1>❤️ Heart Disease Prediction</h1>
        <p>KNN classifier · enter the patient's details and get an instant assessment</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.form("patient_form"):
    st.subheader("Patient details")
 
    c1, c2, c3 = st.columns(3)
 
    with c1:
        st.markdown("**Demographics**")
        age = st.slider("Age", 20, 90, 50)
        sex_label = st.radio("Sex", ["Male", "Female"], horizontal=True)
        sex = 1 if sex_label == "Male" else 0
        cp_label = st.selectbox(
            "Chest pain type (cp)",
            [
                "0 – Typical angina",
                "1 – Atypical angina",
                "2 – Non-anginal pain",
                "3 – Asymptomatic",
            ],
        )
        cp = int(cp_label[0])
 
    with c2:
        st.markdown("**Vitals & blood tests**")
        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)", 80, 220, 130
        )
        chol = st.number_input("Serum cholesterol (mg/dl)", 100, 600, 240)
        fbs_label = st.radio(
            "Fasting blood sugar > 120 mg/dl", ["No", "Yes"], horizontal=True
        )
        fbs = 1 if fbs_label == "Yes" else 0
        restecg_label = st.selectbox(
            "Resting ECG (restecg)",
            [
                "0 – Normal",
                "1 – ST-T wave abnormality",
                "2 – Left ventricular hypertrophy",
            ],
        )
        restecg = int(restecg_label[0])
 
    with c3:
        st.markdown("**Exercise test**")
        thalach = st.slider("Max heart rate achieved (thalach)", 60, 220, 150)
        exang_label = st.radio(
            "Exercise-induced angina", ["No", "Yes"], horizontal=True
        )
        exang = 1 if exang_label == "Yes" else 0
        oldpeak = st.number_input(
            "ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1
        )
        slope_label = st.selectbox(
            "Slope of peak exercise ST segment",
            ["0 – Upsloping", "1 – Flat", "2 – Downsloping"],
        )
        slope = int(slope_label[0])
 
    c4, c5 = st.columns(2)
    with c4:
        ca = st.selectbox("Major vessels colored by fluoroscopy (ca)", [0, 1, 2, 3, 4])
    with c5:
        thal_label = st.selectbox(
            "Thalassemia (thal)",
            [
                "0 – Null / unknown",
                "1 – Fixed defect",
                "2 – Normal",
                "3 – Reversible defect",
            ],
            index=2,
        )
        thal = int(thal_label[0])
 
    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)
 
# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg,
          thalach, exang, oldpeak, slope, ca, thal]],
        columns=FEATURE_ORDER,
    )
 
    scaled = scaler.transform(input_df)
    pred = int(model.predict(scaled)[0])
 
    st.divider()
    left, right = st.columns([1, 1])
 
    with left:
        st.subheader("Result")
        if pred == DISEASE_LABEL:
            st.markdown(
                '<div class="result-card risk-high">⚠️ Heart disease likely<br>'
                '<span style="font-size:.95rem;font-weight:400">'
                "Please consult a cardiologist.</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-card risk-low">✅ Heart disease unlikely<br>'
                '<span style="font-size:.95rem;font-weight:400">'
                "Keep up regular check-ups.</span></div>",
                unsafe_allow_html=True,
            )
 
    with right:
        st.subheader("Confidence")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled)[0]
            classes = list(model.classes_)
            p_disease = float(proba[classes.index(DISEASE_LABEL)])
            st.metric("Probability of heart disease", f"{p_disease * 100:.1f}%")
            st.progress(min(max(p_disease, 0.0), 1.0))
            st.caption(
                f"Based on the {model.n_neighbors} nearest neighbours "
                "in the training data."
            )
 
    with st.expander("See the values sent to the model"):
        st.dataframe(input_df, use_container_width=True)
 
st.divider()
st.caption(
    "⚠️ This tool is for educational purposes only and is not a substitute "
    "for professional medical advice, diagnosis, or treatment."
)