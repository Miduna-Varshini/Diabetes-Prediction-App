import streamlit as st
import pickle
import numpy as np

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩸",
    layout="centered"
)

# ================== LOAD MODEL ==================
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# ================== HEADER ==================
st.markdown("<h1 style='text-align:center;'>🩸 Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-Assisted Diabetes Risk Prediction</p>", unsafe_allow_html=True)
st.markdown("---")

# ================== INPUT FIELDS ==================
age = st.number_input("Age (years)", 1, 100, 25)
bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 25.0)
insulin = st.number_input("Insulin Level (µU/mL)", 0, 900, 100)
glucose = st.number_input("Plasma Glucose Level (mg/dL)", 50, 200, 120)

# ================== PREDICTION ==================
if st.button("🔍 Predict Diabetes Risk"):

    input_data = np.array([[age, bmi, insulin, glucose]])

    # Prediction (0 or 1)
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1] * 100  # Diabetic %

    st.markdown("### 🧪 Prediction Result")

    if probability >= 70:
        st.error(f"⚠️ **High Risk of Diabetes**\n\n"
                 f"Confidence: **{probability:.2f}%**")
    elif probability >= 40:
        st.warning(f"⚠️ **Moderate Risk of Diabetes**\n\n"
                   f"Confidence: **{probability:.2f}%**")
    else:
        st.success(f"✅ **Low Risk of Diabetes**\n\n"
                   f"Confidence: **{probability:.2f}%**")

# ================== OPTIONAL INFO ==================
if st.checkbox("📊 Show Model Accuracy"):
    st.info("Model Accuracy on Training Data: **76.69%**")

if st.checkbox("ℹ️ Medical Disclaimer"):
    st.warning(
        "This prediction is for **educational purposes only** and "
        "should not be considered as medical advice."
    )
