app_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# LOAD MODEL, SCALER, AND THRESHOLD
# ============================================================

# Get the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Load saved model and scaler
model = joblib.load(BASE_DIR / "water_quality_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")

# Load saved threshold
with open(BASE_DIR / "threshold.txt", "r") as f:
    threshold = float(f.read().strip())

# ============================================================
# RULE-BASED STANDARDS EVALUATION (WHO / EPA)
# ============================================================

def evaluate_water_quality(ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, thm, turbidity):
    health_issues = []
    aesthetic_issues = []

    # --- Health-Based Standards ---
    if chloramines > 4.0:
        health_issues.append(f"Chloramines ({chloramines} ppm) exceed EPA health limit (4.0 ppm).")
    if sulfate > 500:
        health_issues.append(f"Sulfate ({sulfate} mg/L) exceed WHO health limit (500 mg/L).")
    if thm > 80:
        health_issues.append(f"Trihalomethanes ({thm} µg/L) exceed EPA health limit (80 µg/L).")
    if turbidity > 5.0:
        health_issues.append(f"Turbidity ({turbidity} NTU) exceeds EPA health limit for effective disinfection (5.0 NTU).")
    if ph < 6.5 or ph > 8.5:
        health_issues.append(f"pH ({ph}) is outside the safe range (6.5 - 8.5).")

    # --- Aesthetic Standards (Taste, Odor, Color, Scaling) ---
    if solids > 500:
        aesthetic_issues.append(f"Total Dissolved Solids ({solids} ppm) exceed EPA aesthetic limit (500 ppm).")
    if sulfate > 250:
        aesthetic_issues.append(f"Sulfate ({sulfate} mg/L) may cause a bitter taste or laxative effect (>250 mg/L).")
    if chloramines > 3.0:
        aesthetic_issues.append(f"Chloramines ({chloramines} ppm) may cause an unpleasant taste/odor (>3.0 ppm).")
    if organic_carbon > 2.0:
        aesthetic_issues.append(f"Organic Carbon ({organic_carbon} ppm) is above the operational guideline (2.0 ppm).")
    if turbidity > 1.0:
        aesthetic_issues.append(f"Turbidity ({turbidity} NTU) is above the ideal aesthetic level (1.0 NTU).")

    return health_issues, aesthetic_issues

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Water Quality Predictor",
    page_icon="💧",
    layout="centered"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💧 Water Quality App")

st.sidebar.header("👩‍🔬 About Me")
st.sidebar.write("**Iloh Fransisca Onyinyechukwu**")
st.sidebar.write("Data Scientist Enthusiast")
st.sidebar.write(
    "I am passionate about using data to solve real-world problems, "
    "especially in environmental science and public health. "
    "This app is a blend of my machine learning skills and my "
    "interest in making water safety information accessible to everyone."
)

st.sidebar.markdown("---")

st.sidebar.header("📱 About the App")
st.sidebar.write(
    "This application uses a **Random Forest Classifier** trained on "
    "3,276 water samples to predict potability. "
    "It goes a step further by combining the ML prediction with "
    "**WHO and EPA health & aesthetic standards** to give you a "
    "comprehensive assessment."
)
st.sidebar.write("**How to use:** Enter your water sample measurements on the main screen and click 'Predict Water Safety'.")

st.sidebar.markdown("---")

st.sidebar.header("💬 Feedback")
st.sidebar.write("We value your feedback to improve this tool!")
feedback = st.sidebar.text_area("Leave your suggestions or comments here:", height=100)

if st.sidebar.button("Submit Feedback"):
    if feedback.strip():
        st.sidebar.success("Thank you! Your feedback helps improve this app. Please also feel free to email me directly at the address below.")
    else:
        st.sidebar.warning("Please enter some feedback before submitting.")

st.sidebar.markdown("---")

st.sidebar.header("📧 Contact")
st.sidebar.write("For collaborations, questions, or feedback:")
st.sidebar.markdown("[Send an Email](mailto:ilohfransisca2014@gmail.com)")
st.sidebar.write("ilohfransisca2014@gmail.com")

# ============================================================
# MAIN PAGE
# ============================================================

st.title("💧 Water Quality Predictor")

st.write(
    "Enter water sample measurements below to predict if the water is safe to drink."
)

st.info(
    "This model was trained on 3,276 water samples using a Random Forest "
    "classifier. It prioritizes safety by minimizing false positives."
)

st.header("Water Sample Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=196.0, step=1.0)
    solids = st.number_input("Solids (ppm)", min_value=0.0, value=22000.0, step=100.0)

with col2:
    chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.1, step=0.1)
    sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=333.0, step=1.0)
    conductivity = st.number_input("Conductivity (uS/cm)", min_value=0.0, value=426.0, step=1.0)

with col3:
    organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=14.3, step=0.1)
    trihalomethanes = st.number_input("Trihalomethanes (ug/L)", min_value=0.0, value=66.4, step=0.1)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=4.0, step=0.1)

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("Predict Water Safety", type="primary"):

    # Build input data in the SAME feature order used during model training
    input_data = pd.DataFrame(
        [[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]],
        columns=["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Get probability
    prob = model.predict_proba(input_scaled)[0, 1]

    # Apply saved threshold
    prediction = 1 if prob >= threshold else 0

    # Get Rule-Based Evaluation
    health_issues, aesthetic_issues = evaluate_water_quality(
        ph, hardness, solids, chloramines, sulfate, 
        conductivity, organic_carbon, trihalomethanes, turbidity
    )

    # ========================================================
    # SHOW RESULTS
    # ========================================================

    st.header("Result")

    # 1. Model Prediction
    if prediction == 1:
        st.success(f"✅ MODEL PREDICTION: SAFE TO DRINK (Predicted probability: {prob:.1%})")
    else:
        st.error(f"❌ MODEL PREDICTION: NOT SAFE TO DRINK (Predicted probability of not safe: {(1 - prob):.1%})")

    # 2. Health Warnings
    if health_issues:
        st.warning("**Health-Based Standards Violated (WHO/EPA):**")
        for issue in health_issues:
            st.write(f"- {issue}")

    # 3. Aesthetic Warnings
    if aesthetic_issues:
        st.info("**Aesthetic / Quality Concerns (Taste, Odor, Scaling):**")
        for issue in aesthetic_issues:
            st.write(f"- {issue}")

    # 4. All Clear
    if not health_issues and not aesthetic_issues:
        st.info("This water sample meets all health and aesthetic guidelines checked.")

    # ========================================================
    # PREDICTION DETAILS
    # ========================================================

    with st.expander("See prediction details"):
        st.write(f"Model threshold: {threshold:.2f}")
        st.write(f"Probability of being safe: {prob:.4f}")
        st.write(f"Probability of being not safe: {(1 - prob):.4f}")
        st.write("Model: Random Forest")

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.warning(
        "⚠️ This is a machine learning prediction combined with "
        "rule-based guidelines, not a certified water test. Always verify "
        "with proper laboratory testing before drinking."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("Built with Streamlit | Model: Random Forest | Dataset: Kaggle Water Potability")
'''
