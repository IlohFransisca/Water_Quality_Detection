
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
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Water Quality Predictor",
    page_icon="💧",
    layout="centered"
)

st.title("💧 Water Quality Predictor")

st.write(
    "Enter water sample measurements below to predict if the water is safe to drink."
)

st.info(
    "This model was trained on 3,276 water samples using a Random Forest "
    "classifier. It prioritizes safety by minimizing false positives."
)

# ============================================================
# INPUT FIELDS
# ============================================================

st.header("Water Sample Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        step=0.1
    )

    hardness = st.number_input(
        "Hardness (mg/L)",
        min_value=0.0,
        value=196.0,
        step=1.0
    )

    solids = st.number_input(
        "Solids (ppm)",
        min_value=0.0,
        value=22000.0,
        step=100.0
    )

with col2:
    chloramines = st.number_input(
        "Chloramines (ppm)",
        min_value=0.0,
        value=7.1,
        step=0.1
    )

    sulfate = st.number_input(
        "Sulfate (mg/L)",
        min_value=0.0,
        value=333.0,
        step=1.0
    )

    conductivity = st.number_input(
        "Conductivity (uS/cm)",
        min_value=0.0,
        value=426.0,
        step=1.0
    )

with col3:
    organic_carbon = st.number_input(
        "Organic Carbon (ppm)",
        min_value=0.0,
        value=14.3,
        step=0.1
    )

    trihalomethanes = st.number_input(
        "Trihalomethanes (ug/L)",
        min_value=0.0,
        value=66.4,
        step=0.1
    )

    turbidity = st.number_input(
        "Turbidity (NTU)",
        min_value=0.0,
        value=4.0,
        step=0.1
    )

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("Predict Water Safety", type="primary"):

    # Build input data in the SAME feature order
    # used during model training
    input_data = pd.DataFrame(
        [[
            ph,
            hardness,
            solids,
            chloramines,
            sulfate,
            conductivity,
            organic_carbon,
            trihalomethanes,
            turbidity
        ]],
        columns=[
            "ph",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic_carbon",
            "Trihalomethanes",
            "Turbidity"
        ]
    )

    # ========================================================
    # SCALE INPUT
    # ========================================================

    input_scaled = scaler.transform(input_data)

    # ========================================================
    # GET PROBABILITY
    # ========================================================

    # Probability that the sample belongs to class 1 (Safe)
    prob = model.predict_proba(input_scaled)[0, 1]

    # ========================================================
    # APPLY SAVED THRESHOLD
    # ========================================================

    prediction = 1 if prob >= threshold else 0

    # ========================================================
    # SHOW RESULT
    # ========================================================

    st.header("Result")

    if prediction == 1:
        st.success(
            f"✅ SAFE TO DRINK "
            f"(Predicted probability: {prob:.1%})"
        )
    else:
        st.error(
            f"❌ NOT SAFE TO DRINK "
            f"(Predicted probability of not safe: {(1 - prob):.1%})"
        )

    # ========================================================
    # PREDICTION DETAILS
    # ========================================================

    with st.expander("See prediction details"):

        st.write(f"Model threshold: {threshold:.2f}")

        st.write(
            f"Probability of being safe: {prob:.4f}"
        )

        st.write(
            f"Probability of being not safe: {(1 - prob):.4f}"
        )

        st.write(
            "Model: Random Forest"
        )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.warning(
        "⚠️ This is a machine learning prediction, not a certified "
        "water test. Always verify with proper laboratory testing "
        "before drinking."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Built with Streamlit | Model: Random Forest | "
    "Dataset: Kaggle Water Potability"
)
