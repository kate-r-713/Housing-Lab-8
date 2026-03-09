import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import os

# -----------------------------
# Load model and artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    # Extract model if not already
    os.makedirs("artifacts", exist_ok=True)
    zip_path = "artifacts/housing_model.zip"
    pkl_path = "artifacts/housing_model.pkl"
    
    if not os.path.exists(pkl_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("artifacts/")
    
    # Load artifacts
    model = joblib.load(pkl_path)
    scaler = joblib.load("artifacts/scaler.pkl")
    feature_names = joblib.load("artifacts/feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# -----------------------------
# App title and description
# -----------------------------
st.title("🏠 Hamilton Housing Price Estimator")

st.markdown(
    """
⚠️ **Disclaimer:**  
This prediction is an approximate estimate based on historical property data.
Actual appraised values may differ.
"""
)

# -----------------------------
# User inputs
# -----------------------------
calc_acres = st.number_input(
    "Lot Size (Acres)", min_value=0.0, max_value=50.0, value=0.25, step=0.01
)

land_use = st.selectbox(
    "Land Use Type", ["Single Family", "Condo", "Multi-Family", "Unknown"]
)

property_type = st.selectbox(
    "Property Type", ["Residential", "Commercial", "Vacant", "Unknown"]
)

# -----------------------------
# Build input row
# -----------------------------
input_dict = {col: 0 for col in feature_names}
input_dict["CALC_ACRES"] = calc_acres

land_col = f"LAND_USE_CODE_DESC_{land_use}"
prop_col = f"PROPERTY_TYPE_CODE_DESC_{property_type}"

if land_col in input_dict:
    input_dict[land_col] = 1
if prop_col in input_dict:
    input_dict[prop_col] = 1

input_df = pd.DataFrame([input_dict])

# Scale input
scaled_input = scaler.transform(input_df)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict House Price"):
    prediction = model.predict(scaled_input)[0]
    st.subheader("💰 Estimated Appraised Value")
    st.success(f"${prediction:,.0f}")
