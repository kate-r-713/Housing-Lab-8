import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --------------------------
# Load artifacts
# --------------------------
model = load_model("artifacts/housing_model.keras")
scaler = joblib.load("artifacts/scaler.pkl")
features = joblib.load("artifacts/feature_names.pkl")

st.title("🏠 Hamilton Housing Price Estimator")
st.markdown(
    "⚠️ **Note:** This is a rough estimate. Actual appraised values may vary."
)

# --------------------------
# Streamlit inputs
# --------------------------
calc_acres = st.number_input(
    "Lot Size (Acres)", min_value=0.0, max_value=50.0, value=0.25, step=0.01
)

land_use_options = ["Single Family", "Condo", "Multi-Family", "Unknown"]
land_use = st.selectbox("Land Use Type", land_use_options)

property_type_options = ["Residential", "Commercial", "Vacant", "Unknown"]
property_type = st.selectbox("Property Type", property_type_options)

# --------------------------
# Build input DataFrame
# --------------------------
input_dict = {feat: 0 for feat in features}
input_dict["CALC_ACRES"] = calc_acres

land_use_col = f"LAND_USE_CODE_DESC_{land_use}"
if land_use_col in input_dict:
    input_dict[land_use_col] = 1

property_type_col = f"PROPERTY_TYPE_CODE_DESC_{property_type}"
if property_type_col in input_dict:
    input_dict[property_type_col] = 1

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)

# --------------------------
# Predict
# --------------------------
predicted_value = model.predict(input_scaled)[0][0]

# --------------------------
# Display result
# --------------------------
st.subheader("💰 Estimated Appraised Value")
st.success(f"${predicted_value:,.0f}")
