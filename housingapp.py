import streamlit as st
import pandas as pd
import numpy as np
import joblib
import onnxruntime as rt

# -----------------------------
# Load ONNX model and preprocessing
# -----------------------------
@st.cache_resource
def load_artifacts():
    session = rt.InferenceSession("artifacts/housing_model.onnx")
    scaler = joblib.load("artifacts/scaler.pkl")
    feature_names = joblib.load("artifacts/feature_names.pkl")
    return session, scaler, feature_names

session, scaler, feature_names = load_artifacts()

# -----------------------------
# App title and description
# -----------------------------
st.title("🏠 Hamilton Housing Price Estimator")
st.markdown("⚠️ **Note:** This is a rough estimate. Actual values may vary.")

# -----------------------------
# Inputs
# -----------------------------
calc_acres = st.number_input("Lot Size (Acres)", min_value=0.0, max_value=50.0, value=0.25, step=0.01)
land_use_options = ["Single Family", "Condo", "Multi-Family", "Unknown"]
land_use = st.selectbox("Land Use Type", land_use_options)
property_type_options = ["Residential", "Commercial", "Vacant", "Unknown"]
property_type = st.selectbox("Property Type", property_type_options)

# -----------------------------
# Build input DataFrame
# -----------------------------
input_dict = {feat: 0 for feat in feature_names}
input_dict["CALC_ACRES"] = calc_acres

# One-hot encode categorical features
land_use_col = f"LAND_USE_CODE_DESC_{land_use}"
if land_use_col in input_dict: input_dict[land_use_col] = 1

property_type_col = f"PROPERTY_TYPE_CODE_DESC_{property_type}"
if property_type_col in input_dict: input_dict[property_type_col] = 1

input_df = pd.DataFrame([input_dict])

# Scale input
scaled_input = scaler.transform(input_df).astype(np.float32)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict House Price"):
    input_name = session.get_inputs()[0].name
    predicted_value = session.run(None, {input_name: scaled_input})[0][0]
    st.subheader("💰 Estimated Appraised Value")
    st.success(f"${predicted_value:,.0f}")
