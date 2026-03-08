import streamlit as st
import pandas as pd
import numpy as np
import joblib
import onnxruntime as rt

# Load ONNX model and artifacts
@st.cache_resource
def load_artifacts():
    sess = rt.InferenceSession("artifacts/housing_model.onnx")
    scaler = joblib.load("artifacts/scaler.pkl")
    features = joblib.load("artifacts/feature_names.pkl")
    return sess, scaler, features

sess, scaler, features = load_artifacts()

st.title("🏠 Hamilton Housing Price Estimator")
st.markdown("⚠️ **Note:** This is a rough estimate. Actual values may vary.")

# Inputs
calc_acres = st.number_input("Lot Size (Acres)", min_value=0.0, max_value=50.0, value=0.25, step=0.01)
land_use_options = ["Single Family", "Condo", "Multi-Family", "Unknown"]
land_use = st.selectbox("Land Use Type", land_use_options)
property_type_options = ["Residential", "Commercial", "Vacant", "Unknown"]
property_type = st.selectbox("Property Type", property_type_options)

# Build input DataFrame
input_dict = {feat: 0 for feat in features}
input_dict["CALC_ACRES"] = calc_acres
land_use_col = f"LAND_USE_CODE_DESC_{land_use}"
if land_use_col in input_dict: input_dict[land_use_col] = 1
property_type_col = f"PROPERTY_TYPE_CODE_DESC_{property_type}"
if property_type_col in input_dict: input_dict[property_type_col] = 1
input_df = pd.DataFrame([input_dict])

# Scale and predict
input_scaled = scaler.transform(input_df)
input_name = sess.get_inputs()[0].name
predicted_value = sess.run(None, {input_name: input_scaled.astype(np.float32)})[0][0]

# Display
st.subheader("💰 Estimated Appraised Value")
st.success(f"${predicted_value:,.0f}")
