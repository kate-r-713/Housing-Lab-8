import streamlit as st
import onnxruntime as rt
import joblib
import numpy as np

@st.cache_resource
def load_artifacts():
    session = rt.InferenceSession("./artifacts/housing_model.onnx")
    scaler = joblib.load("./artifacts/scaler.pkl")
    feature_names = joblib.load("./artifacts/feature_names.pkl")
    return session, scaler, feature_names

sess, scaler, feature_names = load_artifacts()

st.title("🏠 Hamilton Housing Price Estimator")

st.markdown(
"""
⚠️ **Disclaimer:**  
This prediction is an approximate estimate based on historical property data.
Actual appraised values may differ.
"""
)

# Inputs
calc_acres = st.number_input(
    "Lot Size (Acres)",
    min_value=0.0,
    max_value=50.0,
    value=0.25,
    step=0.01
)

land_use = st.selectbox(
    "Land Use Type",
    ["Single Family", "Condo", "Multi-Family", "Unknown"]
)

property_type = st.selectbox(
    "Property Type",
    ["Residential", "Commercial", "Vacant", "Unknown"]
)

# Build feature row
input_dict = {col: 0 for col in feature_names}
input_dict["CALC_ACRES"] = calc_acres

land_col = f"LAND_USE_CODE_DESC_{land_use}"
prop_col = f"PROPERTY_TYPE_CODE_DESC_{property_type}"

if land_col in input_dict:
    input_dict[land_col] = 1

if prop_col in input_dict:
    input_dict[prop_col] = 1

input_df = pd.DataFrame([input_dict])

# Scale features
scaled_input = scaler.transform(input_df)

# Predict
input_name = sess.get_inputs()[0].name
prediction = sess.run(None, {input_name: scaled_input.astype(np.float32)})[0][0]

# Display result
st.subheader("💰 Estimated Appraised Value")

st.success(f"${prediction:,.0f}")

