import streamlit as st
import pandas as pd
import numpy as np
import joblib
import onnxruntime as rt

# -----------------------------
# Load model artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    session = rt.InferenceSession("./artifacts/housing_model.onnx")
    scaler = joblib.load("./artifacts/scaler.pkl")
    feature_names = joblib.load("./artifacts/feature_names.pkl")
    return session, scaler, feature_names

session, scaler, feature_names = load_artifacts()

st.title("🏠 Housing Price Prediction")

st.write("Enter housing characteristics to estimate median house value.")

# -----------------------------
# User Inputs
# -----------------------------
longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=41)
total_rooms = st.number_input("Total Rooms", value=880)
total_bedrooms = st.number_input("Total Bedrooms", value=129)
population = st.number_input("Population", value=322)
households = st.number_input("Households", value=126)
median_income = st.number_input("Median Income", value=8.3252)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

# -----------------------------
# Prepare Input Data
# -----------------------------
input_dict = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
}

input_df = pd.DataFrame([input_dict])

# One-hot encode ocean proximity
input_df = pd.get_dummies(input_df)

# Ensure same columns as training data
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_names]

# Scale input
scaled_input = scaler.transform(input_df)

# Convert to float32 for ONNX
scaled_input = scaled_input.astype(np.float32)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict House Price"):

    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: scaled_input})

    prediction = pred[0][0][0]

    st.success(f"Estimated Median House Value: ${prediction:,.2f}")

