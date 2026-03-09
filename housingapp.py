import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load model and preprocessing objects
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("./artifacts/housing_model.h5")
    scaler = joblib.load("./artifacts/scaler.pkl")
    feature_names = joblib.load("./artifacts/feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

st.title("🏠 Housing Price Prediction App")

st.write("Enter housing details to estimate the median house value.")

# --- User Inputs ---
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

# --- Prepare Input Data ---
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

# One-hot encode to match training
input_df = pd.get_dummies(input_df)

# Ensure columns match training features
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_names]

# Scale numeric values
scaled_input = scaler.transform(input_df)

# --- Prediction ---
if st.button("Predict House Price"):
    prediction = model.predict(scaled_input)
    st.success(f"Estimated Median House Value: ${prediction[0][0]:,.2f}")
