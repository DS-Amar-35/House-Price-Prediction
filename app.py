import streamlit as st
import joblib
import numpy as np
model = joblib.load("linear_regression_model.pkl")
st.title("House Price Prediction App")
st.write("Enter house details to predict price.")
area = st.number_input("Area (sqft)", min_value=500, max_value=10000, value=1200)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
if st.button("Predict Price"):
 features = np.array([[area,bedrooms,bathrooms]])
 prediction = model.predict(features)[0]
 st.success(f"Predicted House Price: ${prediction:,.2f}")




