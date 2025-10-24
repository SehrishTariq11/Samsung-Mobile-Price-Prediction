import streamlit as st
import numpy as np
import joblib

model = joblib.load('model_poly.pkl')

st.title("Polynomial Regression Price Predictor")

feature = st.number_input("Enter input value", min_value=0.0, max_value=10.0, value=5.0)

if st.button("Predict"):
    X_input = np.array([[feature]])
    prediction = model.predict(X_input)[0]
    st.success(f"Predicted Value: {prediction:.2f}")
