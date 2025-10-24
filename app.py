import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline model
model = joblib.load("model_poly.pkl")

st.set_page_config(page_title="Samsung Price Predictor", layout="centered")
st.title(" Samsung Mobile Price Prediction App")

st.write("Enter mobile specifications to predict its price (PKR):")

# Inputs
build_os = st.number_input("Build OS (Android Version)", min_value=10, max_value=20, value=14)
sim_type = st.selectbox("SIM Type", ["Single SIM", "Dual SIM"])
cpu_cores = st.number_input("CPU Cores", min_value=4, max_value=12, value=8)
display_type = st.selectbox("Display Type", ["Dynamic Amoled", "Foldable", "Super Amoled"])
ram_gb = st.number_input("RAM (GB)", min_value=4, max_value=24, value=12)
storage_gb = st.number_input("Storage (GB)", min_value=64, max_value=1024, value=256, step=64)
camera_mp = st.number_input("Camera (MP)", min_value=12, max_value=200, value=50)
battery_mAh = st.number_input("Battery (mAh)", min_value=3000, max_value=7000, value=5000, step=100)
wifi_version = st.number_input("WiFi Version", min_value=5, max_value=7, value=7)

# Yes/No Features
def yn_feature(label):
    return st.selectbox(label, ["Yes", "No"])

dual_band = yn_feature("Dual Band")
tri_band = yn_feature("Tri Band")
hotspot = yn_feature("Hotspot")
wifi_direct = yn_feature("WiFi Direct")
accelerometer = yn_feature("Accelerometer")
compass = yn_feature("Compass")
fingerprint = yn_feature("Fingerprint")
barometer = yn_feature("Barometer")
heart_rate = yn_feature("Heart Rate Sensor")

# Predict Button
if st.button("🔍 Predict Price"):
    df = pd.DataFrame([[
        build_os, sim_type, cpu_cores, display_type, ram_gb,
        storage_gb, camera_mp, battery_mAh, wifi_version,
        dual_band, tri_band, hotspot, wifi_direct,
        accelerometer, compass, fingerprint, barometer, heart_rate
    ]], columns=[
        "Build_OS","SIM_Type","CPU_Cores","Display_Type","RAM_GB",
        "Storage_GB","Camera_MP","Battery_mAh","WiFi_Version",
        "Dual_Band","Tri_Band","Hotspot","WiFi_Direct",
        "Accelerometer","Compass","Fingerprint","Barometer","HeartRate"
    ])

    predicted_price = model.predict(df)[0]

    st.success(f" Predicted Price: **{round(predicted_price):,} PKR** 💰")

st.caption("Polynomial Regression + Streamlit 🚀")




