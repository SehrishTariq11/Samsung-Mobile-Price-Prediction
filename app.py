import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('model_poly.pkl')

st.title("Samsung Mobile Price Prediction App")

st.write("Enter Samsung mobile specifications below to predict its price (in PKR):")

# Numeric input columns
build_os = st.number_input("Build OS (Android Version)", min_value=10, max_value=20, step=1)
sim_type = st.selectbox("SIM Type", ["Single SIM", "Dual SIM"])
cpu_cores = st.number_input("CPU Cores", min_value=4, max_value=12, step=1)
display_type = st.selectbox("Display Type", ["Dynamic Amoled", "Foldable", "Super Amoled"])
ram_gb = st.number_input("RAM (GB)", min_value=4, max_value=24, step=1)
storage_gb = st.number_input("Storage (GB)", min_value=64, max_value=1024, step=64)
camera_mp = st.number_input("Camera (MP)", min_value=12, max_value=200, step=1)
battery_mAh = st.number_input("Battery (mAh)", min_value=3000, max_value=7000, step=100)
wifi_version = st.number_input("WiFi Version", min_value=5, max_value=7, step=1)

# Binary features
def yes_no_input(label):
    return 1 if st.selectbox(label, ["Yes", "No"]) == "Yes" else 0

dual_band = yes_no_input("Dual Band")
tri_band = yes_no_input("Tri Band")
hotspot = yes_no_input("Hotspot")
wifi_direct = yes_no_input("WiFi Direct")
accelerometer = yes_no_input("Accelerometer")
compass = yes_no_input("Compass")
fingerprint = yes_no_input("Fingerprint")
barometer = yes_no_input("Barometer")
heart_rate = yes_no_input("Heart Rate Sensor")

# Encoding SIM & Display to match dataset numeric format
sim_type_map = {"Single SIM": 0, "Dual SIM": 1}
sim_type = sim_type_map[sim_type]

display_map = {"Dynamic Amoled": 0, "Foldable": 1, "Super Amoled": 3}
display_type = display_map[display_type]

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[
        build_os, sim_type, cpu_cores,
        display_type, ram_gb, storage_gb, camera_mp, battery_mAh,
        wifi_version, dual_band, tri_band, hotspot, wifi_direct,
        accelerometer, compass, fingerprint, barometer, heart_rate
    ]], columns=[
        'Build_OS', 'SIM_Type', 'CPU_Cores', 'Display_Type', 'RAM_GB',
        'Storage_GB', 'Camera_MP', 'Battery_mAh', 'WiFi_Version',
        'Dual_Band', 'Tri_Band', 'Hotspot', 'WiFi_Direct',
        'Accelerometer', 'Compass', 'Fingerprint', 'Barometer', 'HeartRate'
    ])

    price = model.predict(input_data)[0]
    st.success(f"Predicted Price: {price:.2f} PKR")

st.caption("Samsung Mobile Price Prediction App ")
