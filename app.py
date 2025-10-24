import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
model = joblib.load('/content/model.pkl')

st.title(" Samsung Price Prediction")

# Inputs
build_os = st.number_input("Build OS", min_value=10, max_value=20, value=14)

sim_type = st.selectbox("SIM Type", ["Single SIM", "Dual SIM"])
sim_type = 0 if sim_type == "Single SIM" else 1  # numeric encoding

cpu_cores = st.number_input("CPU Cores", 4, 12, 8)

display_type = st.selectbox("Display Type", ["Dynamic Amoled", "Foldable", "Super Amoled"])
display_map = {"Dynamic Amoled": 0, "Foldable": 1, "Super Amoled": 3}
display_type = display_map[display_type]

ram_gb = st.number_input("RAM (GB)", 4, 24, 12)
storage_gb = st.number_input("Storage (GB)", 64, 1024, 256, 64)
camera_mp = st.number_input("Camera (MP)", 12, 200, 50)
battery_mAh = st.number_input("Battery (mAh)", 3000, 7000, 5000, 100)
wifi_version = st.number_input("WiFi Version", 5, 7, 7)

def yn(label):
    return 1 if st.radio(label, ["Yes", "No"]) == "Yes" else 0

dual_band = yn("Dual Band")
tri_band = yn("Tri Band")
hotspot = yn("Hotspot")
wifi_direct = yn("WiFi Direct")
accelerometer = yn("Accelerometer")
compass = yn("Compass")
fingerprint = yn("Fingerprint")
barometer = yn("Barometer")
heart_rate = yn("Heart Rate Sensor")

# Predict
if st.button("Predict Price"):
    df = pd.DataFrame([[
        build_os, sim_type, cpu_cores, display_type, ram_gb,
        storage_gb, camera_mp, battery_mAh, wifi_version,
        dual_band, tri_band, hotspot, wifi_direct,
        accelerometer, compass, fingerprint, barometer, heart_rate
    ]], columns=[
        'Build_OS','SIM_Type','CPU_Cores','Display_Type','RAM_GB',
        'Storage_GB','Camera_MP','Battery_mAh','WiFi_Version',
        'Dual_Band','Tri_Band','Hotspot','WiFi_Direct',
        'Accelerometer','Compass','Fingerprint','Barometer','HeartRate'
    ])

    price = model.predict(df)[0]
    st.success(f" Predicted Price: **{round(price):,} PKR** ")

