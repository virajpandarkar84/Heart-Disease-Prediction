import streamlit as st
import pandas as pd
import joblib
import os

# Folder where app2.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load files using absolute paths
model = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))



st.title("Heart Disease Prediction System")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate", min_value=50, max_value=250, value=150)
exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


#Q.10
if st.button("Predict"):
    sample = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    sample = pd.get_dummies(sample)
    sample = sample.reindex(columns=columns, fill_value=0)

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("Heart Disease: Yes")
    else:
        st.success("Heart Disease: No")




