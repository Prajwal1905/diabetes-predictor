import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("best_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter the following details to check the diabetes risk:")

with st.form("prediction_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=1)
    insulin = st.number_input("Insulin", min_value=1)
    bmi = st.number_input("BMI", min_value=1.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.2f")
    age = st.number_input("Age", min_value=1)
    submit = st.form_submit_button("Predict")


if submit:
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]],
                                columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                         'SkinThickness', 'Insulin', 'BMI',
                                         'DiabetesPedigreeFunction', 'Age'])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("The person is likely **Diabetic**.")
    else:
        st.success("The person is likely **Not Diabetic**.")

    if probability is not None:
        st.info(f"Prediction confidence: {probability*100:.2f}%")

st.markdown("---")
st.caption("Made by Prajwal")
