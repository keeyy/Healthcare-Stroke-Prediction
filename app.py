import pandas as pd
import numpy as np
import joblib
import streamlit as st

with open('pipeline_knn.pkl', 'rb') as file_1:
  full_imbhandling = joblib.load(file_1)

st.header('Stroke Prediction')

age = st.number_input('Age : ', 1, 82)
average_glucose_level = st.number_input('Average Glucose Level : ', 55.12, 271.74, step=0.1)
bmi = st.number_input('BMI : ', 10.3, 97.6, step=0.1)
hypertension = st.radio('Hypertension 0=yes 1=no : ', (0, 1))
heart_disease = st.radio('Heart Disease 0=yes 1=no : ', (0, 1))
ever_married = st.radio('Married Status : ', ('Yes', 'No'))
smoking_status = st.selectbox('Smoking Status : ', ('formerly smoked', 'never smoked', 'smokes', 'Unknown'))

if st.button('Predict'):
    data_inf = pd.DataFrame({'age' : [age], 
                            'avg_glucose_level' : [average_glucose_level], 
                            'bmi' : [bmi], 
                            'hypertension' : [hypertension], 
                            'heart_disease' : [heart_disease], 
                            'ever_married' : [ever_married], 
                            'smoking_status' : [smoking_status]})

    hasil = 'Not Stroke' if full_imbhandling.predict(data_inf) == 0 else 'Stroke'
    st.header(f'Prediksi = {hasil}')