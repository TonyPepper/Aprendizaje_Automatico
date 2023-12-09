import streamlit as st
import numpy as np
import joblib

st.title('Predicción del clima')

pipeline_entrenado = joblib.load('C:/Users/Usuario/Documents/tuia/aprendizaje automatico/Unidad 6_ MLOps/ejemplo-streamlit-iris/clima_pipeline.joblib')



MinTemp = st.slider('MinTemp', -4.8, 29.7, 13.18)
MaxTemp = st.slider('MaxTemp', 6.6, 47.3, 23.75)
Rainfall = st.slider('Rainfall', 0.0, 371.0, 2.7)
Evaporation = st.slider('Evaporation', 0.0, 145.0, 5.62)
Sunshine = st.slider('Sunshine', 0.0, 14.10, 7.58)
WindGustDir = st.slider('WindGustDir', 0.0, 15.0, 7.28)
WindGustSpeed = st.slider('WindGustSpeed', 7.0, 135.0, 38.92)
WindDir9am = st.slider('WindDir9am', 0.0, 15.0, 8.2)
WindSpeed3pm = st.slider('WindSpeed3pm', 0.0, 15.0, 7.12)
WindSpeed9am = st.slider('WindSpeed9am', 0.0, 130.0, 13.15)
WindSpeed3pm = st.slider('WindSpeed3pm', 4.0, 8.0, 5.0)
Humidity9am = st.slider('Humidity9am', 2.0, 4.5, 3.0)
Humidity3pm = st.slider('Humidity3pm', 1.0, 100.0, 52.6)
Pressure9am = st.slider('Pressure9am', 980.5, 1039.9, 1018.29)
Pressure3pm = st.slider('Pressure3pm', 979.0, 1037.8, 1015.75)
Cloud9am = st.slider('Cloud9am', 0.0, 9.0, 4.36)
Cloud3am = st.slider('Cloud3am', 0.0, 8.0, 4.44)
Temp9am = st.slider('Temp9am', 0.0, 37.7, 17.68)
Temp3pm = st.slider('Temp3pm', 6.0, 46.7, 22.28)
RainToday = st.slider('RainToday', 0.0, 1.0, 0.22)




#data_para_predecir = np.array([[MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindSpeed3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3am,Temp9am,Temp3pm,RainToday]])#se usa array porque entra ese dato a la tuberia

#prediccion = pipeline_entrenado.predict(data_para_predecir)

#st.write('Predicción:', prediccion)
# Agregar un botón para predecir
if st.button('Predict'):
    data_para_predecir = np.array([[MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindSpeed3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3am,Temp9am,Temp3pm,RainToday]])#se usa array porque entra ese dato a la tuberia

    prediccion = pipeline_entrenado.predict(data_para_predecir)
    st.write('Predicción:', prediccion)