import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import streamlit as st
import joblib
import sklearn

# Path del modelo preentrenado
MODEL_PATH = 'models/RandomForestClassifier.pkl'
SCALER_PATH = 'models/MinMaxScaler.pkl'


# Se reciben los valores y el modelo, devuelve la predicción
def model_prediction(x_in, model):

    x = np.asarray(x_in).reshape(1,-1)
    preds=model.predict(x)

    return preds


def main():
    
    model=''

    # Se carga el modelo
    if model=='':
        model = joblib.load(MODEL_PATH)
    
    scaler = joblib.load(SCALER_PATH)
    
    # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">Algoritmo En Spark</h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Lectura de datos
    data_size_mb = st.text_input("Valor de data_size_mb:")
    user = st.text_input("Valor de user:")
    system = st.text_input("Valor de max_heart_rate_achieved:")
    softirq = st.text_input("Valor de system:")
    used = st.text_input("Valor de softirq:")
    recv_packets = st.text_input("Valor de recv_packets:")
    load10 = st.text_input("Valor de load10:")
    
    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción :"):         
        x_in =np.array([[np.float_(data_size_mb.title()),
                    np.float_(user.title()),
                    np.float_(system.title()),
                    np.float_(softirq.title()),
                    np.float_(used.title()),
                    np.float_(recv_packets.title()),
                    np.float_(load10.title())]])
        
        x_in = pd.DataFrame(x_in, columns=['data_size_mb', 'user', 'system', 'softirq', 'used', 'recv_packets', 'load10'])
        x_in = scaler.transform(x_in)
        predictS = model_prediction(x_in, model)
        st.success('La predicción de dolencia cardiaca es: {}'.format(predictS[0]).upper())

if __name__ == '__main__':
    main()