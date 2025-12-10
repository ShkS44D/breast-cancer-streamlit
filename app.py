import streamlit as st
import pandas as pd
import numpy as np
import pickle

model, scaler, feature_names = pickle.load(open("model.pkl","rb"))

st.title("Breast Cancer Prediction App (SVM)")

inputs=[]
for f in feature_names:
    value = st.number_input(f, value=0.0)
    inputs.append(value)

if st.button("Predict"):
    x = np.array(inputs).reshape(1,-1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    st.success("Prediction: " + ("Malignant" if pred=="M" else "Benign"))
