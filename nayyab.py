import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

# Model aur encoders load 
model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as f:
    geo_encoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# App ka title
st.title("Simple Salary Predictor")

# Inputs section
geo = st.selectbox("Country", geo_encoder.categories_[0])
gender = st.selectbox("Gender", gender_encoder.classes_)
age = st.slider("Age", 18, 90)
score = st.number_input("Credit Score")
balance = st.number_input("Balance")
tenure = st.slider("Tenure (Years)", 0, 10)
products = st.slider("Products", 1, 4)
card = st.selectbox("Has Credit Card?", [0, 1])
active = st.selectbox("Is Active Member?", [0, 1])

# Input ka DataFrame bana lein
data = pd.DataFrame({
    "CreditScore": [score],
    "Gender": [gender_encoder.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [products],
    "HasCrCard": [card],
    "IsActiveMember": [active]
})

# Geography encode kar ke add karein
geo_encoded = geo_encoder.transform([[geo]]).toarray()
geo_cols = geo_encoder.get_feature_names_out(["Geography"])
geo_df = pd.DataFrame(geo_encoded, columns=geo_cols)

# Sab ko combine kar ke final input banaya
final_input = pd.concat([data, geo_df], axis=1)

# Scaling aur prediction
scaled = scaler.transform(final_input)
pred = model.predict(scaled)
salary = pred[0][0]

# Result show 
st.write(f"Predicted Salary: {salary:.2f}")

if salary >= 400000:
    st.success("Range: 400k to 500k")
elif salary >= 300000:
    st.info("Range: 300k to 400k")
elif salary >= 200000:
    st.warning("Range: 200k to 300k")
else:
    st.error("Range: Below 100k")


