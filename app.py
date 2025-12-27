import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="Churn Predictor App", page_icon="📊", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    .main {background-color: #F7FAFC;}
    .title {text-align:center; font-size:40px; font-weight:700; color:#2C5282;}
    .sub {text-align:center; font-size:18px; color:#4A5568;}
    .footer {text-align:center; font-size:13px; color:#718096; padding-top:20px;}
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='title'>📊 Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Enter customer details to predict churn probability 🚀</p>", unsafe_allow_html=True)

# Load Model & Encoders
model = tf.keras.models.load_model('model.h5')
label_encoder_gender = pickle.load(open('label_encoder_gender.pkl', 'rb'))
onehot_encoder_geo = pickle.load(open('onehot_encoder_geo.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Input fields grouped in columns
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('🧑 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92)
        tenure = st.slider('⌛ Tenure (Years)', 0, 10)
        num_of_products = st.slider('🛍️ Number of Products', 1, 4)

    with col2:
        credit_score = st.number_input('💳 Credit Score')
        balance = st.number_input('💰 Balance')
        estimated_salary = st.number_input('💼 Estimated Salary')
        has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
        is_active_member = st.selectbox('🔥 Active Member', [0, 1])

# Prepare Input Data
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode geography
geo_encoded = onehot_encoder_geo.transform(pd.DataFrame({'Geography':[geography]})).toarray()
geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
geo_df = pd.DataFrame(geo_encoded, columns=geo_cols)

input_df = pd.concat([input_df, geo_df], axis=1)
scaled = scaler.transform(input_df)

# Predict Button
st.markdown("---")
if st.button("🔍 Predict Churn"):
    prediction = float(model.predict(scaled)[0][0])

    st.write("### 📈 Churn Probability Score")
    st.progress(prediction)
    st.write(f"**Prob = `{prediction:.2f}`**")

    if prediction > 0.5:
        st.error("⚠️ High Risk – Customer is likely to churn!")
    else:
        st.success("✔️ Low Risk – Customer is not likely to churn")

