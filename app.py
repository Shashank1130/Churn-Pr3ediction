import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle as pkl
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the trained model
model = load_model('model.h5')

# Load the Gender Label Encoder
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pkl.load(file)

# Load the Geography OneHotEncoder
with open('ohe_geo.pkl', 'rb') as file:
    ohe_geo = pkl.load(file)

# Load the Scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pkl.load(file)

# Streamlit app title
st.title('CUSTOMER CHURN PREDICTION')

# User Inputs
geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Button for triggering the prediction
if st.button('Predict Churn'):
    # Prepare the input data without Geography
    input_data = pd.DataFrame({
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

    # Encode Geography
    geo_encoded = ohe_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

    # Combine OHE encoded Geography with input data
    input_df = pd.concat([input_data, geo_encoded_df], axis=1)

    # Scale the input data
    input_df_scaled = scaler.transform(input_df)

    # Predict Churn
    prediction = model.predict(input_df_scaled)
    prediction_prob = prediction[0][0]

    # Display the result
    st.write(f"Churn Probability: {prediction_prob:.2f}")
    if prediction_prob > 0.5:
        st.write("The customer is **likely** to churn.")
    else:
        st.write("The customer is **not likely** to churn.")
