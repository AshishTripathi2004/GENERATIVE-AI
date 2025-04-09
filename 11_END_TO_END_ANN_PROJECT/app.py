import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

import os

# Get path to the directory where the current script (app.py) is
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Append the model filename to that path
model_path = os.path.join(BASE_DIR, "model.h5")

# Load the model
model = tf.keras.models.load_model(model_path)



## load the encoders and scalers
with open(os.path.join(BASE_DIR, "onehot_encoder_geo.pkl"), "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open(os.path.join(BASE_DIR, "label_encoder_gender.pkl"), "rb") as file:
    label_encoder_gender = pickle.load(file)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as file:
    scaler = pickle.load(file)

## set the title
st.title("CUSTOMER CHURN PREDICTION")

## specify the inputs
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.select_slider("Age",options=list(range(18, 101)), value=25)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",0,10)
num_of_products = st.selectbox("Number of Products",[1,2,3,4])
has_cr_card = st.selectbox("Has Credit Card?",[0,1])
is_active_member = st.selectbox("Is Member?", [0,1])


## convert it into dataframe
input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

## label encode the gender 
input_df["Gender"] = label_encoder_gender.transform(input_df["Gender"])

## also one hot encode the geography values
geo_encoded = onehot_encoder_geo.transform([input_df["Geography"]]).toarray()
lables = onehot_encoder_geo.get_feature_names_out()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=lables)

## append this
input_df = input_df.drop(columns=["Geography"],axis=1)
input_df = pd.concat([input_df,geo_encoded_df],axis=1)


## scale the data
input_df = scaler.transform(input_df)


## make prediction
prediction = model.predict(input_df)
st.text(f"The probability is : {prediction[0][0]:.2f}")
if(prediction>0.5):
    st.text("The customer will Churn")
else:
    st.text("The customer will Not Churn")
