import streamlit as st
import pandas as pd
import numpy as np

## take the input
name = st.text_input("Enter your name")
if name:
    st.write(f"Your name is {name}")

age = st.slider("Select your age:",0,100,18)
st.write(f"Your age is {age}")

options = ["Java","C++","Python","Javascript"]
choice = st.selectbox("Choose a suitable option",options)
st.write("You chose",choice)

uploaded_file = st.file_uploader("Choose csv",type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)