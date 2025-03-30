import streamlit as st
import pandas as pd
import numpy as np


## simple title
st.title("Hello Streamlit")


## plain text
st.write("This is simple text")


## display data frame
df = pd.DataFrame({
    'first column':[1,2,3,4,5],
    'second column':[1,4,9,16,25]
})

st.write(df)

## chart data
chart_data = pd.DataFrame(
    np.random.randn(20,4), columns=[1,2,3,4]
)

st.line_chart(chart_data)