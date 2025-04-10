import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import numpy as np
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "simple_rnn_model.h5")
model = load_model(model_path)


## streamlit title
st.title("Movie Review Sentiment Analysis")


## take the input as the movie review
review = st.text_input("Please enter your movie review here:")

## use the imdb dataset to simply preprocess the input
word_index = imdb.get_word_index()
def preprocessInput(review):
    review = review.lower()
    encoded_review = [word_index.get(word,2)+3 for word in review.split()]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500, padding="pre")
    return padded_review

## make prediction fucntion
def predictSentiment(review):
    processed_review = preprocessInput(review)
    prediction = model.predict(processed_review)
    score = prediction[0][0]
    sentiment = 'Positive Review' if  score>0.5 else 'Negative Review'
    return sentiment, score

## displaying the results
sentiment, score = predictSentiment(review)

st.write(f"Results : {sentiment}")
st.write(f"Sentiment Score : {score}")


