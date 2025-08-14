import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding,Dense,SimpleRNN

word_index = imdb.get_word_index()
reversed_word_index = {v: k for k, v in word_index.items()}
model = load_model('imdb_rnn_model.h5')

def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])
def preprocess_review(review):
    words = review.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in review.split()]
    return sequence.pad_sequences([encoded_review], maxlen=400)
def predict_review(review):
    encoded_review = preprocess_review(review)
    prediction = model.predict(encoded_review)
    prediction_label = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return prediction_label, prediction[0][0]


import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative):")

user_input = st.text_area("movie review")
if st.button("predict"):
    preprocessed_input = preprocess_review(user_input)
    prediction_label, prediction_score = predict_review(user_input)
    st.write(f"Prediction: {prediction_label} (Score: {prediction_score:.2f})")
    # st.write("Decoded Review:", user_input)

    # predction = model.predict(preprocessed_input)
else:
    st.write("Please enter a review to get the prediction.")
