import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index=imdb.get_word_index()
reverse_word_index={v:k for k,v in word_index.items()}
model=load_model('simple_rnn_imbd.h5')
## helper functins
# function to decode the review
def decode_review(encode_review):
    return " ".join([reverse_word_index.get(i-3,"?") for i in encode_review])


def preprocess_text(text):
    print(f"{text}=")
    words=text.lower().split()
    enocode_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([enocode_review],maxlen=500)
    return padded_review

# prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)

    sentiment='Positive' if prediction[0][0]>.5 else 'Negative'
    return sentiment,prediction[0][0]



## streamlit app
import streamlit as st
st.title("IMDB Moive Review Sentiment Analysis")
st.write("Enter a movie review")

user_input=st.text_area('Moview Review')


if st.button('Classify'):
    preprocess_text=preprocess_text(user_input)
    prediction=model.predict(preprocess_text)
    sentiment='Positive' if  prediction[0][0]>.5 else 'Negative'

    st.write(f'{sentiment=}')
    st.write(f'Score: {prediction[0][0]}')
else:
    st.write('Please Enter the movie review')