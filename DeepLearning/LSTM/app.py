import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model


# load model
model=load_model('nextword_lstm.h5')

with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)
    # pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)


def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list[-(max_sequence_len):] # ensure token len matches max squences
    token_list=pad_sequences([token_list],maxlen=max_sequence_len,padding="pre")
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)
    for word, index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None



# Streamlist
st.title("Next word predcion ")
input_text=st.text_input("Enter the Sequence","TO be or not to be great")

if st.button("Predict next word"):
    max_sequence_len=model.input_shape[1]+1
    nextword=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write("Next word:",nextword)

















