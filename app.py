import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model
model = tf.keras.models.load_model("text_gen_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_seq_len = 50  # same as training

st.title("🧠 Text Generator (RNN + LSTM)")

seed_text = st.text_input("Enter starting text")

if st.button("Generate"):
    next_words = 20

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')

        predicted = np.argmax(model.predict(token_list), axis=-1)

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break

    st.write("Generated Text:")
    st.success(seed_text)