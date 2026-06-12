import streamlit as st
import tensorflow as tf

# ----------------------------------
# LOAD MODEL
# ----------------------------------

model = tf.keras.models.load_model("models/bilstm_model.keras")

# ----------------------------------
# PAGE
# ----------------------------------

st.title("Bidirectional LSTM")

st.write("IMDb Review Classification")

review = st.text_area("Enter Review")

if st.button("Predict"):
    st.success("Bi-LSTM Model Loaded Successfully")
