import streamlit as st
import pandas as pd
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

# Load Model dan TF-IDF
model = joblib.load('naive_bayes_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Fungsi Preprocessing
def preprocess_text(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = text.lower()
    text = re.sub(r'[^\w\s]|@\w+|#\w+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    tokens = text.split()
    stemmed = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed)

# Tampilan Streamlit
st.title("🔎 Analisis Sentimen Tweet Valorant")
st.write("Prediksi sentimen (positif/negatif) menggunakan model Naive Bayes")

input_text = st.text_area("Masukkan tweet tentang Valorant:")
if st.button("Prediksi"):
    cleaned_text = preprocess_text(input_text)
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    st.success(f"Sentimen diprediksi: **{prediction.upper()}**")
