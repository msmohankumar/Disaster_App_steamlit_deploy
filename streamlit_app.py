import streamlit as st
import pandas as pd
import sqlite3
import os
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from utils import simple_tokenize



nltk.download('punkt')
nltk.download('wordnet')

# Streamlit App Config
st.set_page_config(page_title="Disaster Rescue App", page_icon="🚨")
st.title("🚨 Disaster Rescue Message Classifier")

# Paths
DB_PATH = 'DisasterResponse.db'
MODEL_PATH = 'classifier.pkl'

# Load model if exists
if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found! Please run train_classifier.py to generate classifier.pkl.")
else:
    model = joblib.load(MODEL_PATH)

    # Tokenizer function (same as in train_classifier)
    def tokenize(text):
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
        return tokens

    # Load DB if exists
    if not os.path.exists(DB_PATH):
        st.error("Database not found! Please run process_data.py to generate DisasterResponse.db.")
    else:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM disaster_messages", conn)
        conn.close()

        # Sidebar Info
        st.sidebar.header("About")
        st.sidebar.info("This app helps classify disaster-related messages into multiple emergency-related categories!")

        # Input Message
        txt = st.text_area("Paste a Disaster-related Message here:")

        if st.button("Classify 🚀"):
            if txt:
                # Actual model prediction
                prediction = model.predict([txt])[0]
                categories = df.columns[4:]  # Skip id, message, original, genre
                
                st.success("✅ Classification complete!")
                st.write(f"**Message:** {txt}")
                st.write("**Predicted Categories:**")
                for cat, label in zip(categories, prediction):
                    if label == 1:
                        st.markdown(f"- ✅ **{cat}**")
            else:
                st.warning("Please input a message to classify.")

        # Show the database
        toggle = st.checkbox("Show Dataset 📊")
        if toggle:
            st.dataframe(df.head(20))
