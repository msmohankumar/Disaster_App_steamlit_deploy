import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from utils import tokenize

# Load pre-trained model
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

# Load data for visuals
DATA_PATH = "DisasterResponse.db"
df = pd.read_sql_table('DisasterResponse', f'sqlite:///{DATA_PATH}')

# Streamlit UI
st.title("Disaster Response Message Classifier")
st.markdown("Enter a message to classify it into relevant disaster response categories.")

# Sidebar for data exploration
st.sidebar.title("Data Overview")
if st.sidebar.checkbox("Show dataset"):
    st.write(df.head())

# Visualization 1 - Distribution of message genres
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)

st.sidebar.subheader("Message Genre Distribution")
fig_genre = px.bar(x=genre_names, y=genre_counts.values, labels={'x': 'Genre', 'y': 'Count'})
st.sidebar.plotly_chart(fig_genre)

# Text input box for user message
txt = st.text_area("Enter a disaster-related message:", "")

if st.button("Classify Message"):
    if txt:
        # Predict using the model
        prediction = model.predict([txt])[0]
        prediction_labels = df.columns[4:]
        prediction_results = dict(zip(prediction_labels, prediction))

        st.subheader("Classification Results:")
        for category, value in prediction_results.items():
            st.write(f"{category}: {'✅' if value == 1 else '❌'}")
    else:
        st.warning("Please enter a message to classify.")

# Optional: Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
