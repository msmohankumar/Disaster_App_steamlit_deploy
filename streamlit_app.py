import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from utils import simple_tokenize

# Define absolute paths (based on deployment structure)
BASE_PATH = "/mount/src/disaster_app_steamlit_deploy/"
MODEL_PATH = BASE_PATH + "model.pkl"
DATA_PATH = BASE_PATH + "DisasterResponse.db"

# Load pre-trained model
model = joblib.load(MODEL_PATH)

# Load data for visuals
df = pd.read_sql_table('DisasterResponse', f'sqlite:///{DATA_PATH}')

# Streamlit UI
st.title("🌍 Disaster Response Message Classifier")
st.markdown("Classify disaster-related messages into multiple response categories!")

# Sidebar - Data exploration
st.sidebar.title("📊 Data Overview")
if st.sidebar.checkbox("Show dataset"):
    st.dataframe(df.head())

# Visualization 1 - Distribution of message genres
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)

st.sidebar.subheader("Message Genre Distribution")
fig_genre = px.bar(
    x=genre_names,
    y=genre_counts.values,
    labels={'x': 'Genre', 'y': 'Number of Messages'},
    title="Messages per Genre"
)
st.sidebar.plotly_chart(fig_genre, use_container_width=True)

# User Input
txt = st.text_area("✍️ Enter a disaster-related message:", "")

if st.button("🚀 Classify Message"):
    if txt.strip():
        prediction = model.predict([txt])[0]
        prediction_labels = df.columns[4:]
        prediction_results = dict(zip(prediction_labels, prediction))

        st.subheader("🧩 Classification Results:")
        for category, value in prediction_results.items():
            st.write(f"**{category}**: {'✅ Relevant' if value == 1 else '❌ Not Relevant'}")
    else:
        st.warning("⚠️ Please enter a message to classify.")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Plotly.")
