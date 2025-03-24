import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from joblib import load

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Disaster Rescue App", layout="wide")

st.title("🚨 Disaster Rescue App 🆘")
st.markdown("This app helps visualize disaster response data and classify messages. ⚠️")

# ---------- Sidebar ----------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["📊 Dashboard", "💬 Classify Message"])

# ---------- Connect to Database ----------
@st.cache_data
def load_data():
    conn = sqlite3.connect('DisasterResponse.db')
    df = pd.read_sql("SELECT * FROM disaster_messages", conn)
    conn.close()
    return df

df = load_data()

# ---------- Dashboard ----------
if page == "📊 Dashboard":
    st.subheader("📂 Dataset Overview")
    st.write(df.head())

    st.subheader("📈 Message Genre Distribution")
    genre_count = df.groupby('genre').count()['message']
    fig = px.bar(genre_count, x=genre_count.index, y=genre_count.values,
                 labels={'x': 'Genre', 'y': 'Message Count'}, color=genre_count.index)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 Most Frequent Categories")
    category_cols = df.columns[4:]  # Assuming first 4 cols are id, message, original, genre
    category_counts = df[category_cols].sum().sort_values(ascending=False)[:10]
    fig2 = px.bar(category_counts, x=category_counts.index, y=category_counts.values,
                  labels={'x': 'Category', 'y': 'Count'}, color=category_counts.index)
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Classify Message ----------
elif page == "💬 Classify Message":
    st.subheader("🚀 Classify an Incoming Message")

    user_message = st.text_area("Paste a disaster-related message:")
    model = load("classifier_model.joblib")  # Make sure you have a trained model

    if st.button("Classify"):
        if user_message.strip() == "":
            st.warning("Please enter a message to classify!")
        else:
            # Fake preprocessing for demo
            # Replace with your actual text preprocessing
            X_input = pd.Series([user_message])
            preds = model.predict([user_message])[0]  # Assuming multilabel output

            result = dict(zip(category_cols, preds))
            st.success("✅ Classification Result:")
            for cat, val in result.items():
                if val == 1:
                    st.markdown(f"- **{cat}**")

# ---------- Footer ----------
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Mohan Kumar 🚀")
