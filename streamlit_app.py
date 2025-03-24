import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(page_title="Disaster Rescue App", page_icon="🚨")
st.title("🚨 Disaster Rescue Message Classifier")

# Connect to DB
conn = sqlite3.connect('DisasterResponse.db')
df = pd.read_sql("SELECT * FROM disaster_messages", conn)
conn.close()

# Sidebar Info
st.sidebar.header("About")
st.sidebar.info("This app helps classify disaster-related messages and show their categories.")

# Input Message
txt = st.text_area("Paste a Disaster-related Message here:")

if st.button("Classify 🚀"):
    if txt:
        # Dummy Classification (replace with ML model prediction later)
        st.success("This is a mock classification. In production, connect your ML model here.")
        st.write(f"Message: {txt}")
        st.write("Predicted Categories: Flood, Aid Related, Infrastructure Damage")
    else:
        st.warning("Please input a message to classify.")

# Show the database
toggle = st.checkbox("Show Dataset 📊")
if toggle:
    st.dataframe(df.head(20))
