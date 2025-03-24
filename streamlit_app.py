import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.express as px

# Load Data
@st.cache_data
def load_data():
    conn = sqlite3.connect('DisasterResponse.db')
    df = pd.read_sql("SELECT * FROM messages", conn)
    conn.close()
    return df

# Load Trained Model
@st.cache_resource
def load_model():
    model = joblib.load('classifier.pkl')  # Update if model filename is different
    return model

# App Title
st.title("🚨 Disaster Response App 🌍")
st.write("Classify disaster messages and provide the right category for rescue efforts! 🚑")

# Sidebar Filters / Upload
st.sidebar.header("📥 Input Message")
user_input = st.sidebar.text_area("Enter a disaster-related message:", "")

# Load Data and Model
df = load_data()
model = load_model()

# Show Data Sample
with st.expander("📊 View Sample Data"):
    st.dataframe(df.head())

# Predict Button
if st.sidebar.button("Classify Message"):
    if user_input != "":
        classification_labels = model.predict([user_input])[0]
        classification_results = dict(zip(df.columns[4:], classification_labels))  # Assuming label cols start from 5th col
        st.subheader("📌 Classification Results:")
        for category, value in classification_results.items():
            st.write(f"**{category}:** {'✅' if value else '❌'}")
    else:
        st.warning("Please enter a message!")

# Optional: Data Visualization
with st.expander("📈 View Message Categories Distribution"):
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    fig = px.bar(category_counts, orientation='h', title="Category Distribution in Dataset")
    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
