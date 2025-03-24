# utils.py
import re
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')  # WordNet typically works without external files on Streamlit Cloud

def simple_tokenize(text):
    # Simple regex tokenizer: split on non-word characters
    tokens = re.findall(r'\b\w+\b', text.lower())
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token).strip() for token in tokens]
    return tokens
