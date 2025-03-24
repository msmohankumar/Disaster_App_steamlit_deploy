import re
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')  # Optional, for better lemmatization in some setups

def tokenize(text):
    """
    Tokenizes and lemmatizes input text.

    Steps:
    1. Lowercases and splits on non-word characters (basic regex tokenizer).
    2. Lemmatizes each token.

    Args:
        text (str): Input text.

    Returns:
        list: List of lemmatized tokens.
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token).strip() for token in tokens]
    return tokens
