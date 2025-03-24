import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None")

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

DATABASE_FILEPATH = '/workspaces/Disaster_App_steamlit_deploy/DisasterResponse.db'
MODEL_FILEPATH = '/workspaces/Disaster_App_steamlit_deploy/classifier.pkl'

def load_data(database_filepath):
    print(f"Loading data from {database_filepath}...")
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Correct table name as per your DB
    df = pd.read_sql_table('disaster_messages', engine)
    
    X = df['message']
    y = df.iloc[:, 4:]  # Assuming first columns: id, message, original, genre
    category_names = y.columns.tolist()
    
    print(f"Loaded {df.shape[0]} records.")
    return X, y, category_names

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    return tokens

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(f'\nCategory: {category}')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
    print(f"Model saved to {model_filepath}")

def main():
    X, Y, category_names = load_data(DATABASE_FILEPATH)

    print('Splitting data...')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...')
    save_model(model, MODEL_FILEPATH)

    print('Pipeline complete!')

if __name__ == '__main__':
    main()
