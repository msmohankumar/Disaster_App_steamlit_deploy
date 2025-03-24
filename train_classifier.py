
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None")

nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load data from the SQLite database.

    Parameters:
    database_filepath (str): Filepath of the SQLite database.

    Returns:
    X (pandas.DataFrame): Feature variables.
    y (pandas.DataFrame): Target variables.
    category_names (list): List of category names.
    """
    try:
        engine = create_engine(f'sqlite:///{database_filepath}')
        df = pd.read_sql_table('disaster_data', engine)
        X = df['message']
        y = df.iloc[:, 4:]
        category_names = y.columns.tolist()
    except Exception as e:
        print(f'Error loading data: {e}')
        return None, None, None

    return X, y, category_names

def tokenize(text):
    """
    Tokenize and lemmatize the text.

    Parameters:
    text (str): Text to be tokenized and lemmatized.

    Returns:
    tokens (list): List of tokens.
    """
    try:
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    except Exception as e:
        print(f'Error tokenizing text: {e}')
        return []

    return tokens

def build_model():
    """
    Build and configure the machine learning model.

    Returns:
    grid_search (GridSearchCV): GridSearchCV object with the model pipeline and parameter grid.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__max_features': ['auto', 'sqrt', 'log2'],
        'clf__estimator__max_depth': [3, 5, 10, None],
        'clf__estimator__min_samples_split': [2, 5, 10],
        'clf__estimator__min_samples_leaf': [1, 5, 10]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1)
    
    return grid_search

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the model.

    Parameters:
    model: Trained model.
    X_test (pandas.DataFrame): Testing features.
    Y_test (pandas.DataFrame): Testing targets.
    category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    
    for category_name in category_names:
        print(f'{category_name}:')
        print(classification_report(Y_test[category_name], Y_pred[:, category_names.index(category_name)]))
        print('--------------------------------------')

def save_model(model, model_filepath):
    """
    Save the trained model.

    Parameters:
    model: Trained model.
    model_filepath (str): Filepath to save the model.
    """
    joblib.dump(model, model_filepath)

def main():
    """
    Main function to execute the training script.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...')
        X, Y, category_names = load_data(database_filepath)

        print('Splitting data...')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'and the filepath to save the model as the second argument. '\
              '\n\nExample: python train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
