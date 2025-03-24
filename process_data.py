import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge the messages and categories datasets.

    Args:
    - messages_filepath (str): Path to the messages CSV file.
    - categories_filepath (str): Path to the categories CSV file.

    Returns:
    - df (DataFrame): Merged DataFrame containing messages and categories.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)

    # Load categories dataset
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = pd.merge(messages, categories, on='id')

    return df

def clean_data(df):
    """
    Clean the merged DataFrame.

    Args:
    - df (DataFrame): Merged DataFrame containing messages and categories.

    Returns:
    - df (DataFrame): Cleaned DataFrame.
    """
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)

    # Use the first row to create column names for the categories data
    category_colnames = categories.iloc[0].apply(lambda x: x.split('-')[0])

    # Rename columns of categories with new column names
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories.columns:
        # Handle the 'related' column separately to ensure it contains only 0s and 1s
        if column == 'related':
            # Convert all values other than 1 to 0
            categories[column] = categories[column].apply(lambda x: 1 if int(x.split('-')[1]) == 1 else 0)
        else:
            # Convert values to strings, split, and then keep the last character as before
            categories[column] = categories[column].astype(str).str.split('-').str[-1].astype(int)

    print("Unique values in the 'related' column after conversion:")
    print(categories['related'].unique())

    # Drop the original 'categories' column from the df dataframe
    df = df.drop('categories', axis=1)

    # Concatenate df and categories dataframes
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to an SQLite database.

    Args:
    - df (DataFrame): Cleaned DataFrame.
    - database_filename (str): Path to the SQLite database file.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_data', engine, index=False, if_exists='replace')

    # Connect to the SQLite database
    conn = sqlite3.connect(database_filename)

    # Create a cursor object
    cur = conn.cursor()

    # Query to get the list of tables in the database
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = cur.execute(tables_query).fetchall()

    # Print the list of tables
    print("Tables in the database:")
    for table in tables:
        print(table[0])

    # Assuming your table name is 'disaster_data', replace it with the actual table name
    table_name = 'disaster_data'

    # Execute a query to fetch data from the specified table
    query = f"SELECT * FROM {table_name};"
    df_query = pd.read_sql_query(query, conn)

    # Close the cursor and connection
    cur.close()
    conn.close()

    # Display the content of the dataframe
    print(f"\nContent of {table_name} table:")
    print(df_query.head())

def main():
    """
    Main ETL pipeline script.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...')
        save_data(df, database_filepath)

        print(f'Data saved to {database_filepath}')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as well as the filepath of the database to save the '\
              'cleaned data to. \n\nExample: python etl_pipeline.py '\
              'messages.csv categories.csv DisasterResponse.db')

if __name__ == '__main__':
    main()
