import pandas as pd
import sqlite3

# Load CSV files
messages = pd.read_csv('disaster_messages.csv')
categories = pd.read_csv('disaster_categories.csv')

# Merge on 'id'
df = messages.merge(categories, on='id')

# Clean 'categories' column
categories_expanded = df['categories'].str.split(';', expand=True)
row = categories_expanded.iloc[0]
category_colnames = row.apply(lambda x: x.split('-')[0])
categories_expanded.columns = category_colnames

for column in categories_expanded:
    categories_expanded[column] = categories_expanded[column].apply(lambda x: int(x.split('-')[1]))

# Drop original categories and concatenate new columns
df.drop('categories', axis=1, inplace=True)
df = pd.concat([df, categories_expanded], axis=1)

# Save to SQLite database
conn = sqlite3.connect('DisasterResponse.db')
df.to_sql('disaster_messages', conn, index=False, if_exists='replace')
conn.close()

print("✅ Database created successfully with table 'disaster_messages'")
