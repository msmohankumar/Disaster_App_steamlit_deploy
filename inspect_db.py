import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_FILEPATH = './DisasterResponse.db'

engine = create_engine(f'sqlite:///{DATABASE_FILEPATH}')

# Correct way to execute raw SQL queries
query = text("SELECT name FROM sqlite_master WHERE type='table';")

with engine.connect() as conn:
    result = conn.execute(query)
    print("Tables found in database:")
    for row in result:
        print(row[0])
