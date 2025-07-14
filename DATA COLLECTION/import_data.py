import sqlite3
import pandas as pd
from datetime import datetime

# Read the CSV file
df = pd.read_csv('data/derivs_hist/open-interest.csv')

# Convert timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create SQLite connection
conn = sqlite3.connect('../db.sqlite3')
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS open_interest (
    timestamp TIMESTAMP PRIMARY KEY,
    close_settlement REAL,
    close_quote REAL)
''')

# Import data
df.to_sql('open_interest', conn, if_exists='replace', index=False)

# Commit changes and close connection
conn.commit()
conn.close()

print("Data imported successfully!") 