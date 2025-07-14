import sqlite3
import pandas as pd
import os

# Define paths
db_path = '../db.sqlite3'
csv_path = 'data/Kraken_Trading_History/XBTUSD.csv'

# Create database directory if it doesn't exist
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Connect to SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Drop existing table if it exists
cursor.execute("DROP TABLE IF EXISTS XBTUSD")

# Define column names for the CSV data
column_names = [
    'timestamp',
    'price',
    'volume'
]

# Create table with defined column names
create_table_sql = f"""
CREATE TABLE XBTUSD (
    {', '.join([f'"{col}" TEXT' for col in column_names])}
)
"""
cursor.execute(create_table_sql)

# Read CSV file in chunks without headers
chunk_size = 100000  # Process 100,000 rows at a time
for chunk in pd.read_csv(csv_path, chunksize=chunk_size, header=None, names=column_names):
    # Append data to the table
    chunk.to_sql('XBTUSD', conn, if_exists='append', index=False)
    print(f"Processed {len(chunk)} rows")

# Commit changes and close connection
conn.commit()
conn.close()

print("Data import completed successfully!") 