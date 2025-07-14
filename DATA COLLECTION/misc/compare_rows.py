import sqlite3
import pandas as pd
import os

# Define paths
db_path = '../db.sqlite3'
csv_path = 'data/Kraken_Trading_History/XBTUSD.csv'

# Count rows in CSV file
print("Counting rows in CSV file...")
csv_rows = 0
for chunk in pd.read_csv(csv_path, chunksize=100000, header=None):
    csv_rows += len(chunk)
print(f"CSV file has {csv_rows:,} rows")

# Count rows in SQLite table
print("\nCounting rows in SQLite table...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM XBTUSD")
table_rows = cursor.fetchone()[0]
print(f"SQLite table has {table_rows:,} rows")

# Compare the counts
if csv_rows == table_rows:
    print("\n✅ The number of rows match!")
else:
    print(f"\n❌ The number of rows do not match!")
    print(f"Difference: {abs(csv_rows - table_rows):,} rows")

conn.close() 