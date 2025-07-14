import sqlite3
from datetime import datetime

# Connect to the SQLite database
conn = sqlite3.connect('../db.sqlite3')
cursor = conn.cursor()

# Query to get the latest date from XBTUSD table
cursor.execute("SELECT MAX(timestamp) FROM XBTUSD")
latest_timestamp = cursor.fetchone()[0]

# Convert string timestamp to float and then to datetime
latest_date = datetime.fromtimestamp(float(latest_timestamp))

print(f"Latest timestamp in XBTUSD table: {latest_timestamp}")
print(f"Latest date in XBTUSD table: {latest_date}")

# Close the connection
conn.close() 