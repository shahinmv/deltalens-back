import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Connect to SQLite database
conn = sqlite3.connect('../db.sqlite3')

# Query to get trades for March and April 2025
query = """
SELECT 
    strftime('%Y-%m-%d', timestamp, 'unixepoch') as date,
    COUNT(*) as trade_count
FROM XBTUSD
WHERE 
    strftime('%Y', timestamp, 'unixepoch') = '2025' AND
    strftime('%m', timestamp, 'unixepoch') IN ('03', '04')
GROUP BY date
ORDER BY date
"""

# Execute query and load into DataFrame
df = pd.read_sql_query(query, conn)

# Convert date string to datetime
df['date'] = pd.to_datetime(df['date'])

# Create the plot
plt.figure(figsize=(15, 7))
bars = plt.bar(df['date'], df['trade_count'], width=0.8)

# Customize the plot
plt.title('Number of Trades per Day - March and April 2025', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Trades', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Close database connection
conn.close() 