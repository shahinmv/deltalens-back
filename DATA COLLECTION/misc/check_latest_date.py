import pandas as pd
import os
from datetime import datetime

# Path to the XBTUSD.csv file
file_path = os.path.join('data', 'Kraken_Trading_History', 'XBTUSD.csv')

# Read the CSV file without header, specifying the first column as timestamp
df = pd.read_csv(file_path, header=None, usecols=[0], names=['timestamp'])
latest_timestamp = df['timestamp'].max()

# Convert timestamp to datetime
latest_date = datetime.fromtimestamp(latest_timestamp)

print(f"Latest timestamp in XBTUSD.csv: {latest_timestamp}")
print(f"Latest date in XBTUSD.csv: {latest_date}")
