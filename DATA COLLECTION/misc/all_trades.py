import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load your CSV file
# Replace 'kraken_btc_data.csv' with the path to your file
df = pd.read_csv('data/Kraken_Trading_History/XBTUSD.csv', header=None, names=['timestamp', 'price', 'volume'])

# Convert UNIX timestamp to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# Sort by datetime (just in case)
df = df.sort_values('datetime')

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(df['datetime'], df['price'], label='BTC Price', color='blue', linewidth=0.6)
plt.title('BTC Trade Price Over Time (Kraken Historical Data)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
