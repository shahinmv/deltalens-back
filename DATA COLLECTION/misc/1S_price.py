import pandas as pd

# Load raw trade data
# Replace with your actual file if needed
df = pd.read_csv('data/Kraken_Trading_History/XBTUSD.csv', header=None, names=['timestamp', 'price', 'volume'])

# Convert UNIX timestamp to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# Set datetime as index
df.set_index('datetime', inplace=True)

# Resample to 1-second timeframe and calculate OHLCV
second_df = df['price'].resample('1s').ohlc()
second_df['volume'] = df['volume'].resample('1s').sum()

second_df.to_csv('data/seconds/btc_second_ohlcv.csv')

# Optional: Plot closing price
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 6))
plt.plot(second_df.index, second_df['close'], label='1-Second Close Price')
plt.title('BTC 1-Second Close Price (Aggregated from Tick Data)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
