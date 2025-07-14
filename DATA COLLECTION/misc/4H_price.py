import pandas as pd

# Load raw trade data
# Replace with your actual file if needed
df = pd.read_csv('data/Kraken_Trading_History/XBTUSD.csv', header=None, names=['timestamp', 'price', 'volume'])

# Convert UNIX timestamp to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# Set datetime as index
df.set_index('datetime', inplace=True)

# Resample to 4-hour timeframe and calculate OHLCV
four_hour_df = df['price'].resample('4h').ohlc()
four_hour_df['volume'] = df['volume'].resample('4h').sum()

four_hour_df.to_csv('data/hourly/btc_4h_ohlcv.csv')

# Optional: Plot closing price
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 6))
plt.plot(four_hour_df.index, four_hour_df['close'], label='4-Hour Close Price')
plt.title('BTC 4-Hour Close Price (Aggregated from Tick Data)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
