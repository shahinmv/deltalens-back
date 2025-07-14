import pandas as pd

# Load raw trade data
# Replace with your actual file if needed
df = pd.read_csv('data/Kraken_Trading_History/XBTUSD.csv', header=None, names=['timestamp', 'price', 'volume'])

# Convert UNIX timestamp to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# Set datetime as index
df.set_index('datetime', inplace=True)

# Resample to 15-minute timeframe and calculate OHLCV
fifteen_min_df = df['price'].resample('15min').ohlc()
fifteen_min_df['volume'] = df['volume'].resample('15min').sum()

fifteen_min_df.to_csv('data/minutes/btc_15min_ohlcv.csv')

# Optional: Plot closing price
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 6))
plt.plot(fifteen_min_df.index, fifteen_min_df['close'], label='15-Minute Close Price')
plt.title('BTC 15-Minute Close Price (Aggregated from Tick Data)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
