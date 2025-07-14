import pandas as pd

# Load raw trade data
# Replace with your actual file if needed
df = pd.read_csv('data/Kraken_Trading_History/XBTUSD.csv', header=None, names=['timestamp', 'price', 'volume'])

# Convert UNIX timestamp to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# Set datetime as index
df.set_index('datetime', inplace=True)

# Resample to weekly timeframe and calculate OHLCV
weekly_df = df['price'].resample('1W').ohlc()
weekly_df['volume'] = df['volume'].resample('1W').sum()

weekly_df.to_csv('data/weekly/btc_weekly_ohlcv.csv')

# Optional: Plot closing price
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 6))
plt.plot(weekly_df.index, weekly_df['close'], label='Weekly Close Price')
plt.title('BTC Weekly Close Price (Aggregated from Tick Data)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
