import pandas as pd

# Load raw trade data
# Replace with your actual file if needed
df = pd.read_csv('data/Kraken_Trading_History/XBTUSD.csv', header=None, names=['timestamp', 'price', 'volume'])

# Convert UNIX timestamp to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# Set datetime as index
df.set_index('datetime', inplace=True)

# Resample to daily timeframe and calculate OHLCV
daily_df = df['price'].resample('1D').ohlc()
daily_df['volume'] = df['volume'].resample('1D').sum()

daily_df.to_csv('data/daily/btc_daily_ohlcv.csv')

# Optional: Plot closing price
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 6))
plt.plot(daily_df.index, daily_df['close'], label='Daily Close Price')
plt.title('BTC Daily Close Price (Aggregated from Tick Data)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
