import requests
import pandas as pd
from datetime import datetime, timedelta

# Load your existing CSV
existing_df = pd.read_csv('data/hourly_history/btc_hourly_ohlcv.csv', parse_dates=True, index_col='datetime')

# Get the last date in your CSV
last_date = existing_df.index[-1]
# Add 1 hour instead of 1 day to ensure continuous data
start_date = last_date + pd.Timedelta(hours=1)
since_timestamp = int(start_date.timestamp())

# Fetch new data from Kraken API
def fetch_new_ohlc(pair='XBTUSD', interval=60, since=since_timestamp):
    url = 'https://api.kraken.com/0/public/OHLC'
    params = {
        'pair': pair,
        'interval': interval,
        'since': since
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if response.status_code != 200 or 'result' not in data:
        raise Exception("Error fetching OHLC data from Kraken")

    key = list(data['result'].keys())[0]
    ohlc = data['result'][key]

    new_df = pd.DataFrame(ohlc, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
    ])
    new_df['datetime'] = pd.to_datetime(new_df['timestamp'], unit='s')
    new_df.set_index('datetime', inplace=True)
    new_df = new_df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
    
    return new_df[['open', 'high', 'low', 'close', 'volume']]

# Fetch and combine
new_data_df = fetch_new_ohlc()
full_df = pd.concat([existing_df, new_data_df])
full_df = full_df[~full_df.index.duplicated(keep='first')]  # Remove any accidental duplicates

# today = pd.Timestamp.now().normalize()
# full_df = full_df[full_df.index < today]

# Save updated data
full_df.to_csv('data/hourly_history/btc_hourly_ohlcv.csv')

print("✅ Data updated and saved to 'btc_hourly_ohlcv.csv'")
