import pandas as pd
import requests
import time
from datetime import datetime

# Load your existing tick-level CSV
df_old = pd.read_csv('data/Kraken_Trading_History/XBTUSD.csv', header=None, names=['timestamp', 'price', 'volume'])

# Get the last timestamp in your data
last_ts = int(df_old['timestamp'].max())

def fetch_trades(pair='XBTUSD', since=last_ts, max_batches=10):
    all_trades = []
    current_since = since
    batch_count = 0

    print("Fetching new trades from Kraken...")

    while batch_count < max_batches:
        url = f'https://api.kraken.com/0/public/Trades'
        params = {'pair': pair, 'since': current_since}
        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code != 200 or 'result' not in data:
            raise Exception("API request failed")

        pair_key = list(data['result'].keys())[0]
        trades = data['result'][pair_key]

        if not trades:
            print("No new trades found in this batch")
            break

        # Parse trades
        batch_trades = []
        for t in trades:
            timestamp = int(float(t[2]))
            price = float(t[0])
            volume = float(t[1])
            batch_trades.append([timestamp, price, volume])

        # Add batch trades to main list
        all_trades.extend(batch_trades)
        
        # Print progress
        print(f"Batch {batch_count + 1}: Processed {len(batch_trades)} trades from {datetime.fromtimestamp(batch_trades[0][0])} to {datetime.fromtimestamp(batch_trades[-1][0])}")

        # Prepare for next batch
        current_since = int(data['result']['last'])
        batch_count += 1
        
        # Save progress after each batch
        if all_trades:
            df_new = pd.DataFrame(all_trades, columns=['timestamp', 'price', 'volume'])
            df_all = pd.concat([df_old, df_new])
            df_all = df_all.drop_duplicates().sort_values(by='timestamp')
            df_all.to_csv('data/Kraken_Trading_History/XBTUSD.csv', header=False, index=False)
            print(f"✅ Saved {len(all_trades)} new trades to CSV")

        time.sleep(1)  # Respect Kraken rate limit

    return pd.DataFrame(all_trades, columns=['timestamp', 'price', 'volume'])

# Fetch new trades with a limit of 10 batches
df_new = fetch_trades(max_batches=10)

print("Update complete!")
