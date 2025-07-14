import pandas as pd
import requests
import time
import sqlite3
from datetime import datetime

# Connect to SQLite database
conn = sqlite3.connect('../db.sqlite3')
cursor = conn.cursor()

def get_last_timestamp():
    cursor.execute('SELECT MAX(timestamp) FROM XBTUSD')
    result = cursor.fetchone()
    return result[0] if result[0] is not None else 0

def fetch_trades(pair='XBTUSD', since=None):
    if since is None:
        since = get_last_timestamp()
    
    all_trades = []
    current_since = since
    batch_count = 0

    print("Fetching new trades from Kraken...")

    while True:
        try:
            url = f'https://api.kraken.com/0/public/Trades'
            params = {'pair': pair, 'since': current_since}
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
                
            data = response.json()
            
            if 'error' in data and data['error']:
                raise Exception(f"Kraken API returned errors: {data['error']}")
                
            if 'result' not in data:
                raise Exception("Invalid response format: 'result' field missing")

            pair_key = list(data['result'].keys())[0]
            trades = data['result'][pair_key]

            if not trades:
                print("No new trades found")
                break

            # Parse trades
            batch_trades = []
            for t in trades:
                timestamp = int(float(t[2]))
                price = float(t[0])
                volume = float(t[1])
                batch_trades.append((timestamp, price, volume))

            # Add batch trades to main list
            all_trades.extend(batch_trades)
            
            # Print progress
            print(f"Batch {batch_count + 1}: Processed {len(batch_trades)} trades from {datetime.fromtimestamp(batch_trades[0][0])} to {datetime.fromtimestamp(batch_trades[-1][0])}")

            # Prepare for next batch
            current_since = int(data['result']['last'])
            batch_count += 1
            
            # Save progress after each batch
            if all_trades:
                # Insert new trades into database
                cursor.executemany('''
                INSERT OR IGNORE INTO XBTUSD (timestamp, price, volume)
                VALUES (?, ?, ?)
                ''', all_trades)
                conn.commit()
                print(f"✅ Saved {len(all_trades)} new trades to database")

            time.sleep(0.5)  # Respect Kraken rate limit

        except Exception as e:
            print("API failed, retrying...")
            continue

    return pd.DataFrame(all_trades, columns=['timestamp', 'price', 'volume'])

# Fetch new trades
df_new = fetch_trades()

# Close database connection
conn.close()

print("Update complete!")
