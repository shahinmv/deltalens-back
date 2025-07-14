import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def fetch_agg_trades(symbol, start_ts, end_ts):
    url = "https://api.binance.com/api/v3/aggTrades"
    params = {
        "symbol": symbol,
        "startTime": start_ts,
        "endTime": end_ts,
        "limit": 1000
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code} — {response.text}")
            return []
        return response.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def to_dataframe(trades):
    df = pd.DataFrame(trades)
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['T'], unit='ms')
    df = df.rename(columns={'p': 'price', 'q': 'volume'})
    return df[['timestamp', 'price', 'volume']]

def process_time_window(args):
    current, next_window, symbol = args
    start_ts = int(current.timestamp() * 1000)
    end_ts = int(next_window.timestamp() * 1000)
    
    trades = fetch_agg_trades(symbol, start_ts, end_ts)
    if trades:
        return to_dataframe(trades)
    return pd.DataFrame()

def download_all_trades(symbol='BTCUSDT', start_date='2017-08-17', output_file='binance_btcusdt_trades.csv', max_workers=10):
    start_time = datetime.strptime(start_date, "%Y-%m-%d")
    end_time = datetime.utcnow()
    delta = timedelta(hours=1)  # Increased to 1-hour windows
    
    print(f"🔄 Fetching trades for {symbol} from {start_time.date()} to {end_time.date()}...")
    print(f"Time window size: {delta}")
    print(f"Using {max_workers} concurrent workers")

    # Generate all time windows
    time_windows = []
    current = start_time
    while current < end_time:
        next_window = min(current + delta, end_time)
        time_windows.append((current, next_window, symbol))
        current = next_window

    all_dfs = []
    
    # Process time windows concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_time_window, window) for window in time_windows]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading data"):
            df = future.result()
            if not df.empty:
                all_dfs.append(df)

    if all_dfs:
        final_df = pd.concat(all_dfs)
        final_df = final_df.sort_values('timestamp')  # Ensure data is sorted by timestamp
        final_df.to_csv(output_file, index=False)
        print(f"✅ Done! Saved to {output_file}")
        print(f"Total records collected: {len(final_df)}")
    else:
        print("⚠️ No data collected.")

if __name__ == "__main__":
    download_all_trades()
