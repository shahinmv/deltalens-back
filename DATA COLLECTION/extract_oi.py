import os
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import random

# ─── Configuration "─────────────────────────────────────────────────────────────
API_KEY    = "bf6f393d7bd74fd98c4ba52037b4a966c9e348d972bdf7f605872d5d95604047"
if not API_KEY:
    raise ValueError("Set CCDATA_API_KEY in your environment")

BASE_URL   = "https://data-api.coindesk.com/futures/v1/historical/open-interest/minutes"
SYMBOL     = "BTC-USDT-VANILLA-PERPETUAL"
START      = int(pd.Timestamp("2025-06-10 07:10:00", tz="UTC").timestamp())
# END        = int(time.time())  # now
END        = int(pd.Timestamp("2025-06-15 20:30:00", tz="UTC").timestamp())
OUTPUT_CSV = "oi_history_dataapi_3.csv"
BATCH_SIZE = 400  # Increased batch size
MAX_WORKERS = 5    # Number of parallel workers
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 2  # seconds between requests

# ─── Fetch one batch with retry logic ──────────────────────────────────────────
def fetch_batch_with_retry(to_ts: int, retry_count=0) -> pd.DataFrame:
    if retry_count >= MAX_RETRIES:
        print(f"Max retries reached for timestamp {to_ts}")
        return pd.DataFrame()
    
    try:
        params = {
            "market": "binance",
            "instrument": SYMBOL,
            "limit": BATCH_SIZE,
            "to_ts": to_ts,
            "aggregate": 5,
        }
        headers = {"authorization": f"Apikey {API_KEY}"}
        r = requests.get(BASE_URL, params=params, headers=headers)
        
        if r.status_code == 429:  # Rate limit
            wait_time = RATE_LIMIT_DELAY * (2 ** retry_count)  # Exponential backoff
            print(f"Rate limited. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return fetch_batch_with_retry(to_ts, retry_count + 1)
            
        r.raise_for_status()
        data = r.json()["Data"]
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], unit="s", utc=True)
        return df[["timestamp", "CLOSE_SETTLEMENT", "CLOSE_QUOTE"]].rename(columns={
            "CLOSE_SETTLEMENT": "close_settlement",
            "CLOSE_QUOTE": "close_quote"
        })
    except requests.exceptions.RequestException as e:
        print(f"Request error for timestamp {to_ts}: {e}")
        time.sleep(RATE_LIMIT_DELAY)
        return fetch_batch_with_retry(to_ts, retry_count + 1)

# ─── Parallel backfill ─────────────────────────────────────────────────────────
def process_time_range(start_ts: int, end_ts: int) -> pd.DataFrame:
    frames = []
    cur = end_ts
    while cur > start_ts:
        print(f"Fetching up to {pd.to_datetime(cur, unit='s', utc=True)} …")
        try:
            batch = fetch_batch_with_retry(cur)
            if not batch.empty:
                frames.append(batch)
                min_timestamp = batch["timestamp"].min()
                cur = int(min_timestamp.timestamp()) - 1
            else:
                # If no data, create a null entry and move back by 1 hour
                null_entry = pd.DataFrame({
                    'timestamp': [pd.to_datetime(cur, unit='s', utc=True)],
                    'close_settlement': [None],
                    'close_quote': [None]
                })
                frames.append(null_entry)
                cur -= 3600  # Move back 1 hour
            
            # Add a small random delay between requests
            time.sleep(RATE_LIMIT_DELAY + random.uniform(0, 1))
            
        except Exception as e:
            print(f"Error fetching data for timestamp {cur}: {e}")
            null_entry = pd.DataFrame({
                'timestamp': [pd.to_datetime(cur, unit='s', utc=True)],
                'close_settlement': [None],
                'close_quote': [None]
            })
            frames.append(null_entry)
            cur -= 3600
            time.sleep(RATE_LIMIT_DELAY)
            
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def main():
    # Split the time range into chunks for parallel processing
    time_ranges = []
    chunk_size = (END - START) // MAX_WORKERS
    for i in range(MAX_WORKERS):
        start = START + (i * chunk_size)
        end = min(START + ((i + 1) * chunk_size), END)
        time_ranges.append((start, end))

    # Process chunks in parallel
    frames = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_range = {executor.submit(process_time_range, start, end): (start, end) 
                          for start, end in time_ranges}
        
        for future in as_completed(future_to_range):
            start, end = future_to_range[future]
            try:
                df = future.result()
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                print(f"Error processing range {start}-{end}: {e}")

    # Combine and save results
    if frames:
        oi_hist = (pd.concat(frames, ignore_index=True)
                   .drop_duplicates("timestamp")
                   .sort_values("timestamp"))
        oi_hist.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved data from {oi_hist.timestamp.min()} to {oi_hist.timestamp.max()} → {OUTPUT_CSV}")
    else:
        print("No data was retrieved for the specified time range.")

if __name__ == "__main__":
    main()
