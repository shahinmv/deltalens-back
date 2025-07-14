import requests
import datetime
import pandas as pd
import sqlite3
from tqdm import tqdm
import concurrent.futures
from typing import List, Dict
import time
from datetime import datetime as dt
import logging
from requests.exceptions import RequestException
import random
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)

class RateLimiter:
    def __init__(self, max_requests_per_minute=1200):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                # Wait until the oldest request is 1 minute old
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.requests.append(now)

rate_limiter = RateLimiter()

def get_btcusdt_data(datetime_str: str, max_retries: int = 3) -> Dict:
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()  # Ensure we don't exceed rate limits
            
            # Convert the datetime string to a timestamp
            timestamp = int(datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
            
            # Binance API endpoint for historical klines
            url = f'https://api.binance.com/api/v3/klines'
            
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1s',
                'startTime': timestamp,
                'endTime': timestamp + 1000
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data:
                kline = data[0]
                return {
                    'datetime': datetime.datetime.fromtimestamp(kline[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                }
            return None
            
        except RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.random()  # Exponential backoff with jitter
                logging.warning(f"Request failed for {datetime_str}, attempt {attempt + 1}/{max_retries}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to fetch data for {datetime_str} after {max_retries} attempts: {str(e)}")
                return None
        except Exception as e:
            logging.error(f"Unexpected error for {datetime_str}: {str(e)}")
            return None

def batch_append_to_db(data_list: List[Dict]):
    if not data_list:
        return
    
    try:
        conn = sqlite3.connect('../db.sqlite3')
        cursor = conn.cursor()
        
        # Create the table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS btc_second_ohlcv (
                datetime TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        ''')
        
        # Prepare the data for batch insertion
        values = [(d['datetime'], d['open'], d['high'], d['low'], d['close'], d['volume']) 
                  for d in data_list if d is not None]
        
        if values:
            cursor.executemany('''
                INSERT OR IGNORE INTO btc_second_ohlcv (datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', values)
        
        conn.commit()
        logging.info(f"Successfully inserted {len(values)} records into database")
        
    except sqlite3.Error as e:
        logging.error(f"Database error: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

def process_batch(datetime_list: List[str], batch_size: int = 100) -> List[Dict]:
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:  # Reduced from 50 to 20
        # Submit all tasks
        future_to_datetime = {executor.submit(get_btcusdt_data, dt): dt for dt in datetime_list}
        
        # Process results as they complete
        with tqdm(concurrent.futures.as_completed(future_to_datetime), 
                 total=len(datetime_list),
                 desc="Fetching data",
                 unit="records") as pbar:
            for future in pbar:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.set_postfix({
                        'success': len(results),
                        'rate': f"{len(results)/(time.time()-start_time):.1f} rec/s"
                    })
                except Exception as e:
                    logging.error(f"Error processing future: {str(e)}")
    
    return results

def main():
    # Read the missing dates from the CSV file
    missing_dates = pd.read_csv('missing_timestamps_final .csv')
    datetime_list = missing_dates['missing_timestamps'].tolist()
    
    total_records = len(datetime_list)
    logging.info(f"Starting data collection for {total_records} records")
    logging.info(f"Start time: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Process in smaller batches to avoid overwhelming the API
    batch_size = 500  # Reduced from 1000 to 500
    total_batches = (total_records + batch_size - 1) // batch_size
    
    start_time = time.time()
    total_processed = 0
    total_successful = 0
    
    with tqdm(total=total_records, desc="Overall Progress", unit="records") as pbar:
        for i in range(0, len(datetime_list), batch_size):
            batch = datetime_list[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            
            logging.info(f"\nProcessing batch {current_batch}/{total_batches}")
            
            # Fetch data in parallel
            results = process_batch(batch)
            total_successful += len(results)
            
            # Batch insert into database
            if results:
                logging.info(f"Inserting {len(results)} records into database...")
                batch_append_to_db(results)
            
            # Update progress
            total_processed += len(batch)
            pbar.update(len(batch))
            pbar.set_postfix({
                'success': total_successful,
                'rate': f"{total_processed/(time.time()-start_time):.1f} rec/s"
            })
            
            # Add a small delay between batches
            time.sleep(2)  # Increased from 1 to 2 seconds
    
    end_time = time.time()
    duration = end_time - start_time
    
    logging.info(f"\nProcessing completed!")
    logging.info(f"End time: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total duration: {duration:.2f} seconds")
    logging.info(f"Total records processed: {total_processed}")
    logging.info(f"Successfully inserted: {total_successful}")
    logging.info(f"Average processing rate: {total_processed/duration:.2f} records/second")

if __name__ == "__main__":
    main()
