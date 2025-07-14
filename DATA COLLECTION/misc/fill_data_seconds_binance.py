import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import concurrent.futures
from typing import List, Dict, Optional
import numpy as np
import sqlite3
import os
import random
from requests.exceptions import RequestException
import websocket
import json
import threading
from queue import Queue

def get_earliest_available_date(symbol='BTCUSDT', interval='1s'):
    url = 'https://api.binance.com/api/v3/klines'
    early_timestamp = int(datetime(2010, 1, 1).timestamp() * 1000)
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': early_timestamp,
        'limit': 1
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if response.status_code != 200:
        raise Exception(f"Error fetching earliest date from Binance: {data}")
    
    if not data:
        raise Exception("No data available for the specified parameters")
    
    earliest_timestamp = int(data[0][0])
    earliest_date = datetime.fromtimestamp(earliest_timestamp / 1000)
    
    return earliest_date

def get_total_records(symbol='BTCUSDT', interval='1s', start_time=None):
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': int(datetime.now().timestamp() * 1000),
        'limit': 1
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if response.status_code != 200 or not data:
        return 0
    
    last_timestamp = int(data[0][0])
    total_seconds = (datetime.now().timestamp() - (last_timestamp / 1000))
    return int(total_seconds)

def fetch_chunk(args: Dict) -> List:
    """Fetch a chunk of data with retry logic"""
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': args['symbol'],
        'interval': args['interval'],
        'startTime': args['start_time'],
        'endTime': args['end_time'],
        'limit': 1000
    }
    
    max_retries = 3
    retry_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            # Add random jitter to avoid synchronized retries
            time.sleep(retry_delay + random.uniform(0, 1))
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 429:  # Rate limit exceeded
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Rate limit hit. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
                
            if response.status_code == 200:
                return response.json()
            
            print(f"Request failed with status {response.status_code}. Attempt {attempt + 1}/{max_retries}")
            retry_delay *= 2  # Exponential backoff
            
        except RequestException as e:
            print(f"Request error: {str(e)}. Attempt {attempt + 1}/{max_retries}")
            retry_delay *= 2
            continue
            
    print(f"Failed to fetch chunk after {max_retries} attempts")
    return []

def write_chunk_to_database(chunk_data: List, conn: sqlite3.Connection):
    """Write a chunk of data to the database with error handling"""
    if not chunk_data:
        return
    
    try:
        # Convert chunk to DataFrame
        df = pd.DataFrame(chunk_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Convert string values to float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Write to database
        df[numeric_columns].to_sql('btc_second_ohlcv', conn, if_exists='append', index=True, index_label='datetime')
        conn.commit()
        
    except Exception as e:
        print(f"Error writing chunk to database: {str(e)}")
        conn.rollback()
        raise

def write_websocket_data_to_db(data: List, conn: sqlite3.Connection):
    """Write WebSocket data to database"""
    if not data:
        return
    
    try:
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Write to database
        df[['open', 'high', 'low', 'close', 'volume']].to_sql(
            'btc_second_ohlcv', 
            conn, 
            if_exists='append', 
            index=True, 
            index_label='datetime'
        )
        conn.commit()
    except Exception as e:
        print(f"Error writing WebSocket data to database: {str(e)}")
        conn.rollback()

def fetch_ohlc_data(symbol='BTCUSDT', interval='1s', start_time=None, num_workers=20):
    """Fetch OHLC data in parallel using multiple workers and write to database in chunks"""
    end_time = int(datetime.now().timestamp() * 1000)
    total_records = get_total_records(symbol, interval, start_time)
    print(f"Estimated total records to fetch: {total_records:,}")
    
    # Calculate chunk sizes and create tasks
    chunk_size = 1000 * 1000  # 1000 records per request * 1000ms
    tasks = []
    current_time = start_time
    
    while current_time < end_time:
        chunk_end = min(current_time + chunk_size, end_time)
        tasks.append({
            'symbol': symbol,
            'interval': interval,
            'start_time': current_time,
            'end_time': chunk_end
        })
        current_time = chunk_end
    
    print(f"Total chunks to process: {len(tasks)}")
    
    # Create database connection that will be shared across threads
    db_path = '../db.sqlite3'
    conn = sqlite3.connect(db_path)
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(fetch_chunk, task): task for task in tasks}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc='Fetching and writing data'):
                task = futures[future]
                try:
                    chunk_data = future.result()
                    if chunk_data:
                        write_chunk_to_database(chunk_data, conn)
                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")
                    continue
                    
    finally:
        conn.close()

def get_latest_timestamp_from_db() -> Optional[int]:
    """Get the latest timestamp from the database"""
    db_path = '../db.sqlite3'
    if not os.path.exists(db_path):
        return None
        
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(datetime) FROM btc_second_ohlcv")
        result = cursor.fetchone()
        if result and result[0]:
            # Convert datetime string to timestamp
            latest_dt = datetime.fromisoformat(result[0])
            return int(latest_dt.timestamp() * 1000)
        return None
    finally:
        conn.close()

def create_database():
    """Create SQLite database and table if they don't exist"""
    db_path = '../db.sqlite3'
    
    # Create database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS btc_second_ohlcv (
        datetime TIMESTAMP PRIMARY KEY,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL
    )
    ''')
    
    conn.commit()
    conn.close()

class BinanceWebSocket:
    def __init__(self, symbol='btcusdt', interval='1s', start_from_timestamp=None):
        self.symbol = symbol.lower()
        self.interval = interval
        self.start_from_timestamp = start_from_timestamp
        self.ws = None
        self.data_queue = Queue()
        self.running = False
        self.thread = None
        self.last_processed_timestamp = start_from_timestamp

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if 'k' in data:  # kline data
                kline = data['k']
                if kline['x']:  # if kline is closed
                    timestamp = kline['t']
                    
                    # Only process data after our start timestamp
                    if self.start_from_timestamp and timestamp <= self.start_from_timestamp:
                        return
                        
                    open_price = float(kline['o'])
                    high_price = float(kline['h'])
                    low_price = float(kline['l'])
                    close_price = float(kline['c'])
                    volume = float(kline['v'])
                    
                    self.data_queue.put({
                        'timestamp': timestamp,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume
                    })
                    self.last_processed_timestamp = timestamp
        except Exception as e:
            print(f"WebSocket message error: {str(e)}")

    def on_error(self, ws, error):
        print(f"WebSocket error: {str(error)}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed")
        self.running = False

    def on_open(self, ws):
        print("WebSocket connection established")
        self.running = True

    def start(self):
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_{self.interval}",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        if self.ws:
            self.ws.close()
        self.running = False
        if self.thread:
            self.thread.join()

    def get_latest_data(self):
        data = []
        while not self.data_queue.empty():
            data.append(self.data_queue.get())
        return data

try:
    print("Setting up database...")
    create_database()
    
    # Check for existing data
    latest_timestamp = get_latest_timestamp_from_db()
    if latest_timestamp:
        print(f"Found existing data in database. Latest timestamp: {datetime.fromtimestamp(latest_timestamp/1000)}")
        start_timestamp = latest_timestamp + 1000  # Add 1 second to avoid duplicates
    else:
        print("No existing data found. Starting from earliest available date...")
        print("Finding earliest available date...")
        earliest_date = get_earliest_available_date()
        print(f"Earliest available date: {earliest_date}")
        start_timestamp = int(earliest_date.timestamp() * 1000)
    
    print("Fetching historical data from Binance API...")
    fetch_ohlc_data(start_time=start_timestamp, num_workers=20)
    
    # # Get the latest timestamp after API data collection
    # api_end_timestamp = get_latest_timestamp_from_db()
    # if api_end_timestamp:
    #     print(f"API data collection completed. Latest timestamp: {datetime.fromtimestamp(api_end_timestamp/1000)}")
        
    #     print("Starting WebSocket connection for real-time data...")
    #     # Start WebSocket from the last API timestamp
    #     ws = BinanceWebSocket(start_from_timestamp=api_end_timestamp)
    #     ws.start()
        
    #     # Create a new database connection for WebSocket data
    #     db_path = '../db.sqlite3'
    #     conn = sqlite3.connect(db_path)
        
    #     try:
    #         print("Collecting real-time data...")
    #         while True:  # Run indefinitely
    #             time.sleep(0.2)
    #             data = ws.get_latest_data()
    #             if data:
    #                 write_websocket_data_to_db(data, conn)
    #                 latest_timestamp = data[-1]['timestamp']
    #                 print(f"Added {len(data)} new records from WebSocket (up to {datetime.fromtimestamp(latest_timestamp/1000)})")
    #     except KeyboardInterrupt:
    #         print("\nStopping data collection...")
    #     finally:
    #         ws.stop()
    #         conn.close()
    # else:
    #     print("Error: Could not determine end timestamp of API data")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    if 'ws' in locals():
        ws.stop() 