import pandas as pd
import sqlite3
from datetime import datetime, timezone
import os
from binance.client import Client
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlite3 import Connection
import contextlib
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import backoff
from ratelimit import limits, sleep_and_retry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate limiting constants
CALLS = 1200  # Binance allows 1200 requests per minute
RATE_LIMIT_PERIOD = 60

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT_PERIOD)
def rate_limited_api_call(func, *args, **kwargs):
    return func(*args, **kwargs)

def format_timestamp(dt: datetime) -> str:
    """Format datetime to string without timezone info"""
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

# Connection pool for database operations
class ConnectionPool:
    def __init__(self, max_connections=10):  # Increased pool size
        self.max_connections = max_connections
        self.pool = []
        self._create_pool()

    def _create_pool(self):
        for _ in range(self.max_connections):
            conn = sqlite3.connect('../db.sqlite3', check_same_thread=False)
            conn.execute('PRAGMA journal_mode = WAL')  # Enable WAL mode for better concurrency
            conn.execute('PRAGMA synchronous = NORMAL')  # Reduce synchronous mode for better performance
            self.pool.append(conn)

    def get_connection(self) -> Connection:
        if not self.pool:
            self._create_pool()
        return self.pool.pop()

    def return_connection(self, conn: Connection):
        self.pool.append(conn)

    def close_all(self):
        for conn in self.pool:
            conn.close()
        self.pool.clear()

# Initialize connection pool
connection_pool = ConnectionPool()

@contextlib.contextmanager
def get_db_connection():
    conn = connection_pool.get_connection()
    try:
        yield conn
    finally:
        connection_pool.return_connection(conn)

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def get_binance_data_batch(timestamps: List[datetime]) -> Dict[datetime, Dict[str, float]]:
    """Fetch 1s data from Binance for a batch of timestamps"""
    try:
        client = Client()
        results = {}
        
        logger.info(f"Processing batch of {len(timestamps)} timestamps")
        
        # Process timestamps in larger batches
        batch_size = 200  # Increased batch size
        total_sub_batches = (len(timestamps) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(timestamps), batch_size), 
                     desc="Processing sub-batches",
                     total=total_sub_batches,
                     unit="sub-batch"):
            batch = timestamps[i:i+batch_size]
            
            for timestamp in tqdm(batch, 
                                desc=f"Sub-batch {i//batch_size + 1}/{total_sub_batches}",
                                leave=False,
                                unit="timestamp"):
                try:
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    
                    start_time = int(timestamp.timestamp() * 1000)
                    end_time = start_time + 1000
                    
                    klines = rate_limited_api_call(
                        client.get_klines,
                        symbol='BTCUSDT',
                        interval=Client.KLINE_INTERVAL_1SECOND,
                        startTime=start_time,
                        endTime=end_time
                    )
                    
                    if klines and len(klines) > 0:
                        kline = klines[0]
                        timestamp_str = format_timestamp(timestamp)
                        results[timestamp_str] = {
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5])
                        }
                except Exception as e:
                    logger.error(f"Error processing timestamp {timestamp}: {e}")
                    continue
        
        logger.info(f"Successfully processed {len(results)} timestamps in this batch")
        return results
    except Exception as e:
        logger.error(f"Error fetching Binance data batch: {e}")
        return {}

def insert_batch_data(data_batch: Dict[str, Dict[str, float]]):
    """Insert a batch of data into the database"""
    if not data_batch:
        logger.warning("No data to insert")
        return
    
    with get_db_connection() as conn:
        try:
            cur = conn.cursor()
            query = """
            INSERT OR IGNORE INTO btc_second_ohlcv (datetime, open, high, low, close, volume, imputed)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """
            
            # Prepare batch data
            batch_data = [
                (timestamp, data['open'], data['high'], data['low'], data['close'], data['volume'])
                for timestamp, data in data_batch.items()
            ]
            
            # Use executemany for better performance
            cur.executemany(query, batch_data)
            conn.commit()
            logger.info(f"Successfully inserted {len(batch_data)} records")
            
        except Exception as e:
            logger.error(f"Error inserting batch data: {e}")

def process_timestamps(timestamps: List[datetime]):
    """Process a batch of timestamps"""
    try:
        data = get_binance_data_batch(timestamps)
        if data:
            insert_batch_data(data)
        return len(data)
    except Exception as e:
        logger.error(f"Error in process_timestamps: {e}")
        return 0

def main():
    try:
        # Read the CSV file
        logger.info("Reading CSV file...")
        df = pd.read_csv('missing_timestamps_final.csv')
        timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in df['missing_timestamps'].tolist()]
        
        logger.info(f"Found {len(timestamps)} timestamps to process")
        
        # Process timestamps in parallel batches
        batch_size = 1000  # Increased batch size
        total_processed = 0
        total_batches = (len(timestamps) + batch_size - 1) // batch_size
        
        logger.info(f"Starting processing with {total_batches} batches")
        
        with ThreadPoolExecutor(max_workers=8) as executor:  # Increased thread pool size
            futures = []
            for i in range(0, len(timestamps), batch_size):
                batch = timestamps[i:i+batch_size]
                batch_num = i//batch_size + 1
                logger.info(f"Submitting batch {batch_num}/{total_batches}")
                futures.append(executor.submit(process_timestamps, batch))
            
            for i, future in enumerate(tqdm(as_completed(futures), 
                                          total=total_batches,
                                          desc="Overall Progress",
                                          unit="batch"), 1):
                try:
                    processed = future.result()
                    total_processed += processed
                    logger.info(f"Completed batch {i}/{total_batches}, processed {processed} timestamps")
                except Exception as e:
                    logger.error(f"Error in batch {i}: {e}")
        
        logger.info(f"Successfully processed {total_processed} timestamps in total")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
    finally:
        connection_pool.close_all()

if __name__ == "__main__":
    main()
