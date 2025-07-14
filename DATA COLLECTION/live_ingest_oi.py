import sqlite3
import time
import logging
from datetime import datetime, timedelta, UTC
import pandas as pd
from binance.client import Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_latest_timestamp():
    conn = sqlite3.connect('../db.sqlite3')
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(timestamp) FROM open_interest')
    result = cursor.fetchone()
    conn.close()
    if result[0] is not None:
        # Convert UTC datetime string to Unix timestamp (milliseconds)
        dt = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    return 0

def create_table():
    conn = sqlite3.connect('../db.sqlite3')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS open_interest (
            timestamp TEXT PRIMARY KEY,
            close_settlement REAL,
            close_quote REAL
        )
    ''')
    conn.commit()
    conn.close()

def fetch_open_interest():
    # Initialize Binance client
    client = Client()
    
    try:
        # Get open interest statistics
        oi_stats = client.futures_open_interest_hist(
            symbol='BTCUSDT',
            period='5m',
            limit=1
        )
        
        if oi_stats and len(oi_stats) > 0:
            return oi_stats[0]  # Return the most recent data point
        return None
    except Exception as e:
        logging.error(f"Error fetching data from Binance: {str(e)}")
        return None

def store_data(data):
    if data:
        conn = sqlite3.connect('../db.sqlite3')
        cursor = conn.cursor()
        
        # Convert Unix timestamp to UTC datetime string using timezone-aware approach
        timestamp = int(data['timestamp'])
        dt = datetime.fromtimestamp(timestamp/1000, UTC)
        timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        
        close_settlement = float(data['sumOpenInterest'])
        close_quote = float(data['sumOpenInterestValue'])
        
        # Only store if the timestamp is newer than the latest in the database
        if timestamp > get_latest_timestamp():
            cursor.execute('''
                INSERT OR REPLACE INTO open_interest (timestamp, close_settlement, close_quote)
                VALUES (?, ?, ?)
            ''', (timestamp_str, close_settlement, close_quote))
            
            conn.commit()
            logging.info(f"New data stored - Timestamp: {timestamp_str} UTC, OI: {close_settlement:,.2f}, OI Value: {close_quote:,.2f} USDT")
        else:
            latest_db_time = datetime.fromtimestamp(get_latest_timestamp()/1000, UTC).strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"No new data - Latest in DB: {latest_db_time} UTC, Fetched: {timestamp_str} UTC")
        
        conn.close()

def main():
    create_table()
    latest_timestamp = get_latest_timestamp()
    if latest_timestamp > 0:
        latest_time = datetime.fromtimestamp(latest_timestamp/1000, UTC).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"Latest timestamp in database: {latest_time} UTC")
    
    data = fetch_open_interest()
    if data:
        store_data(data)
        logging.info("Successfully processed open interest data")
    else:
        logging.error("Failed to fetch open interest data")

if __name__ == "__main__":
    main()
