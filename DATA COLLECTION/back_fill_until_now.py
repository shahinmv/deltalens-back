#!/usr/bin/env python3
import requests, aiosqlite, asyncio
import os
from datetime import datetime, timedelta, timezone
from dateutil import parser
import logging
from requests.exceptions import RequestException
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('startup_logs/ohlcv_backfill.log')
    ]
)
logger = logging.getLogger(__name__)

# DB_PATH = os.getenv('DB_PATH', '/app/db.sqlite3')
DB_PATH = '../db.sqlite3'
API_URL = "https://api.binance.com/api/v3/klines"
SYMBOL  = "BTCUSDT"
INTERVAL= "1s"
MAX_LIMIT = 1000  # Binance API limit per request

async def fetch_klines(start_time, end_time):
    """Fetch klines data with pagination"""
    all_data = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": int(current_start.timestamp()*1000),
            "endTime": int(end_time.timestamp()*1000),
            "limit": MAX_LIMIT
        }
        
        try:
            resp = requests.get(API_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if not data:
                break
                
            all_data.extend(data)
            
            # Update start time for next request
            last_timestamp = data[-1][0]
            current_start = datetime.fromtimestamp(last_timestamp/1000, timezone.utc) + timedelta(seconds=1)
            
            if len(data) < MAX_LIMIT:
                break
                
        except RequestException as e:
            logger.error(f"Failed to fetch data from Binance API: {str(e)}")
            return None
        except ValueError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return None
            
    return all_data

async def get_latest_db_timestamp(db):
    try:
        cursor = await db.execute("SELECT MAX(datetime) FROM btc_second_ohlcv")
        row = await cursor.fetchone()
        await cursor.close()
        if row and row[0]:
            # Parse the timestamp from DB (assume it's in '%Y-%m-%d %H:%M:%S' format)
            return datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        else:
            # If no data, start from a default (e.g., 1 hour ago)
            return datetime.now(timezone.utc) - timedelta(hours=1)
    except Exception as e:
        logger.error(f"Error fetching latest timestamp from DB: {str(e)}")
        return datetime.now(timezone.utc) - timedelta(hours=1)

async def backfill_until_now():
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            while True:
                # Get the latest timestamp in the DB
                latest_db_time = await get_latest_db_timestamp(db)
                now = datetime.now(timezone.utc)
                if latest_db_time >= now - timedelta(seconds=1):
                    logger.info("Database is up to date. Exiting.")
                    break
                # Binance API allows up to 1000 seconds per request
                fetch_end = min(latest_db_time + timedelta(seconds=MAX_LIMIT), now)
                logger.info(f"Backfilling: {latest_db_time} → {fetch_end}")
                data = await fetch_klines(latest_db_time + timedelta(seconds=1), fetch_end)
                if not data:
                    logger.info("No new data to backfill in this chunk.")
                    break
                logger.info(f"Received {len(data)} records from API")
                insert_data = [
                    (
                        datetime.fromtimestamp(c[0]/1000, timezone.utc)
                            .strftime("%Y-%m-%d %H:%M:%S"),
                        float(c[1]), float(c[2]), float(c[3]),
                        float(c[4]), float(c[5])
                    ) for c in data
                ]
                # Log the first and last timestamps we're trying to insert
                logger.info(f"First timestamp to insert: {insert_data[0][0]}")
                logger.info(f"Last timestamp to insert: {insert_data[-1][0]}")
                # Check how many rows were actually inserted
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM btc_second_ohlcv WHERE datetime BETWEEN ? AND ?",
                    (insert_data[0][0], insert_data[-1][0])
                )
                count_before = await cursor.fetchone()
                await cursor.close()
                # Insert data
                cursor = await db.executemany(
                    """
                    INSERT OR IGNORE INTO btc_second_ohlcv
                      (datetime, open, high, low, close, volume, imputed)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    """,
                    insert_data
                )
                await cursor.close()
                await db.commit()
                # Check how many rows exist after insertion
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM btc_second_ohlcv WHERE datetime BETWEEN ? AND ?",
                    (insert_data[0][0], insert_data[-1][0])
                )
                count_after = await cursor.fetchone()
                await cursor.close()
                rows_appended = count_after[0] - count_before[0]
                rows_ignored = len(data) - rows_appended
                logger.info(f"Rows before insertion: {count_before[0]}")
                logger.info(f"Rows after insertion: {count_after[0]}")
                logger.info(f"Rows actually appended: {rows_appended}")
                logger.info(f"Rows ignored (already existed): {rows_ignored}")
                logger.info(f"Total rows in database for this period: {count_after[0]}")
                logger.info(f"Successfully processed {len(data)} rows from API")
                logger.info(f"================================================")
                # If we reached 'now', break
                if fetch_end >= now:
                    logger.info("Reached current time. Backfill complete.")
                    break
    except Exception as e:
        logger.error(f"Unexpected error during backfill: {str(e)}")
        return

if __name__ == "__main__":
    try:
        asyncio.run(backfill_until_now())
    except KeyboardInterrupt:
        logger.info("Backfill process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error in main process: {str(e)}")
        sys.exit(1) 