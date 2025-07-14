import sqlite3
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data_collection.log',
    filemode='a'
)

def forward_fill_missing_data():
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('../db.sqlite3')
        cursor = conn.cursor()
        
        # Check if imputed column exists, if not add it
        cursor.execute("PRAGMA table_info(btc_second_ohlcv)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'imputed' not in columns:
            cursor.execute("ALTER TABLE btc_second_ohlcv ADD COLUMN imputed BOOLEAN DEFAULT FALSE")
            conn.commit()
            logging.info("Added imputed column to btc_second_ohlcv table")
        
        # Read missing timestamps from CSV
        missing_timestamps = pd.read_csv('missing_timestamps.csv')
        missing_timestamps['missing_timestamps'] = pd.to_datetime(missing_timestamps['missing_timestamps'])
        
        # Sort timestamps to ensure we process them in chronological order
        missing_timestamps = missing_timestamps.sort_values('missing_timestamps')
        
        # Process each missing timestamp
        for _, row in missing_timestamps.iterrows():
            missing_dt = row['missing_timestamps']
            # Convert datetime to string format that SQLite can understand
            missing_dt_str = missing_dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Find the last known price before the missing timestamp
            cursor.execute('''
                SELECT datetime, open, high, low, close, volume
                FROM btc_second_ohlcv
                WHERE datetime < ?
                ORDER BY datetime DESC
                LIMIT 1
            ''', (missing_dt_str,))
            
            last_row = cursor.fetchone()
            
            if last_row:
                # Insert the new row with forward-filled data
                cursor.execute('''
                    INSERT INTO btc_second_ohlcv (datetime, open, high, low, close, volume, imputed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    missing_dt_str,
                    last_row[1],  # open
                    last_row[2],  # high
                    last_row[3],  # low
                    last_row[4],  # close
                    0,            # volume
                    True          # imputed flag
                ))
                
                logging.info(f"Forward filled data for {missing_dt_str}")
        
        # Commit the changes
        conn.commit()
        logging.info("Successfully forward filled all missing data")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    forward_fill_missing_data()
