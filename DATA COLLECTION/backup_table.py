#!/usr/bin/env python3
import sqlite3
import os
from datetime import datetime

def backup_table():
    # Database paths
    db_path = '../db.sqlite3'
    backup_dir = '../backups'
    
    # Create backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)
    
    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f'btc_second_ohlcv_backup_{timestamp}.db')
    
    try:
        # Connect to source database
        source_conn = sqlite3.connect(db_path)
        source_cursor = source_conn.cursor()
        
        # Create backup database
        backup_conn = sqlite3.connect(backup_path)
        backup_cursor = backup_conn.cursor()
        
        # Create the table in backup database
        source_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='btc_second_ohlcv'")
        create_table_sql = source_cursor.fetchone()[0]
        backup_cursor.execute(create_table_sql)
        
        # Copy all data from source to backup
        source_cursor.execute("SELECT * FROM btc_second_ohlcv")
        rows = source_cursor.fetchall()
        
        # Insert data into backup table
        backup_cursor.executemany("INSERT INTO btc_second_ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        
        # Commit changes and close connections
        backup_conn.commit()
        backup_conn.close()
        source_conn.close()
        
        print(f"Backup created successfully at: {backup_path}")
        print(f"Number of rows backed up: {len(rows)}")
        
    except Exception as e:
        print(f"Error creating backup: {str(e)}")
        if 'backup_conn' in locals():
            backup_conn.close()
        if 'source_conn' in locals():
            source_conn.close()

if __name__ == "__main__":
    backup_table() 