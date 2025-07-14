import sqlite3
import pandas as pd
from datetime import datetime

def create_database():
    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect('../db.sqlite3')
    cursor = conn.cursor()
    
    # Create the funding_rates table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS funding_rates (
        symbol TEXT NOT NULL,
        funding_time TIMESTAMP NOT NULL,
        funding_rate REAL,
        mark_price REAL,
        PRIMARY KEY (symbol, funding_time)
    )
    ''')
    
    conn.commit()
    return conn

def import_data(conn):
    # Read the CSV file
    df = pd.read_csv('data/derivs_hist/funding.csv')
    
    # Convert fundingTime to datetime
    df['fundingTime'] = pd.to_datetime(df['fundingTime'])
    
    # Rename columns to match database schema
    df = df.rename(columns={
        'fundingTime': 'funding_time',
        'fundingRate': 'funding_rate',
        'markPrice': 'mark_price'
    })
    
    # Insert data into the database
    df.to_sql('funding_rates', conn, if_exists='append', index=False)
    
    print(f"Successfully imported {len(df)} records into the database")

def main():
    try:
        # Create database and get connection
        conn = create_database()
        
        # Import the data
        import_data(conn)
        
        # Close the connection
        conn.close()
        print("Database operation completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 