import sqlite3
import pandas as pd

# Path to your CSV and SQLite DB
csv_path = 'oi_history_dataapi_3.csv'
db_path = '../db.sqlite3'

def main():
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert data into the open_interest table
    df.to_sql('open_interest', conn, if_exists='append', index=False)

    # Commit and close
    conn.commit()
    conn.close()
    print("Data appended successfully.")

if __name__ == "__main__":
    main() 