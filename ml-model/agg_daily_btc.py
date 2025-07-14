import sqlite3
import pandas as pd

# Path to the SQLite database
DB_PATH = '../db.sqlite3'
TABLE_NAME = 'btc_second_ohlcv'
DAILY_TABLE = 'btc_daily_ohlcv'
OUTPUT_CSV = 'btc_daily_ohlcv_from_db.csv'
CHUNKSIZE = 10**6  # Adjust based on your memory

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check if btc_daily_ohlcv exists and get the latest date
try:
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{DAILY_TABLE}'")
    if cursor.fetchone():
        cursor.execute(f"SELECT MAX(datetime) FROM {DAILY_TABLE}")
        latest_date = cursor.fetchone()[0]
        if latest_date:
            print(f"Latest date in {DAILY_TABLE}: {latest_date}")
            start_date = pd.to_datetime(latest_date) + pd.Timedelta(days=1)
            query = f"SELECT * FROM {TABLE_NAME} WHERE datetime >= '{start_date.strftime('%Y-%m-%d')}'"
        else:
            print(f"No data in {DAILY_TABLE}, fetching all data.")
            query = f"SELECT * FROM {TABLE_NAME}"
    else:
        print(f"Table {DAILY_TABLE} does not exist, fetching all data.")
        query = f"SELECT * FROM {TABLE_NAME}"
except Exception as e:
    print(f"Error checking {DAILY_TABLE}: {e}")
    query = f"SELECT * FROM {TABLE_NAME}"

# Prepare an empty DataFrame for aggregation
agg_df = None
chunk_num = 0

for chunk in pd.read_sql_query(query, conn, parse_dates=['datetime'], chunksize=CHUNKSIZE):
    chunk_num += 1
    print(f"\nProcessing chunk {chunk_num} with {len(chunk)} rows...")
    if not pd.api.types.is_datetime64_any_dtype(chunk['datetime']):
        print("Converting 'datetime' to pandas datetime...")
        chunk['datetime'] = pd.to_datetime(chunk['datetime'])
    chunk.set_index('datetime', inplace=True)
    print("Resampling chunk to daily OHLCV...")
    ohlc = chunk[['open', 'high', 'low', 'close']].resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    volume = chunk['volume'].resample('1D').sum()
    daily_chunk = pd.concat([ohlc, volume], axis=1)
    daily_chunk = daily_chunk.dropna(subset=['open', 'high', 'low', 'close'])
    print(f"Chunk {chunk_num} daily rows: {len(daily_chunk)}")
    if agg_df is None:
        agg_df = daily_chunk
    else:
        agg_df = pd.concat([agg_df, daily_chunk])
    print(f"Aggregated DataFrame now has {len(agg_df)} daily rows.")

if agg_df is not None and not agg_df.empty:
    # Reset index to keep 'datetime' as a column
    agg_df = agg_df.reset_index()
    # Convert datetime to date only
    agg_df['datetime'] = agg_df['datetime'].dt.date
    # Remove rows with datetimes already in the table
    existing_dates = pd.read_sql_query(f"SELECT datetime FROM {DAILY_TABLE}", conn)
    if not existing_dates.empty:
        # Robustly convert to date, even if time is present
        existing_dates['datetime'] = pd.to_datetime(existing_dates['datetime'], errors='coerce').dt.date
        agg_df = agg_df[~agg_df['datetime'].isin(existing_dates['datetime'])]
    # Remove any duplicate datetimes within agg_df itself
    agg_df = agg_df.drop_duplicates(subset=['datetime'], keep='last')
    if not agg_df.empty:
        print(f"Appending aggregated daily OHLCV to table '{DAILY_TABLE}' in the database...")
        agg_df.to_sql(DAILY_TABLE, conn, if_exists='append', index=False)
        print(f"✅ Resampled daily OHLCV appended to table '{DAILY_TABLE}' in the database.")
    else:
        print("No new data to append after removing duplicates.")
else:
    print("No new data to append.")

conn.close()
