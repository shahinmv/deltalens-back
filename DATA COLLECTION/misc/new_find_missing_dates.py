import sqlite3
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# --- Settings ---
db_path = "../../db.sqlite3"
table_name = "btc_second_ohlcv"
datetime_column = "datetime"
start_time = datetime.strptime("2017-08-17 04:00:28", "%Y-%m-%d %H:%M:%S")
end_time = datetime.strptime("2025-04-25 12:52:18", "%Y-%m-%d %H:%M:%S")

# --- Connect to database ---
print("[1/5] Connecting to database...")
conn = sqlite3.connect(db_path)

# --- Fetch all existing datetimes in one go ---
print("[2/5] Reading existing timestamps from database...")
df_existing = pd.read_sql_query(
    f"SELECT {datetime_column} FROM {table_name}",
    conn,
    parse_dates=[datetime_column]
)
existing_times = df_existing[datetime_column]

print(f"     ➔ Retrieved {len(existing_times):,} timestamps from database.")

# --- Close connection (no need to keep open) ---
conn.close()
print("     ➔ Database connection closed.")

# --- Process into a set for fast lookup ---
print("[3/5] Preparing lookup set...")
existing_times_set = set(existing_times)

# --- Generate full expected timestamps ---
print("[4/5] Generating full expected timestamps...")
full_times = pd.date_range(start=start_time, end=end_time, freq="S")
print(f"     ➔ Generated {len(full_times):,} full timestamps.")

# --- Find missing timestamps ---
print("[5/5] Finding missing timestamps (set difference)...")
missing_times = list(set(full_times) - existing_times_set)

# --- Output ---
print("\n✅ Finished checking!")

print(f"Total missing timestamps: {len(missing_times):,}")

if missing_times:
    print("Saving missing timestamps to 'missing_timestamps.csv'...")
    pd.Series(missing_times, name="missing_datetime").to_csv("missing_timestamps.csv", index=False)
    print("Saved successfully!")

print("🎯 All done.")
