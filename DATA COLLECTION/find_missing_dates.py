import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import calendar
from tqdm import tqdm
import logging

# ─── Configuration ────────────────────────────────────────────────────────────
DB_PATH      = '../db.sqlite3'
TABLE_NAME   = 'btc_second_ohlcv'
COL_NAME     = 'datetime'
CHUNK_SIZE   = 5_000_000         # adjust up/down for your RAM
START_STR    = "2017-08-17 04:00:30"
OUTPUT_CSV   = 'missing_timestamps_final.csv'

# ─── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─── Step 1: Read existing timestamps in UTC seconds ───────────────────────────
logger.info("1/5 ▶ Connecting to database…")
conn = sqlite3.connect(DB_PATH)

existing_secs = set()
sql = f"SELECT {COL_NAME} FROM {TABLE_NAME} ORDER BY {COL_NAME} ASC"

logger.info(f"2/5 ▶ Reading `{COL_NAME}` in chunks of {CHUNK_SIZE:,} rows…")
for chunk in tqdm(
    pd.read_sql_query(sql, conn, parse_dates=[COL_NAME], chunksize=CHUNK_SIZE),
    desc="Loading DB chunks",
    unit="chunk"
):
    # chunk[COL_NAME] is datetime64[ns], naive UTC
    secs = chunk[COL_NAME].values.astype('datetime64[s]').astype(int)
    existing_secs.update(secs)

conn.close()
logger.info(f"   ➔ Loaded {len(existing_secs):,} distinct timestamps from DB.")

# ─── Step 2: Compute UTC start/end seconds ─────────────────────────────────────
start_dt = datetime.strptime(START_STR, "%Y-%m-%d %H:%M:%S")
start_sec = calendar.timegm(start_dt.timetuple())  
# (treat start_dt as UTC)

end_sec = max(existing_secs)
utc_end_dt = datetime.utcfromtimestamp(end_sec)

logger.info(
    f"3/5 ▶ Generating expected UTC range: "
    f"{start_dt.isoformat()} → {utc_end_dt.isoformat()} "
    f"({end_sec - start_sec:,} seconds)"
)

# ─── Step 3: Build expected array & diff ───────────────────────────────────────
expected_secs = np.arange(start_sec, end_sec + 1, dtype=int)
logger.info("4/5 ▶ Computing missing timestamps (set difference)…")
missing_secs = np.setdiff1d(expected_secs, np.fromiter(existing_secs, dtype=int))
logger.info(f"   ➔ Found {len(missing_secs):,} missing timestamps.")

# ─── Step 4: Convert back to datetimes & save ─────────────────────────────────
if missing_secs.size:
    missing_dt = pd.to_datetime(missing_secs, unit='s', origin='unix')
    missing_dt.to_series(name='missing_timestamps').to_csv(OUTPUT_CSV, index=False)
    logger.info(f"5/5 ▶ Saved missing timestamps to `{OUTPUT_CSV}`")
else:
    logger.info("5/5 ▶ ✅ No missing gaps found.")

logger.info("All done.")
