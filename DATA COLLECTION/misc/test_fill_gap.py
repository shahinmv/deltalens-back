# kraken_range_to_1s_ohlcv.py
# -----------------------------------------------------------
# pip install requests pandas python-dateutil tqdm
import requests, pandas as pd, time
from datetime import datetime, timezone, timedelta
from dateutil import parser
from tqdm import tqdm

BASE   = "https://api.kraken.com/0/public/Trades"
PAIR   = "XBTUSD"
SLEEP  = 1.0      # stay polite (Kraken recommends 1 req/s)

def iso_utc(dt_str: str) -> datetime:
    """ISO-8601 → timezone-aware UTC datetime"""
    dt = parser.parse(dt_str)
    return dt.astimezone(timezone.utc)

def fetch_trades_range(ts_start: datetime, ts_end: datetime):
    """Fetch all trades between ts_start and ts_end"""
    since_ns = int(ts_start.timestamp() * 1_000_000_000)  # start in nanoseconds
    all_trades = []

    while True:
        params = {"pair": PAIR, "since": since_ns}
        r = requests.get(BASE, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        trades = data["result"][list(data["result"].keys())[0]]
        last = data["result"]["last"]          # next "since" value (ns)

        # Filter only inside the window
        filtered = [t for t in trades if ts_start.timestamp() <= float(t[2]) < ts_end.timestamp()]
        all_trades.extend(filtered)

        if not trades or (float(trades[-1][2]) >= ts_end.timestamp()):
            break

        since_ns = int(last)       # move pagination cursor
        time.sleep(SLEEP)

    return all_trades

def rebuild_1s_ohlcv(trades, ts_start: datetime, ts_end: datetime):
    """Group raw trades into 1-second OHLCV bars"""
    idx = pd.date_range(ts_start, ts_end, freq='1s', inclusive="left", tz="UTC")

    df = pd.DataFrame(trades, columns=["price", "volume", "time", "side", "type", "misc", "trade_id"])
    df["time"] = pd.to_datetime(df["time"].astype(float), unit="s", utc=True)
    df["price"] = df["price"].astype(float)
    df["volume"] = df["volume"].astype(float)

    grouped = (df.groupby(pd.Grouper(key="time", freq="1S"))
                 .agg(open=("price","first"),
                      high=("price","max"),
                      low =("price","min"),
                      close=("price","last"),
                      volume=("volume","sum"))
              )

    # reindex to ensure every second exists
    grouped = grouped.reindex(idx)
    grouped["imputed"] = grouped["open"].isna()
    return grouped.reset_index().rename(columns={"index":"timestamp"})

if __name__ == "__main__":
    # ── change these two datetime strings ─────────────
    start_dt = "2017-09-06 19:55:00"
    end_dt   = "2017-09-06 20:05:00"
    # ──────────────────────────────────────────────────
    ts_start = iso_utc(start_dt)
    ts_end   = iso_utc(end_dt)

    print(f"Fetching Kraken {PAIR} trades between {ts_start} → {ts_end}...")

    trades = fetch_trades_range(ts_start, ts_end)
    print(f"Fetched {len(trades)} trades.")

    ohlcv = rebuild_1s_ohlcv(trades, ts_start, ts_end)
    print("\nSample output:")
    print(ohlcv.head())

    # Save if you want
    ohlcv.to_csv("kraken_1s_ohlcv_patch.csv", index=False)
    print("\nSaved to kraken_1s_ohlcv_patch.csv ✅")
