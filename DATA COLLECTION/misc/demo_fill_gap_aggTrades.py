# one_second_aggtrades_no_key.py
# -----------------------------------------------------------
# pip install requests python-dateutil pandas
import requests, pandas as pd
from dateutil import parser
from datetime import timezone

BASE_URL = "https://api.binance.com"
SYMBOL   = "BTCUSDT"

def utc_ms(dt_str: str) -> int:
    """ISO-8601 → UTC epoch-milliseconds"""
    dt = parser.parse(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)     # assume already UTC
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)

def get_aggtrades(symbol: str, ts_ms: int):
    url = f"{BASE_URL}/api/v3/aggTrades"
    params = {
        "symbol": symbol,
        "startTime": ts_ms,
        "endTime": ts_ms + 999,   # inclusive
        "limit": 1000
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def to_ohlcv(trades):
    if not trades:
        return "NO TRADES IN THIS SECOND"
    prices  = pd.Series([float(t["p"]) for t in trades])
    volume  = sum(float(t["q"]) for t in trades)
    return dict(open=prices.iloc[0],
                high=prices.max(),
                low =prices.min(),
                close=prices.iloc[-1],
                volume=volume)

if __name__ == "__main__":
    # ↘ Put any datetime here (local or with timezone offset)
    dt_query = "2021-08-13 06:50:27"
    ts = utc_ms(dt_query)

    trades = get_aggtrades(SYMBOL, ts)
    print(f"\nRaw aggTrades ({len(trades)} rows):")
    for t in trades:
        print(t)

    print("\nOHLCV summary:")
    print(to_ohlcv(trades))
