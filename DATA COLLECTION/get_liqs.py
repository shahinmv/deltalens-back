import aiohttp, asyncio, pandas as pd, time, pathlib, logging

BASE      = "https://fapi.binance.com"
SYMBOL    = "BTCUSDT"
OUTDIR    = pathlib.Path("data/derivs_hist")
OUTDIR.mkdir(parents=True, exist_ok=True)

async def get_json(session, path, params):
    async with session.get(f"{BASE}{path}", params=params, timeout=10) as r:
        r.raise_for_status()
        return await r.json()

async def backfill_liquidations(session, start_ms, end_ms):
    """
    Backfill liquidation events (allForceOrders) from end_ms back to start_ms.
    """
    logger = logging.getLogger("backfill_liqs")
    logger.info(f"Backfilling liquidations from {pd.to_datetime(end_ms,unit='ms')} to {pd.to_datetime(start_ms,unit='ms')}")
    window = 200 * 24 * 3600 * 1000  # 200 days in ms
    cur_end = end_ms
    rows = []
    
    while cur_end > start_ms:
        cur_start = max(start_ms, cur_end - window)
        params = {
            "symbol": SYMBOL,
            "startTime": cur_start,
            "endTime":   cur_end,
            "limit":     1000
        }
        data = await get_json(session, "/fapi/v1/allForceOrders", params)
        if not data:
            break
        rows.extend(data)
        # move back to 1ms before the earliest timestamp in this batch
        earliest = min(item["time"] for item in data)
        cur_end = earliest - 1
        await asyncio.sleep(0.2)

    if not rows:
        logger.warning("No liquidation data fetched.")
        return

    # Normalize into DataFrame
    df = pd.DataFrame(rows)
    # time is in ms UTC
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    
    out = OUTDIR / "liquidations_hist.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved {len(df)} historical liquidations → {out}")

async def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    start = int(pd.Timestamp("2019-09-10T08:00:00Z").timestamp() * 1000)
    end   = int(time.time() * 1000)
    async with aiohttp.ClientSession() as sess:
        await backfill_liquidations(sess, start, end)

if __name__ == "__main__":
    asyncio.run(main())
