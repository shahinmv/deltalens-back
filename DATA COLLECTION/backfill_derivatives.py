# backfill_binance_derivs.py
import aiohttp, asyncio, pandas as pd, time, math, pathlib, logging, numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
OUTDIR = pathlib.Path("data/derivs_hist")
OUTDIR.mkdir(parents=True, exist_ok=True)

async def get_json(session, path, params, weight=1):
    """Make API request"""
    logger.debug(f"Making request to {path} with params: {params}")
    async with session.get(f"{BASE}{path}", params=params, timeout=10) as r:
        r.raise_for_status()
        return await r.json()

async def backfill_funding(session, start_ms, end_ms):
    """
    Pulls funding history in 1000-row chunks (= ~41 days) backward to start_ms.
    """
    logger.info(f"Starting funding rate backfill from {pd.Timestamp(end_ms, unit='ms')} to {pd.Timestamp(start_ms, unit='ms')}")
    cur_end = end_ms
    dfs = []
    total_records = 0
    step = 30 * 24 * 60 * 60 * 1000  # 30 days in milliseconds
    
    while cur_end > start_ms:
        current_start = max(start_ms, cur_end - step)
        params = {
            "symbol": SYMBOL,
            "limit": 1000,
            "startTime": current_start,
            "endTime": cur_end
        }
        logger.debug(f"Requesting data from {pd.Timestamp(current_start, unit='ms')} to {pd.Timestamp(cur_end, unit='ms')}")
        
        try:
            data = await get_json(session, "/fapi/v1/fundingRate", params)
            if not data:
                logger.info(f"No data found for period {pd.Timestamp(current_start, unit='ms')} to {pd.Timestamp(cur_end, unit='ms')}")
                break
                
            df = pd.DataFrame(data)
            if len(df) == 0:
                logger.info(f"Empty response for period {pd.Timestamp(current_start, unit='ms')} to {pd.Timestamp(cur_end, unit='ms')}")
                cur_end = current_start - 1
                continue
                
            dfs.append(df)
            total_records += len(df)
            logger.info(f"Retrieved {len(df)} funding rate records for period {pd.Timestamp(current_start, unit='ms')} to {pd.Timestamp(cur_end, unit='ms')} (Total: {total_records})")
            
            # Move to the next period
            cur_end = current_start - 1
            await asyncio.sleep(0.2)
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            break
    
    if dfs:
        hist = pd.concat(dfs, ignore_index=True)
        # Convert timestamp columns to datetime
        hist['fundingTime'] = pd.to_datetime(hist['fundingTime'], unit='ms')
        # Sort by time
        hist = hist.sort_values('fundingTime')
        # Remove duplicates
        hist = hist.drop_duplicates(subset=['fundingTime'])
        # Save to CSV
        csv_path = OUTDIR / "funding.csv"
        hist.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(hist)} total funding rate records to {csv_path}")
        logger.info(f"Data range: {hist['fundingTime'].min()} to {hist['fundingTime'].max()}")
    else:
        logger.warning("No funding rate data was retrieved")

async def backfill_oi(session, start_ms, end_ms):
    """
    5-minute open-interest history. Binance wants startTime/endTime in ms,
    limit=200 points per call → ~16.6 hours of data at 5m resolution.
    """
    logger.info(f"Starting open interest backfill from {pd.Timestamp(end_ms, unit='ms')} to {pd.Timestamp(start_ms, unit='ms')}")
    cur_end = end_ms
    dfs = []
    total_records = 0

    # 16h window (5m × 200 points = 16.6 h)
    window_ms = 200 * 5 * 60 * 1000

    while cur_end > start_ms:
        current_start = max(start_ms, cur_end - window_ms)
        params = {
            "symbol": SYMBOL,
            "period": "5m",
            "limit": 200,
            "startTime": current_start,
            "endTime":   cur_end
        }
        logger.debug(f"Requesting OI from {pd.Timestamp(current_start, unit='ms')} to {pd.Timestamp(cur_end, unit='ms')}")
        try:
            data = await get_json(session, "/futures/data/openInterestHist", params)
            if not data:
                logger.info("No more data; ending backfill loop.")
                break

            df = pd.DataFrame(data)
            if df.empty:
                logger.info("Empty frame; stepping back.")
                cur_end = current_start - 1
                continue

            # API returns a 'timestamp' in ms already
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            dfs.append(df)
            total_records += len(df)
            logger.info(f"Got {len(df)} records; total so far: {total_records}")

            # move the window back
            cur_end = int(df['timestamp'].min().timestamp() * 1000) - 1
            await asyncio.sleep(0.2)

        except Exception as e:
            logger.error(f"Error fetching OI: {e}")
            break

    if dfs:
        hist = pd.concat(dfs, ignore_index=True)
        hist = hist.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        out = OUTDIR / "oi_5m_corrected.csv"
        hist.to_csv(out, index=False)
        logger.info(f"Saved {len(hist)} OI records to {out}")
        logger.info(f"Range: {hist['timestamp'].min()} → {hist['timestamp'].max()}")
    else:
        logger.warning("No open interest history retrieved")


async def main():
    logger.info("Starting Binance derivatives data backfill")
    # Set start time to the earliest possible date for BTCUSDT futures
    start_ms = int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)
    end_ms   = int(time.time() * 1000)
    logger.info(f"Data collection period: {pd.Timestamp(start_ms, unit='ms')} to {pd.Timestamp(end_ms, unit='ms')}")
    
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            backfill_funding(session, start_ms, end_ms),
            backfill_oi(session,      start_ms, end_ms)
        )
    logger.info("Data backfill completed successfully")

if __name__ == "__main__":
    asyncio.run(main())
