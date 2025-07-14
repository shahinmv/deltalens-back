# derivatives_stream.py
import asyncio, aiohttp, websockets, json, time
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BINANCE_REST = "https://fapi.binance.com"
SYMBOL       = "BTCUSDT"
OUT_DIR      = Path("data/derivs")  # bronze layer

async def fetch_json(session, url, params=None):
    try:
        async with session.get(url, params=params, timeout=10) as r:
            r.raise_for_status()
            data = await r.json()
            logger.debug(f"Fetched data from {url}: {data}")
            return data
    except Exception as e:
        logger.error(f"Error fetching data from {url}: {str(e)}")
        raise

async def periodic_rest(session, endpoint, params, out_csv, every_secs):
    logger.info(f"Starting periodic collection for {endpoint}")
    while True:
        try:
            data = await fetch_json(session, f"{BINANCE_REST}{endpoint}", params)
            # Handle both list and dictionary responses
            if isinstance(data, list):
                data = data[0]  # Take the first item if it's a list
            epoch_ms = int(time.time() * 1000)
            data["ts"] = epoch_ms
            data["datetime"] = datetime.fromtimestamp(epoch_ms/1000).strftime('%Y-%m-%d %H:%M:%S')
            (OUT_DIR / out_csv.parent).mkdir(parents=True, exist_ok=True)
            pd.DataFrame([data]).to_csv(out_csv, mode="a", index=False, header=not out_csv.exists())
            logger.info(f"Saved data to {out_csv}")
            await asyncio.sleep(every_secs)
        except Exception as e:
            logger.error(f"Error in periodic_rest for {endpoint}: {str(e)}")
            await asyncio.sleep(every_secs)  # Wait before retrying

async def liquidation_ws(out_csv):
    url = f"wss://stream.binance.com:9443/ws/!forceOrder@arr"
    logger.info("Starting liquidation websocket connection")
    async for ws in websockets.connect(url, ping_interval=20):
        try:
            async for msg in ws:
                event = json.loads(msg)
                if event["o"]["s"] != SYMBOL:       # filter BTC only
                    continue
                epoch_ms = event["E"]
                row = {
                    "ts": epoch_ms,
                    "datetime": datetime.fromtimestamp(epoch_ms/1000).strftime('%Y-%m-%d %H:%M:%S'),
                    "side": event["o"]["S"],         # BUY = longs liquidated
                    "qty": float(event["o"]["q"]),
                    "price": float(event["o"]["p"]),
                }
                pd.DataFrame([row]).to_csv(out_csv, mode="a", index=False,
                                           header=not out_csv.exists())
                logger.debug(f"Saved liquidation event: {row}")
        except websockets.ConnectionClosed:
            logger.warning("Websocket connection closed, attempting to reconnect...")
            continue  # auto-reconnect
        except Exception as e:
            logger.error(f"Error in liquidation_ws: {str(e)}")
            continue

async def main():
    logger.info("Starting data collection process")
    async with aiohttp.ClientSession() as session:
        tasks = [
            # funding + mark price
            periodic_rest(session,
                "/fapi/v1/premiumIndex",
                {"symbol": SYMBOL},
                OUT_DIR / "funding.csv",
                every_secs=60),

            # open interest (USD)
            periodic_rest(session,
                "/fapi/v1/openInterest",
                {"symbol": SYMBOL},
                OUT_DIR / "open_interest.csv",
                every_secs=60),

            # long/short ratio (5-min window, 12 = last hour etc.)
            periodic_rest(session,
                "/futures/data/globalLongShortAccountRatio",
                {"symbol": SYMBOL, "period": "5m", "limit": 1},
                OUT_DIR / "ls_ratio.csv",
                every_secs=300),

            # real-time liquidations
            liquidation_ws(OUT_DIR / "liquidations.csv")
        ]
        logger.info("All tasks started, waiting for completion...")
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Data collection stopped by user")
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
