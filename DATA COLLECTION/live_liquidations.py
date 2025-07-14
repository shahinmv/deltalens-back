#!/usr/bin/env python3
import asyncio
import json
import logging
import sqlite3
import time
import os
import websockets

# ─── Configuration ─────────────────────────────────────────────────────────────
WS_URL       = "wss://fstream.binance.com/ws/btcusdt@forceOrder"
RECONNECT_DELAY = 5  # seconds
HEARTBEAT_INTERVAL = 30  # seconds
DB_PATH         = os.getenv('DB_PATH', '/app/db.sqlite3')


# ─── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("liquidations_stream")

# Add a filter to clean up websocket messages
class WebSocketFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # Filter out PING/PONG and binary messages
        return not any(pattern in msg for pattern in [
            'PING', 'PONG', '[binary', 'bytes]'
        ])

logger.addFilter(WebSocketFilter())

# ─── Database setup ────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
cur  = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS liquidations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exec_time TIMESTAMP,
    side       TEXT,
    qty        REAL,
    price      REAL,
    notional   REAL
)
""")
conn.commit()

# ─── Stream handler ────────────────────────────────────────────────────────────
async def heartbeat(ws):
    while True:
        try:
            await ws.ping()
            logger.debug("Heartbeat sent")
            await asyncio.sleep(HEARTBEAT_INTERVAL)
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            break

async def stream_liquidations():
    while True:
        try:
            logger.info(f"Connecting to {WS_URL}")
            async with websockets.connect(
                WS_URL,
                ping_interval=60,
                ping_timeout=10,
            ) as ws:
                logger.info("Connected, starting to receive messages")
                
                # Start heartbeat task
                heartbeat_task = asyncio.create_task(heartbeat(ws))
                
                try:
                    async for message in ws:
                        logger.debug(f"Received message: {message[:200]}...")  # Log first 200 chars of message
                        data = json.loads(message).get("o", {})
                        # Parse fields
                        exec_time  = int(data.get("T", 0))
                        side       = data.get("S")
                        qty        = float(data.get("l", 0.0))
                        price      = float(data.get("ap", 0.0))
                        notional   = qty * price

                        exec_ts = time.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            time.gmtime(exec_time / 1000)
                        )

                        # Insert into SQLite
                        cur.execute("""
                            INSERT INTO liquidations (
                                exec_time, side, qty, price, notional
                            ) VALUES ( ?, ?, ?, ?, ?)
                        """, (exec_ts, side, qty, price, notional))
                        conn.commit()
                        logger.info(
                            f"Liquidation: {side} | Size: {qty:.4f} BTC | Price: ${price:,.2f} | Value: ${notional:,.2f}"
                        )
                finally:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

        except Exception as e:
            logger.error(f"WebSocket error: {e}, reconnecting in {RECONNECT_DELAY}s")
            await asyncio.sleep(RECONNECT_DELAY)

# ─── Main entrypoint ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(stream_liquidations())
    except KeyboardInterrupt:
        logger.info("Shutting down stream...")
    finally:
        conn.close()
