#!/usr/bin/env python3
"""
live_ingest.py
───────────────
Continuously ingest Binance BTCUSDT 1-second candles into SQLite.

Features:
 • Async WebSocket via python-binance
 • aiosqlite for non-blocking SQLite writes
 • Batch commits every N candles
 • Automatic reconnect on errors
 • Detailed logging
"""

import asyncio
import logging
import time
import aiosqlite
import os
from datetime import datetime, timezone
from binance import AsyncClient, BinanceSocketManager
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

# ─── Configuration ────────────────────────────────────────────────────────────
DB_PATH         = os.getenv('DB_PATH', '/app/db.sqlite3')
PAIR            = "BTCUSDT"
FLUSH_EVERY     = 5        # commit every 5 candles
RECONNECT_DELAY = 5        # seconds to wait before reconnecting
LOG_LEVEL       = logging.INFO
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS btc_second_ohlcv (
    datetime TEXT PRIMARY KEY,    -- UTC timestamp 'YYYY-MM-DD HH:MM:SS'
    open     REAL NOT NULL,
    high     REAL NOT NULL,
    low      REAL NOT NULL,
    close    REAL NOT NULL,
    volume   REAL NOT NULL,
    imputed  INTEGER DEFAULT 0    -- 0 = real candle
);
"""

INSERT_SQL = """
INSERT OR IGNORE INTO btc_second_ohlcv
  (datetime, open, high, low, close, volume, imputed)
VALUES (?, ?, ?, ?, ?, ?, 0);
"""

async def ingest_loop():
    # Ensure the DB + table exists
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_TABLE_SQL)
        await db.execute("PRAGMA journal_mode=DELETE;")  # Changed from WAL to DELETE
        await db.execute("PRAGMA synchronous=NORMAL;")   # Added for better performance
        await db.commit()
    logging.info("Initialized SQLite DB at %s", DB_PATH)

    while True:
        client = await AsyncClient.create()
        try:
            bsm = BinanceSocketManager(client)
            socket = bsm.kline_socket(
                symbol=PAIR,
                interval="1s"
            )

            async with aiosqlite.connect(DB_PATH) as db, socket as stream:
                buffer = []
                logging.info("WebSocket connected for %s@1s", PAIR)
                logging.info("Starting to receive data...")

                while True:
                    msg = await stream.recv()
                    logging.debug("Received message: %s", msg)
                    # 1) Filter for kline events
                    if msg.get("e") != "kline":
                        logging.debug("Skipping non-kline message")
                        continue

                    k = msg["k"]
                    # 2) Only process the *closed* candle
                    if not k.get("x", False):
                        logging.debug("Skipping non-closed candle")
                        continue

                    # 3) Build our row
                    ts = datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc)\
                                 .strftime("%Y-%m-%d %H:%M:%S")
                    row = (
                        ts,
                        float(k["o"]),
                        float(k["h"]),
                        float(k["l"]),
                        float(k["c"]),
                        float(k["v"])
                    )
                    buffer.append(row)
                    logging.debug("Buffered candle %s (open=%.2f close=%.2f)",
                                  ts, row[1], row[4])

                    # 4) Flush every N rows
                    if len(buffer) >= FLUSH_EVERY:
                        try:
                            await db.executemany(INSERT_SQL, buffer)
                            await db.commit()
                            # Verify the insert
                            cursor = await db.execute(
                                "SELECT COUNT(*) FROM btc_second_ohlcv WHERE datetime = ?",
                                (buffer[-1][0],)
                            )
                            count = await cursor.fetchone()
                            await cursor.close()
                            if count[0] > 0:
                                logging.info("Successfully inserted %d candles, last ts=%s (verified)",
                                             len(buffer), buffer[-1][0])
                            else:
                                logging.error("Failed to insert candles, last ts=%s",
                                             buffer[-1][0])
                        except Exception as e:
                            logging.error("Error inserting data: %s", str(e))
                        buffer.clear()

        except (ConnectionClosedError, ConnectionClosedOK) as e:
            logging.warning("WebSocket disconnected: %s – reconnecting in %ds",
                            e, RECONNECT_DELAY)
            await client.close_connection()
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception as e:
            logging.exception("Unexpected error in ingest loop – reconnecting in %ds",
                              RECONNECT_DELAY)
            await client.close_connection()
            await asyncio.sleep(RECONNECT_DELAY)
        else:
            # if we ever exit the socket context cleanly, restart
            logging.info("Socket context ended cleanly – restarting in %ds",
                         RECONNECT_DELAY)
            await client.close_connection()
            await asyncio.sleep(RECONNECT_DELAY)

if __name__ == "__main__":
    try:
        asyncio.run(ingest_loop())
    except KeyboardInterrupt:
        logging.info("Interrupted by user; exiting.")
