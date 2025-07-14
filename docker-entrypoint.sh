#!/bin/bash
set -e

# Start cron service
echo "Starting cron service..."
sudo service cron start

# Verify cron is running (using ps instead of pgrep)
if ! ps aux | grep -q "[c]ron"; then
    echo "Error: Cron service failed to start"
    exit 1
fi
echo "Cron service started successfully"

# Run initial backfill
echo "Running initial backfill 1s BTC..."
python3 "DATA COLLECTION/back_fill_last_hour.py"

echo "Running initial backfill funding..."
python3 "DATA COLLECTION/back_fill_funding.py"

echo "Running initial backfill oi..."
python3 "DATA COLLECTION/back_fill_oi.py"

# Start both live processes in the background
echo "Starting live ingest process..."
python3 "DATA COLLECTION/live_ingest.py" &

echo "Starting live liquidations process..."
python3 "DATA COLLECTION/live_liquidations.py" &

# Wait for any background process to exit
wait 