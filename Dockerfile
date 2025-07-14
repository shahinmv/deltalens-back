FROM python:3.11-slim

WORKDIR /app

# Install cron and other required packages
RUN apt-get update && apt-get install -y cron sudo procps && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app && \
    echo "appuser ALL=(ALL) NOPASSWD: /usr/sbin/service cron start" >> /etc/sudoers

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create DATA COLLECTION directory and copy only required files
RUN mkdir -p "startup_logs"

RUN mkdir -p "DATA COLLECTION"
COPY ["DATA COLLECTION/live_ingest.py", "DATA COLLECTION/back_fill_last_hour.py", "DATA COLLECTION/back_fill_funding.py", "DATA COLLECTION/back_fill_oi.py", "DATA COLLECTION/live_liquidations.py", "./DATA COLLECTION/"]

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DB_PATH=/app/db.sqlite3
ENV PATH="/usr/local/bin:${PATH}"

# Create a script to run both services
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Set up cron jobs for the non-root user with full python path and proper escaping
RUN (echo "0 * * * * cd /app && /usr/local/bin/python3 'DATA COLLECTION/back_fill_last_hour.py' >> /app/logs/ohlcv_cron.log 2>&1" && \
     echo "0 */8 * * * cd /app && /usr/local/bin/python3 'DATA COLLECTION/back_fill_funding.py' >> /app/logs/funding_cron.log 2>&1" && \
     echo "*/5 * * * * cd /app && /usr/local/bin/python3 'DATA COLLECTION/back_fill_oi.py' >> /app/logs/oi_cron.log 2>&1") | crontab -u appuser -

# Create log files with proper permissions
RUN touch /app/logs/ohlcv_cron.log /app/logs/funding_cron.log /app/logs/oi_cron.log && \
    chown appuser:appuser /app/logs/*.log && \
    chmod 664 /app/logs/*.log

# Create a dummy database file to ensure permissions are set correctly
RUN touch /app/db.sqlite3 && \
    chown appuser:appuser /app/db.sqlite3 && \
    chmod 664 /app/db.sqlite3

# Switch to non-root user
USER appuser

ENTRYPOINT ["/app/docker-entrypoint.sh"] 