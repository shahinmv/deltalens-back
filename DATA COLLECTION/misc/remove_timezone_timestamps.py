import sqlite3
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_to_db():
    """Establish connection to the SQLite database"""
    try:
        conn = sqlite3.connect('../../db.sqlite3')
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def find_timezone_timestamps(conn):
    """Find all timestamps with timezone information"""
    try:
        cur = conn.cursor()
        # Find timestamps that contain '+00:00'
        query = """
        SELECT datetime FROM btc_second_ohlcv 
        WHERE datetime LIKE '%+00:00'
        """
        cur.execute(query)
        return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error finding timezone timestamps: {e}")
        return []

def remove_timezone_timestamps(conn, timestamps):
    """Remove timestamps with timezone information"""
    try:
        cur = conn.cursor()
        # Delete timestamps that contain '+00:00'
        query = """
        DELETE FROM btc_second_ohlcv 
        WHERE datetime LIKE '%+00:00'
        """
        cur.execute(query)
        conn.commit()
        return cur.rowcount
    except Exception as e:
        logger.error(f"Error removing timezone timestamps: {e}")
        return 0

def main():
    conn = connect_to_db()
    if not conn:
        return

    try:
        # Find timestamps with timezone information
        logger.info("Finding timestamps with timezone information...")
        timezone_timestamps = find_timezone_timestamps(conn)
        
        if not timezone_timestamps:
            logger.info("No timestamps with timezone information found.")
            return
        
        logger.info(f"Found {len(timezone_timestamps)} timestamps with timezone information")
        
        # Remove the timestamps
        logger.info("Removing timestamps with timezone information...")
        removed_count = remove_timezone_timestamps(conn, timezone_timestamps)
        
        logger.info(f"Successfully removed {removed_count} timestamps with timezone information")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main() 