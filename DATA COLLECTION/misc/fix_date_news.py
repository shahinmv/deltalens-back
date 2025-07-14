import sqlite3
from datetime import datetime
import re

DB_PATH = '../../db.sqlite3'

def parse_date(date_str):
    # If it's a relative time (e.g., '8 hours ago', '23 minutes ago'), return today
    if re.match(r"\\d+ (minute|hour|day|week|month|year)s? ago", date_str):
        return datetime.now().strftime('%Y-%m-%d')
    try:
        # Try parsing format like 'Jun 27, 2025'
        dt = datetime.strptime(date_str, '%b %d, %Y')
        return dt.strftime('%Y-%m-%d')
    except Exception:
        # If parsing fails, return None
        return None

def migrate_news_dates():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Add new column if not exists
    cur.execute("PRAGMA table_info(news);")
    columns = [row[1] for row in cur.fetchall()]
    if 'date_fixed' not in columns:
        cur.execute("ALTER TABLE news ADD COLUMN date_fixed DATE;")
        conn.commit()

    # 2. Update new column with converted dates
    cur.execute("SELECT id, date FROM news;")
    rows = cur.fetchall()
    for row in rows:
        id_, date_str = row
        fixed_date = parse_date(date_str)
        if fixed_date:
            cur.execute("UPDATE news SET date_fixed = ? WHERE id = ?;", (fixed_date, id_))
    conn.commit()

    # 3. (Optional) Drop old column and rename new one
    # SQLite doesn't support DROP COLUMN directly, so we need to recreate the table
    cur.execute("PRAGMA table_info(news);")
    columns = [row[1] for row in cur.fetchall()]
    if 'date_fixed' in columns:
        # Create new table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS news_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE,
                description TEXT,
                date DATE
            );
        ''')
        # Copy data
        cur.execute('''
            INSERT OR REPLACE INTO news_new (id, title, description, date)
            SELECT id, title, description, date_fixed FROM news;
        ''')
        conn.commit()
        # Drop old table and rename
        cur.execute('DROP TABLE news;')
        cur.execute('ALTER TABLE news_new RENAME TO news;')
        conn.commit()

    conn.close()
    print('Migration complete!')

if __name__ == '__main__':
    migrate_news_dates()
