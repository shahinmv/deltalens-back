import requests
import sqlite3
from datetime import datetime, date
from tqdm import tqdm

API_URL = "https://conpletus.cointelegraph.com/v1/"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0"
}

QUERY = '''query TagPageQuery($short: String, $slug: String!, $order: String, $offset: Int!, $length: Int!) {
  locale(short: $short) {
    tag(slug: $slug) {
      posts(order: $order, offset: $offset, length: $length) {
        data {
          id
          slug
          postTranslate {
            title
            published
            leadText
          }
        }
        postsCount
      }
    }
  }
}'''

DB_PATH = '../db.sqlite3'
CUTOFF_DATE = datetime(2020, 5, 6).date()

def fetch_news(offset=0, length=15):
    variables = {
        "short": "en",
        "slug": "bitcoin",
        "order": "postPublishedTime",
        "offset": offset,
        "length": length
    }
    payload = {
        "operationName": "TagPageQuery",
        "query": QUERY,
        "variables": variables
    }
    resp = requests.post(API_URL, json=payload, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()

def get_latest_news_date(db_path="../db.sqlite3"):
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM news")
    result = cursor.fetchone()
    conn.close()
    if result and result[0]:
        return result[0]  # 'YYYY-MM-DD'
    return None

def get_all_news(latest_date=None):
    all_news = []
    offset = 0
    length = 15
    stop = False
    today = date.today()
    total_days = (today - CUTOFF_DATE).days
    earliest_date = today
    pbar = tqdm(total=total_days, desc="Collecting news by date", unit="days")
    
    print(f"Fetching news from {latest_date or 'beginning'} onwards (inclusive)...")
    
    while not stop:
        data = fetch_news(offset, length)
        posts = data['data']['locale']['tag']['posts']['data']
        if not posts:
            break
        for post in posts:
            title = post['postTranslate']['title']
            published = post['postTranslate']['published']
            description = post['postTranslate']['leadText']
            # Parse published date
            try:
                pub_dt = datetime.fromisoformat(published.replace('Z', '+00:00')).date()
                pub_date_str = pub_dt.strftime('%Y-%m-%d')
            except Exception:
                pub_dt = None
                pub_date_str = published[:10]
            
            # Modified logic: Stop when we encounter news OLDER than latest_date
            # This ensures we include news FROM the latest_date
            if latest_date and pub_date_str < latest_date:
                print(f"Reached cutoff date. Stopping at {pub_date_str} (before {latest_date})")
                stop = True
                break
            
            # Include all news from latest_date onwards (or all news if no latest_date)
            if not latest_date or pub_date_str >= latest_date:
                all_news.append({
                    "title": title,
                    "time": published,
                    "description": description,
                    "date": pub_date_str
                })
            
            # Update progress bar
            if pub_dt and pub_dt < earliest_date:
                days_covered = (today - pub_dt).days
                pbar.n = days_covered
                pbar.refresh()
                earliest_date = pub_dt
        offset += length
    pbar.n = total_days
    pbar.refresh()
    pbar.close()
    return all_news

def insert_news_to_db(news, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    inserted = 0
    updated = 0
    
    for item in news:
        try:
            date_str = item.get('date') or (item['time'][:10] if 'time' in item else None)
            
            # Check if record already exists
            cursor.execute(
                "SELECT COUNT(*) FROM news WHERE title = ? AND date = ?",
                (item['title'], date_str)
            )
            exists = cursor.fetchone()[0] > 0
            
            if exists:
                # Update existing record
                cursor.execute(
                    "UPDATE news SET description = ? WHERE title = ? AND date = ?",
                    (item['description'], item['title'], date_str)
                )
                if cursor.rowcount > 0:
                    updated += 1
            else:
                # Insert new record
                cursor.execute(
                    "INSERT INTO news (title, description, date) VALUES (?, ?, ?)",
                    (item['title'], item['description'], date_str)
                )
                if cursor.rowcount > 0:
                    inserted += 1
                    
        except Exception as e:
            print(f"Error processing: {item['title']}\n  {e}")
    
    conn.commit()
    conn.close()
    print(f"Inserted {inserted} new news items and updated {updated} existing items in the database.")

if __name__ == "__main__":
    latest_date = get_latest_news_date()
    print(f"Latest news date in DB: {latest_date}")
    news = get_all_news(latest_date=latest_date)
    print(f"Total news to process: {len(news)}")
    insert_news_to_db(news)