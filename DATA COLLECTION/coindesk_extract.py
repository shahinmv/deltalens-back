import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin
import json
from datetime import datetime
import sqlite3
from tqdm import tqdm
import re

max_pages = 20  # Global variable to control the maximum number of pages to scrape

class CoinDeskBTCScraper:
    def __init__(self):
        self.base_url = "https://www.coindesk.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_page_content(self, url):
        """Fetch page content with error handling"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def normalize_date(self, date_str):
        """Normalize date string to 'YYYY-MM-DD'. If relative (e.g., '11 hours ago'), use today's date."""
        months = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
            'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
        }
        # Try to match absolute date like 'Jul 8, 2025'
        abs_date_match = re.match(r"([A-Za-z]{3}) (\d{1,2}), (\d{4})", date_str)
        if abs_date_match:
            month = months.get(abs_date_match.group(1), '01')
            day = abs_date_match.group(2).zfill(2)
            year = abs_date_match.group(3)
            return f"{year}-{month}-{day}"
        # If it's relative (e.g., '11 hours ago', '2 days ago', 'just now', etc.)
        if (
            'ago' in date_str.lower() or
            'just now' in date_str.lower() or
            'minute' in date_str.lower() or
            'hour' in date_str.lower() or
            'day' in date_str.lower()
        ):
            return datetime.now().strftime('%Y-%m-%d')
        # If it's already in YYYY-MM-DD or similar, try to parse it
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return dt.strftime('%Y-%m-%d')
        except Exception:
            pass
        # Fallback: return as is
        return date_str
    
    def parse_headlines(self, html_content):
        """Parse headlines from HTML content - specifically from div.flex.flex-col.w-full.gap-4, extracting h2 (headline), p (description), and span (date) elements under each div."""
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        headlines = []
        
        # Target the specific div class and extract h2, p, and span
        target_divs = soup.find_all('div', class_=['flex', 'flex-col', 'gap-4'])
        
        if not target_divs:
            print("    No target divs found with class 'flex flex-col w-full gap-4'")
            return []
        
        # print(f"    Found {len(target_divs)} divs with class 'flex flex-col w-full gap-4'")
        
        for div_idx, div in enumerate(target_divs):
            # Find the first h2, p, and span with the correct classes directly under this div
            h2 = div.find('h2')
            description_p = div.find('p', class_='font-body text-charcoal-600 mb-4')
            date_span = div.find('span', class_='font-metadata text-color-charcoal-600 uppercase')

            headline_text = h2.get_text(strip=True) if h2 else None
            description = description_p.get_text(strip=True) if description_p else 'No description found'
            date = date_span.get_text(strip=True) if date_span else 'No date found'
            date = self.normalize_date(date)

            # Filter and add valid headlines
            if (headline_text and 
                len(headline_text) > 5 and 
                headline_text.lower() not in ['more', 'read more', 'continue reading', 'advertisement']):
                headline_data = {
                    'headline': headline_text,
                    'description': description,
                    'date': date,
                    'scraped_at': datetime.now().isoformat()
                }
                # Avoid duplicates based on headline text
                if not any(existing['headline'] == headline_text for existing in headlines):
                    headlines.append(headline_data)
        
        return headlines
    
    def scrape_bitcoin_news(self, latest_date=None):
        """Scrape Bitcoin news from all available pages, only including news with date >= latest_date (if provided)"""
        all_headlines = []
        page_num = 1
        consecutive_empty_pages = 0
        max_consecutive_empty = 3  # Stop after 3 consecutive empty pages
        global max_pages
        pbar = tqdm(total=max_pages, desc="Scraping pages", unit="page")
        total_news_found = 0
        while consecutive_empty_pages < max_consecutive_empty:
            if page_num == 1:
                url = "https://www.coindesk.com/tag/bitcoin"
            else:
                url = f"https://www.coindesk.com/tag/bitcoin/{page_num}"
            html_content = self.get_page_content(url)
            headlines = self.parse_headlines(html_content)
            # Filter by latest_date if provided
            if latest_date:
                headlines = [h for h in headlines if h['date'] >= latest_date]
            if headlines:
                all_headlines.extend(headlines)
                total_news_found += len(headlines)
                consecutive_empty_pages = 0
            else:
                consecutive_empty_pages += 1
                if page_num == 1:
                    print("WARNING: No headlines found on the first page. Check if the website structure changed.")
                    break
            page_num += 1
            pbar.update(1)
            pbar.set_postfix(news_found=total_news_found)
            time.sleep(2)
            if page_num > max_pages:
                print(f"Reached maximum page limit ({max_pages}). Stopping.")
                pbar.close()
                break
        pbar.close()
        print(f"\nScraping completed. Processed {page_num - 1} pages total.")
        print(f"Stopped after {consecutive_empty_pages} consecutive empty pages.")
        # Remove duplicates based on headline text
        unique_headlines = []
        seen_headlines = set()
        for headline in all_headlines:
            headline_text = headline['headline'].lower().strip()
            if headline_text not in seen_headlines:
                seen_headlines.add(headline_text)
                unique_headlines.append(headline)
        print(f"Total headlines before deduplication: {len(all_headlines)}")
        print(f"Unique headlines after deduplication: {len(unique_headlines)}")
        return unique_headlines
    
    def save_to_json(self, headlines, filename="btc_news.json"):
        """Save headlines to JSON file"""
        data = {
            'scraped_at': datetime.now().isoformat(),
            'total_headlines': len(headlines),
            'headlines': headlines
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(headlines)} headlines to {filename}")
    
    def print_headlines(self, headlines):
        """Print headlines in a formatted way (headline, description, date only)"""
        print(f"\n{'='*80}")
        print(f"BITCOIN NEWS HEADLINES - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"Total unique headlines found: {len(headlines)}\n")
        
        for i, headline in enumerate(headlines, 1):
            print(f"{i:2d}. Title: {headline['headline']}")
            print(f"    Description: {headline['description']}")
            print(f"    Date: {headline['date']}")
            print()
    
    def insert_headlines_to_db(self, headlines, db_path="../db.sqlite3"):
        """Insert headlines into the SQLite database, skipping duplicates by title."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        inserted = 0
        for headline in headlines:
            try:
                cursor.execute(
                    "INSERT OR IGNORE INTO news (title, description, date) VALUES (?, ?, ?)",
                    (headline.get('headline') or headline.get('title'), headline['description'], headline['date'])
                )
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                print(f"Error inserting headline: {headline.get('headline') or headline.get('title')}\n  {e}")
        conn.commit()
        conn.close()
        print(f"Inserted {inserted} new headlines into the database.")

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

def main():
    """Main function to run the scraper"""
    scraper = CoinDeskBTCScraper()
    print("Starting CoinDesk Bitcoin News Scraper...")
    print("=" * 50)
    # Get latest date from DB
    latest_date = get_latest_news_date()
    print(f"Latest news date in DB: {latest_date}")
    # Scrape headlines
    headlines = scraper.scrape_bitcoin_news(latest_date=latest_date)
    if headlines:
        scraper.save_to_json(headlines)
        scraper.insert_headlines_to_db(headlines)
        print(f"\nScraping completed successfully!")
        print(f"Found {len(headlines)} unique Bitcoin news headlines")
    else:
        print("No headlines found. The website structure might have changed.")

if __name__ == "__main__":
    main()