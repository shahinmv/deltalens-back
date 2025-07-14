import sys
import os
sys.path.append(os.path.dirname(__file__))

from coindesk_extract import CoinDeskBTCScraper, get_latest_news_date
from cointelegraph_extract import get_all_news, insert_news_to_db

def run_coindesk(latest_date):
    print("\n=== Running CoinDesk Extractor ===")
    scraper = CoinDeskBTCScraper()
    headlines = scraper.scrape_bitcoin_news(latest_date=latest_date)
    print(f"CoinDesk: {len(headlines)} new headlines found.")
    if headlines:
        scraper.save_to_json(headlines, filename="btc_news_coindesk.json")
        scraper.insert_headlines_to_db(headlines)
    else:
        print("No new CoinDesk headlines to insert.")

def run_cointelegraph(latest_date):
    print("\n=== Running CoinTelegraph Extractor ===")
    news = get_all_news(latest_date=latest_date)
    print(f"CoinTelegraph: {len(news)} new news items found.")
    if news:
        import json
        with open("btc_news_cointelegraph.json", "w", encoding="utf-8") as f:
            json.dump(news, f, indent=2, ensure_ascii=False)
        insert_news_to_db(news)
    else:
        print("No new CoinTelegraph news to insert.")

def main():
    latest_date = get_latest_news_date()
    print(f"Latest news date in DB: {latest_date}")
    run_coindesk(latest_date)
    run_cointelegraph(latest_date)
    print("\nAll news extraction complete.")

if __name__ == "__main__":
    main() 