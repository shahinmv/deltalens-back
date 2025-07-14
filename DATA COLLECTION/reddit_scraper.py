#!/usr/bin/env python3
"""
reddit_full_year_scraper.py

Fetches **all** submissions and comments from given subreddits over the past 365 days
using PRAW (no Pushshift). Outputs a JSONL file with one record per line.

Usage:
    pip install praw python-dateutil tqdm
    export REDDIT_CLIENT_ID=···
    export REDDIT_CLIENT_SECRET=···
    export REDDIT_USER_AGENT="your_app_name by /u/your_username"
    python reddit_full_year_scraper.py \
        --subs bitcoin ethereum crypto_currency \
        --out reddit_year.jsonl \
        --window-days 30
"""

import os
import time
import argparse
import json
from datetime import datetime, timedelta
from dateutil import tz
import praw
from tqdm import tqdm

def init_reddit():
    try:
        reddit = praw.Reddit(
            client_id='yHf2LQhwL7mLzu7TfNVa_Q',
            client_secret='nOquQg8aJcnNpkJ9YbrhYXqYBIUAug',
            user_agent='shahin',
            ratelimit_seconds=60
        )
        # Test the connection
        print("Testing Reddit API connection...")
        reddit.user.me()
        print("Successfully connected to Reddit API")
        return reddit
    except Exception as e:
        print(f"Error connecting to Reddit API: {str(e)}")
        raise


def epoch(dt: datetime) -> int:
    return int(dt.replace(tzinfo=tz.UTC).timestamp())

def fetch_window(reddit, subreddit: str, start_ts: int, end_ts: int):
    """
    Yield submissions in [start_ts, end_ts), sorted by score (top posts).
    """
    query = f"timestamp:{start_ts}..{end_ts}"
    for submission in reddit.subreddit(subreddit).search(
        query,
        sort="top",          # <-- change here: fetch by top score
        syntax="cloudsearch",
        limit=None           # unlimited (up to Reddit cap)
    ):
        yield submission

def fetch_comments(submission):
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        yield comment

def serialize_submission(sub):
    return {
        "type": "submission",
        "subreddit": sub.subreddit.display_name,
        "id": sub.id,
        "created_utc": sub.created_utc,
        "author": str(sub.author),
        "title": sub.title,
        "selftext": sub.selftext or "",
        "score": sub.score,
        "num_comments": sub.num_comments,
        "url": sub.url,
    }

def serialize_comment(cm):
    return {
        "type": "comment",
        "subreddit": cm.subreddit.display_name,
        "id": cm.id,
        "parent_id": cm.parent_id,
        "created_utc": cm.created_utc,
        "author": str(cm.author),
        "body": cm.body,
        "score": cm.score,
    }

def scrape_subreddit(reddit, subreddit: str, out_f, window_days: int):
    end_dt   = datetime.now(tz=tz.UTC)
    start_dt = end_dt - timedelta(days=60)
    current  = start_dt

    print(f"\nScraping r/{subreddit} top posts from {start_dt.date()} to {end_dt.date()}")
    while current < end_dt:
        nxt = min(current + timedelta(days=window_days), end_dt)
        start_ts, end_ts = epoch(current), epoch(nxt)

        print(f"\n  Window: {current.date()} → {nxt.date()} [top posts]")
        for sub in tqdm(fetch_window(reddit, subreddit, start_ts, end_ts), desc="    submissions"):
            out_f.write(json.dumps(serialize_submission(sub)) + "\n")
            for cm in fetch_comments(sub):
                out_f.write(json.dumps(serialize_comment(cm)) + "\n")

        time.sleep(2)  # rate-limit guard
        current = nxt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subs",       nargs="+", required=True, help="Target subreddits")
    parser.add_argument("--out",        required=True,         help="Output JSONL file")
    parser.add_argument("--window-days",type=int, default=30,   help="Days per window")
    args = parser.parse_args()

    reddit = init_reddit()
    with open(args.out, "w", encoding="utf-8") as fout:
        for s in args.subs:
            scrape_subreddit(reddit, s, fout, args.window_days)
    print(f"\nDone! Data in {args.out}")

if __name__ == "__main__":
    main()
