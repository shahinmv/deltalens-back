from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import HttpResponseRedirect, JsonResponse, StreamingHttpResponse
import requests
import time
import sqlite3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from .llm_stream import StreamingQwenAgent
from langchain_core.messages import HumanMessage
import json
from django.views.decorators.csrf import csrf_exempt

# Create your views here.


def get_market_stats(request):
    def get_bitcoin_marketcap_dominance():
        MAX_RETRIES = 3
        RETRY_DELAY = 3  # seconds
        def coingecko_get(url):
            for attempt in range(MAX_RETRIES):
                try:
                    response = requests.get(url)
                    if response.status_code == 429:
                        print(f"CoinGecko rate limit hit, waiting {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
                        continue
                    response.raise_for_status()
                    return response
                except requests.exceptions.HTTPError as e:
                    if response.status_code == 429 and attempt < MAX_RETRIES - 1:
                        print(f"CoinGecko rate limit hit, waiting {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        print(f"HTTP error from CoinGecko: {e}")
                        return None
                except Exception as e:
                    print(f"Error fetching data from CoinGecko: {e}")
                    return None
            return None
        try:
            url = "https://api.coingecko.com/api/v3/global"
            try:
                response = coingecko_get(url)
                if response is None:
                    return None
                data = response.json()
            except Exception as e:
                print(f"Error fetching global data from CoinGecko: {e}")
                return None
            btc_url = "https://api.coingecko.com/api/v3/coins/bitcoin"
            try:
                btc_response = coingecko_get(btc_url)
                if btc_response is None:
                    return None
                btc_response.raise_for_status()
                btc_data = btc_response.json()
            except Exception as e:
                print(f"Error fetching BTC data from CoinGecko: {e}")
                return None
            try:
                market_data = btc_data["market_data"]
            except Exception as e:
                print(f"Error extracting market_data from BTC data: {e}")
                return None
            headers = {
                'X-CMC_PRO_API_KEY': 'ac4bc071-2576-46de-8aab-718cb1d02fb6',  # Replace with your actual API key
                'Accept': 'application/json'
            }
            global_url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
            try:
                global_response = requests.get(global_url, headers=headers)
                global_response.raise_for_status()
                global_data = global_response.json()
            except Exception as e:
                print(f"Error fetching global metrics from CoinMarketCap: {e}")
                return None
            try:
                global_info = global_data['data']
            except Exception as e:
                print(f"Error extracting 'data' from CoinMarketCap response: {e}")
                return None
            try:
                return {
                    "market_cap": market_data["market_cap"]["usd"],
                    "market_cap_change_24h": market_data["market_cap_change_24h"],
                    "market_cap_change_percentage_24h": market_data["market_cap_change_percentage_24h"],
                    "btc_dominance": global_info['btc_dominance'],
                    "btc_dominance_change_24h": global_info['btc_dominance_24h_percentage_change']
                }
            except Exception as e:
                print(f"Error extracting fields for return value: {e}")
                return None
        except Exception as e:
            print(f"Unexpected error in get_bitcoin_marketcap_dominance: {e}")
            return None

    def get_bitcoin_data_binance():
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            params = {"symbol": "BTCUSDT"}
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                print(f"Error fetching BTC data from Binance: {e}")
                return None
            try:
                return {
                    "price": float(data["lastPrice"]),
                    "volume_24h": float(data["volume"]),
                    "volume_24h_usdt": float(data["quoteVolume"]),
                    "price_change_24h": float(data["priceChange"]),
                    "price_change_percent_24h": float(data["priceChangePercent"])
                }
            except Exception as e:
                print(f"Error extracting fields from Binance data: {e}")
                return None
        except Exception as e:
            print(f"Unexpected error in get_bitcoin_data_binance: {e}")
            return None

    try:
        market_data = get_bitcoin_marketcap_dominance()
        binance_data = get_bitcoin_data_binance()
    except Exception as e:
        print(f"Error calling data functions: {e}")
        return JsonResponse({"error": "Failed to fetch market data"}, status=500)

    if not market_data or not binance_data:
        return JsonResponse({"error": "Failed to fetch market data"}, status=500)

    try:
        response = {
            "market_cap": market_data["market_cap"],
            "market_cap_change_24h": market_data["market_cap_change_24h"],
            "market_cap_change_percentage_24h": market_data["market_cap_change_percentage_24h"],
            "btc_dominance": market_data["btc_dominance"],
            "btc_dominance_change_24h": market_data["btc_dominance_change_24h"],
            "volume_24h": binance_data["volume_24h_usdt"],
            "volume_24h_change": binance_data["price_change_percent_24h"],
        }
    except Exception as e:
        print(f"Error constructing response: {e}")
        return JsonResponse({"error": "Failed to construct response"}, status=500)
    return JsonResponse(response)


def get_today_news_sentiment(request):
    db_path = 'db.sqlite3'
    today = datetime.now().strftime('%Y-%m-%d')
    analyzer = SentimentIntensityAnalyzer()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT title, description FROM news WHERE date = ?", (today,))
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return JsonResponse({"sentiment": None, "message": "No news for today."})
        sentiments = []
        for title, description in rows:
            text = (title or '') + ' ' + (description or '')
            score = analyzer.polarity_scores(text)
            sentiments.append(score['compound'])
        avg_sentiment = sum(sentiments) / len(sentiments)
        return JsonResponse({"sentiment": avg_sentiment})
    except Exception as e:
        print(f"Error in get_today_news_sentiment: {e}")
        return JsonResponse({"error": "Failed to fetch sentiment"}, status=500)


def get_news(request):
    db_path = 'db.sqlite3'
    page = int(request.GET.get('page', 1))
    page_size = int(request.GET.get('page_size', 3))
    offset = (page - 1) * page_size
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM news")
        total_count = cursor.fetchone()[0]
        cursor.execute("SELECT title, description, date FROM news ORDER BY date DESC LIMIT ? OFFSET ?", (page_size, offset))
        rows = cursor.fetchall()
        conn.close()
        news_list = [
            {"title": title, "description": description, "date": date}
            for title, description, date in rows
        ]
        return JsonResponse({
            "news": news_list,
            "total": total_count,
            "page": page,
            "page_size": page_size
        })
    except Exception as e:
        print(f"Error in get_news: {e}")
        return JsonResponse({"error": "Failed to fetch news"}, status=500)


@csrf_exempt
def stream_llm_response(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        data = json.loads(request.body.decode())
        user_message = data.get("message", "")
        if not user_message:
            return JsonResponse({"error": "No message provided"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Invalid JSON: {str(e)}"}, status=400)

    agent = StreamingQwenAgent()
    messages = [HumanMessage(content=user_message)]
    
    def token_stream():
        for token in agent.stream_chat(messages):
            if isinstance(token, tuple):
                # For tool responses, you may want to format differently
                # if token[0] == "tool_name":
                #     yield f"[TOOL_NAME] {token[1]}\n"
                if token[0] == "ai_intermediate":
                    yield token[1]
            else:
                yield token
    
    return StreamingHttpResponse(token_stream(), content_type="text/plain")


