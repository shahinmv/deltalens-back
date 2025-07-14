import requests

def get_btc_usd_price():
    url = 'https://api.kraken.com/0/public/Ticker?pair=XBTUSD'
    response = requests.get(url)
    data = response.json()
    print(data)

    if response.status_code == 200 and 'result' in data:
        result = data['result']
        pair = list(result.keys())[0]  # typically "XXBTZUSD"
        price = result[pair]['c'][0]   # 'c' is the last trade closed price
        return float(price)
    else:
        raise Exception("Failed to fetch price data")

price = get_btc_usd_price()
print(f"Current BTC/USD price: ${price}")
