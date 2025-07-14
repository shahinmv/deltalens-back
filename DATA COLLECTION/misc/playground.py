import requests
from datetime import datetime
import pandas as pd

def get_specific_btc_data(target_datetime):
    """
    Fetch specific BTCUSDT data point from Binance API
    
    Args:
        target_datetime (str): Target datetime in format 'YYYY-MM-DD HH:MM:SS'
    
    Returns:
        dict: Dictionary containing the requested data
    """
    # Convert target datetime to milliseconds timestamp
    dt = datetime.strptime(target_datetime, '%Y-%m-%d %H:%M:%S')
    timestamp = int(dt.timestamp() * 1000)
    
    # Binance API endpoint
    url = 'https://api.binance.com/api/v3/klines'
    
    # Parameters for the request
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1s',
        'startTime': timestamp,
        'endTime': timestamp + 1000,  # Add 1 second to get the specific candle
        'limit': 100
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        print(data)
        if not data:
            return None
            
        # Extract the data
        candle = data[0]
        result = {
            'datetime': float(candle[0]),
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low': float(candle[3]),
            'close': float(candle[4]),
            'volume': float(candle[5])
        }
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    # Get data for the specific datetime
    target_time = "2025-04-25 17:08:58"
    data = get_specific_btc_data(target_time)
    
    if data:
        print("\nBTCUSDT Data for", target_time)
        print("--------------------------------")
        print(f"Datetime: {data['datetime']}")
        print(f"Open: {data['open']}")
        print(f"High: {data['high']}")
        print(f"Low: {data['low']}")
        print(f"Close: {data['close']}")
        print(f"Volume: {data['volume']}")
    else:
        print("No data found for the specified time")