import requests
import json
from datetime import datetime

def get_open_interest_history(symbol="BTCUSDT", period="5m", limit=30):
    """
    Fetch open interest history data from Binance API
    
    Args:
        symbol (str): Trading pair symbol (default: BTCUSDT)
        period (str): Time period for data points (default: 5m)
        limit (int): Number of data points to fetch (default: 30)
    
    Returns:
        dict: JSON response containing open interest history data
    """
    # Binance API endpoint for open interest history
    url = f"https://fapi.binance.com/futures/data/openInterestHist"
    
    # Parameters for the request
    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit
    }
    
    try:
        # Make the GET request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse and return the JSON response
        data = response.json()
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def main():
    # Get open interest history data
    data = get_open_interest_history()
    
    if data:
        # Print the JSON data in a formatted way
        print(json.dumps(data, indent=2))
        
        # Print a summary of the latest data point
        if data:
            latest = data[0]
            print("\nLatest Data Point:")
            print(f"Timestamp: {datetime.fromtimestamp(latest['timestamp']/1000)}")
            print(f"Sum Open Interest: {latest['sumOpenInterest']}")
            print(f"Sum Open Interest Value: {latest['sumOpenInterestValue']}")
    else:
        print("Failed to fetch open interest history data")

if __name__ == "__main__":
    main()
