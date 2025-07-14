import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_historical_oi(symbol, start_time, end_time, interval='5min'):
    """
    Fetch historical open interest data from CoinAnalyze API
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT_PERP.A')
        start_time (str): Start time in ISO format
        end_time (str): End time in ISO format
        interval (str): Time interval (default: '5min')
    
    Returns:
        pd.DataFrame: DataFrame containing open interest data with columns:
            - timestamp: datetime
            - open: float
            - high: float
            - low: float
            - close: float
    """
    # API key
    api_key = "7afaf567-924e-4876-9ee8-0590b35ab6cf"
    
    # API endpoint
    url = "https://api.coinalyze.net/v1/open-interest"
    
    # Request headers
    headers = {
        "api_key": api_key,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    # Request parameters
    params = {
        "symbols": symbol,
        "from": start_time,
        "to": end_time,
        "interval": interval
    }
    
    try:
        # Make API request
        response = requests.get(url, headers=headers, params=params)
        
        # Print response details for debugging
        print(f"Response Status Code: {response.status_code}")
        
        if response.status_code == 401:
            print("Authentication failed. Please check your API key.")
            print("Response content:", response.text)
            return None
            
        response.raise_for_status()  # Raise exception for other bad status codes
        
        # Parse response
        data = response.json()
        print(data)
        
        # Extract the history array from the response
        if not data or not isinstance(data, list) or len(data) == 0:
            print("No data received from API")
            return None
            
        # Get the first symbol's data (since we're querying one symbol)
        symbol_data = data[0]
        if 'history' not in symbol_data:
            print("No history data found in response")
            return None
            
        # Print first few raw timestamps for debugging
        print("\nFirst few raw timestamps:")
        for i, entry in enumerate(symbol_data['history'][:3]):
            print(f"Entry {i}: {entry['t']}")
            
        # Convert history array to DataFrame
        df = pd.DataFrame(symbol_data['history'], columns=['t', 'o', 'h', 'l', 'c'])
        
        # Rename columns to be more descriptive
        df.columns = ['timestamp', 'open', 'high', 'low', 'close']
        
        # Convert timestamp from seconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        if hasattr(e.response, 'text'):
            print(f"Error response: {e.response.text}")
        return None

if __name__ == "__main__":
    # Example usage
    symbol = "BTCUSDT_PERP.A"
    start_time = "2025-05-09T00:00:00Z"
    end_time = "2025-05-09T00:05:00Z"
    
    # Get historical open interest data
    df = get_historical_oi(symbol, start_time, end_time)
    
    if df is not None:
        print(f"\nSuccessfully fetched {len(df)} records")
        print("\nFirst few rows:")
        print(df.head())
        
        # Save to CSV
        output_file = f"historical_oi_{symbol.replace(',', '_')}_{start_time.split('T')[0]}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
