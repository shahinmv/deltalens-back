import requests

response = requests.get('https://api.binance.com/api/v3/time')
print(response.json())