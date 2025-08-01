from django.urls import path
from . import views

urlpatterns = [
    path('api/market-stats/', views.get_market_stats, name='market-stats'),
    path('api/news-sentiment/', views.get_today_news_sentiment, name='news-sentiment'),
    path('api/news/', views.get_news, name='news'),
    path('api/trading-signal/', views.get_latest_trading_signal, name='trading-signal'),
    path('api/llm-stream/', views.stream_llm_response, name='llm-stream'),
] 