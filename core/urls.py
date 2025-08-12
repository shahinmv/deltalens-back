from django.urls import path
from . import views
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    # Existing API endpoints
    path('api/market-stats/', views.get_market_stats, name='market-stats'),
    path('api/news-sentiment/', views.get_today_news_sentiment, name='news-sentiment'),
    path('api/news/', views.get_news, name='news'),
    # path('api/trading-signal/', views.get_latest_trading_signal, name='trading-signal'),
    path('api/iterative-trading-signals/', views.get_iterative_trading_signals, name='iterative-trading-signals'),
    path('api/signal-performance/', views.get_signal_performance, name='signal-performance'),
    path('api/llm-stream/', views.stream_llm_response, name='llm-stream'),
    
    # Authentication endpoints
    path('api/auth/register/', views.UserRegistrationView.as_view(), name='user-register'),
    path('api/auth/login/', views.UserLoginView.as_view(), name='user-login'),
    path('api/auth/token/refresh/', TokenRefreshView.as_view(), name='token-refresh'),
    path('api/auth/profile/', views.UserProfileView.as_view(), name='user-profile'),
    
    # Admin endpoints
    path('api/admin/users/', views.UserListView.as_view(), name='user-list'),
    path('api/admin/users/<int:user_id>/role/', views.UpdateUserRoleView.as_view(), name='update-user-role'),
    
    # Conversation endpoints
    path('api/conversations/', views.ConversationSessionListView.as_view(), name='conversation-sessions'),
    path('api/conversations/<uuid:session_id>/', views.ConversationSessionDetailView.as_view(), name='conversation-detail'),
] 