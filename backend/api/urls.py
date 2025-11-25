from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from . import views

urlpatterns = [
    path('auth/register/', views.register, name='register'),
    path('auth/login/', views.login, name='login'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/me/', views.get_current_user, name='current_user'),
    
    path('status/', views.check_api_status, name='api_status'),
    path('dashboard/', views.get_dashboard, name='dashboard'),
    path('analytics/', views.get_analytics, name='analytics'),
    
    path('settings/', views.get_settings, name='get_settings'),
    path('settings/update/', views.update_settings, name='update_settings'),
    path('settings/toggle-ai/', views.toggle_ai, name='toggle_ai'),
    
    path('trades/', views.get_trades, name='trades'),
    path('trades/open/', views.get_open_trades, name='open_trades'),
    path('trades/history/', views.get_trade_history, name='trade_history'),
    path('trades/<int:trade_id>/', views.get_trade_detail, name='trade_detail'),
    
    path('market/prices/', views.get_market_prices, name='market_prices'),
    path('market/data/<str:pair>/<str:timeframe>/', views.get_market_data, name='market_data'),
    
    path('ml/status/', views.get_ml_status, name='ml_status'),
    path('ml/models/', views.get_ml_models, name='ml_models'),
    
    path('events/', views.get_economic_events, name='economic_events'),
    
    path('account/', views.get_account_info, name='account_info'),
]
