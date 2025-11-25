from django.contrib import admin
from .models import (
    User, TradingSettings, Trade, MLModel, 
    MarketData, EconomicEvent, TradingSession, PerformanceMetric
)

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ['email', 'username', 'is_active', 'created_at']
    search_fields = ['email', 'username']

@admin.register(TradingSettings)
class TradingSettingsAdmin(admin.ModelAdmin):
    list_display = ['user', 'ai_enabled', 'current_capital']

@admin.register(Trade)
class TradeAdmin(admin.ModelAdmin):
    list_display = ['pair', 'direction', 'status', 'profit_loss', 'opened_at']
    list_filter = ['status', 'direction', 'pair']
    search_fields = ['pair']

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ['user', 'model_version', 'accuracy', 'is_active', 'trained_at']

@admin.register(EconomicEvent)
class EconomicEventAdmin(admin.ModelAdmin):
    list_display = ['currency', 'event_name', 'impact', 'scheduled_at']
    list_filter = ['currency', 'impact']
