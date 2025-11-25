from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
import json


class User(AbstractUser):
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']
    
    def __str__(self):
        return self.email


class TradingSettings(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='trading_settings')
    ai_enabled = models.BooleanField(default=False)
    initial_capital = models.DecimalField(max_digits=15, decimal_places=2, default=1000.00)
    current_capital = models.DecimalField(max_digits=15, decimal_places=2, default=1000.00)
    forex_limit_percent = models.DecimalField(max_digits=5, decimal_places=2, default=20.00)
    gold_limit_percent = models.DecimalField(max_digits=5, decimal_places=2, default=15.00)
    last_ai_toggle = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"Settings for {self.user.email}"


class Trade(models.Model):
    STATUS_CHOICES = [
        ('open', 'Open'),
        ('closed', 'Closed'),
        ('pending', 'Pending'),
    ]
    
    DIRECTION_CHOICES = [
        ('buy', 'Buy'),
        ('sell', 'Sell'),
    ]
    
    OUTCOME_CHOICES = [
        ('win', 'Win'),
        ('loss', 'Loss'),
        ('pending', 'Pending'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='trades')
    pair = models.CharField(max_length=20)
    direction = models.CharField(max_length=10, choices=DIRECTION_CHOICES)
    entry_price = models.DecimalField(max_digits=15, decimal_places=5)
    exit_price = models.DecimalField(max_digits=15, decimal_places=5, null=True, blank=True)
    current_price = models.DecimalField(max_digits=15, decimal_places=5, null=True, blank=True)
    position_size = models.DecimalField(max_digits=15, decimal_places=2)
    stop_loss = models.DecimalField(max_digits=15, decimal_places=5, null=True, blank=True)
    take_profit = models.DecimalField(max_digits=15, decimal_places=5, null=True, blank=True)
    profit_loss = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    profit_loss_pips = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='open')
    outcome = models.CharField(max_length=10, choices=OUTCOME_CHOICES, default='pending')
    strategy_used = models.CharField(max_length=100, blank=True)
    ml_prediction = models.FloatField(null=True, blank=True)
    ml_confidence = models.FloatField(null=True, blank=True)
    technical_signals = models.JSONField(default=dict, blank=True)
    capital_api_deal_id = models.CharField(max_length=100, blank=True)
    opened_at = models.DateTimeField(auto_now_add=True)
    closed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-opened_at']
    
    def __str__(self):
        return f"{self.pair} {self.direction} - {self.status}"
    
    def close_trade(self, exit_price, actual_profit_loss=None):
        from decimal import Decimal
        self.exit_price = exit_price
        self.status = 'closed'
        self.closed_at = timezone.now()
        
        if actual_profit_loss is not None:
            self.profit_loss = Decimal(str(actual_profit_loss))
        else:
            exit_decimal = Decimal(str(exit_price))
            
            if self.direction == 'buy':
                raw_pnl = (exit_decimal - self.entry_price) * self.position_size
            else:
                raw_pnl = (self.entry_price - exit_decimal) * self.position_size
            
            if self.pair.startswith('USD/'):
                self.profit_loss = raw_pnl / exit_decimal
            else:
                self.profit_loss = raw_pnl
        
        self.outcome = 'win' if self.profit_loss > 0 else 'loss'
        self.save()


class MLModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ml_models')
    pair = models.CharField(max_length=20, default='ALL')
    model_version = models.IntegerField(default=1)
    model_data = models.BinaryField(null=True, blank=True)
    accuracy = models.FloatField(default=0)
    precision = models.FloatField(default=0)
    recall = models.FloatField(default=0)
    f1_score = models.FloatField(default=0)
    trades_trained_on = models.IntegerField(default=0)
    feature_importance = models.JSONField(default=dict, blank=True)
    training_metrics = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    trained_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-trained_at']
    
    def __str__(self):
        return f"Model v{self.model_version} for {self.user.email}"


class MarketData(models.Model):
    pair = models.CharField(max_length=20)
    timeframe = models.CharField(max_length=10)
    open_price = models.DecimalField(max_digits=15, decimal_places=5)
    high_price = models.DecimalField(max_digits=15, decimal_places=5)
    low_price = models.DecimalField(max_digits=15, decimal_places=5)
    close_price = models.DecimalField(max_digits=15, decimal_places=5)
    volume = models.DecimalField(max_digits=20, decimal_places=2, default=0)
    timestamp = models.DateTimeField()
    
    class Meta:
        ordering = ['-timestamp']
        unique_together = ['pair', 'timeframe', 'timestamp']
        indexes = [
            models.Index(fields=['pair', 'timeframe', 'timestamp']),
        ]
    
    def __str__(self):
        return f"{self.pair} {self.timeframe} @ {self.timestamp}"


class TechnicalIndicator(models.Model):
    market_data = models.ForeignKey(MarketData, on_delete=models.CASCADE, related_name='indicators')
    indicator_name = models.CharField(max_length=50)
    value = models.FloatField()
    
    class Meta:
        unique_together = ['market_data', 'indicator_name']


class EconomicEvent(models.Model):
    IMPACT_CHOICES = [
        ('high', 'High'),
        ('medium', 'Medium'),
        ('low', 'Low'),
    ]
    
    currency = models.CharField(max_length=3)
    event_name = models.CharField(max_length=200)
    impact = models.CharField(max_length=10, choices=IMPACT_CHOICES)
    forecast = models.CharField(max_length=50, blank=True)
    previous = models.CharField(max_length=50, blank=True)
    actual = models.CharField(max_length=50, blank=True)
    scheduled_at = models.DateTimeField()
    is_notified = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['scheduled_at']
    
    def __str__(self):
        return f"{self.currency} - {self.event_name}"


class TradingSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    total_trades = models.IntegerField(default=0)
    winning_trades = models.IntegerField(default=0)
    losing_trades = models.IntegerField(default=0)
    total_profit_loss = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"Session for {self.user.email} at {self.started_at}"


class PerformanceMetric(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='metrics')
    date = models.DateField()
    total_trades = models.IntegerField(default=0)
    winning_trades = models.IntegerField(default=0)
    losing_trades = models.IntegerField(default=0)
    win_rate = models.FloatField(default=0)
    profit_loss = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    best_trade = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    worst_trade = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    avg_trade_duration = models.DurationField(null=True, blank=True)
    strategy_breakdown = models.JSONField(default=dict, blank=True)
    pair_breakdown = models.JSONField(default=dict, blank=True)
    
    class Meta:
        unique_together = ['user', 'date']
        ordering = ['-date']
    
    def __str__(self):
        return f"Metrics for {self.user.email} on {self.date}"


class CacheEntry(models.Model):
    key = models.CharField(max_length=255, unique=True)
    value = models.JSONField()
    expires_at = models.DateTimeField()
    
    class Meta:
        indexes = [
            models.Index(fields=['key']),
            models.Index(fields=['expires_at']),
        ]
    
    @classmethod
    def get(cls, key):
        try:
            entry = cls.objects.get(key=key)
            if entry.expires_at > timezone.now():
                return entry.value
            entry.delete()
        except cls.DoesNotExist:
            pass
        return None
    
    @classmethod
    def set(cls, key, value, timeout=60):
        expires_at = timezone.now() + timezone.timedelta(seconds=timeout)
        cls.objects.update_or_create(
            key=key,
            defaults={'value': value, 'expires_at': expires_at}
        )
