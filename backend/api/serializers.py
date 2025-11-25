from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from .models import (
    TradingSettings, Trade, MLModel, MarketData, 
    EconomicEvent, TradingSession, PerformanceMetric
)

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'email', 'username', 'created_at']
        read_only_fields = ['id', 'created_at']


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, validators=[validate_password])
    password_confirm = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = ['email', 'username', 'password', 'password_confirm']
    
    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({"password": "Passwords don't match"})
        return attrs
    
    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User.objects.create_user(
            email=validated_data['email'],
            username=validated_data['username'],
            password=validated_data['password']
        )
        TradingSettings.objects.create(user=user)
        return user


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()


class TradingSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = TradingSettings
        fields = [
            'ai_enabled', 'initial_capital', 'current_capital',
            'forex_limit_percent', 'gold_limit_percent', 'last_ai_toggle'
        ]
        read_only_fields = ['last_ai_toggle']


class TradeSerializer(serializers.ModelSerializer):
    duration = serializers.SerializerMethodField()
    
    class Meta:
        model = Trade
        fields = [
            'id', 'pair', 'direction', 'entry_price', 'exit_price',
            'current_price', 'position_size', 'stop_loss', 'take_profit',
            'profit_loss', 'profit_loss_pips', 'status', 'outcome',
            'strategy_used', 'ml_prediction', 'ml_confidence',
            'technical_signals', 'capital_api_deal_id',
            'opened_at', 'closed_at', 'duration'
        ]
        read_only_fields = ['id', 'opened_at', 'closed_at']
    
    def get_duration(self, obj):
        if obj.closed_at and obj.opened_at:
            delta = obj.closed_at - obj.opened_at
            return str(delta)
        elif obj.opened_at:
            from django.utils import timezone
            delta = timezone.now() - obj.opened_at
            return str(delta)
        return None


class TradeCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Trade
        fields = [
            'pair', 'direction', 'entry_price', 'position_size',
            'stop_loss', 'take_profit', 'strategy_used'
        ]


class MLModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModel
        fields = [
            'id', 'model_version', 'accuracy', 'precision', 'recall',
            'f1_score', 'trades_trained_on', 'feature_importance',
            'training_metrics', 'is_active', 'trained_at'
        ]
        read_only_fields = ['id', 'trained_at']


class MarketDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = MarketData
        fields = [
            'id', 'pair', 'timeframe', 'open_price', 'high_price',
            'low_price', 'close_price', 'volume', 'timestamp'
        ]


class EconomicEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = EconomicEvent
        fields = [
            'id', 'currency', 'event_name', 'impact',
            'forecast', 'previous', 'actual', 'scheduled_at'
        ]


class TradingSessionSerializer(serializers.ModelSerializer):
    win_rate = serializers.SerializerMethodField()
    
    class Meta:
        model = TradingSession
        fields = [
            'id', 'started_at', 'ended_at', 'total_trades',
            'winning_trades', 'losing_trades', 'total_profit_loss',
            'is_active', 'win_rate'
        ]
    
    def get_win_rate(self, obj):
        if obj.total_trades > 0:
            return round((obj.winning_trades / obj.total_trades) * 100, 2)
        return 0


class PerformanceMetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = PerformanceMetric
        fields = [
            'id', 'date', 'total_trades', 'winning_trades',
            'losing_trades', 'win_rate', 'profit_loss',
            'best_trade', 'worst_trade', 'avg_trade_duration',
            'strategy_breakdown', 'pair_breakdown'
        ]


class DashboardSerializer(serializers.Serializer):
    account_balance = serializers.DecimalField(max_digits=15, decimal_places=2)
    available_capital = serializers.DecimalField(max_digits=15, decimal_places=2)
    total_profit_loss = serializers.DecimalField(max_digits=15, decimal_places=2)
    win_rate = serializers.FloatField()
    ai_status = serializers.BooleanField()
    total_trades = serializers.IntegerField()
    open_trades = serializers.IntegerField()
    ml_model_active = serializers.BooleanField()
    ml_accuracy = serializers.FloatField()
    trades_until_ml = serializers.IntegerField()
    trades_until_retrain = serializers.IntegerField()


class AnalyticsSerializer(serializers.Serializer):
    total_trades = serializers.IntegerField()
    winning_trades = serializers.IntegerField()
    losing_trades = serializers.IntegerField()
    win_rate = serializers.FloatField()
    total_profit_loss = serializers.DecimalField(max_digits=15, decimal_places=2)
    best_pair = serializers.CharField()
    worst_pair = serializers.CharField()
    ml_accuracy = serializers.FloatField()
    avg_trade_duration = serializers.CharField()
    strategy_performance = serializers.DictField()
    pair_performance = serializers.DictField()
    daily_performance = serializers.ListField()
    trade_distribution = serializers.DictField()
