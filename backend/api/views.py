from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.utils import timezone
from django.db.models import Sum, Count, Avg, Q
from django.db.models.functions import TruncDate
from datetime import timedelta
from decimal import Decimal

from .models import (
    User, TradingSettings, Trade, MLModel, 
    MarketData, EconomicEvent, TradingSession, PerformanceMetric, CacheEntry
)
from .serializers import (
    UserSerializer, RegisterSerializer, LoginSerializer,
    TradingSettingsSerializer, TradeSerializer, MLModelSerializer,
    MarketDataSerializer, EconomicEventSerializer, DashboardSerializer,
    AnalyticsSerializer
)
from .capital_api import CapitalComAPI
from .trading_engine import TradingEngine


@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    serializer = RegisterSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        return Response({
            'user': UserSerializer(user).data,
            'access': str(refresh.access_token),
            'refresh': str(refresh),
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    serializer = LoginSerializer(data=request.data)
    if serializer.is_valid():
        user = authenticate(
            username=serializer.validated_data['email'],
            password=serializer.validated_data['password']
        )
        if user:
            refresh = RefreshToken.for_user(user)
            return Response({
                'user': UserSerializer(user).data,
                'access': str(refresh.access_token),
                'refresh': str(refresh),
            })
        return Response(
            {'error': 'Invalid credentials'},
            status=status.HTTP_401_UNAUTHORIZED
        )
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_current_user(request):
    return Response(UserSerializer(request.user).data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_dashboard(request):
    user = request.user
    settings = TradingSettings.objects.get_or_create(user=user)[0]
    
    total_trades = Trade.objects.filter(user=user).count()
    closed_trades = Trade.objects.filter(user=user, status='closed')
    open_trades = Trade.objects.filter(user=user, status='open').count()
    winning_trades = closed_trades.filter(outcome='win').count()
    
    total_pnl = closed_trades.aggregate(
        total=Sum('profit_loss')
    )['total'] or Decimal('0')
    
    win_rate = 0
    if closed_trades.count() > 0:
        win_rate = (winning_trades / closed_trades.count()) * 100
    
    ml_model = MLModel.objects.filter(user=user, is_active=True).first()
    ml_accuracy = ml_model.accuracy if ml_model else 0
    ml_active = ml_model is not None and closed_trades.count() >= 30
    
    trades_until_ml = max(0, 30 - closed_trades.count())
    
    trades_since_last_train = 0
    if ml_model:
        trades_since_last_train = closed_trades.count() - ml_model.trades_trained_on
    trades_until_retrain = max(0, 10 - trades_since_last_train) if ml_active else 0
    
    account_balance = Decimal('0')
    available_capital = Decimal('0')
    api_connected = False
    
    api = CapitalComAPI()
    if api.authenticate():
        api_connected = True
        account_info = api.get_account_info()
        if account_info:
            account_balance = Decimal(str(account_info.get('balance', 0)))
            available_capital = Decimal(str(account_info.get('available', 0)))
            
            settings.current_capital = account_balance
            settings.save()
    
    if not api_connected:
        allocated_capital = Trade.objects.filter(
            user=user, status='open'
        ).aggregate(total=Sum('position_size'))['total'] or Decimal('0')
        account_balance = settings.current_capital
        available_capital = settings.current_capital - allocated_capital
    
    data = {
        'account_balance': account_balance,
        'available_capital': available_capital,
        'total_profit_loss': total_pnl,
        'win_rate': round(win_rate, 2),
        'ai_status': settings.ai_enabled,
        'total_trades': total_trades,
        'open_trades': open_trades,
        'ml_model_active': ml_active,
        'ml_accuracy': round(ml_accuracy * 100, 2),
        'trades_until_ml': trades_until_ml,
        'trades_until_retrain': trades_until_retrain,
        'api_connected': api_connected,
    }
    
    return Response(data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_analytics(request):
    user = request.user
    
    all_trades = Trade.objects.filter(user=user)
    closed_trades = all_trades.filter(status='closed')
    
    total_trades = all_trades.count()
    winning_trades = closed_trades.filter(outcome='win').count()
    losing_trades = closed_trades.filter(outcome='loss').count()
    
    win_rate = 0
    if closed_trades.count() > 0:
        win_rate = (winning_trades / closed_trades.count()) * 100
    
    total_pnl = closed_trades.aggregate(
        total=Sum('profit_loss')
    )['total'] or Decimal('0')
    
    pair_stats = closed_trades.values('pair').annotate(
        count=Count('id'),
        wins=Count('id', filter=Q(outcome='win')),
        total_pnl=Sum('profit_loss')
    ).order_by('-total_pnl')
    
    best_pair = pair_stats.first()['pair'] if pair_stats else 'N/A'
    worst_pair = pair_stats.last()['pair'] if pair_stats else 'N/A'
    
    pair_performance = {}
    for stat in pair_stats:
        pair_performance[stat['pair']] = {
            'trades': stat['count'],
            'wins': stat['wins'],
            'win_rate': round((stat['wins'] / stat['count']) * 100, 2) if stat['count'] > 0 else 0,
            'profit_loss': float(stat['total_pnl'] or 0)
        }
    
    strategy_stats = closed_trades.values('strategy_used').annotate(
        count=Count('id'),
        wins=Count('id', filter=Q(outcome='win')),
        total_pnl=Sum('profit_loss')
    )
    
    strategy_performance = {}
    for stat in strategy_stats:
        strategy_name = stat['strategy_used'] or 'Unknown'
        strategy_performance[strategy_name] = {
            'trades': stat['count'],
            'wins': stat['wins'],
            'win_rate': round((stat['wins'] / stat['count']) * 100, 2) if stat['count'] > 0 else 0,
            'profit_loss': float(stat['total_pnl'] or 0)
        }
    
    daily_stats = closed_trades.annotate(
        date=TruncDate('closed_at')
    ).values('date').annotate(
        trades=Count('id'),
        wins=Count('id', filter=Q(outcome='win')),
        pnl=Sum('profit_loss')
    ).order_by('date')[:30]
    
    daily_performance = [
        {
            'date': str(stat['date']),
            'trades': stat['trades'],
            'wins': stat['wins'],
            'profit_loss': float(stat['pnl'] or 0)
        }
        for stat in daily_stats
    ]
    
    ml_model = MLModel.objects.filter(user=user, is_active=True).first()
    ml_accuracy = ml_model.accuracy if ml_model else 0
    
    avg_duration = None
    durations = []
    for trade in closed_trades:
        if trade.closed_at and trade.opened_at:
            durations.append((trade.closed_at - trade.opened_at).total_seconds())
    if durations:
        avg_seconds = sum(durations) / len(durations)
        hours = int(avg_seconds // 3600)
        minutes = int((avg_seconds % 3600) // 60)
        avg_duration = f"{hours}h {minutes}m"
    else:
        avg_duration = "N/A"
    
    trade_distribution = {
        'buy': closed_trades.filter(direction='buy').count(),
        'sell': closed_trades.filter(direction='sell').count(),
    }
    
    data = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': round(win_rate, 2),
        'total_profit_loss': total_pnl,
        'best_pair': best_pair,
        'worst_pair': worst_pair,
        'ml_accuracy': round(ml_accuracy * 100, 2),
        'avg_trade_duration': avg_duration,
        'strategy_performance': strategy_performance,
        'pair_performance': pair_performance,
        'daily_performance': daily_performance,
        'trade_distribution': trade_distribution,
    }
    
    return Response(data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_settings(request):
    settings = TradingSettings.objects.get_or_create(user=request.user)[0]
    return Response(TradingSettingsSerializer(settings).data)


@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_settings(request):
    settings = TradingSettings.objects.get_or_create(user=request.user)[0]
    serializer = TradingSettingsSerializer(settings, data=request.data, partial=True)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([AllowAny])
def check_api_status(request):
    api = CapitalComAPI()
    api_configured = all([api.api_key, api.password, api.identifier])
    api_connected = False
    account_info = None
    
    if api_configured:
        api_connected = api.authenticate()
        if api_connected:
            account_info = api.get_account_info()
    
    return Response({
        'api_configured': api_configured,
        'api_connected': api_connected,
        'account_info': account_info,
        'missing_credentials': [] if api_configured else [
            'CAPITAL_COM_API_KEY' if not api.api_key else None,
            'CAPITAL_COM_PASSWORD' if not api.password else None,
            'CAPITAL_COM_IDENTIFIER' if not api.identifier else None,
        ]
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def toggle_ai(request):
    api = CapitalComAPI()
    api_configured = all([api.api_key, api.password, api.identifier])
    
    settings = TradingSettings.objects.get_or_create(user=request.user)[0]
    
    if not settings.ai_enabled:
        if not api_configured:
            return Response({
                'error': 'Capital.com API credentials not configured',
                'message': 'Please provide CAPITAL_COM_API_KEY, CAPITAL_COM_PASSWORD, and CAPITAL_COM_IDENTIFIER',
                'ai_enabled': False
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if not api.authenticate():
            return Response({
                'error': 'Failed to connect to Capital.com API',
                'message': 'Please verify your API credentials are correct',
                'ai_enabled': False
            }, status=status.HTTP_400_BAD_REQUEST)
    
    was_enabled = settings.ai_enabled
    settings.ai_enabled = not settings.ai_enabled
    settings.last_ai_toggle = timezone.now()
    settings.save()
    
    closed_positions = 0
    total_pnl = Decimal('0')
    if was_enabled and not settings.ai_enabled:
        authenticated = api.authenticate()
        positions_map = {}
        if authenticated:
            positions = api.get_open_positions()
            if positions:
                positions_map = {p['dealId']: p for p in positions}
        
        open_trades = Trade.objects.filter(user=request.user, status='open')
        for trade in open_trades:
            current_price = trade.entry_price
            
            if trade.capital_api_deal_id and trade.capital_api_deal_id in positions_map:
                pos = positions_map[trade.capital_api_deal_id]
                current_price = Decimal(str(pos.get('currentLevel', trade.entry_price)))
            
            if trade.capital_api_deal_id and authenticated:
                api.close_position(trade.capital_api_deal_id)
            
            trade.close_trade(current_price)
            total_pnl += trade.profit_loss
            closed_positions += 1
        
        settings.current_capital += total_pnl
        settings.save()
    
    if settings.ai_enabled:
        TradingSession.objects.filter(user=request.user, is_active=True).update(
            is_active=False, ended_at=timezone.now()
        )
        TradingSession.objects.create(user=request.user)
    else:
        TradingSession.objects.filter(user=request.user, is_active=True).update(
            is_active=False, ended_at=timezone.now()
        )
    
    message = 'AI trading started' if settings.ai_enabled else 'AI trading stopped'
    if closed_positions > 0:
        message += f' - Closed {closed_positions} open position(s)'
    
    return Response({
        'ai_enabled': settings.ai_enabled,
        'message': message,
        'closed_positions': closed_positions,
        'closed': closed_positions,
        'total_pnl': float(total_pnl)
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_trades(request):
    trades = Trade.objects.filter(user=request.user)
    serializer = TradeSerializer(trades, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_open_trades(request):
    trades = Trade.objects.filter(user=request.user, status='open')
    serializer = TradeSerializer(trades, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_trade_history(request):
    limit = int(request.query_params.get('limit', 50))
    trades = Trade.objects.filter(
        user=request.user, status='closed'
    ).order_by('-closed_at')[:limit]
    serializer = TradeSerializer(trades, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_trade_detail(request, trade_id):
    try:
        trade = Trade.objects.get(id=trade_id, user=request.user)
        serializer = TradeSerializer(trade)
        return Response(serializer.data)
    except Trade.DoesNotExist:
        return Response(
            {'error': 'Trade not found'},
            status=status.HTTP_404_NOT_FOUND
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def close_position(request, trade_id):
    try:
        trade = Trade.objects.get(id=trade_id, user=request.user, status='open')
    except Trade.DoesNotExist:
        return Response(
            {'error': 'Trade not found or already closed', 'closed': 0, 'total_pnl': 0},
            status=status.HTTP_404_NOT_FOUND
        )
    
    api = CapitalComAPI()
    current_price = trade.entry_price
    authenticated = api.authenticate()
    
    if trade.capital_api_deal_id and authenticated:
        positions = api.get_open_positions()
        if positions:
            for pos in positions:
                if pos.get('dealId') == trade.capital_api_deal_id:
                    current_price = Decimal(str(pos.get('currentLevel', trade.entry_price)))
                    break
        api.close_position(trade.capital_api_deal_id)
    
    trade.close_trade(current_price)
    
    settings = TradingSettings.objects.get_or_create(user=request.user)[0]
    settings.current_capital += trade.profit_loss
    settings.save()
    
    return Response({
        'message': 'Position closed successfully',
        'trade_id': trade.id,
        'pair': trade.pair,
        'profit_loss': float(trade.profit_loss),
        'outcome': trade.outcome,
        'closed': 1,
        'total_pnl': float(trade.profit_loss)
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def close_all_profitable(request):
    open_trades = list(Trade.objects.filter(user=request.user, status='open'))
    
    api = CapitalComAPI()
    authenticated = api.authenticate()
    
    positions_map = {}
    if authenticated:
        positions = api.get_open_positions()
        if positions:
            positions_map = {p['dealId']: p for p in positions}
    
    for trade in open_trades:
        if trade.capital_api_deal_id and trade.capital_api_deal_id in positions_map:
            pos = positions_map[trade.capital_api_deal_id]
            trade.current_price = Decimal(str(pos.get('currentLevel', trade.entry_price)))
            trade.profit_loss = Decimal(str(pos.get('profitLoss', 0)))
            trade.save()
    
    profitable_trades = [t for t in open_trades if t.profit_loss and t.profit_loss > 0]
    
    if not profitable_trades:
        return Response({
            'message': 'No profitable trades to close',
            'closed': 0,
            'total_profit': 0,
            'total_pnl': 0
        })
    
    closed_count = 0
    total_profit = Decimal('0')
    
    for trade in profitable_trades:
        current_price = trade.current_price or trade.entry_price
        
        if trade.capital_api_deal_id and authenticated:
            api.close_position(trade.capital_api_deal_id)
        
        trade.close_trade(current_price)
        total_profit += trade.profit_loss
        closed_count += 1
    
    settings = TradingSettings.objects.get_or_create(user=request.user)[0]
    settings.current_capital += total_profit
    settings.save()
    
    return Response({
        'message': f'Successfully closed {closed_count} profitable trade(s)',
        'closed': closed_count,
        'total_profit': float(total_profit),
        'total_pnl': float(total_profit)
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def close_all_positions(request):
    open_trades = list(Trade.objects.filter(user=request.user, status='open'))
    
    if not open_trades:
        return Response({
            'message': 'No open positions to close',
            'closed': 0,
            'total_pnl': 0,
            'total_profit': 0
        })
    
    api = CapitalComAPI()
    authenticated = api.authenticate()
    
    positions_map = {}
    if authenticated:
        positions = api.get_open_positions()
        if positions:
            positions_map = {p['dealId']: p for p in positions}
    
    closed_count = 0
    total_pnl = Decimal('0')
    
    for trade in open_trades:
        current_price = trade.entry_price
        
        if trade.capital_api_deal_id and trade.capital_api_deal_id in positions_map:
            pos = positions_map[trade.capital_api_deal_id]
            current_price = Decimal(str(pos.get('currentLevel', trade.entry_price)))
        
        if trade.capital_api_deal_id and authenticated:
            api.close_position(trade.capital_api_deal_id)
        
        trade.close_trade(current_price)
        total_pnl += trade.profit_loss
        closed_count += 1
    
    settings = TradingSettings.objects.get_or_create(user=request.user)[0]
    settings.current_capital += total_pnl
    settings.save()
    
    return Response({
        'message': f'Successfully closed {closed_count} position(s)',
        'closed': closed_count,
        'total_pnl': float(total_pnl),
        'total_profit': float(total_pnl)
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_market_prices(request):
    cache_key = 'market_prices'
    cached = CacheEntry.get(cache_key)
    if cached:
        return Response(cached)
    
    api = CapitalComAPI()
    if api.authenticate():
        prices = api.get_prices()
        if prices:
            CacheEntry.set(cache_key, prices, timeout=60)
            return Response(prices)
    
    return Response({'error': 'Unable to fetch market prices'}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_market_data(request, pair, timeframe):
    cache_key = f'market_data_{pair}_{timeframe}'
    cached = CacheEntry.get(cache_key)
    if cached:
        return Response(cached)
    
    api = CapitalComAPI()
    if api.authenticate():
        data = api.get_historical_prices(pair, timeframe)
        if data:
            CacheEntry.set(cache_key, data, timeout=60)
            return Response(data)
    
    return Response({'error': 'Unable to fetch market data'}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_ml_status(request):
    user = request.user
    closed_trades = Trade.objects.filter(user=user, status='closed').count()
    ml_model = MLModel.objects.filter(user=user, is_active=True).first()
    
    data = {
        'ml_enabled': closed_trades >= 30 and ml_model is not None,
        'total_closed_trades': closed_trades,
        'trades_until_ml': max(0, 30 - closed_trades),
        'model': MLModelSerializer(ml_model).data if ml_model else None,
        'last_trained': ml_model.trained_at.isoformat() if ml_model else None,
        'trades_until_retrain': 0,
    }
    
    if ml_model:
        trades_since = closed_trades - ml_model.trades_trained_on
        data['trades_until_retrain'] = max(0, 10 - trades_since)
    
    return Response(data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_ml_models(request):
    models = MLModel.objects.filter(user=request.user)
    serializer = MLModelSerializer(models, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_economic_events(request):
    cache_key = 'economic_events'
    cached = CacheEntry.get(cache_key)
    if cached:
        return Response(cached)
    
    events = EconomicEvent.objects.filter(
        currency__in=['USD', 'GBP'],
        scheduled_at__gte=timezone.now(),
        scheduled_at__lte=timezone.now() + timedelta(days=7)
    ).order_by('scheduled_at')[:20]
    
    serializer = EconomicEventSerializer(events, many=True)
    CacheEntry.set(cache_key, serializer.data, timeout=300)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_account_info(request):
    cache_key = f'account_info_{request.user.id}'
    cached = CacheEntry.get(cache_key)
    if cached:
        return Response(cached)
    
    api = CapitalComAPI()
    if api.authenticate():
        account = api.get_account_info()
        if account:
            CacheEntry.set(cache_key, account, timeout=60)
            return Response(account)
    
    settings = TradingSettings.objects.get_or_create(user=request.user)[0]
    return Response({
        'balance': float(settings.current_capital),
        'available': float(settings.current_capital),
        'deposit': float(settings.initial_capital),
        'profitLoss': float(settings.current_capital - settings.initial_capital),
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def run_trading_cycle(request):
    settings = TradingSettings.objects.get_or_create(user=request.user)[0]
    
    if not settings.ai_enabled:
        return Response({
            'success': False,
            'message': 'AI trading is not enabled. Turn on AI first.',
            'trades_executed': 0
        })
    
    try:
        engine = TradingEngine(request.user)
        if not engine.initialize():
            return Response({
                'success': False,
                'message': 'Failed to connect to Capital.com API',
                'trades_executed': 0
            })
        
        result = engine.run_trading_cycle()
        
        return Response({
            'success': True,
            'message': 'Trading cycle completed',
            'result': result
        })
    except Exception as e:
        return Response({
            'success': False,
            'message': f'Error running trading cycle: {str(e)}',
            'trades_executed': 0
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
