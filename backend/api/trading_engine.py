import logging
from decimal import Decimal
from typing import Dict, Optional
from django.utils import timezone
import pandas as pd

from .models import Trade, TradingSettings, TradingSession
from .capital_api import CapitalComAPI
from .technical_analysis import TechnicalAnalysis, analyze_multi_timeframe, Signal
from .ml_engine import MLTradingEngine

logger = logging.getLogger(__name__)


class TradingEngine:
    TRADING_PAIRS = [
        'GBP/USD', 'EUR/USD', 'USD/JPY', 'AUD/USD',
        'USD/CAD', 'NZD/USD', 'USD/CHF', 'XAU/USD'
    ]
    
    TIMEFRAMES = ['15m', '30m', '1H', '4H']
    
    def __init__(self, user):
        self.user = user
        self.settings = TradingSettings.objects.get_or_create(user=user)[0]
        self.api = CapitalComAPI()
        self.ml_engine = MLTradingEngine(user)
        self.authenticated = False
    
    def initialize(self) -> bool:
        self.authenticated = self.api.authenticate()
        if not self.authenticated:
            logger.warning("Failed to authenticate with Capital.com API")
        return self.authenticated
    
    def get_capital_limit(self, pair: str) -> Decimal:
        if pair == 'XAU/USD':
            limit_percent = self.settings.gold_limit_percent
        else:
            limit_percent = self.settings.forex_limit_percent
        
        return (self.settings.current_capital * limit_percent) / 100
    
    def get_allocated_capital(self, pair: str) -> Decimal:
        open_trades = Trade.objects.filter(
            user=self.user,
            pair=pair,
            status='open'
        )
        return sum(trade.position_size for trade in open_trades)
    
    def get_available_capital(self, pair: str) -> Decimal:
        limit = self.get_capital_limit(pair)
        allocated = self.get_allocated_capital(pair)
        return max(Decimal('0'), limit - allocated)
    
    def calculate_position_size(self, pair: str, entry_price: float, stop_loss: float) -> Decimal:
        available = self.get_available_capital(pair)
        
        if available <= 0:
            return Decimal('0')
        
        risk_per_trade = available * Decimal('0.02')
        
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            return Decimal('0')
        
        position_size = risk_per_trade / Decimal(str(stop_distance))
        
        position_size = min(position_size, available)
        
        return position_size.quantize(Decimal('0.01'))
    
    def get_market_data(self, pair: str) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        
        for timeframe in self.TIMEFRAMES:
            candles = self.api.get_historical_prices(pair, timeframe)
            if candles:
                df = pd.DataFrame(candles)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                data_dict[timeframe] = df
        
        return data_dict
    
    def analyze_pair(self, pair: str) -> Optional[Dict]:
        if not self.authenticated:
            return None
        
        data_dict = self.get_market_data(pair)
        
        if not data_dict or len(data_dict) < 2:
            logger.warning(f"Insufficient data for {pair}")
            return None
        
        analysis = analyze_multi_timeframe(data_dict)
        
        closed_trades = Trade.objects.filter(user=self.user, status='closed').count()
        
        if closed_trades >= 30:
            if self.ml_engine.should_retrain():
                logger.info(f"Retraining ML model for user {self.user.email}")
                self.ml_engine.train()
            
            entry_result = analysis.get('timeframe_results', {}).get('15m', {})
            indicators = entry_result.get('indicators', {})
            
            ml_result = self.ml_engine.get_combined_signal(
                {'signal': analysis['entry_signal'].value, 'confidence': analysis['confidence']},
                indicators
            )
            
            analysis['ml_prediction'] = ml_result['ml_prediction']
            analysis['ml_confidence'] = ml_result['ml_confidence']
            analysis['combined_signal'] = ml_result['signal']
            analysis['combined_confidence'] = ml_result['confidence']
            analysis['signals_aligned'] = ml_result['aligned']
        else:
            analysis['ml_prediction'] = None
            analysis['ml_confidence'] = None
            analysis['combined_signal'] = analysis['entry_signal'].value
            analysis['combined_confidence'] = analysis['confidence']
            analysis['signals_aligned'] = True
        
        return analysis
    
    def should_trade(self, analysis: Dict, pair: str) -> bool:
        if not analysis:
            return False
        
        available = self.get_available_capital(pair)
        if available <= 0:
            logger.info(f"No available capital for {pair}")
            return False
        
        signal = analysis.get('combined_signal', 0)
        confidence = analysis.get('combined_confidence', 0)
        
        if signal == 0:
            return False
        
        if confidence < 0.6:
            return False
        
        if not analysis.get('aligned', True):
            return False
        
        open_trades = Trade.objects.filter(
            user=self.user,
            pair=pair,
            status='open'
        ).count()
        
        if open_trades >= 2:
            return False
        
        return True
    
    def execute_trade(self, pair: str, analysis: Dict) -> Optional[Trade]:
        signal = analysis.get('combined_signal', 0)
        direction = 'buy' if signal > 0 else 'sell'
        
        entry_result = analysis.get('timeframe_results', {}).get('15m', {})
        entry_price = entry_result.get('entry_price', 0)
        stop_loss = entry_result.get('stop_loss', 0)
        take_profit = entry_result.get('take_profit', 0)
        
        if not entry_price:
            return None
        
        position_size = self.calculate_position_size(pair, entry_price, stop_loss)
        
        if position_size <= 0:
            return None
        
        result = None
        if self.authenticated:
            result = self.api.open_position(
                pair=pair,
                direction=direction,
                size=float(position_size),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        
        strategy_parts = []
        if analysis.get('ml_prediction') is not None:
            strategy_parts.append('ML')
        
        ta_strategies = entry_result.get('strategies', {})
        for strategy, signal_name in ta_strategies.items():
            if signal_name in ['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL']:
                strategy_parts.append(strategy)
        
        strategy_used = ' + '.join(strategy_parts[:3]) if strategy_parts else 'Technical'
        
        trade = Trade.objects.create(
            user=self.user,
            pair=pair,
            direction=direction,
            entry_price=Decimal(str(entry_price)),
            position_size=position_size,
            stop_loss=Decimal(str(stop_loss)) if stop_loss else None,
            take_profit=Decimal(str(take_profit)) if take_profit else None,
            status='open',
            strategy_used=strategy_used,
            ml_prediction=analysis.get('ml_prediction'),
            ml_confidence=analysis.get('ml_confidence'),
            technical_signals={
                'indicators': entry_result.get('indicators', {}),
                'strategies': ta_strategies,
                'trend': analysis.get('trend', Signal.NEUTRAL).name if hasattr(analysis.get('trend'), 'name') else str(analysis.get('trend')),
                'confidence': analysis.get('combined_confidence', 0),
            },
            capital_api_deal_id=result.get('dealId', '') if result else '',
        )
        
        session = TradingSession.objects.filter(user=self.user, is_active=True).first()
        if session:
            session.total_trades += 1
            session.save()
        
        logger.info(f"Opened trade: {pair} {direction} at {entry_price}")
        
        return trade
    
    def update_open_trades(self):
        if not self.authenticated:
            return
        
        positions = self.api.get_open_positions()
        if not positions:
            return
        
        position_map = {p['dealId']: p for p in positions}
        
        open_trades = Trade.objects.filter(user=self.user, status='open')
        
        for trade in open_trades:
            if trade.capital_api_deal_id and trade.capital_api_deal_id in position_map:
                pos = position_map[trade.capital_api_deal_id]
                trade.current_price = Decimal(str(pos['currentLevel']))
                trade.profit_loss = Decimal(str(pos['profitLoss']))
                trade.save()
    
    def check_and_close_trades(self):
        if not self.authenticated:
            return
        
        open_trades = Trade.objects.filter(user=self.user, status='open')
        
        for trade in open_trades:
            if not trade.capital_api_deal_id:
                continue
            
            if trade.current_price:
                should_close = False
                
                if trade.direction == 'buy':
                    if trade.stop_loss and trade.current_price <= trade.stop_loss:
                        should_close = True
                    elif trade.take_profit and trade.current_price >= trade.take_profit:
                        should_close = True
                else:
                    if trade.stop_loss and trade.current_price >= trade.stop_loss:
                        should_close = True
                    elif trade.take_profit and trade.current_price <= trade.take_profit:
                        should_close = True
                
                if should_close:
                    result = self.api.close_position(trade.capital_api_deal_id)
                    if result:
                        trade.close_trade(trade.current_price)
                        self._update_session_stats(trade)
    
    def _update_session_stats(self, trade: Trade):
        session = TradingSession.objects.filter(user=self.user, is_active=True).first()
        if session:
            if trade.outcome == 'win':
                session.winning_trades += 1
            else:
                session.losing_trades += 1
            session.total_profit_loss += trade.profit_loss
            session.save()
        
        self.settings.current_capital += trade.profit_loss
        self.settings.save()
    
    def run_trading_cycle(self):
        if not self.settings.ai_enabled:
            return
        
        if not self.authenticated:
            self.initialize()
        
        self.update_open_trades()
        self.check_and_close_trades()
        
        for pair in self.TRADING_PAIRS:
            try:
                analysis = self.analyze_pair(pair)
                
                if self.should_trade(analysis, pair):
                    self.execute_trade(pair, analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing {pair}: {str(e)}")
                continue
