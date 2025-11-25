import logging
from decimal import Decimal
from typing import Dict, Optional
from django.utils import timezone
import pandas as pd

from .models import Trade, TradingSettings, TradingSession
from .capital_api import CapitalComAPI
from .technical_analysis import TechnicalAnalysis, analyze_multi_timeframe, Signal
from .ml_engine import MLTradingEngine, MIN_TRADES_PER_PAIR

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Trading Engine with Bootstrap Mode and ML Mode
    
    Bootstrap Mode (0-30 trades):
    Uses comprehensive technical analysis including:
    - Price Action (Supply/Demand, Candlesticks, Market Structure)
    - SMC/MSC (Liquidity Grabs, BOS, FVG, Order Blocks, Premium/Discount)
    - Trend-Following (MA Crossover, Breakouts, RSI+Trend)
    - Range-Bound (Bollinger Bounce, RSI Range)
    
    Multi-timeframe approach:
    - Market Structure (Trend) -> 4H
    - Zones (Supply & Demand) -> 4H and 1H
    - Key Levels (Support/Resistance) -> 1H
    - Entry -> 15m and 30m
    
    ML Mode (30+ trades):
    Combines TA signals with ML predictions
    """
    
    TRADING_PAIRS = [
        'GBP/USD', 'EUR/USD', 'USD/JPY', 'AUD/USD',
        'USD/CAD', 'NZD/USD', 'USD/CHF', 'XAU/USD'
    ]
    
    TIMEFRAMES = ['15m', '30m', '1H', '4H']
    
    BOOTSTRAP_TRADE_THRESHOLD = 30  # Legacy - now using per-pair threshold
    MIN_TRADES_PER_PAIR_THRESHOLD = MIN_TRADES_PER_PAIR
    
    MIN_POSITION_SIZES = {
        'GBP/USD': 500,
        'EUR/USD': 500,
        'USD/JPY': 500,
        'AUD/USD': 500,
        'USD/CAD': 500,
        'NZD/USD': 500,
        'USD/CHF': 500,
        'XAU/USD': 0.5,
    }
    
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
    
    def get_trade_count(self) -> int:
        return Trade.objects.filter(user=self.user, status='closed').count()
    
    def get_pair_trade_count(self, pair: str) -> int:
        return Trade.objects.filter(user=self.user, pair=pair, status='closed').count()
    
    def is_bootstrap_mode(self) -> bool:
        return self.get_trade_count() < self.BOOTSTRAP_TRADE_THRESHOLD
    
    def is_pair_ml_ready(self, pair: str) -> bool:
        return self.ml_engine.has_model_for_pair(pair)
    
    def train_pair_if_ready(self, pair: str):
        pair_count = self.get_pair_trade_count(pair)
        if pair_count >= self.MIN_TRADES_PER_PAIR_THRESHOLD:
            if self.ml_engine.should_retrain_pair(pair):
                logger.info(f"Training ML model for {pair} (has {pair_count} closed trades)")
                self.ml_engine.train_pair(pair)
    
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
        
        min_size = Decimal(str(self.MIN_POSITION_SIZES.get(pair, 500)))
        
        leverage = Decimal('30')
        
        if pair == 'XAU/USD':
            margin_for_min = (min_size * Decimal(str(entry_price))) / leverage
        else:
            margin_for_min = min_size / leverage
        
        if margin_for_min > available:
            logger.warning(f"  {pair}: Insufficient margin for minimum position (need ${margin_for_min:.2f}, have ${available:.2f})")
            return Decimal('0')
        
        risk_per_trade = available * Decimal('0.05')
        
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            stop_distance = entry_price * 0.01
        
        pip_value = Decimal('1') if 'JPY' in pair else Decimal('0.0001')
        
        if pair == 'XAU/USD':
            risk_based_size = risk_per_trade / Decimal(str(stop_distance))
            position_size = max(risk_based_size, min_size)
            position_size = position_size.quantize(Decimal('0.01'))
        else:
            risk_based_size = risk_per_trade / Decimal(str(stop_distance))
            position_size = max(risk_based_size, min_size)
            position_size = max(position_size, Decimal('500'))
            position_size = position_size.quantize(Decimal('1'))
        
        if pair == 'XAU/USD':
            margin_required = (position_size * Decimal(str(entry_price))) / leverage
        else:
            margin_required = position_size / leverage
        
        max_margin = available * Decimal('0.5')
        if margin_required > max_margin:
            if pair == 'XAU/USD':
                position_size = (max_margin * leverage) / Decimal(str(entry_price))
                position_size = position_size.quantize(Decimal('0.01'))
            else:
                position_size = max_margin * leverage
                position_size = position_size.quantize(Decimal('1'))
            
            if position_size < min_size:
                logger.warning(f"  {pair}: Position size {position_size} below minimum {min_size}")
                return Decimal('0')
        
        logger.info(f"  {pair}: Position size = {position_size} units (margin ~${margin_required:.2f}, min={min_size})")
        return position_size
    
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
    
    def analyze_pair_bootstrap(self, pair: str, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Bootstrap mode analysis using comprehensive TA strategies
        
        Strategy Categories:
        1. SMC/MSC - Can trade independently (Order Blocks, FVG, Liquidity Grabs, BOS, Premium/Discount)
        2. Price Action - Can trade independently (Supply/Demand, Candlesticks, Market Structure)
        3. Trend-Following - Requires 2 aligned strategies (MA Crossover, Breakout, RSI+Trend)
        4. Range-Bound - Requires 2 aligned strategies (Bollinger Bounce, RSI Range)
        
        Multi-timeframe approach:
        - 4H: Market structure and trend identification
        - 4H + 1H: Supply/Demand zones
        - 1H: Key support/resistance levels
        - 15m + 30m: Entry signals
        """
        logger.info(f"Analyzing {pair} - Available timeframes: {list(data_dict.keys())}")
        for tf, df in data_dict.items():
            logger.info(f"  {tf}: {len(df)} candles")
        
        analysis = analyze_multi_timeframe(data_dict)
        
        trade_decision = {
            'should_trade': False,
            'direction': None,
            'confidence': 0,
            'reasons': [],
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'strategies_used': [],
            'strategy_category': None,
        }
        
        trend = analysis.get('trend', Signal.NEUTRAL)
        trend_confidence = analysis.get('trend_confidence', 0)
        structure = analysis.get('structure', {})
        entry_signal = analysis.get('entry_signal', Signal.NEUTRAL)
        entry_result = analysis.get('entry_result')
        aligned = analysis.get('aligned', False)
        confidence = analysis.get('confidence', 0)
        
        logger.info(f"  {pair} Analysis: trend={trend.name if hasattr(trend, 'name') else trend}, entry={entry_signal.name if hasattr(entry_signal, 'name') else entry_signal}, aligned={aligned}, confidence={confidence:.2f}")
        
        if trend == Signal.NEUTRAL:
            aligned = True
            logger.info(f"  {pair}: Trend is neutral, allowing entry signal to drive decision")
        
        if not aligned:
            trade_decision['reasons'].append('Timeframes not aligned')
            logger.info(f"  {pair}: Skipping - timeframes not aligned")
            return {**analysis, 'trade_decision': trade_decision}
        
        min_confidence = 0.50
        if confidence < min_confidence:
            trade_decision['reasons'].append(f'Confidence {confidence:.2f} < {min_confidence}')
            logger.info(f"  {pair}: Skipping - confidence too low ({confidence:.2f} < {min_confidence})")
            return {**analysis, 'trade_decision': trade_decision}
        
        smc_strategies = []
        price_action_strategies = []
        trend_following_strategies = []
        range_bound_strategies = []
        
        tf_strategies = {}
        if entry_result:
            tf_strategies = analysis.get('timeframe_results', {}).get('15m', {}).get('strategies', {})
            if not tf_strategies:
                tf_strategies = analysis.get('timeframe_results', {}).get('30m', {}).get('strategies', {})
        
        smc_details = {}
        if entry_result:
            smc_details = entry_result.zones.get('smc_details', {})
        
        if smc_details.get('order_block'):
            smc_strategies.append(f"order_block_{smc_details['order_block']}")
        if smc_details.get('fvg'):
            smc_strategies.append(f"fvg_{smc_details['fvg']}")
        if smc_details.get('liquidity'):
            smc_strategies.append(f"liquidity_{smc_details['liquidity']}")
        if structure and structure.get('bos'):
            smc_strategies.append(f"bos_{structure['bos'].get('type', 'unknown')}")
        
        fib_zones = analysis.get('fib_zones', {})
        if fib_zones:
            current_zone = fib_zones.get('current_zone', 'neutral')
            if current_zone == 'discount' and entry_signal in [Signal.BUY, Signal.STRONG_BUY]:
                smc_strategies.append('premium_discount_buy_in_discount')
            elif current_zone == 'premium' and entry_signal in [Signal.SELL, Signal.STRONG_SELL]:
                smc_strategies.append('premium_discount_sell_in_premium')
        
        zones = analysis.get('zones_4h', {}) or analysis.get('zones_1h', {})
        current_price = entry_result.entry_price if entry_result else 0
        if zones and current_price:
            for demand in zones.get('demand', []):
                if demand.get('price_low', 0) <= current_price <= demand.get('price_high', 0):
                    price_action_strategies.append('supply_demand_in_demand_zone')
                    break
            for supply in zones.get('supply', []):
                if supply.get('price_low', 0) <= current_price <= supply.get('price_high', 0):
                    price_action_strategies.append('supply_demand_in_supply_zone')
                    break
        
        if tf_strategies.get('candlestick') in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
            price_action_strategies.append(f"candlestick_{tf_strategies['candlestick'].lower()}")
        
        if structure:
            if structure.get('hh') and structure.get('hl'):
                price_action_strategies.append('market_structure_uptrend_hh_hl')
            elif structure.get('lh') and structure.get('ll'):
                price_action_strategies.append('market_structure_downtrend_lh_ll')
        
        if tf_strategies.get('ma_crossover') in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
            trend_following_strategies.append(f"ma_crossover_{tf_strategies['ma_crossover'].lower()}")
        if tf_strategies.get('breakout') in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
            trend_following_strategies.append(f"breakout_{tf_strategies['breakout'].lower()}")
        if tf_strategies.get('rsi_trend') in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
            trend_following_strategies.append(f"rsi_trend_{tf_strategies['rsi_trend'].lower()}")
        if tf_strategies.get('macd') in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
            trend_following_strategies.append(f"macd_{tf_strategies['macd'].lower()}")
        
        if tf_strategies.get('bollinger') in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
            range_bound_strategies.append(f"bollinger_{tf_strategies['bollinger'].lower()}")
        if tf_strategies.get('rsi_range') in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
            range_bound_strategies.append(f"rsi_range_{tf_strategies['rsi_range'].lower()}")
        
        should_trade = False
        strategy_category = None
        strategies_used = []
        reasons = []
        
        logger.info(f"  {pair} Strategies: SMC={smc_strategies}, PA={price_action_strategies}, TF={trend_following_strategies}, RB={range_bound_strategies}")
        
        if smc_strategies:
            should_trade = True
            strategy_category = 'SMC'
            strategies_used = smc_strategies
            reasons.append(f"SMC signal: {smc_strategies[0]}")
        
        elif price_action_strategies:
            should_trade = True
            strategy_category = 'Price Action'
            strategies_used = price_action_strategies
            reasons.append(f"Price Action signal: {price_action_strategies[0]}")
        
        elif len(trend_following_strategies) >= 2:
            should_trade = True
            strategy_category = 'Trend-Following'
            strategies_used = trend_following_strategies
            reasons.append(f"Trend-Following aligned: {', '.join(trend_following_strategies[:2])}")
        
        elif len(range_bound_strategies) >= 2:
            should_trade = True
            strategy_category = 'Range-Bound'
            strategies_used = range_bound_strategies
            reasons.append(f"Range-Bound aligned: {', '.join(range_bound_strategies[:2])}")
        
        elif len(trend_following_strategies) >= 1 and len(range_bound_strategies) >= 1:
            should_trade = True
            strategy_category = 'Mixed'
            strategies_used = trend_following_strategies + range_bound_strategies
            reasons.append(f"Mixed strategies aligned: {trend_following_strategies[0]} + {range_bound_strategies[0]}")
        
        elif len(trend_following_strategies) >= 1 and entry_signal in [Signal.BUY, Signal.STRONG_BUY, Signal.SELL, Signal.STRONG_SELL]:
            should_trade = True
            strategy_category = 'Trend-Following-Single'
            strategies_used = trend_following_strategies
            reasons.append(f"Single trend-following with strong signal: {trend_following_strategies[0]}")
        
        elif len(range_bound_strategies) >= 1 and entry_signal in [Signal.BUY, Signal.STRONG_BUY, Signal.SELL, Signal.STRONG_SELL]:
            should_trade = True
            strategy_category = 'Range-Bound-Single'
            strategies_used = range_bound_strategies
            reasons.append(f"Single range-bound with strong signal: {range_bound_strategies[0]}")
        
        elif entry_signal in [Signal.BUY, Signal.STRONG_BUY, Signal.SELL, Signal.STRONG_SELL] and confidence >= 0.3:
            should_trade = True
            strategy_category = 'Signal-Confidence'
            strategies_used = ['high_confidence_signal']
            reasons.append(f"Strong {entry_signal.name} signal with confidence {confidence:.2f}")
        
        else:
            if trend_following_strategies:
                reasons.append(f"Trend-Following needs 2 aligned, only have: {trend_following_strategies}")
            if range_bound_strategies:
                reasons.append(f"Range-Bound needs 2 aligned, only have: {range_bound_strategies}")
            if not (smc_strategies or price_action_strategies or trend_following_strategies or range_bound_strategies):
                reasons.append("No strategy signals detected")
        
        if should_trade:
            if entry_signal in [Signal.BUY, Signal.STRONG_BUY]:
                trade_decision['direction'] = 'buy'
            elif entry_signal in [Signal.SELL, Signal.STRONG_SELL]:
                trade_decision['direction'] = 'sell'
            else:
                should_trade = False
                reasons.append("No clear directional signal")
        
        trade_decision['should_trade'] = should_trade
        trade_decision['strategy_category'] = strategy_category
        trade_decision['strategies_used'] = strategies_used[:5]
        trade_decision['confidence'] = analysis.get('confidence', 0)
        trade_decision['reasons'] = reasons
        
        if entry_result and should_trade:
            trade_decision['entry_price'] = entry_result.entry_price
            trade_decision['stop_loss'] = entry_result.stop_loss
            trade_decision['take_profit'] = entry_result.take_profit
        
        return {**analysis, 'trade_decision': trade_decision}
    
    def analyze_pair_ml(self, pair: str, data_dict: Dict[str, pd.DataFrame], ta_analysis: Dict) -> Dict:
        """
        ML mode analysis combining TA with ML predictions (per-pair)
        """
        self.train_pair_if_ready(pair)
        
        entry_result = ta_analysis.get('timeframe_results', {}).get('15m', {})
        if not entry_result:
            entry_result = ta_analysis.get('timeframe_results', {}).get('30m', {})
        
        indicators = entry_result.get('indicators', {})
        ta_signal = ta_analysis.get('entry_signal', Signal.NEUTRAL)
        ta_confidence = ta_analysis.get('confidence', 0)
        
        ml_result = self.ml_engine.get_combined_signal(
            {'signal': ta_signal.value if hasattr(ta_signal, 'value') else ta_signal, 
             'confidence': ta_confidence},
            indicators,
            pair=pair
        )
        
        ta_analysis['ml_prediction'] = ml_result['ml_prediction']
        ta_analysis['ml_confidence'] = ml_result['ml_confidence']
        ta_analysis['combined_signal'] = ml_result['signal']
        ta_analysis['combined_confidence'] = ml_result['confidence']
        ta_analysis['signals_aligned'] = ml_result['aligned']
        
        trade_decision = ta_analysis.get('trade_decision', {})
        
        if ml_result['aligned'] and ml_result['confidence'] >= 0.6:
            trade_decision['should_trade'] = True
            trade_decision['direction'] = 'buy' if ml_result['signal'] > 0 else 'sell'
            trade_decision['confidence'] = ml_result['confidence']
            trade_decision['strategies_used'].append('ml_prediction')
            trade_decision['reasons'].append('ML and TA signals aligned')
        elif not ml_result['aligned']:
            trade_decision['should_trade'] = False
            trade_decision['reasons'].append('ML and TA signals conflict')
        
        ta_analysis['trade_decision'] = trade_decision
        return ta_analysis
    
    def analyze_pair(self, pair: str) -> Optional[Dict]:
        if not self.authenticated:
            return None
        
        data_dict = self.get_market_data(pair)
        
        if not data_dict or len(data_dict) < 2:
            logger.warning(f"Insufficient data for {pair}")
            return None
        
        pair_ml_ready = self.is_pair_ml_ready(pair)
        pair_trade_count = self.get_pair_trade_count(pair)
        
        analysis = self.analyze_pair_bootstrap(pair, data_dict)
        
        if pair_ml_ready:
            analysis = self.analyze_pair_ml(pair, data_dict, analysis)
            analysis['mode'] = 'ml'
        else:
            analysis['ml_prediction'] = None
            analysis['ml_confidence'] = None
            analysis['combined_signal'] = analysis.get('entry_signal', Signal.NEUTRAL)
            if hasattr(analysis['combined_signal'], 'value'):
                analysis['combined_signal'] = analysis['combined_signal'].value
            analysis['combined_confidence'] = analysis.get('confidence', 0)
            analysis['signals_aligned'] = True
            analysis['mode'] = 'bootstrap'
            analysis['pair_trades_until_ml'] = max(0, self.MIN_TRADES_PER_PAIR_THRESHOLD - pair_trade_count)
        
        return analysis
    
    def should_trade(self, analysis: Dict, pair: str) -> bool:
        if not analysis:
            logger.info(f"  {pair}: No analysis available")
            return False
        
        trade_decision = analysis.get('trade_decision', {})
        if not trade_decision.get('should_trade', False):
            logger.info(f"  {pair}: trade_decision.should_trade is False - reasons: {trade_decision.get('reasons', [])}")
            return False
        
        available = self.get_available_capital(pair)
        if available <= 0:
            logger.info(f"  {pair}: No available capital")
            return False
        
        confidence = trade_decision.get('confidence', 0)
        min_trade_confidence = 0.50
        if confidence < min_trade_confidence:
            logger.info(f"  {pair}: Confidence {confidence:.2f} < {min_trade_confidence}")
            return False
        
        open_trades = Trade.objects.filter(
            user=self.user,
            pair=pair,
            status='open'
        ).count()
        
        if open_trades >= 2:
            logger.info(f"  {pair}: Already has {open_trades} open trades")
            return False
        
        logger.info(f"  {pair}: TRADE APPROVED - confidence={confidence:.2f}, direction={trade_decision.get('direction')}")
        return True
    
    def execute_trade(self, pair: str, analysis: Dict) -> Optional[Trade]:
        trade_decision = analysis.get('trade_decision', {})
        direction = trade_decision.get('direction', 'buy')
        
        entry_price = trade_decision.get('entry_price', 0)
        stop_loss = trade_decision.get('stop_loss', 0)
        take_profit = trade_decision.get('take_profit', 0)
        
        if not entry_price:
            return None
        
        position_size = self.calculate_position_size(pair, entry_price, stop_loss)
        
        if position_size <= 0:
            return None
        
        result = None
        deal_id = ''
        if self.authenticated:
            result = self.api.open_position(
                pair=pair,
                direction=direction,
                size=float(position_size),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            if result and result.get('dealId'):
                deal_id = str(result.get('dealId'))
                logger.info(f"  {pair}: API returned deal_id={deal_id}")
            else:
                logger.warning(f"  {pair}: API call failed or no deal_id returned")
        
        strategies_used = trade_decision.get('strategies_used', [])
        strategy_str = ' + '.join(strategies_used[:3]) if strategies_used else 'Technical'
        
        if analysis.get('ml_prediction') is not None:
            strategy_str = 'ML + ' + strategy_str
        
        entry_result = analysis.get('timeframe_results', {}).get('15m', {})
        if not entry_result:
            entry_result = analysis.get('timeframe_results', {}).get('30m', {})
        
        trade = Trade.objects.create(
            user=self.user,
            pair=pair,
            direction=direction,
            entry_price=Decimal(str(entry_price)),
            position_size=position_size,
            stop_loss=Decimal(str(stop_loss)) if stop_loss else None,
            take_profit=Decimal(str(take_profit)) if take_profit else None,
            status='open',
            strategy_used=strategy_str,
            ml_prediction=analysis.get('ml_prediction'),
            ml_confidence=analysis.get('ml_confidence'),
            technical_signals={
                'indicators': entry_result.get('indicators', {}),
                'strategies': entry_result.get('strategies', {}),
                'strategies_triggered': strategies_used,
                'trend': str(analysis.get('trend', 'neutral')),
                'structure': analysis.get('structure', {}),
                'confidence': analysis.get('combined_confidence', 0),
                'mode': 'bootstrap' if self.is_bootstrap_mode() else 'ml',
                'reasons': trade_decision.get('reasons', []),
            },
            capital_api_deal_id=deal_id,
        )
        
        session = TradingSession.objects.filter(user=self.user, is_active=True).first()
        if session:
            session.total_trades += 1
            session.save()
        
        logger.info(f"Opened trade: {pair} {direction} at {entry_price} using {strategy_str}")
        
        return trade
    
    def update_open_trades(self):
        if not self.authenticated:
            return
        
        positions = self.api.get_open_positions(force_refresh=True)
        if positions is None:
            return
        
        position_map = {p['dealId']: p for p in positions}
        position_deal_ids = set(position_map.keys())
        
        open_trades = Trade.objects.filter(user=self.user, status='open')
        
        trades_closed_by_broker = []
        
        for trade in open_trades:
            if trade.capital_api_deal_id:
                if trade.capital_api_deal_id in position_map:
                    pos = position_map[trade.capital_api_deal_id]
                    trade.current_price = Decimal(str(pos['currentLevel']))
                    trade.profit_loss = Decimal(str(pos['profitLoss']))
                    trade.save()
                else:
                    trades_closed_by_broker.append(trade)
        
        for trade in trades_closed_by_broker:
            self._handle_broker_closed_trade(trade)
        
        if trades_closed_by_broker:
            self._check_ml_retrain()
    
    def _handle_broker_closed_trade(self, trade: Trade):
        """Handle a trade that was closed by Capital.com (SL/TP hit)"""
        logger.info(f"Detected broker-closed trade: {trade.pair} {trade.direction} (deal_id: {trade.capital_api_deal_id})")
        
        exit_price = trade.current_price or trade.entry_price
        
        if trade.direction == 'buy':
            if trade.stop_loss and exit_price <= trade.stop_loss:
                exit_price = trade.stop_loss
                outcome_reason = 'stop_loss'
            elif trade.take_profit and exit_price >= trade.take_profit:
                exit_price = trade.take_profit
                outcome_reason = 'take_profit'
            else:
                outcome_reason = 'broker_closed'
        else:
            if trade.stop_loss and exit_price >= trade.stop_loss:
                exit_price = trade.stop_loss
                outcome_reason = 'stop_loss'
            elif trade.take_profit and exit_price <= trade.take_profit:
                exit_price = trade.take_profit
                outcome_reason = 'take_profit'
            else:
                outcome_reason = 'broker_closed'
        
        trade.close_trade(exit_price)
        
        self._update_session_stats(trade)
        
        logger.info(f"Trade closed by broker ({outcome_reason}): {trade.pair} {trade.direction} P/L: ${trade.profit_loss}")
    
    def _check_ml_retrain(self):
        """Check if ML model needs retraining after trades are closed"""
        if self.ml_engine.should_retrain():
            logger.info(f"Triggering ML retrain for user {self.user.email}")
            result = self.ml_engine.train()
            if result:
                logger.info(f"ML model retrained - accuracy: {result.get('accuracy', 0):.2%}")
    
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
        
        self.train_pair_if_ready(trade.pair)
    
    def run_trading_cycle(self):
        if not self.settings.ai_enabled:
            return {'status': 'disabled', 'trades_executed': 0}
        
        if not self.authenticated:
            self.initialize()
        
        self.update_open_trades()
        self.check_and_close_trades()
        
        total_trades = self.get_trade_count()
        ml_ready_pairs = sum(1 for pair in self.TRADING_PAIRS if self.is_pair_ml_ready(pair))
        logger.info(f"Running trading cycle - {total_trades} total trades, {ml_ready_pairs}/{len(self.TRADING_PAIRS)} pairs with ML")
        
        trades_executed = 0
        pairs_analyzed = []
        signals_found = []
        
        for pair in self.TRADING_PAIRS:
            try:
                analysis = self.analyze_pair(pair)
                pairs_analyzed.append(pair)
                
                if analysis and analysis.get('signal') != 'HOLD':
                    signals_found.append({
                        'pair': pair,
                        'signal': analysis.get('signal'),
                        'confidence': analysis.get('confidence', 0)
                    })
                
                if self.should_trade(analysis, pair):
                    result = self.execute_trade(pair, analysis)
                    if result:
                        trades_executed += 1
                        logger.info(f"Trade executed for {pair}: {analysis.get('signal')}")
                
            except Exception as e:
                logger.error(f"Error analyzing {pair}: {str(e)}")
                continue
        
        return {
            'status': 'completed',
            'total_trades': total_trades,
            'ml_ready_pairs': ml_ready_pairs,
            'pairs_analyzed': len(pairs_analyzed),
            'trades_executed': trades_executed,
            'signals_found': signals_found
        }
