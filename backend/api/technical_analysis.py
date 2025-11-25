import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Signal(Enum):
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class AnalysisResult:
    signal: Signal
    confidence: float
    strategies: Dict[str, Signal]
    indicators: Dict[str, float]
    zones: Dict[str, any]
    entry_price: float
    stop_loss: float
    take_profit: float


class TechnicalAnalysis:
    """
    Advanced Technical Analysis Engine for Bootstrap Mode
    
    Implements:
    1. Price Action Strategies (Supply/Demand, Candlesticks, Market Structure)
    2. SMC/MSC Strategies (Liquidity Grabs, BOS, FVG, Order Blocks, Premium/Discount)
    3. Trend-Following (MA Crossover, Breakouts, RSI+Trend)
    4. Range-Bound (Bollinger Bounce, RSI Range)
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self._calculate_indicators()
    
    def _calculate_indicators(self):
        df = self.data
        
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        
        df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_std'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
        
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        df['highest_high'] = df['high'].rolling(window=20).max()
        df['lowest_low'] = df['low'].rolling(window=20).min()
        
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        
        self.data = df
    
    def identify_trend(self) -> Tuple[str, float]:
        """Identify overall trend using MAs and price action"""
        df = self.data
        current = df.iloc[-1]
        
        scores = []
        
        if current['close'] > current['SMA_50']:
            scores.append(1)
        else:
            scores.append(-1)
        
        if current['SMA_20'] > current['SMA_50']:
            scores.append(1)
        else:
            scores.append(-1)
        
        if current['close'] > current['SMA_200']:
            scores.append(1)
        else:
            scores.append(-1)
        
        if current['EMA_9'] > current['EMA_21']:
            scores.append(1)
        else:
            scores.append(-1)
        
        avg_score = np.mean(scores)
        
        if avg_score > 0.5:
            return 'uptrend', abs(avg_score)
        elif avg_score < -0.5:
            return 'downtrend', abs(avg_score)
        return 'ranging', 0.5
    
    def identify_supply_demand_zones(self) -> Dict:
        """
        Identify institutional supply and demand zones
        Buy in demand zones, sell in supply zones
        """
        df = self.data
        zones = {'supply': [], 'demand': []}
        
        for i in range(20, len(df) - 5):
            high_vol = df.iloc[i]['body'] > df.iloc[i-10:i]['body'].mean() * 1.5
            
            if df.iloc[i]['high'] == df.iloc[i-20:i+1]['high'].max():
                if df.iloc[i+1:i+6]['high'].max() < df.iloc[i]['high']:
                    strength = 1.0 + (0.5 if high_vol else 0)
                    zones['supply'].append({
                        'price_high': float(df.iloc[i]['high']),
                        'price_low': float(df.iloc[i]['open'] if df.iloc[i]['is_bearish'] else df.iloc[i]['close']),
                        'index': i,
                        'strength': strength,
                        'tested': False
                    })
            
            if df.iloc[i]['low'] == df.iloc[i-20:i+1]['low'].min():
                if df.iloc[i+1:i+6]['low'].min() > df.iloc[i]['low']:
                    strength = 1.0 + (0.5 if high_vol else 0)
                    zones['demand'].append({
                        'price_high': float(df.iloc[i]['open'] if df.iloc[i]['is_bullish'] else df.iloc[i]['close']),
                        'price_low': float(df.iloc[i]['low']),
                        'index': i,
                        'strength': strength,
                        'tested': False
                    })
        
        current_price = df.iloc[-1]['close']
        for zone in zones['supply']:
            if zone['price_low'] <= current_price <= zone['price_high']:
                zone['tested'] = True
        for zone in zones['demand']:
            if zone['price_low'] <= current_price <= zone['price_high']:
                zone['tested'] = True
        
        zones['supply'] = zones['supply'][-5:]
        zones['demand'] = zones['demand'][-5:]
        
        return zones
    
    def identify_candlestick_patterns(self) -> List[Dict]:
        """
        Identify key candlestick patterns:
        - Engulfing (Bullish/Bearish)
        - Pin Bar (Hammer/Shooting Star)
        - Morning Star / Evening Star
        """
        df = self.data
        patterns = []
        
        for i in range(3, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            body = current['body']
            avg_body = df.iloc[i-10:i]['body'].mean()
            
            if current['lower_wick'] > body * 2 and current['upper_wick'] < body * 0.5:
                if current['is_bullish']:
                    patterns.append({
                        'pattern': 'bullish_pin_bar',
                        'index': i,
                        'signal': Signal.BUY,
                        'strength': 1.5 if body > avg_body else 1.0
                    })
            
            if current['upper_wick'] > body * 2 and current['lower_wick'] < body * 0.5:
                if current['is_bearish']:
                    patterns.append({
                        'pattern': 'bearish_pin_bar',
                        'index': i,
                        'signal': Signal.SELL,
                        'strength': 1.5 if body > avg_body else 1.0
                    })
            
            if prev['is_bearish'] and current['is_bullish']:
                if (current['body'] > prev['body'] and 
                    current['open'] < prev['close'] and 
                    current['close'] > prev['open']):
                    patterns.append({
                        'pattern': 'bullish_engulfing',
                        'index': i,
                        'signal': Signal.STRONG_BUY,
                        'strength': 2.0
                    })
            
            if prev['is_bullish'] and current['is_bearish']:
                if (current['body'] > prev['body'] and 
                    current['open'] > prev['close'] and 
                    current['close'] < prev['open']):
                    patterns.append({
                        'pattern': 'bearish_engulfing',
                        'index': i,
                        'signal': Signal.STRONG_SELL,
                        'strength': 2.0
                    })
            
            if i >= 3:
                first = df.iloc[i-2]
                middle = df.iloc[i-1]
                third = df.iloc[i]
                
                if (first['is_bearish'] and first['body'] > avg_body and
                    middle['body'] < avg_body * 0.5 and
                    third['is_bullish'] and third['body'] > avg_body and
                    third['close'] > (first['open'] + first['close']) / 2):
                    patterns.append({
                        'pattern': 'morning_star',
                        'index': i,
                        'signal': Signal.STRONG_BUY,
                        'strength': 2.5
                    })
                
                if (first['is_bullish'] and first['body'] > avg_body and
                    middle['body'] < avg_body * 0.5 and
                    third['is_bearish'] and third['body'] > avg_body and
                    third['close'] < (first['open'] + first['close']) / 2):
                    patterns.append({
                        'pattern': 'evening_star',
                        'index': i,
                        'signal': Signal.STRONG_SELL,
                        'strength': 2.5
                    })
        
        return patterns[-5:]
    
    def identify_market_structure(self) -> Dict:
        """
        Identify market structure:
        - Uptrend: Higher Highs (HH), Higher Lows (HL) -> buy pullbacks
        - Downtrend: Lower Highs (LH), Lower Lows (LL) -> sell pullbacks
        """
        df = self.data
        
        swing_highs = []
        swing_lows = []
        
        for i in range(3, len(df) - 3):
            if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and 
                df.iloc[i]['high'] > df.iloc[i-2]['high'] and
                df.iloc[i]['high'] > df.iloc[i-3]['high'] and
                df.iloc[i]['high'] > df.iloc[i+1]['high'] and 
                df.iloc[i]['high'] > df.iloc[i+2]['high']):
                swing_highs.append({'price': float(df.iloc[i]['high']), 'index': i})
            
            if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and 
                df.iloc[i]['low'] < df.iloc[i-2]['low'] and
                df.iloc[i]['low'] < df.iloc[i-3]['low'] and
                df.iloc[i]['low'] < df.iloc[i+1]['low'] and 
                df.iloc[i]['low'] < df.iloc[i+2]['low']):
                swing_lows.append({'price': float(df.iloc[i]['low']), 'index': i})
        
        structure = {
            'swing_highs': swing_highs[-5:],
            'swing_lows': swing_lows[-5:],
            'trend': 'neutral',
            'hh': False,
            'hl': False,
            'lh': False,
            'll': False,
            'bos': None,
            'pullback_zone': None
        }
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1]['price'] > swing_highs[-2]['price']
            hl = swing_lows[-1]['price'] > swing_lows[-2]['price']
            lh = swing_highs[-1]['price'] < swing_highs[-2]['price']
            ll = swing_lows[-1]['price'] < swing_lows[-2]['price']
            
            structure['hh'] = hh
            structure['hl'] = hl
            structure['lh'] = lh
            structure['ll'] = ll
            
            if hh and hl:
                structure['trend'] = 'uptrend'
                structure['pullback_zone'] = {
                    'high': swing_lows[-1]['price'] * 1.005,
                    'low': swing_lows[-1]['price'] * 0.995
                }
            elif lh and ll:
                structure['trend'] = 'downtrend'
                structure['pullback_zone'] = {
                    'high': swing_highs[-1]['price'] * 1.005,
                    'low': swing_highs[-1]['price'] * 0.995
                }
            
            if len(swing_highs) >= 3 and len(swing_lows) >= 3:
                prev_trend_up = swing_highs[-3]['price'] < swing_highs[-2]['price']
                curr_trend_down = swing_highs[-1]['price'] < swing_highs[-2]['price']
                if prev_trend_up and curr_trend_down:
                    structure['bos'] = {
                        'type': 'bearish',
                        'level': swing_lows[-2]['price'],
                        'index': swing_lows[-2]['index']
                    }
                
                prev_trend_down = swing_lows[-3]['price'] > swing_lows[-2]['price']
                curr_trend_up = swing_lows[-1]['price'] > swing_lows[-2]['price']
                if prev_trend_down and curr_trend_up:
                    structure['bos'] = {
                        'type': 'bullish',
                        'level': swing_highs[-2]['price'],
                        'index': swing_highs[-2]['index']
                    }
        
        return structure
    
    def identify_liquidity_grabs(self) -> List[Dict]:
        """
        SMC: Identify liquidity grabs (stop hunts)
        Market takes out equal highs/lows then reverses
        """
        df = self.data
        liquidity_grabs = []
        
        for i in range(10, len(df) - 2):
            recent_highs = df.iloc[i-10:i]['high']
            equal_highs = recent_highs[abs(recent_highs - recent_highs.max()) < recent_highs.max() * 0.001]
            
            if len(equal_highs) >= 2:
                if df.iloc[i]['high'] > recent_highs.max():
                    if df.iloc[i+1]['close'] < df.iloc[i]['open'] and df.iloc[i+1]['is_bearish']:
                        liquidity_grabs.append({
                            'type': 'bearish_grab',
                            'level': float(recent_highs.max()),
                            'index': i,
                            'signal': Signal.SELL
                        })
            
            recent_lows = df.iloc[i-10:i]['low']
            equal_lows = recent_lows[abs(recent_lows - recent_lows.min()) < recent_lows.min() * 0.001]
            
            if len(equal_lows) >= 2:
                if df.iloc[i]['low'] < recent_lows.min():
                    if df.iloc[i+1]['close'] > df.iloc[i]['open'] and df.iloc[i+1]['is_bullish']:
                        liquidity_grabs.append({
                            'type': 'bullish_grab',
                            'level': float(recent_lows.min()),
                            'index': i,
                            'signal': Signal.BUY
                        })
        
        return liquidity_grabs[-5:]
    
    def identify_fair_value_gaps(self) -> List[Dict]:
        """
        SMC: Fair Value Gaps (FVG) - Imbalance where price moved too fast
        Price often returns to fill the gap
        Buy at bullish FVG / sell at bearish FVG
        """
        df = self.data
        fvgs = []
        current_price = df.iloc[-1]['close']
        
        for i in range(2, len(df)):
            if df.iloc[i]['low'] > df.iloc[i-2]['high']:
                fvg = {
                    'type': 'bullish',
                    'upper': float(df.iloc[i]['low']),
                    'lower': float(df.iloc[i-2]['high']),
                    'mid': float((df.iloc[i]['low'] + df.iloc[i-2]['high']) / 2),
                    'index': i,
                    'filled': False,
                    'active': False
                }
                if fvg['lower'] <= current_price <= fvg['upper']:
                    fvg['active'] = True
                fvgs.append(fvg)
            
            if df.iloc[i]['high'] < df.iloc[i-2]['low']:
                fvg = {
                    'type': 'bearish',
                    'upper': float(df.iloc[i-2]['low']),
                    'lower': float(df.iloc[i]['high']),
                    'mid': float((df.iloc[i-2]['low'] + df.iloc[i]['high']) / 2),
                    'index': i,
                    'filled': False,
                    'active': False
                }
                if fvg['lower'] <= current_price <= fvg['upper']:
                    fvg['active'] = True
                fvgs.append(fvg)
        
        return fvgs[-10:]
    
    def identify_order_blocks(self) -> List[Dict]:
        """
        SMC: Order Blocks - Last opposite candle before a big move
        Banks use these to enter, price returns -> touch -> reversal
        """
        df = self.data
        order_blocks = []
        current_price = df.iloc[-1]['close']
        avg_body = df['body'].mean()
        
        for i in range(1, len(df) - 3):
            current = df.iloc[i]
            next_candles = df.iloc[i+1:i+4]
            
            if current['is_bearish'] and current['body'] > avg_body * 0.5:
                bullish_moves = next_candles[next_candles['is_bullish']]
                if len(bullish_moves) >= 2:
                    if next_candles['close'].iloc[-1] > current['high']:
                        ob = {
                            'type': 'bullish',
                            'high': float(current['high']),
                            'low': float(current['low']),
                            'mid': float((current['high'] + current['low']) / 2),
                            'index': i,
                            'mitigated': False,
                            'active': False
                        }
                        if ob['low'] <= current_price <= ob['high']:
                            ob['active'] = True
                        order_blocks.append(ob)
            
            if current['is_bullish'] and current['body'] > avg_body * 0.5:
                bearish_moves = next_candles[next_candles['is_bearish']]
                if len(bearish_moves) >= 2:
                    if next_candles['close'].iloc[-1] < current['low']:
                        ob = {
                            'type': 'bearish',
                            'high': float(current['high']),
                            'low': float(current['low']),
                            'mid': float((current['high'] + current['low']) / 2),
                            'index': i,
                            'mitigated': False,
                            'active': False
                        }
                        if ob['low'] <= current_price <= ob['high']:
                            ob['active'] = True
                        order_blocks.append(ob)
        
        return order_blocks[-10:]
    
    def calculate_premium_discount_zones(self) -> Dict:
        """
        SMC: Premium/Discount Zones using Fibonacci 50-79%
        Only buy in discount zone (below 50%)
        Only sell in premium zone (above 50%)
        """
        df = self.data
        
        lookback = min(100, len(df))
        recent_data = df.iloc[-lookback:]
        
        high = recent_data['high'].max()
        low = recent_data['low'].min()
        diff = high - low
        current_price = df.iloc[-1]['close']
        
        equilibrium = low + (diff * 0.5)
        
        zones = {
            'high': float(high),
            'low': float(low),
            'equilibrium': float(equilibrium),
            '0.236': float(high - diff * 0.236),
            '0.382': float(high - diff * 0.382),
            '0.5': float(high - diff * 0.5),
            '0.618': float(high - diff * 0.618),
            '0.786': float(high - diff * 0.786),
            'premium_zone': {
                'high': float(high),
                'low': float(high - diff * 0.5)
            },
            'discount_zone': {
                'high': float(low + diff * 0.5),
                'low': float(low)
            },
            'optimal_buy_zone': {
                'high': float(high - diff * 0.618),
                'low': float(high - diff * 0.786)
            },
            'optimal_sell_zone': {
                'high': float(high - diff * 0.236),
                'low': float(high - diff * 0.382)
            },
            'current_zone': 'neutral',
            'current_position': float((current_price - low) / diff) if diff > 0 else 0.5
        }
        
        if current_price < equilibrium:
            zones['current_zone'] = 'discount'
        else:
            zones['current_zone'] = 'premium'
        
        return zones
    
    def identify_support_resistance(self) -> Dict:
        """Identify key support and resistance levels"""
        df = self.data
        levels = {'support': [], 'resistance': []}
        
        for i in range(5, len(df) - 5):
            is_resistance = (df.iloc[i]['high'] >= df.iloc[i-5:i]['high'].max() and
                           df.iloc[i]['high'] >= df.iloc[i+1:i+6]['high'].max())
            if is_resistance:
                levels['resistance'].append({
                    'price': float(df.iloc[i]['high']),
                    'index': i,
                    'touches': 1
                })
            
            is_support = (df.iloc[i]['low'] <= df.iloc[i-5:i]['low'].min() and
                        df.iloc[i]['low'] <= df.iloc[i+1:i+6]['low'].min())
            if is_support:
                levels['support'].append({
                    'price': float(df.iloc[i]['low']),
                    'index': i,
                    'touches': 1
                })
        
        def consolidate_levels(level_list, threshold=0.002):
            if not level_list:
                return []
            consolidated = []
            level_list = sorted(level_list, key=lambda x: x['price'])
            current_group = [level_list[0]]
            
            for level in level_list[1:]:
                if abs(level['price'] - current_group[0]['price']) / current_group[0]['price'] < threshold:
                    current_group.append(level)
                else:
                    avg_price = sum(l['price'] for l in current_group) / len(current_group)
                    consolidated.append({
                        'price': avg_price,
                        'touches': len(current_group),
                        'strength': len(current_group)
                    })
                    current_group = [level]
            
            if current_group:
                avg_price = sum(l['price'] for l in current_group) / len(current_group)
                consolidated.append({
                    'price': avg_price,
                    'touches': len(current_group),
                    'strength': len(current_group)
                })
            
            return consolidated
        
        levels['support'] = consolidate_levels(levels['support'])[-5:]
        levels['resistance'] = consolidate_levels(levels['resistance'])[-5:]
        
        return levels
    
    def breakout_signal(self) -> Tuple[Signal, Dict]:
        """
        Trend-Following: Breakout Trading
        Enter when price breaks key support/resistance with momentum
        """
        df = self.data
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        lookback = 20
        highest_high = df.iloc[-lookback-1:-1]['high'].max()
        lowest_low = df.iloc[-lookback-1:-1]['low'].min()
        
        breakout_info = {
            'type': None,
            'level': None,
            'momentum': False
        }
        
        if current['close'] > highest_high and current['body'] > df['body'].mean():
            breakout_info = {
                'type': 'resistance_break',
                'level': float(highest_high),
                'momentum': current['body'] > df['body'].mean() * 1.5
            }
            return Signal.STRONG_BUY if breakout_info['momentum'] else Signal.BUY, breakout_info
        
        if current['close'] < lowest_low and current['body'] > df['body'].mean():
            breakout_info = {
                'type': 'support_break',
                'level': float(lowest_low),
                'momentum': current['body'] > df['body'].mean() * 1.5
            }
            return Signal.STRONG_SELL if breakout_info['momentum'] else Signal.SELL, breakout_info
        
        return Signal.NEUTRAL, breakout_info
    
    def ma_crossover_signal(self) -> Signal:
        """
        Trend-Following: Moving Average Crossover
        Buy when short MA crosses above long MA
        Sell when short MA crosses below long MA
        """
        df = self.data
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if prev['EMA_9'] < prev['EMA_21'] and current['EMA_9'] > current['EMA_21']:
            return Signal.STRONG_BUY
        elif prev['EMA_9'] > prev['EMA_21'] and current['EMA_9'] < current['EMA_21']:
            return Signal.STRONG_SELL
        elif current['EMA_9'] > current['EMA_21'] and current['SMA_20'] > current['SMA_50']:
            return Signal.BUY
        elif current['EMA_9'] < current['EMA_21'] and current['SMA_20'] < current['SMA_50']:
            return Signal.SELL
        return Signal.NEUTRAL
    
    def rsi_trend_signal(self) -> Signal:
        """
        RSI signal - ONLY signals in EXTREME conditions.
        Override to BUY only if RSI < 30 (oversold)
        Override to SELL only if RSI > 70 (overbought)
        Neutral RSI (30-70) does NOT block trades.
        """
        current = self.data.iloc[-1]
        rsi = current['RSI']
        
        # Only signal in extreme conditions
        if rsi < 30:
            return Signal.STRONG_BUY  # Extreme oversold
        elif rsi > 70:
            return Signal.STRONG_SELL  # Extreme overbought
        
        # Neutral RSI (30-70) - does NOT affect trade decision
        return Signal.NEUTRAL
    
    def bollinger_bounce_signal(self) -> Signal:
        """
        Bollinger Band signal - ONLY signals in EXTREME conditions.
        Override to BUY only if price at/below lower band
        Override to SELL only if price at/above upper band
        Price in middle of bands does NOT block trades.
        """
        current = self.data.iloc[-1]
        
        # Only signal when price is at extreme Bollinger levels
        if current['close'] <= current['BB_lower']:
            return Signal.STRONG_BUY  # At lower band - extreme oversold
        elif current['close'] >= current['BB_upper']:
            return Signal.STRONG_SELL  # At upper band - extreme overbought
        
        # Price between bands - does NOT affect trade decision
        return Signal.NEUTRAL
    
    def rsi_range_signal(self) -> Signal:
        """
        RSI Range signal - ONLY signals in EXTREME conditions.
        Override to BUY only if RSI < 30 (extreme oversold)
        Override to SELL only if RSI > 70 (extreme overbought)
        RSI 30-70 is neutral and does NOT block trades.
        """
        current = self.data.iloc[-1]
        rsi = current['RSI']
        
        # Only signal in extreme conditions
        if rsi < 30:
            return Signal.STRONG_BUY  # Extreme oversold
        elif rsi > 70:
            return Signal.STRONG_SELL  # Extreme overbought
        
        # RSI 30-70 - does NOT affect trade decision
        return Signal.NEUTRAL
    
    def macd_signal(self) -> Signal:
        """MACD crossover signal"""
        df = self.data
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if prev['MACD'] < prev['MACD_signal'] and current['MACD'] > current['MACD_signal']:
            return Signal.STRONG_BUY
        elif prev['MACD'] > prev['MACD_signal'] and current['MACD'] < current['MACD_signal']:
            return Signal.STRONG_SELL
        elif current['MACD'] > current['MACD_signal'] and current['MACD_histogram'] > 0:
            return Signal.BUY
        elif current['MACD'] < current['MACD_signal'] and current['MACD_histogram'] < 0:
            return Signal.SELL
        
        return Signal.NEUTRAL
    
    def smc_composite_signal(self) -> Tuple[Signal, float, Dict]:
        """
        Generate composite SMC signal based on:
        - Order Blocks
        - Fair Value Gaps
        - Premium/Discount Zones
        - Liquidity Grabs
        - Market Structure
        """
        current = self.data.iloc[-1]
        current_price = current['close']
        
        order_blocks = self.identify_order_blocks()
        fvgs = self.identify_fair_value_gaps()
        zones = self.calculate_premium_discount_zones()
        liquidity = self.identify_liquidity_grabs()
        structure = self.identify_market_structure()
        
        signals = []
        details = {
            'order_block': None,
            'fvg': None,
            'zone': zones['current_zone'],
            'liquidity': None,
            'structure': structure['trend']
        }
        
        for ob in order_blocks[-3:]:
            if ob['active']:
                if ob['type'] == 'bullish':
                    signals.append(Signal.BUY.value * 1.5)
                    details['order_block'] = 'bullish_active'
                else:
                    signals.append(Signal.SELL.value * 1.5)
                    details['order_block'] = 'bearish_active'
        
        for fvg in fvgs[-3:]:
            if fvg['active']:
                if fvg['type'] == 'bullish':
                    signals.append(Signal.BUY.value)
                    details['fvg'] = 'bullish_active'
                else:
                    signals.append(Signal.SELL.value)
                    details['fvg'] = 'bearish_active'
        
        if zones['current_zone'] == 'discount':
            signals.append(Signal.BUY.value * 0.5)
        elif zones['current_zone'] == 'premium':
            signals.append(Signal.SELL.value * 0.5)
        
        if liquidity:
            last_grab = liquidity[-1]
            if last_grab['index'] >= len(self.data) - 5:
                signals.append(last_grab['signal'].value * 1.5)
                details['liquidity'] = last_grab['type']
        
        if structure['trend'] == 'uptrend':
            signals.append(Signal.BUY.value)
        elif structure['trend'] == 'downtrend':
            signals.append(Signal.SELL.value)
        
        if not signals:
            return Signal.NEUTRAL, 0.0, details
        
        avg_signal = np.mean(signals)
        confidence = min(abs(avg_signal) / 2, 1.0)
        
        if avg_signal >= 1.5:
            return Signal.STRONG_BUY, confidence, details
        elif avg_signal >= 0.5:
            return Signal.BUY, confidence, details
        elif avg_signal <= -1.5:
            return Signal.STRONG_SELL, confidence, details
        elif avg_signal <= -0.5:
            return Signal.SELL, confidence, details
        
        return Signal.NEUTRAL, confidence, details
    
    def analyze(self) -> AnalysisResult:
        """
        Comprehensive analysis combining all strategies
        """
        trend, trend_strength = self.identify_trend()
        zones = self.identify_supply_demand_zones()
        patterns = self.identify_candlestick_patterns()
        structure = self.identify_market_structure()
        fvgs = self.identify_fair_value_gaps()
        order_blocks = self.identify_order_blocks()
        fib_zones = self.calculate_premium_discount_zones()
        support_resistance = self.identify_support_resistance()
        breakout_signal, breakout_info = self.breakout_signal()
        smc_signal, smc_confidence, smc_details = self.smc_composite_signal()
        
        strategies = {
            'ma_crossover': self.ma_crossover_signal(),
            'rsi_trend': self.rsi_trend_signal(),
            'rsi_range': self.rsi_range_signal(),
            'bollinger': self.bollinger_bounce_signal(),
            'macd': self.macd_signal(),
            'breakout': breakout_signal,
            'smc': smc_signal,
        }
        
        recent_patterns = [p for p in patterns if p['index'] >= len(self.data) - 5]
        if recent_patterns:
            strategies['candlestick'] = recent_patterns[-1]['signal']
        else:
            strategies['candlestick'] = Signal.NEUTRAL
        
        if structure['trend'] == 'uptrend':
            strategies['market_structure'] = Signal.BUY
        elif structure['trend'] == 'downtrend':
            strategies['market_structure'] = Signal.SELL
        else:
            strategies['market_structure'] = Signal.NEUTRAL
        
        weights = {
            'smc': 2.0,
            'market_structure': 1.5,
            'ma_crossover': 1.2,
            'macd': 1.2,
            'breakout': 1.3,
            'candlestick': 1.0,
            'rsi_trend': 1.0,
            'bollinger': 0.8,
            'rsi_range': 0.8,
        }
        
        weighted_sum = 0
        total_weight = 0
        for strategy, signal in strategies.items():
            weight = weights.get(strategy, 1.0)
            weighted_sum += signal.value * weight
            total_weight += weight
        
        avg_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        if avg_score >= 1.2:
            signal = Signal.STRONG_BUY
        elif avg_score >= 0.4:
            signal = Signal.BUY
        elif avg_score <= -1.2:
            signal = Signal.STRONG_SELL
        elif avg_score <= -0.4:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL
        
        confidence = min(abs(avg_score) / 2, 1.0)
        
        current = self.data.iloc[-1]
        atr = current['ATR']
        entry_price = current['close']
        
        if signal in [Signal.BUY, Signal.STRONG_BUY]:
            stop_loss = entry_price - (atr * 2)
            take_profit = entry_price + (atr * 2)  # 1:1 risk:reward
            
            for demand in zones['demand'][-2:]:
                if demand['price_low'] < entry_price:
                    stop_loss = min(stop_loss, demand['price_low'] - atr * 0.5)
                    break
        elif signal in [Signal.SELL, Signal.STRONG_SELL]:
            stop_loss = entry_price + (atr * 2)
            take_profit = entry_price - (atr * 2)  # 1:1 risk:reward
            
            for supply in zones['supply'][-2:]:
                if supply['price_high'] > entry_price:
                    stop_loss = max(stop_loss, supply['price_high'] + atr * 0.5)
                    break
        else:
            stop_loss = entry_price
            take_profit = entry_price
        
        indicators = {
            'rsi': float(current['RSI']),
            'macd': float(current['MACD']),
            'macd_signal': float(current['MACD_signal']),
            'macd_histogram': float(current['MACD_histogram']),
            'sma_20': float(current['SMA_20']),
            'sma_50': float(current['SMA_50']),
            'ema_9': float(current['EMA_9']),
            'ema_21': float(current['EMA_21']),
            'bb_upper': float(current['BB_upper']),
            'bb_lower': float(current['BB_lower']),
            'bb_middle': float(current['BB_middle']),
            'atr': float(atr),
            'trend': trend,
            'trend_strength': trend_strength,
        }
        
        return AnalysisResult(
            signal=signal,
            confidence=confidence,
            strategies={k: v.name for k, v in strategies.items()},
            indicators=indicators,
            zones={
                'supply_demand': zones,
                'fibonacci': fib_zones,
                'structure': structure,
                'fvg': [f for f in fvgs[-5:] if f['active'] or f['index'] > len(self.data) - 20],
                'order_blocks': [ob for ob in order_blocks[-5:] if ob['active'] or ob['index'] > len(self.data) - 20],
                'support_resistance': support_resistance,
                'smc_details': smc_details,
            },
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
        )


def analyze_multi_timeframe(data_dict: Dict[str, pd.DataFrame]) -> Dict:
    """
    Multi-timeframe analysis following the specified approach:
    - Market Structure (Trend) -> 4H
    - Zones (Supply & Demand) -> 4H and 1H
    - Key Levels (Support/Resistance) -> 1H
    - Entry -> 15m and 30m
    """
    results = {}
    
    for timeframe, data in data_dict.items():
        if len(data) < 50:
            continue
        analysis = TechnicalAnalysis(data)
        results[timeframe] = analysis.analyze()
    
    trend = Signal.NEUTRAL
    trend_confidence = 0
    structure_info = None
    
    if '4H' in results:
        trend = results['4H'].signal
        trend_confidence = results['4H'].confidence
        structure_info = results['4H'].zones.get('structure', {})
    
    zones_4h = {}
    zones_1h = {}
    
    if '4H' in results:
        zones_4h = results['4H'].zones.get('supply_demand', {})
    if '1H' in results:
        zones_1h = results['1H'].zones.get('supply_demand', {})
    
    key_levels = {}
    fib_zones = {}
    if '1H' in results:
        key_levels = results['1H'].zones.get('support_resistance', {})
        fib_zones = results['1H'].zones.get('fibonacci', {})
    
    entry_signal = Signal.NEUTRAL
    entry_confidence = 0
    entry_result = None
    
    for tf in ['15m', '30m']:
        if tf in results:
            if results[tf].confidence > entry_confidence:
                entry_signal = results[tf].signal
                entry_confidence = results[tf].confidence
                entry_result = results[tf]
    
    aligned = True
    alignment_score = 1.0
    
    if trend in [Signal.BUY, Signal.STRONG_BUY]:
        if entry_signal in [Signal.SELL, Signal.STRONG_SELL]:
            aligned = False
            alignment_score = 0.3
        elif entry_signal == Signal.NEUTRAL:
            alignment_score = 0.6
    elif trend in [Signal.SELL, Signal.STRONG_SELL]:
        if entry_signal in [Signal.BUY, Signal.STRONG_BUY]:
            aligned = False
            alignment_score = 0.3
        elif entry_signal == Signal.NEUTRAL:
            alignment_score = 0.6
    
    final_confidence = entry_confidence * alignment_score
    
    return {
        'trend': trend,
        'trend_confidence': trend_confidence,
        'structure': structure_info,
        'zones_4h': zones_4h,
        'zones_1h': zones_1h,
        'key_levels': key_levels,
        'fib_zones': fib_zones,
        'entry_signal': entry_signal,
        'entry_result': entry_result,
        'confidence': final_confidence,
        'aligned': aligned,
        'alignment_score': alignment_score,
        'timeframe_results': {tf: {
            'signal': r.signal.name,
            'confidence': r.confidence,
            'strategies': r.strategies,
            'indicators': r.indicators,
            'entry_price': r.entry_price,
            'stop_loss': r.stop_loss,
            'take_profit': r.take_profit,
        } for tf, r in results.items()},
    }
