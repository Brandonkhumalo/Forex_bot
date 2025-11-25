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
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self._calculate_indicators()
    
    def _calculate_indicators(self):
        df = self.data
        
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
        
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
        
        self.data = df
    
    def identify_trend(self) -> Tuple[str, float]:
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
        
        avg_score = np.mean(scores)
        
        if avg_score > 0.5:
            return 'uptrend', abs(avg_score)
        elif avg_score < -0.5:
            return 'downtrend', abs(avg_score)
        return 'ranging', 0.5
    
    def identify_supply_demand_zones(self) -> Dict:
        df = self.data
        zones = {'supply': [], 'demand': []}
        
        for i in range(20, len(df) - 5):
            if df.iloc[i]['high'] == df.iloc[i-20:i+1]['high'].max():
                if df.iloc[i+1:i+6]['high'].max() < df.iloc[i]['high']:
                    zones['supply'].append({
                        'price': float(df.iloc[i]['high']),
                        'index': i,
                        'strength': 1.0
                    })
            
            if df.iloc[i]['low'] == df.iloc[i-20:i+1]['low'].min():
                if df.iloc[i+1:i+6]['low'].min() > df.iloc[i]['low']:
                    zones['demand'].append({
                        'price': float(df.iloc[i]['low']),
                        'index': i,
                        'strength': 1.0
                    })
        
        zones['supply'] = zones['supply'][-5:]
        zones['demand'] = zones['demand'][-5:]
        
        return zones
    
    def identify_candlestick_patterns(self) -> List[Dict]:
        df = self.data
        patterns = []
        
        for i in range(2, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            body = abs(current['close'] - current['open'])
            upper_wick = current['high'] - max(current['close'], current['open'])
            lower_wick = min(current['close'], current['open']) - current['low']
            
            if lower_wick > body * 2 and upper_wick < body * 0.5:
                if current['close'] > current['open']:
                    patterns.append({'pattern': 'bullish_pin_bar', 'index': i, 'signal': Signal.BUY})
            
            if upper_wick > body * 2 and lower_wick < body * 0.5:
                if current['close'] < current['open']:
                    patterns.append({'pattern': 'bearish_pin_bar', 'index': i, 'signal': Signal.SELL})
            
            prev_body = abs(prev['close'] - prev['open'])
            if prev['close'] < prev['open'] and current['close'] > current['open']:
                if body > prev_body and current['open'] < prev['close'] and current['close'] > prev['open']:
                    patterns.append({'pattern': 'bullish_engulfing', 'index': i, 'signal': Signal.BUY})
            
            if prev['close'] > prev['open'] and current['close'] < current['open']:
                if body > prev_body and current['open'] > prev['close'] and current['close'] < prev['open']:
                    patterns.append({'pattern': 'bearish_engulfing', 'index': i, 'signal': Signal.SELL})
        
        return patterns[-5:]
    
    def identify_market_structure(self) -> Dict:
        df = self.data
        
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(df) - 2):
            if df.iloc[i]['high'] > df.iloc[i-1]['high'] and df.iloc[i]['high'] > df.iloc[i-2]['high']:
                if df.iloc[i]['high'] > df.iloc[i+1]['high'] and df.iloc[i]['high'] > df.iloc[i+2]['high']:
                    swing_highs.append({'price': float(df.iloc[i]['high']), 'index': i})
            
            if df.iloc[i]['low'] < df.iloc[i-1]['low'] and df.iloc[i]['low'] < df.iloc[i-2]['low']:
                if df.iloc[i]['low'] < df.iloc[i+1]['low'] and df.iloc[i]['low'] < df.iloc[i+2]['low']:
                    swing_lows.append({'price': float(df.iloc[i]['low']), 'index': i})
        
        structure = {
            'swing_highs': swing_highs[-5:],
            'swing_lows': swing_lows[-5:],
            'trend': 'neutral',
            'bos': None,
        }
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1]['price'] > swing_highs[-2]['price'] if len(swing_highs) >= 2 else False
            hl = swing_lows[-1]['price'] > swing_lows[-2]['price'] if len(swing_lows) >= 2 else False
            lh = swing_highs[-1]['price'] < swing_highs[-2]['price'] if len(swing_highs) >= 2 else False
            ll = swing_lows[-1]['price'] < swing_lows[-2]['price'] if len(swing_lows) >= 2 else False
            
            if hh and hl:
                structure['trend'] = 'uptrend'
            elif lh and ll:
                structure['trend'] = 'downtrend'
        
        return structure
    
    def identify_fair_value_gaps(self) -> List[Dict]:
        df = self.data
        fvgs = []
        
        for i in range(2, len(df)):
            if df.iloc[i]['low'] > df.iloc[i-2]['high']:
                fvgs.append({
                    'type': 'bullish',
                    'upper': float(df.iloc[i]['low']),
                    'lower': float(df.iloc[i-2]['high']),
                    'index': i,
                    'filled': False
                })
            
            if df.iloc[i]['high'] < df.iloc[i-2]['low']:
                fvgs.append({
                    'type': 'bearish',
                    'upper': float(df.iloc[i-2]['low']),
                    'lower': float(df.iloc[i]['high']),
                    'index': i,
                    'filled': False
                })
        
        return fvgs[-10:]
    
    def identify_order_blocks(self) -> List[Dict]:
        df = self.data
        order_blocks = []
        
        for i in range(1, len(df) - 3):
            current = df.iloc[i]
            next_candles = df.iloc[i+1:i+4]
            
            if current['close'] < current['open']:
                if all(next_candles['close'] > next_candles['open']) and next_candles['close'].iloc[-1] > current['high']:
                    order_blocks.append({
                        'type': 'bullish',
                        'high': float(current['high']),
                        'low': float(current['low']),
                        'index': i,
                        'mitigated': False
                    })
            
            if current['close'] > current['open']:
                if all(next_candles['close'] < next_candles['open']) and next_candles['close'].iloc[-1] < current['low']:
                    order_blocks.append({
                        'type': 'bearish',
                        'high': float(current['high']),
                        'low': float(current['low']),
                        'index': i,
                        'mitigated': False
                    })
        
        return order_blocks[-10:]
    
    def calculate_fibonacci_zones(self) -> Dict:
        df = self.data
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low
        
        return {
            'high': float(high),
            'low': float(low),
            '0.236': float(high - diff * 0.236),
            '0.382': float(high - diff * 0.382),
            '0.5': float(high - diff * 0.5),
            '0.618': float(high - diff * 0.618),
            '0.786': float(high - diff * 0.786),
            'premium_zone': (float(high - diff * 0.5), float(high)),
            'discount_zone': (float(low), float(high - diff * 0.5)),
        }
    
    def ma_crossover_signal(self) -> Signal:
        df = self.data
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if prev['SMA_20'] < prev['SMA_50'] and current['SMA_20'] > current['SMA_50']:
            return Signal.BUY
        elif prev['SMA_20'] > prev['SMA_50'] and current['SMA_20'] < current['SMA_50']:
            return Signal.SELL
        elif current['SMA_20'] > current['SMA_50']:
            return Signal.BUY
        elif current['SMA_20'] < current['SMA_50']:
            return Signal.SELL
        return Signal.NEUTRAL
    
    def rsi_signal(self) -> Signal:
        current = self.data.iloc[-1]
        rsi = current['RSI']
        
        if rsi < 30:
            return Signal.STRONG_BUY
        elif rsi < 40:
            return Signal.BUY
        elif rsi > 70:
            return Signal.STRONG_SELL
        elif rsi > 60:
            return Signal.SELL
        return Signal.NEUTRAL
    
    def bollinger_signal(self) -> Signal:
        current = self.data.iloc[-1]
        
        if current['close'] < current['BB_lower']:
            return Signal.BUY
        elif current['close'] > current['BB_upper']:
            return Signal.SELL
        return Signal.NEUTRAL
    
    def macd_signal(self) -> Signal:
        df = self.data
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if prev['MACD'] < prev['MACD_signal'] and current['MACD'] > current['MACD_signal']:
            return Signal.BUY
        elif prev['MACD'] > prev['MACD_signal'] and current['MACD'] < current['MACD_signal']:
            return Signal.SELL
        elif current['MACD'] > current['MACD_signal']:
            return Signal.BUY
        elif current['MACD'] < current['MACD_signal']:
            return Signal.SELL
        return Signal.NEUTRAL
    
    def analyze(self) -> AnalysisResult:
        trend, trend_strength = self.identify_trend()
        zones = self.identify_supply_demand_zones()
        patterns = self.identify_candlestick_patterns()
        structure = self.identify_market_structure()
        fvgs = self.identify_fair_value_gaps()
        order_blocks = self.identify_order_blocks()
        fib = self.calculate_fibonacci_zones()
        
        strategies = {
            'ma_crossover': self.ma_crossover_signal(),
            'rsi': self.rsi_signal(),
            'bollinger': self.bollinger_signal(),
            'macd': self.macd_signal(),
        }
        
        if patterns:
            last_pattern = patterns[-1]
            strategies['candlestick'] = last_pattern['signal']
        else:
            strategies['candlestick'] = Signal.NEUTRAL
        
        if structure['trend'] == 'uptrend':
            strategies['market_structure'] = Signal.BUY
        elif structure['trend'] == 'downtrend':
            strategies['market_structure'] = Signal.SELL
        else:
            strategies['market_structure'] = Signal.NEUTRAL
        
        current = self.data.iloc[-1]
        if fvgs:
            for fvg in fvgs[-3:]:
                if fvg['type'] == 'bullish' and fvg['lower'] <= current['close'] <= fvg['upper']:
                    strategies['fvg'] = Signal.BUY
                    break
                elif fvg['type'] == 'bearish' and fvg['lower'] <= current['close'] <= fvg['upper']:
                    strategies['fvg'] = Signal.SELL
                    break
            else:
                strategies['fvg'] = Signal.NEUTRAL
        else:
            strategies['fvg'] = Signal.NEUTRAL
        
        scores = [s.value for s in strategies.values()]
        avg_score = np.mean(scores)
        
        if avg_score >= 1.5:
            signal = Signal.STRONG_BUY
        elif avg_score >= 0.5:
            signal = Signal.BUY
        elif avg_score <= -1.5:
            signal = Signal.STRONG_SELL
        elif avg_score <= -0.5:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL
        
        confidence = min(abs(avg_score) / 2, 1.0)
        
        atr = current['ATR']
        entry_price = current['close']
        
        if signal in [Signal.BUY, Signal.STRONG_BUY]:
            stop_loss = entry_price - (atr * 2)
            take_profit = entry_price + (atr * 3)
        elif signal in [Signal.SELL, Signal.STRONG_SELL]:
            stop_loss = entry_price + (atr * 2)
            take_profit = entry_price - (atr * 3)
        else:
            stop_loss = entry_price
            take_profit = entry_price
        
        indicators = {
            'rsi': float(current['RSI']),
            'macd': float(current['MACD']),
            'macd_signal': float(current['MACD_signal']),
            'sma_20': float(current['SMA_20']),
            'sma_50': float(current['SMA_50']),
            'bb_upper': float(current['BB_upper']),
            'bb_lower': float(current['BB_lower']),
            'atr': float(atr),
        }
        
        return AnalysisResult(
            signal=signal,
            confidence=confidence,
            strategies={k: v.name for k, v in strategies.items()},
            indicators=indicators,
            zones={
                'supply_demand': zones,
                'fibonacci': fib,
                'structure': structure,
                'fvg': fvgs[-3:] if fvgs else [],
                'order_blocks': order_blocks[-3:] if order_blocks else [],
            },
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
        )


def analyze_multi_timeframe(data_dict: Dict[str, pd.DataFrame]) -> Dict:
    results = {}
    
    for timeframe, data in data_dict.items():
        if len(data) < 50:
            continue
        analysis = TechnicalAnalysis(data)
        results[timeframe] = analysis.analyze()
    
    if '4H' in results:
        trend = results['4H'].signal
    else:
        trend = Signal.NEUTRAL
    
    if '1H' in results:
        zones = results['1H'].zones
    else:
        zones = {}
    
    entry_signal = Signal.NEUTRAL
    entry_confidence = 0
    
    for tf in ['15m', '30m']:
        if tf in results:
            entry_signal = results[tf].signal
            entry_confidence = results[tf].confidence
            break
    
    aligned = True
    if trend in [Signal.BUY, Signal.STRONG_BUY] and entry_signal in [Signal.SELL, Signal.STRONG_SELL]:
        aligned = False
    elif trend in [Signal.SELL, Signal.STRONG_SELL] and entry_signal in [Signal.BUY, Signal.STRONG_BUY]:
        aligned = False
    
    return {
        'trend': trend,
        'entry_signal': entry_signal,
        'confidence': entry_confidence if aligned else entry_confidence * 0.5,
        'aligned': aligned,
        'zones': zones,
        'timeframe_results': {tf: r.__dict__ for tf, r in results.items()},
    }
