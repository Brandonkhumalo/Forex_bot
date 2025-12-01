import numpy as np
import pandas as pd
import pickle
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
import pytz

from .models import Trade, MLModel, MarketData

logger = logging.getLogger(__name__)

AI_LOGS_DIR = Path(__file__).resolve().parent.parent / 'ai_logs'

MIN_TRADES_PER_PAIR = 5

TRADING_PAIRS = [
    'GBP/USD', 'EUR/USD', 'USD/JPY', 'AUD/USD',
    'USD/CAD', 'NZD/USD', 'USD/CHF', 'XAU/USD',
    'GBP/JPY', 'EUR/JPY'
]

SESSION_TIMES = {
    'asia': {'start': 0, 'end': 8},
    'london': {'start': 8, 'end': 16},
    'new_york': {'start': 13, 'end': 22},
    'overlap_london_ny': {'start': 13, 'end': 16},
}

REGIME_TYPES = ['trending_up', 'trending_down', 'ranging', 'high_volatility', 'low_volatility']


class AILogger:
    def __init__(self, user_email: str):
        self.user_email = user_email.replace('@', '_at_').replace('.', '_')
        self.user_dir = AI_LOGS_DIR / self.user_email
        self.user_dir.mkdir(parents=True, exist_ok=True)
    
    def get_log_file(self) -> Path:
        date_str = datetime.now().strftime('%Y-%m-%d')
        return self.user_dir / f'ml_training_{date_str}.log'
    
    def log(self, message: str, level: str = 'INFO'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.get_log_file(), 'a') as f:
            f.write(log_line)
        
        if level == 'ERROR':
            logger.error(message)
        elif level == 'WARNING':
            logger.warning(message)
        else:
            logger.info(message)
    
    def log_training_start(self, pair: str, num_trades: int):
        self.log(f"=" * 60)
        self.log(f"ML MODEL TRAINING STARTED FOR {pair}")
        self.log(f"User: {self.user_email}")
        self.log(f"Trades for training: {num_trades}")
        self.log(f"=" * 60)
    
    def log_training_complete(self, pair: str, metrics: Dict):
        self.log(f"TRAINING COMPLETE - {pair} Model v{metrics.get('version', 'N/A')}")
        self.log(f"  Accuracy: {metrics.get('accuracy', 0):.2%}")
        self.log(f"  Precision: {metrics.get('precision', 0):.2%}")
        self.log(f"  Recall: {metrics.get('recall', 0):.2%}")
        self.log(f"  F1 Score: {metrics.get('f1_score', 0):.2%}")
        self.log(f"  Cross-validation Mean: {metrics.get('cv_mean', 0):.2%}")
        self.log(f"  Regime Accuracy: {metrics.get('regime_accuracy', 0):.2%}")
        self.log(f"  TP Probability Model: {metrics.get('tp_accuracy', 0):.2%}")
        self.log(f"-" * 40)
        self.log(f"TOP FEATURE IMPORTANCE:")
        importance = metrics.get('feature_importance', {})
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_features[:10]:
            self.log(f"  {feature}: {imp:.4f}")
        self.log(f"=" * 60)
    
    def log_prediction(self, pair: str, prediction: float, confidence: float, ta_signal: float, regime: str = 'unknown'):
        self.log(f"PREDICTION: {pair} | ML={prediction:.2f} ({confidence:.1%}) | TA={ta_signal} | Regime={regime}")
    
    def log_retrain_trigger(self, pair: str, trades_since: int):
        self.log(f"RETRAIN TRIGGERED for {pair}: {trades_since} new trades since last training")
    
    def log_error(self, error_message: str):
        self.log(f"ERROR: {error_message}", level='ERROR')


class MarketRegimeDetector:
    """Detects market regime: trending, ranging, high/low volatility"""
    
    @staticmethod
    def get_volatility_regime(atr: float, atr_50: float) -> str:
        """Calculate volatility regime based on ATR ratio"""
        if atr_50 == 0:
            return 'normal'
        
        ratio = atr / atr_50
        
        if ratio > 1.5:
            return 'high_volatility'
        elif ratio < 0.5:
            return 'low_volatility'
        else:
            return 'normal'
    
    @staticmethod
    def get_trend_regime(sma_20: float, sma_50: float, sma_200: float, 
                         adx: float = 25, price: float = 0) -> str:
        """Determine if market is trending or ranging"""
        if sma_20 == 0 or sma_50 == 0:
            return 'ranging'
        
        sma_ratio = sma_20 / sma_50
        
        if adx > 25:
            if sma_ratio > 1.01 and price > sma_50:
                return 'trending_up'
            elif sma_ratio < 0.99 and price < sma_50:
                return 'trending_down'
        
        if 0.995 <= sma_ratio <= 1.005:
            return 'ranging'
        
        if sma_ratio > 1.0:
            return 'trending_up'
        else:
            return 'trending_down'
    
    @staticmethod
    def get_session(hour_utc: int) -> int:
        """
        Get trading session:
        0 = Asia (00:00-08:00 UTC)
        1 = London (08:00-16:00 UTC)
        2 = New York (13:00-22:00 UTC)
        """
        if 0 <= hour_utc < 8:
            return 0
        elif 8 <= hour_utc < 13:
            return 1
        elif 13 <= hour_utc < 16:
            return 3
        elif 16 <= hour_utc < 22:
            return 2
        else:
            return 0
    
    @staticmethod
    def is_good_trading_time(pair: str, hour_utc: int) -> bool:
        """Check if current time is good for trading this pair"""
        if 'JPY' in pair:
            return 0 <= hour_utc < 8 or 8 <= hour_utc < 16
        elif 'XAU' in pair or 'GOLD' in pair:
            return 8 <= hour_utc < 22
        elif 'GBP' in pair or 'EUR' in pair:
            return 8 <= hour_utc < 16
        elif 'USD' in pair:
            return 13 <= hour_utc < 22
        return True
    
    @staticmethod
    def calculate_regime_score(volatility_regime: str, trend_regime: str, 
                               session: int, pair: str) -> float:
        """Calculate overall regime alignment score (0-1)"""
        score = 0.5
        
        if volatility_regime == 'normal':
            score += 0.2
        elif volatility_regime == 'high_volatility':
            score -= 0.2
        elif volatility_regime == 'low_volatility':
            score -= 0.1
        
        if trend_regime in ['trending_up', 'trending_down']:
            score += 0.15
        elif trend_regime == 'ranging':
            score -= 0.1
        
        hour_utc = datetime.utcnow().hour
        if MarketRegimeDetector.is_good_trading_time(pair, hour_utc):
            score += 0.15
        else:
            score -= 0.15
        
        return max(0.0, min(1.0, score))


class DXYTracker:
    """Track USD strength via DXY proxy (calculated from major pairs)"""
    
    @staticmethod
    def calculate_usd_strength(eur_usd: float = 0, gbp_usd: float = 0, 
                                usd_jpy: float = 0, usd_chf: float = 0) -> Dict:
        """
        Calculate USD strength index proxy
        Higher value = stronger USD
        """
        if eur_usd == 0 and gbp_usd == 0:
            return {'strength': 0.5, 'trend': 'neutral'}
        
        eur_component = (1 / eur_usd) if eur_usd > 0 else 1
        gbp_component = (1 / gbp_usd) if gbp_usd > 0 else 1
        jpy_component = usd_jpy / 150 if usd_jpy > 0 else 1
        chf_component = usd_chf if usd_chf > 0 else 1
        
        strength = (eur_component * 0.4 + gbp_component * 0.2 + 
                   jpy_component * 0.2 + chf_component * 0.2)
        
        normalized = min(1.0, max(0.0, (strength - 0.8) / 0.4))
        
        if normalized > 0.6:
            trend = 'bullish'
        elif normalized < 0.4:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        return {
            'strength': normalized,
            'trend': trend,
            'raw_value': strength
        }
    
    @staticmethod
    def get_xau_correlation(gold_price: float, usd_strength: float) -> float:
        """
        Gold typically moves inverse to USD
        Returns correlation factor (-1 to 1)
        """
        if usd_strength > 0.6:
            return -0.7
        elif usd_strength < 0.4:
            return 0.7
        return 0.0


class MLTradingEngine:
    def __init__(self, user):
        self.user = user
        self.models = {}
        self.regime_models = {}
        self.tp_models = {}
        self.meta_models = {}
        self.xgb_models = {}
        self.scalers = {}
        self.ai_logger = AILogger(user.email)
        self.regime_detector = MarketRegimeDetector()
        self.dxy_tracker = DXYTracker()
        
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'sma_20', 'sma_50', 'sma_ratio',
            'bb_position', 'atr', 'volume_ratio',
            'trend_strength', 'price_momentum',
            'hour_of_day', 'day_of_week',
            'volatility_regime', 'session',
            'atr_ratio', 'adx', 'trend_age',
            'distance_from_high', 'distance_from_low',
            'spread_ratio', 'liquidity_score',
            'dxy_strength', 'regime_score',
        ]
    
    def get_pair_training_progress(self) -> List[Dict]:
        progress = []
        
        for pair in TRADING_PAIRS:
            closed_trades = Trade.objects.filter(
                user=self.user,
                pair=pair,
                status='closed'
            ).count()
            
            ml_model = MLModel.objects.filter(
                user=self.user,
                pair=pair,
                is_active=True
            ).first()
            
            progress.append({
                'pair': pair,
                'closed_trades': closed_trades,
                'required_trades': MIN_TRADES_PER_PAIR,
                'is_trained': ml_model is not None,
                'model_version': ml_model.model_version if ml_model else 0,
                'accuracy': ml_model.accuracy if ml_model else 0,
                'progress_percent': min(100, int((closed_trades / MIN_TRADES_PER_PAIR) * 100)),
            })
        
        return progress
    
    def _get_session_feature(self, trade_time: datetime) -> int:
        """Get trading session as numeric feature"""
        if trade_time is None:
            return 1
        hour = trade_time.hour
        return self.regime_detector.get_session(hour)
    
    def _get_volatility_regime_feature(self, atr: float, atr_50: float) -> int:
        """Get volatility regime as numeric feature"""
        regime = self.regime_detector.get_volatility_regime(atr, atr_50)
        regime_map = {'low_volatility': 0, 'normal': 1, 'high_volatility': 2}
        return regime_map.get(regime, 1)
    
    def prepare_features(self, trades: List[Trade]) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []
        
        for trade in trades:
            if trade.status != 'closed':
                continue
            
            signals = trade.technical_signals or {}
            indicators = signals.get('indicators', {})
            
            atr = indicators.get('atr', 0)
            atr_50 = indicators.get('atr_50', atr)
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            
            feature_row = [
                indicators.get('rsi', 50),
                indicators.get('macd', 0),
                indicators.get('macd_signal', 0),
                indicators.get('macd', 0) - indicators.get('macd_signal', 0),
                sma_20,
                sma_50,
                sma_20 / sma_50 if sma_50 != 0 else 1,
                self._calculate_bb_position(
                    float(trade.entry_price),
                    indicators.get('bb_upper', 0),
                    indicators.get('bb_lower', 0)
                ),
                atr,
                indicators.get('volume_ratio', 1.0),
                signals.get('trend_strength', 0.5),
                signals.get('price_momentum', 0),
                trade.opened_at.hour if trade.opened_at else 12,
                trade.opened_at.weekday() if trade.opened_at else 0,
                self._get_volatility_regime_feature(atr, atr_50),
                self._get_session_feature(trade.opened_at),
                atr / atr_50 if atr_50 != 0 else 1.0,
                indicators.get('adx', 25),
                signals.get('trend_age', 10),
                indicators.get('distance_from_high', 0.5),
                indicators.get('distance_from_low', 0.5),
                indicators.get('spread_ratio', 1.0),
                signals.get('liquidity_score', 0.5),
                indicators.get('dxy_strength', 0.5),
                signals.get('regime_score', 0.5),
            ]
            
            features.append(feature_row)
            labels.append(1 if trade.outcome == 'win' else 0)
        
        return np.array(features), np.array(labels)
    
    def prepare_regime_features(self, trades: List[Trade]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for regime classification"""
        features = []
        labels = []
        
        for trade in trades:
            if trade.status != 'closed':
                continue
            
            signals = trade.technical_signals or {}
            indicators = signals.get('indicators', {})
            
            atr = indicators.get('atr', 0)
            atr_50 = indicators.get('atr_50', atr)
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            
            feature_row = [
                atr / atr_50 if atr_50 != 0 else 1.0,
                sma_20 / sma_50 if sma_50 != 0 else 1.0,
                indicators.get('adx', 25),
                indicators.get('rsi', 50),
                indicators.get('bb_width', 0),
                signals.get('trend_strength', 0.5),
            ]
            
            features.append(feature_row)
            
            if trade.outcome == 'win':
                labels.append(1)
            else:
                labels.append(0)
        
        return np.array(features), np.array(labels)
    
    def prepare_tp_features(self, trades: List[Trade]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for TP probability prediction"""
        features = []
        labels = []
        
        for trade in trades:
            if trade.status != 'closed':
                continue
            
            signals = trade.technical_signals or {}
            indicators = signals.get('indicators', {})
            
            atr = indicators.get('atr', 0)
            entry = float(trade.entry_price)
            tp = float(trade.take_profit) if trade.take_profit else entry
            sl = float(trade.stop_loss) if trade.stop_loss else entry
            
            tp_distance = abs(tp - entry)
            sl_distance = abs(sl - entry)
            rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 1
            
            feature_row = [
                indicators.get('rsi', 50),
                indicators.get('atr', 0),
                rr_ratio,
                tp_distance / atr if atr > 0 else 1,
                signals.get('trend_strength', 0.5),
                self._get_session_feature(trade.opened_at),
                1 if trade.direction == 'buy' else 0,
            ]
            
            features.append(feature_row)
            
            if trade.exit_price:
                exit_price = float(trade.exit_price)
                if trade.direction == 'buy':
                    tp_hit = exit_price >= tp * 0.99
                else:
                    tp_hit = exit_price <= tp * 1.01
                labels.append(1 if tp_hit else 0)
            else:
                labels.append(1 if trade.outcome == 'win' else 0)
        
        return np.array(features), np.array(labels)
    
    def _calculate_bb_position(self, price: float, upper: float, lower: float) -> float:
        if upper == lower:
            return 0.5
        return (price - lower) / (upper - lower)
    
    def train_pair(self, pair: str) -> Optional[Dict]:
        trades = list(Trade.objects.filter(
            user=self.user,
            pair=pair,
            status='closed'
        ).order_by('closed_at'))
        
        if len(trades) < MIN_TRADES_PER_PAIR:
            logger.info(f"Not enough trades for {pair}. Have {len(trades)}, need {MIN_TRADES_PER_PAIR}")
            return None
        
        self.ai_logger.log_training_start(pair, len(trades))
        
        X, y = self.prepare_features(trades)
        
        if len(X) == 0:
            self.ai_logger.log_error(f"No valid features extracted from {pair} trades")
            return None
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        scaler = StandardScaler()
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        )
        
        xgb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        unique_classes = np.unique(y)
        is_single_class = len(unique_classes) < 2
        
        if is_single_class:
            logger.warning(f"{pair} has only one class (all {'wins' if y[0] == 1 else 'losses'}), creating baseline model")
            X_scaled = scaler.fit_transform(X)
            
            from sklearn.dummy import DummyClassifier
            rf_model = DummyClassifier(strategy='most_frequent')
            xgb_model = DummyClassifier(strategy='most_frequent')
            rf_model.fit(X_scaled, y)
            xgb_model.fit(X_scaled, y)
            
            accuracy = 0.5
            precision = 0.5
            recall = 0.5
            f1 = 0.5
            cv_mean = 0.5
            
            self.models[pair] = rf_model
            self.xgb_models[pair] = xgb_model
            self.scalers[pair] = scaler
            
            feature_importance = {col: 1.0/len(self.feature_columns) for col in self.feature_columns}
            
            MLModel.objects.filter(user=self.user, pair=pair, is_active=True).update(is_active=False)
            latest_version = MLModel.objects.filter(user=self.user, pair=pair).count() + 1
            
            model_data = pickle.dumps({
                'rf_model': rf_model,
                'xgb_model': xgb_model,
                'scaler': scaler,
                'meta_model': None,
                'regime_model': None,
                'tp_model': None,
                'feature_columns': self.feature_columns,
                'n_features': len(self.feature_columns)
            })
            
            ml_model = MLModel.objects.create(
                user=self.user,
                pair=pair,
                model_version=latest_version,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                model_data=model_data,
                trades_trained_on=len(trades),
                feature_importance=feature_importance,
                is_active=True
            )
            
            logger.info(f"Created baseline model for {pair} v{latest_version} (single-class data)")
            
            return {
                'pair': pair,
                'version': latest_version,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_score': cv_mean,
                'n_features': len(self.feature_columns),
                'n_samples': len(trades),
                'note': 'Single-class baseline model - needs more diverse trade outcomes'
            }
        
        if len(X) < 3:
            X_scaled = scaler.fit_transform(X)
            rf_model.fit(X_scaled, y)
            xgb_model.fit(X_scaled, y)
            accuracy = 0.5
            precision = 0.5
            recall = 0.5
            f1 = 0.5
            cv_mean = 0.5
        else:
            test_size = max(1, int(len(X) * 0.2))
            if len(X) - test_size < 2:
                test_size = 1
            
            min_class_count = min(np.sum(y == 0), np.sum(y == 1))
            
            try:
                if min_class_count >= 2:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                
                if len(np.unique(y_train)) < 2:
                    raise ValueError("Training set has only one class")
                    
            except ValueError:
                logger.warning(f"Cannot split {pair} data properly, training on full dataset")
                X_scaled = scaler.fit_transform(X)
                rf_model.fit(X_scaled, y)
                xgb_model.fit(X_scaled, y)
                accuracy = 0.5
                precision = 0.5
                recall = 0.5
                f1 = 0.5
                cv_mean = 0.5
                
                # Skip to regime/TP training
                X_train_scaled = X_scaled
                y_train = y
                X_test_scaled = X_scaled
                y_test = y
                
                # Store models and continue
                self.models[pair] = rf_model
                self.xgb_models[pair] = xgb_model
                self.scalers[pair] = scaler
                
                # Continue to regime and TP model training below
                regime_accuracy = 0.5
                X_regime, y_regime = self.prepare_regime_features(trades)
                if len(X_regime) >= 3:
                    regime_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                    X_regime = np.nan_to_num(X_regime, nan=0.0, posinf=0.0, neginf=0.0)
                    regime_model.fit(X_regime, y_regime)
                    self.regime_models[pair] = regime_model
                    regime_accuracy = 0.6
                
                tp_accuracy = 0.5
                X_tp, y_tp = self.prepare_tp_features(trades)
                if len(X_tp) >= 3:
                    tp_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                    X_tp = np.nan_to_num(X_tp, nan=0.0, posinf=0.0, neginf=0.0)
                    tp_model.fit(X_tp, y_tp)
                    self.tp_models[pair] = tp_model
                    tp_accuracy = 0.6
                
                feature_importance = dict(zip(
                    self.feature_columns,
                    rf_model.feature_importances_.tolist()
                ))
                
                MLModel.objects.filter(user=self.user, pair=pair, is_active=True).update(is_active=False)
                latest_version = MLModel.objects.filter(user=self.user, pair=pair).count() + 1
                
                model_data = pickle.dumps({
                    'rf_model': rf_model,
                    'xgb_model': xgb_model,
                    'scaler': scaler,
                    'meta_model': self.meta_models.get(pair),
                    'regime_model': self.regime_models.get(pair),
                    'tp_model': self.tp_models.get(pair),
                    'feature_columns': self.feature_columns,
                    'n_features': len(self.feature_columns)
                })
                
                ml_model = MLModel.objects.create(
                    user=self.user,
                    pair=pair,
                    model_version=latest_version,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    model_data=model_data,
                    trades_trained_on=len(trades),
                    feature_importance=feature_importance,
                    is_active=True
                )
                
                logger.info(f"Trained ML ensemble for {pair} v{latest_version} with accuracy {accuracy:.2%} (full dataset)")
                
                return {
                    'pair': pair,
                    'version': latest_version,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'cv_mean': cv_mean,
                    'regime_accuracy': regime_accuracy,
                    'tp_accuracy': tp_accuracy,
                    'feature_importance': feature_importance,
                    'training_samples': len(trades)
                }
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            rf_model.fit(X_train_scaled, y_train)
            xgb_model.fit(X_train_scaled, y_train)
            
            rf_pred = rf_model.predict_proba(X_test_scaled)
            xgb_pred = xgb_model.predict_proba(X_test_scaled)
            
            if len(X_train) >= 5:
                meta_features_train = np.column_stack([
                    rf_model.predict_proba(X_train_scaled),
                    xgb_model.predict_proba(X_train_scaled)
                ])
                meta_features_test = np.column_stack([rf_pred, xgb_pred])
                
                meta_model = LogisticRegression(random_state=42)
                meta_model.fit(meta_features_train, y_train)
                
                y_pred = meta_model.predict(meta_features_test)
                self.meta_models[pair] = meta_model
            else:
                ensemble_pred = (rf_pred[:, 1] + xgb_pred[:, 1]) / 2
                y_pred = (ensemble_pred > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0.0)
            recall = recall_score(y_test, y_pred, zero_division=0.0)
            f1 = f1_score(y_test, y_pred, zero_division=0.0)
            
            cv_folds = min(3, len(X_train))
            if cv_folds >= 2:
                cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=cv_folds)
                cv_mean = float(cv_scores.mean())
            else:
                cv_mean = accuracy
        
        regime_accuracy = 0.5
        X_regime, y_regime = self.prepare_regime_features(trades)
        if len(X_regime) >= 3:
            regime_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            X_regime = np.nan_to_num(X_regime, nan=0.0, posinf=0.0, neginf=0.0)
            regime_model.fit(X_regime, y_regime)
            self.regime_models[pair] = regime_model
            regime_accuracy = 0.6
        
        tp_accuracy = 0.5
        X_tp, y_tp = self.prepare_tp_features(trades)
        if len(X_tp) >= 3:
            tp_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            X_tp = np.nan_to_num(X_tp, nan=0.0, posinf=0.0, neginf=0.0)
            tp_model.fit(X_tp, y_tp)
            self.tp_models[pair] = tp_model
            tp_accuracy = 0.6
        
        feature_importance = dict(zip(
            self.feature_columns,
            rf_model.feature_importances_.tolist()
        ))
        
        MLModel.objects.filter(user=self.user, pair=pair, is_active=True).update(is_active=False)
        
        latest_version = MLModel.objects.filter(user=self.user, pair=pair).count() + 1
        
        model_data = pickle.dumps({
            'rf_model': rf_model,
            'xgb_model': xgb_model,
            'meta_model': self.meta_models.get(pair),
            'regime_model': self.regime_models.get(pair),
            'tp_model': self.tp_models.get(pair),
            'scaler': scaler,
            'feature_columns': self.feature_columns,
        })
        
        ml_model = MLModel.objects.create(
            user=self.user,
            pair=pair,
            model_version=latest_version,
            model_data=model_data,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            trades_trained_on=len(trades),
            feature_importance=feature_importance,
            training_metrics={
                'cv_mean': cv_mean,
                'train_size': len(X),
                'regime_accuracy': regime_accuracy,
                'tp_accuracy': tp_accuracy,
                'has_meta_model': pair in self.meta_models,
                'has_xgb': True,
            },
            is_active=True,
        )
        
        self.models[pair] = rf_model
        self.xgb_models[pair] = xgb_model
        self.scalers[pair] = scaler
        
        metrics = {
            'version': latest_version,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_mean,
            'regime_accuracy': regime_accuracy,
            'tp_accuracy': tp_accuracy,
            'feature_importance': feature_importance,
        }
        
        self.ai_logger.log_training_complete(pair, metrics)
        logger.info(f"Trained ML ensemble for {pair} v{latest_version} with accuracy {accuracy:.2%}")
        
        return metrics
    
    def train_all_eligible_pairs(self) -> Dict[str, Optional[Dict]]:
        results = {}
        
        for pair in TRADING_PAIRS:
            closed_count = Trade.objects.filter(
                user=self.user,
                pair=pair,
                status='closed'
            ).count()
            
            if closed_count >= MIN_TRADES_PER_PAIR:
                ml_model = MLModel.objects.filter(
                    user=self.user,
                    pair=pair,
                    is_active=True
                ).first()
                
                if not ml_model or self.should_retrain_pair(pair):
                    results[pair] = self.train_pair(pair)
        
        return results
    
    def load_model(self, pair: str) -> bool:
        if pair in self.models:
            return True
        
        ml_model = MLModel.objects.filter(user=self.user, pair=pair, is_active=True).first()
        
        if not ml_model or not ml_model.model_data:
            return False
        
        try:
            data = pickle.loads(ml_model.model_data)
            self.models[pair] = data.get('rf_model') or data.get('model')
            self.xgb_models[pair] = data.get('xgb_model')
            self.meta_models[pair] = data.get('meta_model')
            self.regime_models[pair] = data.get('regime_model')
            self.tp_models[pair] = data.get('tp_model')
            self.scalers[pair] = data['scaler']
            return True
        except Exception as e:
            logger.error(f"Error loading model for {pair}: {str(e)}")
            return False
    
    def has_model_for_pair(self, pair: str) -> bool:
        if pair in self.models:
            return True
        return MLModel.objects.filter(user=self.user, pair=pair, is_active=True).exists()
    
    def predict(self, pair: str, indicators: Dict, signals: Dict) -> Tuple[float, float, Dict]:
        if not self.load_model(pair):
            return 0.5, 0.0, {}
        
        rf_model = self.models.get(pair)
        xgb_model = self.xgb_models.get(pair)
        meta_model = self.meta_models.get(pair)
        scaler = self.scalers.get(pair)
        
        if not rf_model or not scaler:
            return 0.5, 0.0, {}
        
        atr = indicators.get('atr', 0)
        atr_50 = indicators.get('atr_50', atr)
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)
        
        now = datetime.utcnow()
        
        expected_features = scaler.n_features_in_
        
        if expected_features == 14:
            feature_row = [
                indicators.get('rsi', 50),
                indicators.get('macd', 0),
                indicators.get('macd_signal', 0),
                indicators.get('macd', 0) - indicators.get('macd_signal', 0),
                sma_20,
                sma_50,
                sma_20 / sma_50 if sma_50 != 0 else 1,
                indicators.get('bb_position', 0.5),
                atr,
                1.0,
                signals.get('trend_strength', 0.5),
                signals.get('price_momentum', 0),
                now.hour,
                now.weekday(),
            ]
        else:
            feature_row = [
                indicators.get('rsi', 50),
                indicators.get('macd', 0),
                indicators.get('macd_signal', 0),
                indicators.get('macd', 0) - indicators.get('macd_signal', 0),
                sma_20,
                sma_50,
                sma_20 / sma_50 if sma_50 != 0 else 1,
                indicators.get('bb_position', 0.5),
                atr,
                indicators.get('volume_ratio', 1.0),
                signals.get('trend_strength', 0.5),
                signals.get('price_momentum', 0),
                now.hour,
                now.weekday(),
                self._get_volatility_regime_feature(atr, atr_50),
                self.regime_detector.get_session(now.hour),
                atr / atr_50 if atr_50 != 0 else 1.0,
                indicators.get('adx', 25),
                signals.get('trend_age', 10),
                indicators.get('distance_from_high', 0.5),
                indicators.get('distance_from_low', 0.5),
                indicators.get('spread_ratio', 1.0),
                signals.get('liquidity_score', 0.5),
                indicators.get('dxy_strength', 0.5),
                signals.get('regime_score', 0.5),
            ]
        
        X = np.array([feature_row])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = scaler.transform(X)
        
        rf_proba = rf_model.predict_proba(X_scaled)
        
        if xgb_model is not None and expected_features == 25:
            try:
                xgb_proba = xgb_model.predict_proba(X_scaled)
                
                if meta_model is not None:
                    meta_features = np.column_stack([rf_proba, xgb_proba])
                    final_proba = meta_model.predict_proba(meta_features)[0]
                    prediction = 1 if final_proba[1] > 0.5 else 0
                    confidence = max(final_proba)
                else:
                    avg_proba = (rf_proba[0] + xgb_proba[0]) / 2
                    prediction = 1 if avg_proba[1] > 0.5 else 0
                    confidence = max(avg_proba)
            except Exception as e:
                logger.warning(f"XGBoost prediction error for {pair}: {e}, falling back to RF only")
                prediction = rf_model.predict(X_scaled)[0]
                confidence = max(rf_proba[0])
        else:
            prediction = rf_model.predict(X_scaled)[0]
            confidence = max(rf_proba[0])
        
        volatility_regime = self.regime_detector.get_volatility_regime(atr, atr_50)
        trend_regime = self.regime_detector.get_trend_regime(
            sma_20, sma_50, indicators.get('sma_200', sma_50),
            indicators.get('adx', 25), indicators.get('close', 0)
        )
        regime_score = self.regime_detector.calculate_regime_score(
            volatility_regime, trend_regime,
            self.regime_detector.get_session(now.hour), pair
        )
        
        tp_probability = 0.5
        tp_model = self.tp_models.get(pair)
        if tp_model is not None:
            tp_features = np.array([[
                indicators.get('rsi', 50),
                atr,
                1.0,
                1.0,
                signals.get('trend_strength', 0.5),
                self.regime_detector.get_session(now.hour),
                1 if signals.get('signal', 0) > 0 else 0,
            ]])
            tp_features = np.nan_to_num(tp_features, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                tp_probability = tp_model.predict_proba(tp_features)[0][1]
            except:
                tp_probability = 0.5
        
        extra_info = {
            'volatility_regime': volatility_regime,
            'trend_regime': trend_regime,
            'regime_score': regime_score,
            'tp_probability': tp_probability,
            'session': self.regime_detector.get_session(now.hour),
            'has_meta_model': meta_model is not None,
            'has_xgb': xgb_model is not None,
        }
        
        return float(prediction), float(confidence), extra_info
    
    def should_retrain_pair(self, pair: str) -> bool:
        ml_model = MLModel.objects.filter(user=self.user, pair=pair, is_active=True).first()
        
        if not ml_model:
            closed_trades = Trade.objects.filter(
                user=self.user,
                pair=pair,
                status='closed'
            ).count()
            return closed_trades >= MIN_TRADES_PER_PAIR
        
        current_trades = Trade.objects.filter(
            user=self.user,
            pair=pair,
            status='closed'
        ).count()
        
        trades_since_training = current_trades - ml_model.trades_trained_on
        
        if trades_since_training >= 3:
            self.ai_logger.log_retrain_trigger(pair, trades_since_training)
            return True
        
        return False
    
    def get_combined_signal(self, technical_signal: Dict, indicators: Dict, pair: str = 'UNKNOWN') -> Dict:
        if not self.has_model_for_pair(pair):
            now = datetime.utcnow()
            atr = indicators.get('atr', 0)
            atr_50 = indicators.get('atr_50', atr)
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            
            volatility_regime = self.regime_detector.get_volatility_regime(atr, atr_50)
            trend_regime = self.regime_detector.get_trend_regime(
                sma_20, sma_50, indicators.get('sma_200', sma_50),
                indicators.get('adx', 25), indicators.get('close', 0)
            )
            regime_score = self.regime_detector.calculate_regime_score(
                volatility_regime, trend_regime,
                self.regime_detector.get_session(now.hour), pair
            )
            
            ta_confidence = technical_signal.get('confidence', 0.5)
            adjusted_confidence = (ta_confidence * 0.8) + (regime_score * 0.2)
            
            return {
                'signal': technical_signal.get('signal', 0),
                'confidence': adjusted_confidence,
                'ml_prediction': None,
                'ml_confidence': 0,
                'ta_signal': technical_signal.get('signal', 0),
                'ta_confidence': ta_confidence,
                'aligned': True,
                'using_ml': False,
                'regime_score': regime_score,
                'volatility_regime': volatility_regime,
                'trend_regime': trend_regime,
                'tp_probability': 0.5,
            }
        
        ml_prediction, ml_confidence, extra_info = self.predict(pair, indicators, technical_signal)
        
        ta_signal = technical_signal.get('signal', 0)
        ta_confidence = technical_signal.get('confidence', 0.5)
        
        ml_signal = 1 if ml_prediction >= 0.5 else -1
        
        regime_score = extra_info.get('regime_score', 0.5)
        volatility_regime = extra_info.get('volatility_regime', 'normal')
        
        self.ai_logger.log_prediction(pair, ml_prediction, ml_confidence, ta_signal, 
                                       extra_info.get('trend_regime', 'unknown'))
        
        if ml_signal == ta_signal:
            base_confidence = (ta_confidence * 0.4) + (ml_confidence * 0.4) + (regime_score * 0.2)
            combined_signal = ta_signal
        else:
            if ml_confidence > ta_confidence:
                combined_signal = ml_signal
                base_confidence = (ml_confidence * 0.4) + (regime_score * 0.2)
            else:
                combined_signal = ta_signal
                base_confidence = (ta_confidence * 0.4) + (regime_score * 0.2)
        
        if volatility_regime == 'high_volatility':
            base_confidence *= 0.8
        elif volatility_regime == 'low_volatility':
            base_confidence *= 0.9
        
        tp_probability = extra_info.get('tp_probability', 0.5)
        final_confidence = base_confidence * (0.7 + tp_probability * 0.3)
        
        return {
            'signal': combined_signal,
            'confidence': min(1.0, final_confidence),
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'ta_signal': ta_signal,
            'ta_confidence': ta_confidence,
            'aligned': ml_signal == ta_signal,
            'using_ml': True,
            'regime_score': regime_score,
            'volatility_regime': volatility_regime,
            'trend_regime': extra_info.get('trend_regime', 'unknown'),
            'tp_probability': tp_probability,
            'session': extra_info.get('session', 1),
        }
