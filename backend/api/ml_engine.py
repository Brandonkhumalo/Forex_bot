import numpy as np
import pandas as pd
import pickle
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta

from .models import Trade, MLModel, MarketData

logger = logging.getLogger(__name__)

AI_LOGS_DIR = Path(__file__).resolve().parent.parent / 'ai_logs'

MIN_TRADES_PER_PAIR = 5

TRADING_PAIRS = [
    'GBP/USD', 'EUR/USD', 'USD/JPY', 'AUD/USD',
    'USD/CAD', 'NZD/USD', 'USD/CHF', 'XAU/USD'
]


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
        self.log(f"-" * 40)
        self.log(f"TOP FEATURE IMPORTANCE:")
        importance = metrics.get('feature_importance', {})
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_features[:5]:
            self.log(f"  {feature}: {imp:.4f}")
        self.log(f"=" * 60)
    
    def log_prediction(self, pair: str, prediction: float, confidence: float, ta_signal: float):
        self.log(f"PREDICTION: {pair} | ML={prediction:.2f} ({confidence:.1%}) | TA={ta_signal}")
    
    def log_retrain_trigger(self, pair: str, trades_since: int):
        self.log(f"RETRAIN TRIGGERED for {pair}: {trades_since} new trades since last training")
    
    def log_error(self, error_message: str):
        self.log(f"ERROR: {error_message}", level='ERROR')


class MLTradingEngine:
    def __init__(self, user):
        self.user = user
        self.models = {}
        self.scalers = {}
        self.ai_logger = AILogger(user.email)
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'sma_20', 'sma_50', 'sma_ratio',
            'bb_position', 'atr', 'volume_ratio',
            'trend_strength', 'price_momentum',
            'hour_of_day', 'day_of_week',
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
    
    def prepare_features(self, trades: List[Trade]) -> Tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []
        
        for trade in trades:
            if trade.status != 'closed':
                continue
            
            signals = trade.technical_signals or {}
            indicators = signals.get('indicators', {})
            
            feature_row = [
                indicators.get('rsi', 50),
                indicators.get('macd', 0),
                indicators.get('macd_signal', 0),
                indicators.get('macd', 0) - indicators.get('macd_signal', 0),
                indicators.get('sma_20', 0),
                indicators.get('sma_50', 0),
                indicators.get('sma_20', 1) / indicators.get('sma_50', 1) if indicators.get('sma_50', 0) else 1,
                self._calculate_bb_position(
                    float(trade.entry_price),
                    indicators.get('bb_upper', 0),
                    indicators.get('bb_lower', 0)
                ),
                indicators.get('atr', 0),
                1.0,
                signals.get('trend_strength', 0.5),
                signals.get('price_momentum', 0),
                trade.opened_at.hour if trade.opened_at else 12,
                trade.opened_at.weekday() if trade.opened_at else 0,
            ]
            
            features.append(feature_row)
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
        
        if len(X) < 3:
            X_scaled = scaler.fit_transform(X)
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                class_weight='balanced'
            )
            model.fit(X_scaled, y)
            accuracy = 0.5
            precision = 0.5
            recall = 0.5
            f1 = 0.5
            cv_mean = 0.5
        else:
            test_size = max(1, int(len(X) * 0.2))
            if len(X) - test_size < 2:
                test_size = 1
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0.5)
            recall = recall_score(y_test, y_pred, zero_division=0.5)
            f1 = f1_score(y_test, y_pred, zero_division=0.5)
            
            cv_folds = min(3, len(X_train))
            if cv_folds >= 2:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
                cv_mean = float(cv_scores.mean())
            else:
                cv_mean = accuracy
        
        feature_importance = dict(zip(
            self.feature_columns,
            model.feature_importances_.tolist()
        ))
        
        MLModel.objects.filter(user=self.user, pair=pair, is_active=True).update(is_active=False)
        
        latest_version = MLModel.objects.filter(user=self.user, pair=pair).count() + 1
        
        model_data = pickle.dumps({
            'model': model,
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
            },
            is_active=True,
        )
        
        self.models[pair] = model
        self.scalers[pair] = scaler
        
        metrics = {
            'version': latest_version,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_mean,
            'feature_importance': feature_importance,
        }
        
        self.ai_logger.log_training_complete(pair, metrics)
        logger.info(f"Trained ML model for {pair} v{latest_version} with accuracy {accuracy:.2%}")
        
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
            self.models[pair] = data['model']
            self.scalers[pair] = data['scaler']
            return True
        except Exception as e:
            logger.error(f"Error loading model for {pair}: {str(e)}")
            return False
    
    def has_model_for_pair(self, pair: str) -> bool:
        if pair in self.models:
            return True
        return MLModel.objects.filter(user=self.user, pair=pair, is_active=True).exists()
    
    def predict(self, pair: str, indicators: Dict, signals: Dict) -> Tuple[float, float]:
        if not self.load_model(pair):
            return 0.5, 0.0
        
        model = self.models.get(pair)
        scaler = self.scalers.get(pair)
        
        if not model or not scaler:
            return 0.5, 0.0
        
        feature_row = [
            indicators.get('rsi', 50),
            indicators.get('macd', 0),
            indicators.get('macd_signal', 0),
            indicators.get('macd', 0) - indicators.get('macd_signal', 0),
            indicators.get('sma_20', 0),
            indicators.get('sma_50', 0),
            indicators.get('sma_20', 1) / indicators.get('sma_50', 1) if indicators.get('sma_50', 0) else 1,
            indicators.get('bb_position', 0.5),
            indicators.get('atr', 0),
            1.0,
            signals.get('trend_strength', 0.5),
            signals.get('price_momentum', 0),
            datetime.now().hour,
            datetime.now().weekday(),
        ]
        
        X = np.array([feature_row])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = scaler.transform(X)
        
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = max(probabilities)
        
        return float(prediction), float(confidence)
    
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
            return {
                'signal': technical_signal.get('signal', 0),
                'confidence': technical_signal.get('confidence', 0.5),
                'ml_prediction': None,
                'ml_confidence': 0,
                'ta_signal': technical_signal.get('signal', 0),
                'ta_confidence': technical_signal.get('confidence', 0.5),
                'aligned': True,
                'using_ml': False,
            }
        
        ml_prediction, ml_confidence = self.predict(pair, indicators, technical_signal)
        
        ta_signal = technical_signal.get('signal', 0)
        ta_confidence = technical_signal.get('confidence', 0.5)
        
        ml_signal = 1 if ml_prediction >= 0.5 else -1
        
        self.ai_logger.log_prediction(pair, ml_prediction, ml_confidence, ta_signal)
        
        if ml_signal == ta_signal:
            combined_confidence = (ml_confidence + ta_confidence) / 2
            combined_signal = ta_signal
        else:
            if ml_confidence > ta_confidence:
                combined_signal = ml_signal
                combined_confidence = ml_confidence * 0.6
            else:
                combined_signal = ta_signal
                combined_confidence = ta_confidence * 0.6
        
        return {
            'signal': combined_signal,
            'confidence': combined_confidence,
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'ta_signal': ta_signal,
            'ta_confidence': ta_confidence,
            'aligned': ml_signal == ta_signal,
            'using_ml': True,
        }
