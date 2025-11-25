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
    
    def log_training_start(self, num_trades: int):
        self.log(f"=" * 60)
        self.log(f"ML MODEL TRAINING STARTED")
        self.log(f"User: {self.user_email}")
        self.log(f"Total trades for training: {num_trades}")
        self.log(f"=" * 60)
    
    def log_training_complete(self, metrics: Dict):
        self.log(f"TRAINING COMPLETE - Model v{metrics.get('version', 'N/A')}")
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
    
    def log_retrain_trigger(self, trades_since: int):
        self.log(f"RETRAIN TRIGGERED: {trades_since} new trades since last training")
    
    def log_error(self, error_message: str):
        self.log(f"ERROR: {error_message}", level='ERROR')


class MLTradingEngine:
    def __init__(self, user):
        self.user = user
        self.model = None
        self.scaler = StandardScaler()
        self.ai_logger = AILogger(user.email)
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'sma_20', 'sma_50', 'sma_ratio',
            'bb_position', 'atr', 'volume_ratio',
            'trend_strength', 'price_momentum',
            'hour_of_day', 'day_of_week',
        ]
    
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
    
    def train(self, min_trades: int = 15) -> Optional[Dict]:
        trades = list(Trade.objects.filter(
            user=self.user,
            status='closed'
        ).order_by('closed_at'))
        
        if len(trades) < min_trades:
            logger.info(f"Not enough trades to train. Have {len(trades)}, need {min_trades}")
            return None
        
        self.ai_logger.log_training_start(len(trades))
        
        X, y = self.prepare_features(trades)
        
        if len(X) == 0:
            self.ai_logger.log_error("No valid features extracted from trades")
            return None
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=3)
        
        feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_.tolist()
        ))
        
        MLModel.objects.filter(user=self.user, is_active=True).update(is_active=False)
        
        latest_version = MLModel.objects.filter(user=self.user).count() + 1
        
        model_data = pickle.dumps({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
        })
        
        ml_model = MLModel.objects.create(
            user=self.user,
            model_version=latest_version,
            model_data=model_data,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            trades_trained_on=len(trades),
            feature_importance=feature_importance,
            training_metrics={
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'test_size': len(X_test),
                'train_size': len(X_train),
            },
            is_active=True,
        )
        
        metrics = {
            'version': latest_version,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': float(cv_scores.mean()),
            'feature_importance': feature_importance,
        }
        
        self.ai_logger.log_training_complete(metrics)
        logger.info(f"Trained ML model v{latest_version} with accuracy {accuracy:.2%}")
        
        return metrics
    
    def load_model(self) -> bool:
        ml_model = MLModel.objects.filter(user=self.user, is_active=True).first()
        
        if not ml_model or not ml_model.model_data:
            return False
        
        try:
            data = pickle.loads(ml_model.model_data)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_columns = data.get('feature_columns', self.feature_columns)
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, indicators: Dict, signals: Dict) -> Tuple[float, float]:
        if not self.model:
            if not self.load_model():
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
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = max(probabilities)
        
        return float(prediction), float(confidence)
    
    def should_retrain(self) -> bool:
        ml_model = MLModel.objects.filter(user=self.user, is_active=True).first()
        
        if not ml_model:
            closed_trades = Trade.objects.filter(user=self.user, status='closed').count()
            return closed_trades >= 15
        
        current_trades = Trade.objects.filter(user=self.user, status='closed').count()
        trades_since_training = current_trades - ml_model.trades_trained_on
        
        if trades_since_training >= 10:
            self.ai_logger.log_retrain_trigger(trades_since_training)
            return True
        return False
    
    def get_combined_signal(self, technical_signal: Dict, indicators: Dict, pair: str = 'UNKNOWN') -> Dict:
        ml_prediction, ml_confidence = self.predict(indicators, technical_signal)
        
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
        }
