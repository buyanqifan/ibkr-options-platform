"""Simple volatility prediction model using sklearn (fallback when xgboost not available)."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import pickle
import os


class SimpleVolatilityModel:
    """Simple RandomForest model for volatility prediction (lightweight alternative)."""
    
    def __init__(self, target_horizon: int = 5):
        self.target_horizon = target_horizon
        self.model = None
        self.feature_names: Optional[list] = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix from price data."""
        from core.ml.features.volatility import VolatilityFeatures
        from core.ml.features.technical import TechnicalFeatures
        
        vol_features = VolatilityFeatures.calculate_volatility_features(df)
        tech_features = TechnicalFeatures.calculate_technical_features(df)
        
        features = pd.concat([vol_features, tech_features], axis=1)
        return features
    
    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Prepare target variable (future realized volatility)."""
        from core.ml.features.volatility import VolatilityFeatures
        
        future_prices = df['close'].shift(-self.target_horizon)
        future_returns = np.log(future_prices / df['close'])
        
        target = (
            future_returns
            .rolling(20)
            .std()
            .shift(-self.target_horizon)
            * np.sqrt(252)
            * 100
        )
        
        return target
    
    def prepare_data(self, df: pd.DataFrame, min_history: int = 60) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare complete dataset."""
        X = self.prepare_features(df)
        y = self.prepare_target(df)
        
        data = pd.concat([X, y.rename('target')], axis=1).dropna()
        
        if len(data) < min_history:
            raise ValueError(f"Insufficient data: {len(data)} < {min_history}")
        
        self.feature_names = list(X.columns)
        
        return data.drop('target', axis=1), data['target']
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2, cv_folds: int = 5) -> dict:
        """Train model with time-series cross-validation."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import mean_squared_error, mean_absolute_error
        except ImportError:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        X, y = self.prepare_data(df)
        
        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        
        print(f"Training with {cv_folds}-fold time-series CV...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            cv_scores.append(rmse)
            print(f"  Fold {fold+1}: RMSE = {rmse:.3f}")
        
        # Train final model on all training data
        self.model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)
        
        # Test set evaluation
        test_pred = self.model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTest Set Performance:")
        print(f"  RMSE: {test_rmse:.3f}")
        print(f"  MAE: {test_mae:.3f}")
        print(f"\nTop 10 Features:")
        print(importance.head(10).to_string(index=False))
        
        return {
            'cv_rmse_mean': np.mean(cv_scores),
            'cv_rmse_std': np.std(cv_scores),
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'feature_importance': importance,
        }
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.feature_names:
            features = features[self.feature_names]
        
        return self.model.predict(features)
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'target_horizon': self.target_horizon,
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.target_horizon = data['target_horizon']
        print(f"Model loaded from {path}")
