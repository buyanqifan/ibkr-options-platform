"""XGBoost volatility prediction model."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os


class VolatilityXGBModel:
    """XGBoost model for predicting future realized volatility."""
    
    def __init__(self, target_horizon: int = 5):
        """Initialize model.
        
        Args:
            target_horizon: Days ahead to predict (default: 5)
        """
        self.target_horizon = target_horizon
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: Optional[list] = None
        
    def build_model(self, **kwargs) -> xgb.XGBRegressor:
        """Build XGBoost model with default or custom parameters."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1,
        }
        default_params.update(kwargs)
        return xgb.XGBRegressor(**default_params)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix from price data.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            Feature DataFrame
        """
        from core.ml.features.volatility import VolatilityFeatures
        from core.ml.features.technical import TechnicalFeatures
        
        # Calculate all features
        vol_features = VolatilityFeatures.calculate_volatility_features(df)
        tech_features = TechnicalFeatures.calculate_technical_features(df)
        
        # Combine features
        features = pd.concat([vol_features, tech_features], axis=1)
        
        return features
    
    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Prepare target variable (future realized volatility).
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Target series (future HV)
        """
        from core.ml.features.volatility import VolatilityFeatures
        
        # Calculate future realized volatility
        future_prices = df['close'].shift(-self.target_horizon)
        future_returns = np.log(future_prices / df['close'])
        
        # Use 20-day window for realized vol
        target = (
            future_returns
            .rolling(20)
            .std()
            .shift(-self.target_horizon)  # Align with prediction time
            * np.sqrt(252)
            * 100
        )
        
        return target
    
    def prepare_data(
        self, 
        df: pd.DataFrame,
        min_history: int = 60
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare complete dataset.
        
        Args:
            df: Raw price data
            min_history: Minimum history required
            
        Returns:
            (X, y) feature matrix and target
        """
        X = self.prepare_features(df)
        y = self.prepare_target(df)
        
        # Align and drop NaN
        data = pd.concat([X, y.rename('target')], axis=1).dropna()
        
        # Filter minimum history
        if len(data) < min_history:
            raise ValueError(f"Insufficient data: {len(data)} < {min_history}")
        
        self.feature_names = list(X.columns)
        
        return data.drop('target', axis=1), data['target']
    
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        cv_folds: int = 5,
        **model_params
    ) -> dict:
        """Train model with time-series cross-validation.
        
        Args:
            df: Training data
            test_size: Fraction for test set
            cv_folds: Number of CV folds
            **model_params: XGBoost parameters
            
        Returns:
            Training metrics
        """
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
            
            model = self.build_model(**model_params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            cv_scores.append(rmse)
            print(f"  Fold {fold+1}: RMSE = {rmse:.3f}")
        
        # Train final model on all training data
        self.model = self.build_model(**model_params)
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
        """Make predictions.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure feature order matches training
        if self.feature_names:
            features = features[self.feature_names]
        
        return self.model.predict(features)
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'target_horizon': self.target_horizon,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.target_horizon = data['target_horizon']
        print(f"Model loaded from {path}")
