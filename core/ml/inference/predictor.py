"""Volatility prediction service for integration with backtesting."""

import os
import pickle
from typing import Optional, List, Dict
import numpy as np
import pandas as pd


class VolatilityPredictor:
    """High-level interface for volatility prediction in trading system."""
    
    MODEL_DIR = "data/models"
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize predictor."""
        self.model = None
        self.model_path = model_path or os.path.join(self.MODEL_DIR, "volatility_model.pkl")
        self._is_loaded = False
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            try:
                self._load_model()
                self._is_loaded = True
            except Exception as e:
                print(f"Failed to load model: {e}")
    
    def _load_model(self):
        """Load model from disk (auto-detect type)."""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check model type
        if 'model' in data:
            # SimpleVolatilityModel or XGBoost format
            self.model = data
        else:
            raise ValueError("Unknown model format")
    
    def is_ready(self) -> bool:
        """Check if predictor is ready for inference."""
        return self._is_loaded and self.model is not None
    
    def predict_from_bars(self, bars: List[Dict]) -> Optional[float]:
        """Predict volatility from recent bars."""
        if not self._is_loaded:
            return None
        
        if len(bars) < 60:
            return None
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Calculate features
            from core.ml.features.volatility import VolatilityFeatures
            from core.ml.features.technical import TechnicalFeatures
            
            vol_features = VolatilityFeatures.calculate_volatility_features(df)
            tech_features = TechnicalFeatures.calculate_technical_features(df)
            features = pd.concat([vol_features, tech_features], axis=1)
            
            # Get latest features
            latest_features = features.iloc[-1:]
            
            # Predict using loaded model
            model_obj = self.model['model']
            feature_names = self.model.get('feature_names', list(latest_features.columns))
            
            # Align features
            latest_features = latest_features[feature_names]
            
            prediction = model_obj.predict(latest_features)[0]
            return float(prediction)
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None
