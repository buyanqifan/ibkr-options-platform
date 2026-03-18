"""Volatility feature engineering."""

import numpy as np
import pandas as pd
from typing import List, Optional


class VolatilityFeatures:
    """Calculate volatility-related features for ML models."""
    
    @staticmethod
    def calculate_hv(prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate historical volatility (annualized, percentage)."""
        log_returns = np.log(prices / prices.shift(1))
        hv = log_returns.rolling(window).std() * np.sqrt(252) * 100
        return hv
    
    @staticmethod
    def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all volatility features."""
        features = pd.DataFrame(index=df.index)
        
        for window in [5, 10, 20, 60]:
            features[f'hv_{window}'] = VolatilityFeatures.calculate_hv(
                df['close'], window
            )
        
        features['vol_of_vol_20'] = features['hv_20'].rolling(20).std()
        features['hv_slope_20'] = (
            features['hv_20'] - features['hv_20'].shift(20)
        ) / 20
        
        return features
