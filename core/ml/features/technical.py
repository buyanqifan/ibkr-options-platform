"""Technical indicator features."""

import numpy as np
import pandas as pd


class TechnicalFeatures:
    """Calculate technical indicators as ML features."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical features."""
        features = pd.DataFrame(index=df.index)
        
        for window in [5, 10, 20, 60]:
            features[f'return_{window}d'] = df['close'].pct_change(window)
        
        features['rsi_14'] = TechnicalFeatures.calculate_rsi(df['close'], 14)
        
        if 'volume' in df.columns:
            features['volume_sma_20'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        
        return features
