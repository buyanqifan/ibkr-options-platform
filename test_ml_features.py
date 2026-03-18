#!/usr/bin/env python3
"""Test ML features calculation."""

import sys
sys.path.insert(0, '/mnt/harddisk/lwb/options-trading-platform')

import pandas as pd
import numpy as np
from core.ml.features.volatility import VolatilityFeatures
from core.ml.features.technical import TechnicalFeatures


def generate_sample_data(n_days=100):
    """Generate sample OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='B')
    
    # Generate random walk price
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    data = []
    for i, price in enumerate(prices):
        daily_vol = 0.015
        high = price * (1 + abs(np.random.normal(0, daily_vol)))
        low = price * (1 - abs(np.random.normal(0, daily_vol)))
        open_price = price * (1 + np.random.normal(0, daily_vol * 0.3))
        
        data.append({
            'date': dates[i],
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(price, 2),
            'volume': int(np.random.uniform(1e6, 1e7)),
        })
    
    return pd.DataFrame(data)


def main():
    print("Testing ML Features")
    print("=" * 50)
    
    # Generate sample data
    df = generate_sample_data(100)
    print(f"\nSample data shape: {df.shape}")
    print(df.head())
    
    # Test volatility features
    print("\n" + "-" * 50)
    print("Testing Volatility Features")
    vol_features = VolatilityFeatures.calculate_volatility_features(df)
    print(f"Features shape: {vol_features.shape}")
    print(f"Features: {list(vol_features.columns)}")
    print("\nLatest values:")
    print(vol_features.iloc[-1])
    
    # Test technical features
    print("\n" + "-" * 50)
    print("Testing Technical Features")
    tech_features = TechnicalFeatures.calculate_technical_features(df)
    print(f"Features shape: {tech_features.shape}")
    print(f"Features: {list(tech_features.columns)}")
    print("\nLatest values:")
    print(tech_features.iloc[-1])
    
    # Combine features
    print("\n" + "-" * 50)
    print("Combined Features")
    all_features = pd.concat([vol_features, tech_features], axis=1)
    all_features = all_features.dropna()
    print(f"Combined shape (after dropping NaN): {all_features.shape}")
    print(f"Total features: {len(all_features.columns)}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == '__main__':
    main()
