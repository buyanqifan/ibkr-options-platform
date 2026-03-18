#!/usr/bin/env python3
"""Train volatility prediction model on historical data."""

import argparse
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, '/mnt/harddisk/lwb/options-trading-platform')

from core.backtesting.engine import BacktestEngine
from core.ml.inference.predictor import VolatilityPredictor


def main():
    parser = argparse.ArgumentParser(description='Train volatility prediction model')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon (days)')
    parser.add_argument('--use-synthetic', action='store_true', help='Use synthetic data')
    
    args = parser.parse_args()
    
    # Default dates: 2 years of data
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    if args.start_date is None:
        start = datetime.strptime(args.end_date, '%Y-%m-%d') - timedelta(days=730)
        args.start_date = start.strftime('%Y-%m-%d')
    
    print(f"Training volatility model for {args.symbol}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Prediction horizon: {args.horizon} days")
    print("-" * 50)
    
    # Fetch historical data
    engine = BacktestEngine()
    try:
        bars = engine._get_historical_data(
            args.symbol,
            args.start_date,
            args.end_date,
            use_synthetic=args.use_synthetic
        )
        print(f"Loaded {len(bars)} bars")
    except Exception as e:
        print(f"Failed to load data: {e}")
        if not args.use_synthetic:
            print("Try using --use-synthetic flag")
        return
    
    if len(bars) < 100:
        print(f"Insufficient data: {len(bars)} bars (need at least 100)")
        return
    
    # Train model
    predictor = VolatilityPredictor()
    predictor.model.target_horizon = args.horizon
    
    metrics = predictor.train_and_save(
        bars,
        test_size=0.2,
        cv_folds=5
    )
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Model saved to: {predictor.model_path}")
    print(f"CV RMSE: {metrics['cv_rmse_mean']:.3f} (+/- {metrics['cv_rmse_std']:.3f})")
    print(f"Test RMSE: {metrics['test_rmse']:.3f}")
    print(f"Test MAE: {metrics['test_mae']:.3f}")


if __name__ == '__main__':
    main()
