#!/usr/bin/env python3
"""
Test script to verify ML DTE optimization in binbin_god strategy.
"""

import sys
import os
sys.path.insert(0, '/mnt/harddisk/lwb/options-trading-platform')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.backtesting.strategies.binbin_god import BinbinGodStrategy
from core.backtesting.pricing import OptionsPricer
from core.ml.dte_optimizer import DTEOptimizationConfig


def test_ml_dte_optimization():
    """Test ML DTE optimization in Binbin God strategy."""
    
    print("=" * 80)
    print("Binbin God Strategy ML DTE Optimization Test")
    print("=" * 80)
    
    # Strategy configuration with ML DTE optimization enabled
    config = {
        "symbol": "NVDA",
        "initial_capital": 100000,
        "dte_min": 21,
        "dte_max": 60,
        "delta_target": 0.30,
        "call_delta": 0.30,
        "put_delta": 0.30,
        "max_positions": 1,
        "cc_optimization_enabled": True,
        "ml_delta_optimization": True,      # Enable ML delta optimization
        "ml_dte_optimization": True,        # Enable ML DTE optimization
        "ml_adoption_rate": 0.7,
    }
    
    strategy = BinbinGodStrategy(config)
    
    print(f"Strategy initialized: {strategy.name}")
    print(f"ML Delta optimization: {strategy.ml_delta_optimization}")
    print(f"ML DTE optimization: {strategy.ml_dte_optimization}")
    
    if strategy.ml_dte_optimization and strategy.ml_integration:
        print("✓ ML integration with DTE optimization initialized successfully")
        
        # Test DTE optimization for Sell Put
        print("\n--- Testing Sell Put DTE Optimization ---")
        try:
            dte_result_sp = strategy.ml_integration.optimize_put_dte(
                symbol="NVDA",
                current_price=850.0,
                cost_basis=840.0,
                bars=[],
                options_data=[],
                iv=0.35,
                strategy_phase="SP"
            )
            print(f"✓ Sell Put DTE optimization successful:")
            print(f"  Optimal DTE: {dte_result_sp.optimal_dte_min}-{dte_result_sp.optimal_dte_max} days")
            print(f"  Confidence: {dte_result_sp.confidence:.2f}")
            print(f"  Reasoning: {dte_result_sp.reasoning}")
        except Exception as e:
            print(f"✗ Sell Put DTE optimization failed: {e}")
        
        # Test DTE optimization for Covered Call
        print("\n--- Testing Covered Call DTE Optimization ---")
        try:
            dte_result_cc = strategy.ml_integration.optimize_call_dte(
                symbol="NVDA",
                current_price=850.0,
                cost_basis=840.0,
                bars=[],
                options_data=[],
                iv=0.35,
                strategy_phase="CC"
            )
            print(f"✓ Covered Call DTE optimization successful:")
            print(f"  Optimal DTE: {dte_result_cc.optimal_dte_min}-{dte_result_cc.optimal_dte_max} days")
            print(f"  Confidence: {dte_result_cc.confidence:.2f}")
            print(f"  Reasoning: {dte_result_cc.reasoning}")
        except Exception as e:
            print(f"✗ Covered Call DTE optimization failed: {e}")
        
        # Test Delta optimization still works
        print("\n--- Testing Delta Optimization (should still work) ---")
        try:
            delta_result = strategy.ml_integration.optimize_call_delta(
                symbol="NVDA",
                current_price=850.0,
                cost_basis=840.0,
                bars=[],
                options_data=[],
                iv=0.35
            )
            print(f"✓ Call Delta optimization successful:")
            print(f"  Optimal Delta: {delta_result.optimal_delta:.3f}")
            print(f"  Confidence: {delta_result.confidence:.2f}")
            print(f"  Reasoning: {delta_result.reasoning}")
        except Exception as e:
            print(f"✗ Call Delta optimization failed: {e}")
    else:
        print("✗ ML integration not available")
        if not strategy.ml_dte_optimization:
            print("  - ML DTE optimization is disabled")
        if not strategy.ml_integration:
            print("  - ML integration object is None")


def test_strategy_with_ml_dte():
    """Test the full strategy workflow with ML DTE optimization."""
    print("\n" + "=" * 80)
    print("Full Strategy Workflow Test with ML DTE Optimization")
    print("=" * 80)
    
    config = {
        "symbol": "NVDA",
        "initial_capital": 150000,
        "dte_min": 21,
        "dte_max": 60,
        "delta_target": 0.30,
        "call_delta": 0.30,
        "put_delta": 0.30,
        "max_positions": 3,
        "cc_optimization_enabled": True,
        "ml_delta_optimization": True,
        "ml_dte_optimization": True,
        "ml_adoption_rate": 0.6,
    }
    
    strategy = BinbinGodStrategy(config)
    
    # Simulate market data for a specific date
    current_date = "2025-03-20"
    underlying_price = 850.0
    iv = 0.35
    
    print(f"Testing on {current_date} with price ${underlying_price}, IV {iv:.2f}")
    
    # Test Sell Put signal generation with ML DTE optimization
    print("\n--- Generating Sell Put Signal ---")
    try:
        # Mock position manager
        class MockPositionMgr:
            def calculate_position_size(self, margin_per_contract, max_positions):
                return 1  # Just one contract for testing
        
        mock_pos_mgr = MockPositionMgr()
        
        # Generate Sell Put signal
        put_signals = strategy._generate_backtest_put_signal(
            symbol="NVDA",
            current_date=current_date,
            underlying_price=underlying_price,
            iv=iv,
            position_mgr=mock_pos_mgr
        )
        
        if put_signals:
            signal = put_signals[0]
            print(f"✓ Sell Put signal generated:")
            print(f"  Symbol: {signal.symbol}")
            print(f"  Type: {signal.trade_type}")
            print(f"  Strike: ${signal.strike:.2f}")
            print(f"  Expiry: {signal.expiry}")
            print(f"  Quantity: {signal.quantity}")
            print(f"  Delta: {signal.delta:.3f}")
            print(f"  Premium: ${signal.premium:.2f}")
        else:
            print("✗ No Sell Put signals generated")
            
    except Exception as e:
        print(f"✗ Sell Put signal generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Covered Call signal generation with ML DTE optimization
    print("\n--- Generating Covered Call Signal ---")
    try:
        # Simulate having shares to cover calls
        strategy.stock_holding.shares = 100
        strategy.stock_holding.cost_basis = 840.0
        strategy.phase = "CC"
        
        call_signals = strategy._generate_backtest_call_signal(
            symbol="NVDA",
            current_date=current_date,
            underlying_price=underlying_price,
            iv=iv,
            position_mgr=mock_pos_mgr,
            shares_available=100
        )
        
        if call_signals:
            signal = call_signals[0]
            print(f"✓ Covered Call signal generated:")
            print(f"  Symbol: {signal.symbol}")
            print(f"  Type: {signal.trade_type}")
            print(f"  Strike: ${signal.strike:.2f}")
            print(f"  Expiry: {signal.expiry}")
            print(f"  Quantity: {signal.quantity}")
            print(f"  Delta: {signal.delta:.3f}")
            print(f"  Premium: ${signal.premium:.2f}")
        else:
            print("✗ No Covered Call signals generated")
            
    except Exception as e:
        print(f"✗ Covered Call signal generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ml_dte_optimization()
    test_strategy_with_ml_dte()
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)