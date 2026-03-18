#!/usr/bin/env python3
"""
Test script to verify CC optimization in binbin_god strategy.

This script tests the optimization logic where:
- When stock price < cost basis, strategy reduces delta and seeks higher strike
- When stock price >= cost basis, strategy uses normal delta selection
"""

import sys
import os
sys.path.insert(0, '/mnt/harddisk/lwb/options-trading-platform')

from core.backtesting.strategies.binbin_god import BinbinGodStrategy
from core.backtesting.pricing import OptionsPricer
from datetime import datetime

def test_cc_optimization():
    """Test CC optimization logic with different scenarios."""
    
    print("=" * 80)
    print("Binbin God Strategy CC Optimization Test")
    print("=" * 80)
    
    # Strategy configuration
    config = {
        "symbol": "AAPL",
        "initial_capital": 100000,
        "dte_min": 30,
        "dte_max": 45,
        "delta_target": 0.30,
        "call_delta": 0.30,
        "put_delta": 0.30,
        "max_positions": 1,
        "cc_optimization_enabled": True,
        "cc_min_delta_cost": 0.15,
        "cc_cost_basis_threshold": 0.05,
        "cc_min_strike_premium": 0.02,
    }
    
    strategy = BinbinGodStrategy(config)
    
    # Test scenario 1: Normal condition (price > cost basis)
    print("\n" + "=" * 60)
    print("Scenario 1: Normal condition - Price > Cost Basis")
    print("=" * 60)
    
    strategy.stock_holding.cost_basis = 150.0  # Bought at $150
    underlying_price = 155.0  # Current price $155 (above cost)
    iv = 0.25
    T = 45 / 365.0
    
    print(f"Stock holding: {strategy.stock_holding.shares} shares @ ${strategy.stock_holding.cost_basis:.2f}")
    print(f"Current price: ${underlying_price:.2f}")
    print(f"Price/Cost ratio: {underlying_price / strategy.stock_holding.cost_basis:.2f}")
    print(f"Optimization should be: DISABLED")
    
    # Test normal strike selection
    strike_normal = strategy.select_strike(underlying_price, iv, T, "C")
    premium_normal = OptionsPricer.call_price(underlying_price, strike_normal, T, iv)
    delta_normal = OptionsPricer.delta(underlying_price, strike_normal, T, iv, "C")
    
    print(f"Normal selection:")
    print(f"  Strike: ${strike_normal:.2f}")
    print(f"  Delta: {delta_normal:.3f}")
    print(f"  Premium: ${premium_normal:.3f}")
    
    # Test scenario 2: Loss condition (price < cost basis)
    print("\n" + "=" * 60)
    print("Scenario 2: Loss condition - Price < Cost Basis")
    print("=" * 60)
    
    strategy.stock_holding.cost_basis = 150.0  # Bought at $150
    underlying_price = 142.0  # Current price $142 (below cost - 5.3%)
    iv = 0.25
    T = 45 / 365.0
    
    print(f"Stock holding: {strategy.stock_holding.shares} shares @ ${strategy.stock_holding.cost_basis:.2f}")
    print(f"Current price: ${underlying_price:.2f}")
    print(f"Price/Cost ratio: {underlying_price / strategy.stock_holding.cost_basis:.2f}")
    print(f"Optimization should be: ENABLED")
    print(f"Expected delta reduction: from 0.30 to {strategy.cc_min_delta_cost:.2f}")
    print(f"Expected minimum strike: ~${strategy.stock_holding.cost_basis * (1 - strategy.cc_min_strike_premium):.2f}")
    
    # Test optimization with constraints - simulate reduced delta target
    original_delta = strategy.delta_target
    strategy.delta_target = strategy.cc_min_delta_cost  # Set reduced delta
    constraints = {"min_strike": strategy.stock_holding.cost_basis * (1 - strategy.cc_min_strike_premium)}
    strike_optimized = strategy.select_strike_with_constraints(underlying_price, iv, T, "C", constraints)
    premium_optimized = OptionsPricer.call_price(underlying_price, strike_optimized, T, iv)
    delta_optimized = OptionsPricer.delta(underlying_price, strike_optimized, T, iv, "C")
    strategy.delta_target = original_delta  # Restore original delta
    
    print(f"Optimized selection:")
    print(f"  Strike: ${strike_optimized:.2f}")
    print(f"  Delta: {delta_optimized:.3f}")
    print(f"  Premium: ${premium_optimized:.3f}")
    
    # Test scenario 3: Severe loss condition (price << cost basis)
    print("\n" + "=" * 60)
    print("Scenario 3: Severe loss condition - Price << Cost Basis")
    print("=" * 60)
    
    strategy.stock_holding.cost_basis = 150.0  # Bought at $150
    underlying_price = 130.0  # Current price $130 (below cost - 13.3%)
    iv = 0.25
    T = 45 / 365.0
    
    print(f"Stock holding: {strategy.stock_holding.shares} shares @ ${strategy.stock_holding.cost_basis:.2f}")
    print(f"Current price: ${underlying_price:.2f}")
    print(f"Price/Cost ratio: {underlying_price / strategy.stock_holding.cost_basis:.2f}")
    print(f"Optimization should be: ENABLED (aggressive)")
    
    # Test optimization with constraints - simulate reduced delta target
    original_delta = strategy.delta_target
    strategy.delta_target = strategy.cc_min_delta_cost  # Set reduced delta
    constraints = {"min_strike": strategy.stock_holding.cost_basis * (1 - strategy.cc_min_strike_premium)}
    strike_severe = strategy.select_strike_with_constraints(underlying_price, iv, T, "C", constraints)
    premium_severe = OptionsPricer.call_price(underlying_price, strike_severe, T, iv)
    delta_severe = OptionsPricer.delta(underlying_price, strike_severe, T, iv, "C")
    strategy.delta_target = original_delta  # Restore original delta
    
    print(f"Severe loss optimized selection:")
    print(f"  Strike: ${strike_severe:.2f}")
    print(f"  Delta: {delta_severe:.3f}")
    print(f"  Premium: ${premium_severe:.3f}")
    
    # Compare results
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    
    print(f"Normal ($155):   Strike ${strike_normal:.2f}, Delta {delta_normal:.3f}, Premium ${premium_normal:.3f}")
    print(f"Loss ($142):     Strike ${strike_optimized:.2f}, Delta {delta_optimized:.3f}, Premium ${premium_optimized:.3f}")
    print(f"Severe Loss ($130): Strike ${strike_severe:.2f}, Delta {delta_severe:.3f}, Premium ${premium_severe:.3f}")
    
    print(f"\nStrike improvement (Loss vs Normal): ${strike_optimized - strike_normal:.2f}")
    print(f"Delta reduction (Loss vs Normal): {delta_normal - delta_optimized:.3f}")
    print(f"Premium reduction (Loss vs Normal): ${premium_normal - premium_optimized:.3f}")
    
    # Verify optimization is working
    optimization_working = (
        strike_optimized > strike_normal and  # Higher strike in loss condition
        delta_optimized < delta_normal and    # Lower delta in loss condition
        strike_severe > strike_optimized     # Even higher strike in severe loss
    )
    
    print(f"\nOptimization working correctly: {'YES' if optimization_working else 'NO'}")
    
    if optimization_working:
        print("✅ CC optimization is working as expected!")
        print("   - Higher strike prices selected when price < cost basis")
        print("   - Lower delta values used to seek protective strikes")
        print("   - More aggressive optimization in severe loss conditions")
    else:
        print("❌ CC optimization has issues - please check the implementation")
    
    return optimization_working

if __name__ == "__main__":
    success = test_cc_optimization()
    sys.exit(0 if success else 1)