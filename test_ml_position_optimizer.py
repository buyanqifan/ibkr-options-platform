"""Test script for ML Position Optimizer with Wheel strategy.

This script demonstrates how to use ML-based position sizing optimization
for the Wheel strategy, showing the difference between traditional rule-based
and ML-optimized position sizing.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.ml.position_optimizer import MLPositionOptimizer, WheelPositionIntegration, PositionRecommendation
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_rule_based_position():
    """Test rule-based position sizing (fallback when ML not trained)."""
    print("\n" + "="*60)
    print("Testing Rule-Based Position Sizing")
    print("="*60)

    optimizer = MLPositionOptimizer()  # No model loaded
    integration = WheelPositionIntegration(optimizer=optimizer, enabled=True)

    # Test case 1: High IV environment (should increase position)
    print("\n--- Test Case 1: High IV Environment ---")
    market_data = {
        'price': 150.0,
        'iv': 0.35,
        'iv_rank': 75,  # High IV
        'iv_percentile': 80,
        'historical_volatility': 35,
        'vix': 18,  # Normal VIX
        'momentum': {'momentum_5d': 2.0, 'momentum_10d': 3.0, 'vs_ma20': 1.5, 'vs_ma50': 2.0},
    }
    portfolio_state = {
        'total_capital': 100000,
        'available_margin': 80000,
        'margin_used': 20000,
        'drawdown': 2.0,
    }
    option_info = {
        'underlying_price': 150.0,
        'strike': 140.0,  # OTM put
        'premium': 2.5,
        'dte': 35,
        'delta': 0.30,
    }

    num_contracts, recommendation = integration.get_position_size(
        symbol="AAPL",
        market_data=market_data,
        portfolio_state=portfolio_state,
        strategy_phase="SP",
        option_info=option_info,
        base_position=5,
        max_position=10,
    )

    print(f"Recommended Contracts: {num_contracts}")
    print(f"Position Multiplier: {recommendation.position_multiplier:.2f}x")
    print(f"Confidence: {recommendation.confidence:.0%}")
    print(f"Reasoning: {recommendation.reasoning}")
    print(f"Expected Return: ${recommendation.expected_return:.2f}")
    print(f"Expected Risk: ${recommendation.expected_risk:.2f}")
    print(f"Kelly Fraction: {recommendation.kelly_fraction:.2%}")

    # Test case 2: High VIX environment (should decrease position)
    print("\n--- Test Case 2: High VIX Environment ---")
    market_data['vix'] = 32  # High VIX
    market_data['iv_rank'] = 50

    num_contracts, recommendation = integration.get_position_size(
        symbol="AAPL",
        market_data=market_data,
        portfolio_state=portfolio_state,
        strategy_phase="SP",
        option_info=option_info,
        base_position=5,
        max_position=10,
    )

    print(f"Recommended Contracts: {num_contracts}")
    print(f"Position Multiplier: {recommendation.position_multiplier:.2f}x")
    print(f"Reasoning: {recommendation.reasoning}")

    # Test case 3: Drawdown scenario (should decrease position)
    print("\n--- Test Case 3: Drawdown Scenario ---")
    market_data['vix'] = 20
    portfolio_state['drawdown'] = 12.0  # 12% drawdown

    num_contracts, recommendation = integration.get_position_size(
        symbol="AAPL",
        market_data=market_data,
        portfolio_state=portfolio_state,
        strategy_phase="SP",
        option_info=option_info,
        base_position=5,
        max_position=10,
    )

    print(f"Recommended Contracts: {num_contracts}")
    print(f"Position Multiplier: {recommendation.position_multiplier:.2f}x")
    print(f"Reasoning: {recommendation.reasoning}")

    # Test case 4: CC phase (should have different behavior)
    print("\n--- Test Case 4: CC Phase with Shares ---")
    portfolio_state['drawdown'] = 0
    portfolio_state['cost_basis'] = 145.0
    market_data['iv_rank'] = 55

    num_contracts, recommendation = integration.get_position_size(
        symbol="AAPL",
        market_data=market_data,
        portfolio_state=portfolio_state,
        strategy_phase="CC",
        option_info=option_info,
        base_position=5,
        max_position=10,
    )

    print(f"Recommended Contracts: {num_contracts}")
    print(f"Position Multiplier: {recommendation.position_multiplier:.2f}x")
    print(f"Reasoning: {recommendation.reasoning}")


def test_feature_engineering():
    """Test feature engineering for ML model."""
    print("\n" + "="*60)
    print("Testing Feature Engineering")
    print("="*60)

    optimizer = MLPositionOptimizer()

    market_data = {
        'price': 180.0,
        'iv': 0.28,
        'iv_rank': 45,
        'iv_percentile': 50,
        'historical_volatility': 30,
        'vix': {
            'vix': 22,
            'vix_percentile': 55,
            'vix_rank': 50,
            'vix_change_pct': 5.0,
            'vix_term_structure': -2.0,
        },
        'momentum': {'momentum_5d': 1.5, 'momentum_10d': 2.5, 'vs_ma20': 3.0, 'vs_ma50': 4.0},
    }

    portfolio_state = {
        'total_capital': 100000,
        'available_margin': 60000,
        'margin_used': 40000,
        'drawdown': 5.0,
        'positions': [
            {'market_value': 20000},
            {'market_value': 15000},
        ],
    }

    option_info = {
        'underlying_price': 180.0,
        'strike': 170.0,
        'premium': 3.5,
        'dte': 40,
        'delta': 0.28,
    }

    features = optimizer.build_features(market_data, portfolio_state, "SP", option_info)

    print("\nGenerated Features:")
    print("-" * 40)
    for col in features.columns:
        print(f"  {col}: {features[col].values[0]:.4f}")


def test_assignment_probability():
    """Test assignment probability estimation."""
    print("\n" + "="*60)
    print("Testing Assignment Probability Estimation")
    print("="*60)

    optimizer = MLPositionOptimizer()

    test_cases = [
        (0.30, 30, 5.0, "Normal OTM Put"),
        (0.50, 10, 2.0, "Near ITM, Short DTE"),
        (0.20, 45, 8.0, "Deep OTM, Long DTE"),
        (0.40, 5, 1.0, "Near ATM, Expiring Soon"),
    ]

    print("\nAssignment Probability Estimates:")
    print("-" * 60)
    for delta, dte, distance, description in test_cases:
        prob = optimizer._estimate_assignment_probability(delta, dte, distance)
        print(f"  {description}:")
        print(f"    Delta: {delta}, DTE: {dte}, Distance: {distance}%")
        print(f"    Assignment Probability: {prob:.1%}\n")


def test_wheel_strategy_integration():
    """Test integration with Wheel strategy."""
    print("\n" + "="*60)
    print("Testing Wheel Strategy Integration")
    print("="*60)

    from core.backtesting.strategies.wheel import WheelStrategy

    params = {
        'symbol': 'AAPL',
        'initial_capital': 100000,
        'dte_min': 30,
        'dte_max': 45,
        'put_delta': 0.30,
        'call_delta': 0.30,
        'max_positions': 10,
        'profit_target_pct': 50,
        'stop_loss_pct': 200,
        'ml_position_optimization': True,  # Enable ML position optimization
    }

    strategy = WheelStrategy(params)

    print(f"\nStrategy initialized with ML position optimization: {strategy.ml_position_optimization}")
    print(f"Position optimizer available: {strategy.ml_position_optimizer is not None}")

    # Simulate position sizing call
    print("\nSimulating position sizing for SP phase...")

    # This would normally be called by the backtest engine
    num_contracts = strategy._calculate_ml_position_size(
        underlying_price=150.0,
        iv=0.30,
        strike=140.0,
        premium=2.5,
        dte=35,
        delta=0.30,
        position_mgr=None,  # Would be provided by engine
        strategy_phase="SP",
    )

    print(f"Recommended contracts: {num_contracts}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ML Position Optimizer Test Suite")
    print("="*60)

    test_feature_engineering()
    test_assignment_probability()
    test_rule_based_position()
    test_wheel_strategy_integration()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()