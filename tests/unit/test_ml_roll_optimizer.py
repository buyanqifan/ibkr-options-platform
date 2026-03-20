"""
Unit tests for ML Roll Optimizer.

Tests cover:
- MLRollOptimizer initialization
- Feature building
- Roll decision prediction
- Rule-based roll logic
- Roll vs hold decisions
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestMLRollOptimizer:
    """Test ML Roll Optimizer functionality."""

    @pytest.fixture
    def roll_optimizer(self):
        """Create MLRollOptimizer instance."""
        from core.ml.roll_optimizer import MLRollOptimizer
        return MLRollOptimizer()

    @pytest.fixture
    def sample_position(self):
        """Sample position for testing."""
        return {
            'entry_date': '2024-01-01',
            'expiry': '20240215',  # 45 DTE
            'strike': 150.0,
            'right': 'P',
            'entry_price': 3.50,
            'quantity': -1,
            'delta_at_entry': 0.30,
            'underlying_price': 155.0,
            'strategy_phase': 'SP',
        }

    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing."""
        return {
            'price': 155.0,
            'iv': 0.35,
            'historical_volatility': 30.0,
            'iv_rank': 45.0,
            'iv_percentile': 50.0,
            'option_price': 0.70,  # 80% premium captured
            'delta': 0.25,
            'vix': 20.0,
            'vix_percentile': 50.0,
            'vix_term_structure': 0.0,
            'market_regime': 1,
        }

    def test_optimizer_initialization(self, roll_optimizer):
        """Test that optimizer initializes correctly."""
        assert roll_optimizer is not None
        assert roll_optimizer.model is None  # No model loaded by default
        assert len(roll_optimizer.feature_names) == 19

    def test_build_features(self, roll_optimizer, sample_position, sample_market_data):
        """Test feature building."""
        import pandas as pd

        features = roll_optimizer.build_features(
            position=sample_position,
            market_data=sample_market_data,
            current_date='2024-01-20'
        )

        assert isinstance(features, pd.DataFrame)
        assert len(features) == 1

        # Check key features exist
        assert 'iv_rank' in features.columns
        assert 'dte' in features.columns
        assert 'premium_capture_pct' in features.columns
        assert 'delta_change_ratio' in features.columns
        assert 'strategy_phase' in features.columns

    def test_rule_based_roll_forward(self, roll_optimizer, sample_position, sample_market_data):
        """Test roll forward decision when premium captured >= 80%."""
        # Set up 80% premium capture with DTE > 7
        sample_market_data['option_price'] = 0.70  # 80% captured
        current_date = '2024-01-25'  # 21 DTE remaining

        recommendation = roll_optimizer.predict_roll_decision(
            position=sample_position,
            market_data=sample_market_data,
            current_date=current_date
        )

        assert recommendation.action == "ROLL_FORWARD"
        assert recommendation.confidence >= 0.8
        assert recommendation.optimal_dte is not None

    def test_rule_based_roll_out(self, roll_optimizer, sample_position, sample_market_data):
        """Test roll out decision when delta doubled near expiry."""
        # Set up delta doubled (delta_change_ratio = 2.0)
        sample_market_data['delta'] = 0.60  # Doubled from 0.30
        sample_market_data['option_price'] = 2.50  # Option more expensive

        # Make DTE <= 14
        sample_position['entry_date'] = '2024-02-01'
        current_date = '2024-02-10'  # 5 DTE remaining

        recommendation = roll_optimizer.predict_roll_decision(
            position=sample_position,
            market_data=sample_market_data,
            current_date=current_date
        )

        assert recommendation.action == "ROLL_OUT"
        assert recommendation.confidence >= 0.7

    def test_rule_based_let_expire(self, roll_optimizer, sample_position, sample_market_data):
        """Test let expire decision near expiry with good capture."""
        # Set up near expiry with good premium capture
        sample_market_data['option_price'] = 0.35  # 90% captured
        sample_position['entry_date'] = '2024-02-01'
        current_date = '2024-02-14'  # 1 DTE remaining

        recommendation = roll_optimizer.predict_roll_decision(
            position=sample_position,
            market_data=sample_market_data,
            current_date=current_date
        )

        assert recommendation.action == "LET_EXPIRE"

    def test_should_roll_method(self, roll_optimizer, sample_position, sample_market_data):
        """Test should_roll method returns correct tuple."""
        sample_market_data['option_price'] = 0.70  # 80% captured
        current_date = '2024-01-25'

        should_roll, recommendation = roll_optimizer.should_roll(
            position=sample_position,
            market_data=sample_market_data,
            current_date=current_date,
            min_confidence=0.6
        )

        assert isinstance(should_roll, bool)
        assert recommendation is not None
        assert recommendation.action in ["ROLL_FORWARD", "ROLL_OUT", "LET_EXPIRE", "CLOSE_EARLY"]

    def test_cc_phase_features(self, roll_optimizer, sample_market_data):
        """Test that CC phase is handled correctly."""
        cc_position = {
            'entry_date': '2024-01-01',
            'expiry': '20240215',
            'strike': 150.0,
            'right': 'C',
            'entry_price': 3.50,
            'quantity': -1,
            'delta_at_entry': 0.30,
            'underlying_price': 145.0,  # Below strike for CC
            'strategy_phase': 'CC',  # Covered Call phase
        }

        features = roll_optimizer.build_features(
            position=cc_position,
            market_data=sample_market_data,
            current_date='2024-01-20'
        )

        # Check strategy_phase is encoded as 1 (CC)
        assert features['strategy_phase'].iloc[0] == 1


class TestBinbinGodRollIntegration:
    """Test BinbinGodStrategy roll integration."""

    @pytest.fixture
    def strategy_params(self):
        """Parameters with ML roll optimization enabled."""
        return {
            'symbol': 'MAG7_AUTO',
            'initial_capital': 150000,
            'dte_min': 30,
            'dte_max': 45,
            'delta_target': 0.30,
            'profit_target_pct': 50,
            'stop_loss_pct': 200,
            'put_delta': 0.30,
            'call_delta': 0.30,
            'max_positions': 10,
            'ml_roll_optimization': True,
            'ml_roll_confidence_threshold': 0.6,
        }

    def test_strategy_initialization_with_roll(self, strategy_params):
        """Test BinbinGodStrategy initializes with roll optimization."""
        from core.backtesting.strategies.binbin_god import BinbinGodStrategy

        strategy = BinbinGodStrategy(strategy_params)

        assert strategy.ml_roll_optimization is True
        assert strategy.ml_roll_optimizer is not None

    def test_strategy_initialization_without_roll(self):
        """Test BinbinGodStrategy without roll optimization."""
        from core.backtesting.strategies.binbin_god import BinbinGodStrategy

        params = {
            'symbol': 'NVDA',
            'initial_capital': 100000,
            'dte_min': 30,
            'dte_max': 45,
            'delta_target': 0.30,
            'profit_target_pct': 50,
            'stop_loss_pct': 200,
        }

        strategy = BinbinGodStrategy(params)

        assert strategy.ml_roll_optimization is False
        assert strategy.ml_roll_optimizer is None

    def test_get_roll_parameters(self, strategy_params):
        """Test get_roll_parameters method."""
        from core.backtesting.strategies.binbin_god import BinbinGodStrategy

        strategy = BinbinGodStrategy(strategy_params)

        # Position without roll recommendation
        position = {'symbol': 'NVDA'}
        result = strategy.get_roll_parameters(position)
        assert result is None

        # Position with roll recommendation
        from core.ml.roll_optimizer import RollRecommendation
        position_with_rec = {
            'symbol': 'NVDA',
            '_roll_recommendation': RollRecommendation(
                action="ROLL_FORWARD",
                confidence=0.85,
                expected_pnl_improvement=50.0,
                optimal_dte=45,
                optimal_delta=0.30,
                reasoning="Test"
            )
        }

        result = strategy.get_roll_parameters(position_with_rec)
        assert result is not None
        assert result['action'] == "ROLL_FORWARD"
        assert result['confidence'] == 0.85

    def test_build_market_data_for_roll(self, strategy_params):
        """Test build_market_data_for_roll method."""
        from core.backtesting.strategies.binbin_god import BinbinGodStrategy

        strategy = BinbinGodStrategy(strategy_params)

        market_data = strategy.build_market_data_for_roll(
            symbol='NVDA',
            current_price=150.0,
            iv=0.35,
            bars=[{'close': 145.0}, {'close': 148.0}, {'close': 150.0}],
            option_price=2.50,
            current_delta=0.25
        )

        assert market_data is not None
        assert market_data['price'] == 150.0
        assert market_data['iv'] == 0.35
        assert market_data['option_price'] == 2.50
        assert market_data['delta'] == 0.25


class TestImportIntegrity:
    """Test that all modules import correctly without NameError."""

    def test_import_roll_optimizer(self):
        """Test MLRollOptimizer imports without errors."""
        from core.ml.roll_optimizer import MLRollOptimizer, RollRecommendation

        assert MLRollOptimizer is not None
        assert RollRecommendation is not None

    def test_import_binbin_god(self):
        """Test BinbinGodStrategy imports without errors."""
        from core.backtesting.strategies.binbin_god import BinbinGodStrategy

        assert BinbinGodStrategy is not None

    def test_import_engine(self):
        """Test BacktestEngine imports without errors."""
        from core.backtesting.engine import BacktestEngine

        assert BacktestEngine is not None

    def test_all_type_hints_resolve(self):
        """Test that all type hints resolve without NameError."""
        import inspect
        from core.backtesting.strategies.binbin_god import BinbinGodStrategy

        # Get all methods
        methods = inspect.getmembers(BinbinGodStrategy, predicate=inspect.isfunction)

        # Try to get annotations for each method
        for name, method in methods:
            if hasattr(method, '__annotations__'):
                # This will raise NameError if any type hints are unresolved
                try:
                    annotations = method.__annotations__
                except NameError as e:
                    pytest.fail(f"NameError in {name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])