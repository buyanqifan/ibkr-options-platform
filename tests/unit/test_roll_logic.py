"""
Unit tests for Roll functionality.

Tests cover:
- OptionsPricer.strike_from_delta()
- BinbinGodStrategy.generate_roll_signal()
- Engine roll logic integration
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestOptionsPricerStrikeFromDelta:
    """Test strike_from_delta method."""

    def test_put_strike_otm(self):
        """Test that put strike is OTM (below stock price) for negative delta."""
        from core.backtesting.pricing import OptionsPricer

        S = 150.0  # Stock price
        T = 45 / 365  # 45 DTE
        sigma = 0.35  # 35% IV
        target_delta = -0.30

        strike = OptionsPricer.strike_from_delta(S, T, sigma, target_delta, 'P')

        # Strike should be below stock price for OTM put
        assert strike < S, f"Put strike {strike} should be below stock price {S}"

    def test_call_strike_otm(self):
        """Test that call strike is OTM (above stock price) for positive delta."""
        from core.backtesting.pricing import OptionsPricer

        S = 150.0  # Stock price
        T = 45 / 365  # 45 DTE
        sigma = 0.35  # 35% IV
        target_delta = 0.30

        strike = OptionsPricer.strike_from_delta(S, T, sigma, target_delta, 'C')

        # Strike should be above stock price for OTM call
        assert strike > S, f"Call strike {strike} should be above stock price {S}"

    def test_put_delta_accuracy(self):
        """Test that calculated strike gives approximately target delta for put."""
        from core.backtesting.pricing import OptionsPricer

        S = 150.0
        T = 45 / 365
        sigma = 0.35
        target_delta = -0.30

        strike = OptionsPricer.strike_from_delta(S, T, sigma, target_delta, 'P')
        actual_delta = OptionsPricer.delta(S, strike, T, sigma, 'P')

        # Delta should be within 5% of target
        assert abs(actual_delta - target_delta) < 0.05, \
            f"Actual delta {actual_delta} differs from target {target_delta}"

    def test_call_delta_accuracy(self):
        """Test that calculated strike gives approximately target delta for call."""
        from core.backtesting.pricing import OptionsPricer

        S = 150.0
        T = 45 / 365
        sigma = 0.35
        target_delta = 0.30

        strike = OptionsPricer.strike_from_delta(S, T, sigma, target_delta, 'C')
        actual_delta = OptionsPricer.delta(S, strike, T, sigma, 'C')

        # Delta should be within 5% of target
        assert abs(actual_delta - target_delta) < 0.05, \
            f"Actual delta {actual_delta} differs from target {target_delta}"

    def test_different_dte(self):
        """Test strike calculation with different DTE values."""
        from core.backtesting.pricing import OptionsPricer

        S = 150.0
        sigma = 0.35
        target_delta = -0.30

        # Test different DTE values
        for dte in [21, 30, 45, 60, 90]:
            T = dte / 365
            strike = OptionsPricer.strike_from_delta(S, T, sigma, target_delta, 'P')
            assert strike < S, f"Strike {strike} should be below {S} for DTE {dte}"

    def test_different_iv(self):
        """Test strike calculation with different IV values."""
        from core.backtesting.pricing import OptionsPricer

        S = 150.0
        T = 45 / 365
        target_delta = -0.30

        # Test different IV values
        for sigma in [0.20, 0.30, 0.40, 0.50]:
            strike = OptionsPricer.strike_from_delta(S, T, sigma, target_delta, 'P')
            assert strike < S, f"Strike {strike} should be below {S} for IV {sigma}"

    def test_edge_case_low_delta(self):
        """Test with very low delta (deep OTM)."""
        from core.backtesting.pricing import OptionsPricer

        S = 150.0
        T = 45 / 365
        sigma = 0.35

        # Deep OTM put with delta -0.15
        strike = OptionsPricer.strike_from_delta(S, T, sigma, -0.15, 'P')
        assert strike < S * 0.9, "Deep OTM put strike should be significantly below stock"

    def test_edge_case_high_delta(self):
        """Test with higher delta (closer to ATM)."""
        from core.backtesting.pricing import OptionsPricer

        S = 150.0
        T = 45 / 365
        sigma = 0.35

        # Closer to ATM put with delta -0.40
        strike = OptionsPricer.strike_from_delta(S, T, sigma, -0.40, 'P')
        assert strike > S * 0.9, "Higher delta put strike should be closer to stock price"


class TestBinbinGodRollSignal:
    """Test generate_roll_signal method."""

    @pytest.fixture
    def strategy(self):
        """Create BinbinGodStrategy instance."""
        from core.backtesting.strategies.binbin_god import BinbinGodStrategy

        params = {
            'symbol': 'MAG7_AUTO',
            'initial_capital': 150000,
            'dte_min': 30,
            'dte_max': 45,
            'put_delta': 0.30,
            'call_delta': 0.30,
            'profit_target_pct': 50,
            'stop_loss_pct': 200,
        }
        return BinbinGodStrategy(params)

    def test_roll_signal_on_profit_target(self, strategy):
        """Test roll signal generation when profit target is hit."""
        closed_trade = {
            'symbol': 'NVDA',
            'right': 'P',
            'quantity': -1,
            'strike': 140.0,
            'exit_reason': 'PROFIT_TARGET',
        }

        signal = strategy.generate_roll_signal(
            closed_trade=closed_trade,
            current_date='2024-02-01',
            underlying_price=145.0,
            iv=0.35,
        )

        assert signal is not None, "Should generate roll signal"
        assert signal.symbol == 'NVDA', "Should roll same symbol"
        assert signal.right == 'P', "Should roll same option type"
        assert signal.trade_type == 'BINBIN_PUT', "Should be BINBIN_PUT"

    def test_no_roll_on_assignment(self, strategy):
        """Test no roll signal on assignment."""
        closed_trade = {
            'symbol': 'NVDA',
            'right': 'P',
            'quantity': -1,
            'strike': 140.0,
            'exit_reason': 'ASSIGNMENT',
        }

        signal = strategy.generate_roll_signal(
            closed_trade=closed_trade,
            current_date='2024-02-01',
            underlying_price=145.0,
            iv=0.35,
        )

        assert signal is None, "Should not roll on assignment"

    def test_no_roll_on_stop_loss(self, strategy):
        """Test no roll signal on stop loss."""
        closed_trade = {
            'symbol': 'NVDA',
            'right': 'P',
            'quantity': -1,
            'strike': 140.0,
            'exit_reason': 'STOP_LOSS',
        }

        signal = strategy.generate_roll_signal(
            closed_trade=closed_trade,
            current_date='2024-02-01',
            underlying_price=145.0,
            iv=0.35,
        )

        assert signal is None, "Should not roll on stop loss"

    def test_roll_signal_strike_otm(self, strategy):
        """Test that roll signal strike is OTM."""
        underlying_price = 150.0

        closed_trade = {
            'symbol': 'NVDA',
            'right': 'P',
            'quantity': -1,
            'strike': 140.0,
            'exit_reason': 'PROFIT_TARGET',
        }

        signal = strategy.generate_roll_signal(
            closed_trade=closed_trade,
            current_date='2024-02-01',
            underlying_price=underlying_price,
            iv=0.35,
        )

        assert signal is not None
        # Put strike should be below underlying price
        assert signal.strike < underlying_price, \
            f"Put strike {signal.strike} should be below {underlying_price}"

    def test_roll_signal_dte_in_range(self, strategy):
        """Test that roll signal DTE is within configured range."""
        closed_trade = {
            'symbol': 'NVDA',
            'right': 'P',
            'quantity': -1,
            'strike': 140.0,
            'exit_reason': 'PROFIT_TARGET',
        }

        signal = strategy.generate_roll_signal(
            closed_trade=closed_trade,
            current_date='2024-02-01',
            underlying_price=145.0,
            iv=0.35,
        )

        assert signal is not None

        # Check DTE is in range (30-45 days)
        entry_date = datetime.strptime('2024-02-01', '%Y-%m-%d')
        expiry_date = datetime.strptime(signal.expiry, '%Y%m%d')
        dte = (expiry_date - entry_date).days

        assert 30 <= dte <= 45, f"DTE {dte} should be in range 30-45"

    def test_roll_signal_call_phase(self, strategy):
        """Test roll signal for call in CC phase."""
        # Set strategy to CC phase with shares
        strategy.phase = 'CC'
        strategy.stock_holding.shares = 100
        strategy.stock_holding.cost_basis = 140.0

        closed_trade = {
            'symbol': 'NVDA',
            'right': 'C',
            'quantity': -1,
            'strike': 150.0,
            'exit_reason': 'PROFIT_TARGET',
        }

        signal = strategy.generate_roll_signal(
            closed_trade=closed_trade,
            current_date='2024-02-01',
            underlying_price=145.0,
            iv=0.35,
        )

        assert signal is not None, "Should generate roll signal for call"
        assert signal.right == 'C', "Should roll call"
        assert signal.trade_type == 'BINBIN_CALL', "Should be BINBIN_CALL"
        # Call strike should be above underlying
        assert signal.strike > 145.0, f"Call strike {signal.strike} should be above underlying"

    def test_defensive_put_roll_triggers_on_deep_itm_short_put(self, strategy):
        """Short puts should trigger defensive roll before legacy SP hold behavior."""
        current_dt = datetime(2024, 2, 1)
        expiry_dt = current_dt + timedelta(days=10)
        should_close, reason = strategy.should_exit_position(
            position={
                "symbol": "NVDA",
                "right": "P",
                "strike": 100.0,
                "expiry": expiry_dt,
                "quantity": -1,
                "strategy_phase": "SP",
            },
            current_price=8.0,
            entry_price=2.0,
            current_dt=current_dt,
            market_data={"price": 85.0},
        )

        assert should_close is True
        assert reason == "DEFENSIVE_PUT_ROLL"


class TestRollLogicIntegration:
    """Test roll logic integration with engine."""

    def test_import_integrity(self):
        """Test that all modules import correctly."""
        from core.backtesting.pricing import OptionsPricer
        from core.backtesting.strategies.binbin_god import BinbinGodStrategy
        from core.backtesting.engine import BacktestEngine

        assert hasattr(OptionsPricer, 'strike_from_delta'), \
            "OptionsPricer should have strike_from_delta method"
        assert hasattr(BinbinGodStrategy, 'generate_roll_signal'), \
            "BinbinGodStrategy should have generate_roll_signal method"

    def test_strike_from_delta_returns_reasonable_values(self):
        """Test that strike_from_delta returns reasonable strike prices."""
        from core.backtesting.pricing import OptionsPricer

        S = 100.0
        T = 30 / 365
        sigma = 0.30

        # Test various deltas
        for target in [-0.40, -0.30, -0.20]:
            strike = OptionsPricer.strike_from_delta(S, T, sigma, target, 'P')
            assert 50 < strike < 150, f"Strike {strike} should be in reasonable range"

        for target in [0.20, 0.30, 0.40]:
            strike = OptionsPricer.strike_from_delta(S, T, sigma, target, 'C')
            assert 50 < strike < 150, f"Strike {strike} should be in reasonable range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
