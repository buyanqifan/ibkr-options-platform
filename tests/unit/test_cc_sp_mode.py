"""Unit tests for CC+SP simultaneous mode (binbingod策略优化).

This test module verifies that all components correctly support the CC+SP
simultaneous mode where SP and CC are not mutually exclusive.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import unittest
import logging

logging.basicConfig(level=logging.WARNING)


class TestSignalStrategyPhase(unittest.TestCase):
    """Test Signal class supports strategy_phase field."""

    def test_signal_default_strategy_phase(self):
        """Signal should default to 'SP' strategy_phase."""
        from core.backtesting.strategies.base import Signal

        sig = Signal(
            symbol="AAPL",
            trade_type="BINBIN_PUT",
            right="P",
            strike=145,
            expiry="20240201",
            quantity=-1,
            iv=0.3,
            delta=0.3,
            premium=1.5,
            underlying_price=150,
        )
        self.assertEqual(sig.strategy_phase, "SP")

    def test_signal_cc_sp_strategy_phase(self):
        """Signal should accept 'CC+SP' strategy_phase."""
        from core.backtesting.strategies.base import Signal

        sig = Signal(
            symbol="AAPL",
            trade_type="BINBIN_PUT",
            right="P",
            strike=145,
            expiry="20240201",
            quantity=-1,
            iv=0.3,
            delta=0.3,
            premium=1.5,
            underlying_price=150,
            strategy_phase="CC+SP",
        )
        self.assertEqual(sig.strategy_phase, "CC+SP")

    def test_signal_cc_strategy_phase(self):
        """Signal should accept 'CC' strategy_phase."""
        from core.backtesting.strategies.base import Signal

        sig = Signal(
            symbol="AAPL",
            trade_type="BINBIN_CALL",
            right="C",
            strike=155,
            expiry="20240201",
            quantity=-1,
            iv=0.3,
            delta=0.3,
            premium=1.5,
            underlying_price=150,
            strategy_phase="CC",
        )
        self.assertEqual(sig.strategy_phase, "CC")


class TestOptionPositionStrategyPhase(unittest.TestCase):
    """Test OptionPosition class supports strategy_phase field."""

    def test_option_position_default_strategy_phase(self):
        """OptionPosition should default to 'SP' strategy_phase."""
        from core.backtesting.simulator import OptionPosition

        pos = OptionPosition(
            symbol="AAPL",
            entry_date="2024-01-01",
            expiry="20240201",
            strike=145,
            right="P",
            trade_type="BINBIN_PUT",
            quantity=-1,
            entry_price=1.5,
            underlying_entry=150,
            iv_at_entry=0.3,
            delta_at_entry=0.3,
        )
        self.assertEqual(pos.strategy_phase, "SP")

    def test_option_position_cc_sp_strategy_phase(self):
        """OptionPosition should accept 'CC+SP' strategy_phase."""
        from core.backtesting.simulator import OptionPosition

        pos = OptionPosition(
            symbol="AAPL",
            entry_date="2024-01-01",
            expiry="20240201",
            strike=145,
            right="P",
            trade_type="BINBIN_PUT",
            quantity=-1,
            entry_price=1.5,
            underlying_entry=150,
            iv_at_entry=0.3,
            delta_at_entry=0.3,
            strategy_phase="CC+SP",
        )
        self.assertEqual(pos.strategy_phase, "CC+SP")


class TestTradeRecordStrategyPhase(unittest.TestCase):
    """Test TradeRecord class supports strategy_phase field."""

    def test_trade_record_strategy_phase(self):
        """TradeRecord should store and export strategy_phase."""
        from core.backtesting.simulator import TradeRecord

        trade = TradeRecord(
            symbol="AAPL",
            trade_type="BINBIN_PUT",
            entry_date="2024-01-01",
            exit_date="2024-01-15",
            expiry="20240201",
            strike=145,
            right="P",
            entry_price=1.5,
            exit_price=0.5,
            quantity=-1,
            pnl=100,
            pnl_pct=66.7,
            exit_reason="PROFIT_TARGET",
            underlying_entry=150,
            underlying_exit=152,
            iv_at_entry=0.3,
            delta_at_entry=0.3,
            strategy_phase="CC+SP",
        )
        self.assertEqual(trade.strategy_phase, "CC+SP")

    def test_trade_record_to_dict_includes_strategy_phase(self):
        """TradeRecord.to_dict() should include strategy_phase."""
        from core.backtesting.simulator import TradeRecord

        trade = TradeRecord(
            symbol="AAPL",
            trade_type="BINBIN_PUT",
            entry_date="2024-01-01",
            exit_date="2024-01-15",
            expiry="20240201",
            strike=145,
            right="P",
            entry_price=1.5,
            exit_price=0.5,
            quantity=-1,
            pnl=100,
            pnl_pct=66.7,
            exit_reason="PROFIT_TARGET",
            underlying_entry=150,
            underlying_exit=152,
            iv_at_entry=0.3,
            delta_at_entry=0.3,
            strategy_phase="CC+SP",
        )
        d = trade.to_dict()
        self.assertIn("strategy_phase", d)
        self.assertEqual(d["strategy_phase"], "CC+SP")


class TestMLPositionOptimizerCCSP(unittest.TestCase):
    """Test MLPositionOptimizer supports CC+SP mode."""

    def setUp(self):
        """Set up test fixtures."""
        from core.ml.position_optimizer import MLPositionOptimizer

        self.optimizer = MLPositionOptimizer()
        self.market_data = {
            "iv": 0.3,
            "iv_rank": 50,
            "iv_percentile": 50,
            "historical_volatility": 30,
            "vix": 20,
            "momentum": {},
        }
        self.portfolio_state = {
            "total_capital": 100000,
            "available_margin": 50000,
            "margin_used": 30000,
            "drawdown": 0,
            "positions": [],
        }
        self.option_info = {
            "underlying_price": 100,
            "strike": 95,
            "delta": 0.3,
            "premium": 1.5,
            "dte": 30,
        }

    def test_strategy_phase_encoding(self):
        """strategy_phase should encode: SP=0, CC=1, CC+SP=2."""
        features_sp = self.optimizer.build_features(
            self.market_data, self.portfolio_state, "SP", self.option_info
        )
        features_cc = self.optimizer.build_features(
            self.market_data, self.portfolio_state, "CC", self.option_info
        )
        features_ccsp = self.optimizer.build_features(
            self.market_data, self.portfolio_state, "CC+SP", self.option_info
        )

        self.assertEqual(features_sp["strategy_phase"].values[0], 0)
        self.assertEqual(features_cc["strategy_phase"].values[0], 1)
        self.assertEqual(features_ccsp["strategy_phase"].values[0], 2)

    def test_cc_sp_uses_sp_strike_distance(self):
        """CC+SP should use SP logic for strike_distance_pct (Put: price > strike)."""
        features_sp = self.optimizer.build_features(
            self.market_data, self.portfolio_state, "SP", self.option_info
        )
        features_ccsp = self.optimizer.build_features(
            self.market_data, self.portfolio_state, "CC+SP", self.option_info
        )

        sp_dist = features_sp["strike_distance_pct"].values[0]
        ccsp_dist = features_ccsp["strike_distance_pct"].values[0]

        self.assertEqual(sp_dist, ccsp_dist)

    def test_cc_sp_uses_sp_premium_yield(self):
        """CC+SP should use SP logic for premium_yield (cash-secured margin)."""
        features_sp = self.optimizer.build_features(
            self.market_data, self.portfolio_state, "SP", self.option_info
        )
        features_ccsp = self.optimizer.build_features(
            self.market_data, self.portfolio_state, "CC+SP", self.option_info
        )

        sp_yield = features_sp["premium_yield"].values[0]
        ccsp_yield = features_ccsp["premium_yield"].values[0]

        self.assertEqual(sp_yield, ccsp_yield)

    def test_cc_sp_reasoning(self):
        """CC+SP mode reasoning should mention 'CC+SP simultaneous'."""
        features = self.optimizer.build_features(
            self.market_data, self.portfolio_state, "CC+SP", self.option_info
        )
        rec = self.optimizer._rule_based_predict(features, 1, 10, "CC+SP")

        self.assertIn("CC+SP simultaneous", rec.reasoning)

    def test_cc_sp_multiplier_more_conservative(self):
        """CC+SP multiplier should be more conservative than SP (0.85 vs 0.90)."""
        features_sp = self.optimizer.build_features(
            self.market_data, self.portfolio_state, "SP", self.option_info
        )
        features_ccsp = self.optimizer.build_features(
            self.market_data, self.portfolio_state, "CC+SP", self.option_info
        )

        rec_sp = self.optimizer._rule_based_predict(features_sp, 1, 10, "SP")
        rec_ccsp = self.optimizer._rule_based_predict(features_ccsp, 1, 10, "CC+SP")

        self.assertAlmostEqual(rec_sp.position_multiplier, 0.90, places=2)
        self.assertAlmostEqual(rec_ccsp.position_multiplier, 0.85, places=2)
        self.assertLess(rec_ccsp.position_multiplier, rec_sp.position_multiplier)

    def test_cc_sp_uses_sp_max_loss(self):
        """CC+SP should use SP logic for max_loss calculation."""
        import pandas as pd

        features_sp = pd.DataFrame(
            {
                "strike": [95],
                "premium": [1.5],
            }
        )
        features_ccsp = features_sp.copy()

        max_loss_sp = self.optimizer._calculate_max_loss(features_sp, "SP", 1)
        max_loss_ccsp = self.optimizer._calculate_max_loss(features_ccsp, "CC+SP", 1)

        # SP max_loss = (strike - premium) * 100 = (95 - 1.5) * 100 = 9350
        self.assertEqual(max_loss_sp, max_loss_ccsp)


class TestMLRollOptimizerCCSP(unittest.TestCase):
    """Test MLRollOptimizer supports CC+SP mode."""

    def test_roll_optimizer_strategy_phase_encoding(self):
        """MLRollOptimizer should encode strategy_phase correctly."""
        from core.ml.roll_optimizer import MLRollOptimizer

        optimizer = MLRollOptimizer()

        position_sp = {
            "strategy_phase": "SP",
            "entry_date": "2024-01-01",
            "expiry": "20240201",
            "entry_price": 1.5,
            "strike": 145,
        }
        position_cc = {
            "strategy_phase": "CC",
            "entry_date": "2024-01-01",
            "expiry": "20240201",
            "entry_price": 1.5,
            "strike": 155,
        }
        position_ccsp = {
            "strategy_phase": "CC+SP",
            "entry_date": "2024-01-01",
            "expiry": "20240201",
            "entry_price": 1.5,
            "strike": 145,
        }
        market_data = {
            "price": 150,
            "option_price": 1.0,
            "iv": 0.3,
            "iv_rank": 50,
            "vix": 20,
        }

        features_sp = optimizer.build_features(position_sp, market_data, "2024-01-15")
        features_cc = optimizer.build_features(position_cc, market_data, "2024-01-15")
        features_ccsp = optimizer.build_features(position_ccsp, market_data, "2024-01-15")

        self.assertEqual(features_sp["strategy_phase"].values[0], 0)
        self.assertEqual(features_cc["strategy_phase"].values[0], 1)
        self.assertEqual(features_ccsp["strategy_phase"].values[0], 2)


class TestDTEOptimizerCCSP(unittest.TestCase):
    """Test DTEOptimizerML supports CC+SP mode."""

    def test_dte_optimizer_cc_sp_uses_sp_logic(self):
        """DTEOptimizer should use SP logic for CC+SP mode."""
        from core.ml.dte_optimizer import DTEOptimizerML, DTEOptimizationConfig, DTEMarketContext

        config = DTEOptimizationConfig()
        optimizer = DTEOptimizerML(config)

        context_sp = DTEMarketContext(
            symbol="AAPL",
            current_price=150.0,
            cost_basis=145.0,
            volatility_20d=0.25,
            volatility_30d=0.28,
            momentum_5d=0.02,
            momentum_20d=0.05,
            pe_ratio=25.0,
            iv_rank=50.0,
            market_regime="neutral",
            days_to_earnings=30,
            option_liquidity=0.9,
            strategy_phase="SP",
        )

        context_ccsp = DTEMarketContext(
            symbol="AAPL",
            current_price=150.0,
            cost_basis=145.0,
            volatility_20d=0.25,
            volatility_30d=0.28,
            momentum_5d=0.02,
            momentum_20d=0.05,
            pe_ratio=25.0,
            iv_rank=50.0,
            market_regime="neutral",
            days_to_earnings=30,
            option_liquidity=0.9,
            strategy_phase="CC+SP",
        )

        # Test _optimize_for_strategy_phase
        score_sp = optimizer._optimize_for_strategy_phase(30, "SP", context_sp)
        score_ccsp = optimizer._optimize_for_strategy_phase(30, "CC+SP", context_ccsp)

        self.assertEqual(score_sp, score_ccsp)


class TestBinbinGodStrategyCCSP(unittest.TestCase):
    """Test BinbinGodStrategy supports CC+SP configuration."""

    def test_cc_sp_config_parameters(self):
        """BinbinGodStrategy should accept CC+SP configuration parameters."""
        from core.backtesting.strategies.binbin_god import BinbinGodStrategy

        config = {
            "initial_capital": 100000,
            "allow_sp_in_cc_phase": True,
            "sp_in_cc_margin_threshold": 0.5,
            "sp_in_cc_max_positions": 3,
        }
        strategy = BinbinGodStrategy(config)

        self.assertTrue(strategy.allow_sp_in_cc_phase)
        self.assertEqual(strategy.sp_in_cc_margin_threshold, 0.5)
        self.assertEqual(strategy.sp_in_cc_max_positions, 3)

    def test_generate_sp_in_cc_phase_method_exists(self):
        """BinbinGodStrategy should have _generate_sp_in_cc_phase method."""
        from core.backtesting.strategies.binbin_god import BinbinGodStrategy

        strategy = BinbinGodStrategy({"initial_capital": 100000})
        self.assertTrue(hasattr(strategy, "_generate_sp_in_cc_phase"))

    def test_cc_sp_disabled_by_default(self):
        """CC+SP should be enabled by default (allow_sp_in_cc_phase=True)."""
        from core.backtesting.strategies.binbin_god import BinbinGodStrategy

        strategy = BinbinGodStrategy({"initial_capital": 100000})
        # Default should be True
        self.assertTrue(strategy.allow_sp_in_cc_phase)


class TestBuildTrainingDataCCSP(unittest.TestCase):
    """Test _build_training_data supports CC+SP mode."""

    def test_build_training_data_with_cc_sp(self):
        """_build_training_data should correctly identify CC+SP trades."""
        from core.ml.position_optimizer import MLPositionOptimizer

        optimizer = MLPositionOptimizer()

        market_data = {
            "iv": 0.3,
            "iv_rank": 50,
            "iv_percentile": 50,
            "historical_volatility": 30,
            "vix": 20,
            "momentum": {},
        }
        portfolio_state = {
            "total_capital": 100000,
            "available_margin": 50000,
            "margin_used": 30000,
            "drawdown": 0,
            "positions": [],
        }
        option_info = {
            "underlying_price": 100,
            "strike": 95,
            "delta": 0.3,
            "premium": 1.5,
            "dte": 30,
        }

        historical_trades = [
            {
                "trade_type": "BINBIN_PUT",
                "cc_phase_sp": True,  # CC+SP mode
                "market_data_at_entry": market_data,
                "portfolio_state_at_entry": portfolio_state,
                "option_info": option_info,
                "pnl": 100,
                "max_drawdown": 200,
                "quantity": 1,
            },
            {
                "trade_type": "BINBIN_PUT",
                "cc_phase_sp": False,  # Normal SP mode
                "market_data_at_entry": market_data,
                "portfolio_state_at_entry": portfolio_state,
                "option_info": option_info,
                "pnl": 150,
                "max_drawdown": 100,
                "quantity": 1,
            },
        ]

        X, y = optimizer._build_training_data(historical_trades, "sharpe")

        # First trade should be CC+SP (strategy_phase=2)
        self.assertEqual(X["strategy_phase"].iloc[0], 2)
        # Second trade should be SP (strategy_phase=0)
        self.assertEqual(X["strategy_phase"].iloc[1], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)