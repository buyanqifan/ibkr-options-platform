"""
Unit tests for Options Strategies.

Tests cover:
- BaseStrategy functionality
- SellPutStrategy
- CoveredCallStrategy
- Spread strategies
- IronCondorStrategy
- Straddle/Strangle strategies
- ML Delta optimization integration
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from core.backtesting.strategies.base import BaseStrategy, Signal
from core.backtesting.strategies.sell_put import SellPutStrategy
from core.backtesting.strategies.covered_call import CoveredCallStrategy
from core.backtesting.strategies.spreads import BullPutSpreadStrategy, BearCallSpreadStrategy
from core.backtesting.strategies.iron_condor import IronCondorStrategy
from core.backtesting.strategies.straddle import StraddleStrategy, StrangleStrategy
from core.backtesting.strategies.wheel import WheelStrategy, StockHolding
from core.backtesting.position_manager import PositionManager
from core.backtesting.strategies.binbin_god import BinbinGodStrategy


@pytest.fixture
def base_params():
    """Base parameters for strategies."""
    return {
        'symbol': 'NVDA',
        'initial_capital': 100000,
        'dte_min': 30,
        'dte_max': 45,
        'delta_target': 0.30,
        'profit_target_pct': 50,
        'stop_loss_pct': 200,
        'max_positions': 5,
    }


@pytest.fixture
def ml_params(base_params):
    """Parameters with ML Delta optimization enabled."""
    params = base_params.copy()
    params['ml_delta_optimization'] = True
    params['ml_adoption_rate'] = 0.5
    return params


class TestBaseStrategy:
    """Tests for BaseStrategy class."""
    
    def test_initialization(self, base_params):
        """Test base strategy initialization."""
        strategy = SellPutStrategy(base_params)
        
        assert strategy.params == base_params
        assert strategy.dte_min == 30
        assert strategy.dte_max == 45
        assert strategy.delta_target == 0.30
    
    def test_select_expiry_dte(self, base_params):
        """Test DTE selection."""
        strategy = SellPutStrategy(base_params)
        dte = strategy.select_expiry_dte()
        
        assert dte == 37.5  # Average of 30 and 45
    
    def test_select_strike_put(self, base_params):
        """Test strike selection for put."""
        strategy = SellPutStrategy(base_params)
        
        underlying_price = 150.0
        iv = 0.25
        T = 30 / 365.0
        
        strike = strategy.select_strike(underlying_price, iv, T, 'P')
        
        # For 0.30 delta put, strike should be below current price
        assert strike < underlying_price
        assert strike > underlying_price * 0.85  # Not too far OTM
    
    def test_select_strike_call(self, base_params):
        """Test strike selection for call."""
        strategy = SellPutStrategy(base_params)
        
        underlying_price = 150.0
        iv = 0.25
        T = 30 / 365.0
        
        strike = strategy.select_strike(underlying_price, iv, T, 'C')
        
        # For 0.30 delta call, strike should be above current price
        assert strike > underlying_price
        assert strike < underlying_price * 1.15  # Not too far OTM
    
    def test_ml_optimization_disabled_by_default(self, base_params):
        """Test that ML optimization is disabled by default."""
        strategy = SellPutStrategy(base_params)
        
        assert strategy.ml_delta_optimization is False
        assert strategy.ml_integration is None


class TestSellPutStrategy:
    """Tests for SellPutStrategy."""
    
    def test_name(self, base_params):
        """Test strategy name."""
        strategy = SellPutStrategy(base_params)
        assert strategy.name == 'sell_put'
    
    def test_generate_signals_basic(self, base_params):
        """Test basic signal generation."""
        strategy = SellPutStrategy(base_params)
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[],
        )
        
        assert len(signals) == 1
        assert signals[0].trade_type == 'SELL_PUT'
        assert signals[0].right == 'P'
        assert signals[0].quantity < 0  # Sell (negative)
    
    def test_generate_signals_max_positions(self, base_params):
        """Test that signals stop at max positions."""
        base_params['max_positions'] = 1
        strategy = SellPutStrategy(base_params)
        
        # Simulate existing position
        from dataclasses import dataclass
        @dataclass
        class MockPosition:
            trade_type = 'SELL_PUT'
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[MockPosition()],
        )
        
        assert len(signals) == 0  # Should not generate new signal
    
    def test_ml_optimization_enabled(self, ml_params):
        """Test ML optimization integration."""
        try:
            strategy = SellPutStrategy(ml_params)
            
            if strategy.ml_delta_optimization:
                assert strategy.ml_integration is not None
                assert hasattr(strategy, 'get_optimized_delta')
            else:
                # ML initialization may fail due to dependencies
                assert strategy.ml_integration is None
        except ImportError:
            pytest.skip("ML dependencies not available")
    
    def test_profit_target_disabled(self, base_params):
        """Test profit target disabled flag."""
        base_params['profit_target_pct'] = 999999
        strategy = SellPutStrategy(base_params)
        
        assert strategy._profit_target_disabled is True
    
    def test_stop_loss_disabled(self, base_params):
        """Test stop loss disabled flag."""
        base_params['stop_loss_pct'] = 999999
        strategy = SellPutStrategy(base_params)
        
        assert strategy._stop_loss_disabled is True


class TestCoveredCallStrategy:
    """Tests for CoveredCallStrategy."""
    
    def test_name(self, base_params):
        """Test strategy name."""
        strategy = CoveredCallStrategy(base_params)
        assert strategy.name == 'covered_call'
    
    def test_initialize_stock_position(self, base_params):
        """Test stock position initialization."""
        strategy = CoveredCallStrategy(base_params)
        strategy.initialize_stock_position(150.0)
        
        assert strategy.stock_holding.shares > 0
        assert strategy.stock_holding.cost_basis == 150.0
    
    def test_no_signals_without_shares(self, base_params):
        """Test that no signals generated without shares."""
        strategy = CoveredCallStrategy(base_params)
        # Don't initialize shares
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[],
        )
        
        assert len(signals) == 0
    
    def test_generate_signals_with_shares(self, base_params):
        """Test signal generation with shares."""
        strategy = CoveredCallStrategy(base_params)
        strategy.initialize_stock_position(150.0)
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[],
        )
        
        assert len(signals) == 1
        assert signals[0].trade_type == 'COVERED_CALL'
        assert signals[0].right == 'C'
    
    def test_on_trade_closed_expiry(self, base_params):
        """Test trade closed with expiry."""
        strategy = CoveredCallStrategy(base_params)
        strategy.initialize_stock_position(150.0)
        
        trade = {
            'exit_reason': 'EXPIRY',
            'entry_price': 2.0,
            'quantity': -1,
        }
        
        result = strategy.on_trade_closed(trade)
        
        assert result == 0.0  # No additional stock P&L
        assert strategy.stock_holding.total_premium_collected > 0


class TestBullPutSpreadStrategy:
    """Tests for BullPutSpreadStrategy."""
    
    def test_name(self, base_params):
        """Test strategy name."""
        strategy = BullPutSpreadStrategy(base_params)
        assert strategy.name == 'bull_put_spread'
    
    def test_generate_signals(self, base_params):
        """Test signal generation."""
        strategy = BullPutSpreadStrategy(base_params)
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[],
        )
        
        assert len(signals) == 2  # Short and long leg
        assert signals[0].trade_type == 'BULL_PUT_SHORT'
        assert signals[1].trade_type == 'BULL_PUT_LONG'
        assert signals[0].strike > signals[1].strike
    
    def test_spread_width(self, base_params):
        """Test spread width configuration."""
        base_params['spread_width'] = 10.0
        strategy = BullPutSpreadStrategy(base_params)
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[],
        )
        
        spread = signals[0].strike - signals[1].strike
        assert spread == pytest.approx(10.0, rel=0.1)


class TestBearCallSpreadStrategy:
    """Tests for BearCallSpreadStrategy."""
    
    def test_name(self, base_params):
        """Test strategy name."""
        strategy = BearCallSpreadStrategy(base_params)
        assert strategy.name == 'bear_call_spread'
    
    def test_generate_signals(self, base_params):
        """Test signal generation."""
        strategy = BearCallSpreadStrategy(base_params)
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[],
        )
        
        assert len(signals) == 2
        assert signals[0].trade_type == 'BEAR_CALL_SHORT'
        assert signals[1].trade_type == 'BEAR_CALL_LONG'
        assert signals[1].strike > signals[0].strike


class TestIronCondorStrategy:
    """Tests for IronCondorStrategy."""
    
    def test_name(self, base_params):
        """Test strategy name."""
        strategy = IronCondorStrategy(base_params)
        assert strategy.name == 'iron_condor'
    
    def test_generate_signals(self, base_params):
        """Test signal generation."""
        strategy = IronCondorStrategy(base_params)
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[],
        )
        
        assert len(signals) == 4  # 4 legs
        
        # Check leg types
        leg_types = [s.trade_type for s in signals]
        assert 'IRON_CONDOR_SP' in leg_types
        assert 'IRON_CONDOR_LP' in leg_types
        assert 'IRON_CONDOR_SC' in leg_types
        assert 'IRON_CONDOR_LC' in leg_types
    
    def test_put_side_structure(self, base_params):
        """Test put side structure."""
        strategy = IronCondorStrategy(base_params)
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[],
        )
        
        put_signals = [s for s in signals if s.right == 'P']
        assert len(put_signals) == 2
        
        short_put = [s for s in put_signals if s.quantity < 0][0]
        long_put = [s for s in put_signals if s.quantity > 0][0]
        
        assert short_put.strike > long_put.strike
    
    def test_call_side_structure(self, base_params):
        """Test call side structure."""
        strategy = IronCondorStrategy(base_params)
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[],
        )
        
        call_signals = [s for s in signals if s.right == 'C']
        assert len(call_signals) == 2
        
        short_call = [s for s in call_signals if s.quantity < 0][0]
        long_call = [s for s in call_signals if s.quantity > 0][0]
        
        assert long_call.strike > short_call.strike


class TestStraddleStrategy:
    """Tests for StraddleStrategy."""
    
    def test_name(self, base_params):
        """Test strategy name."""
        strategy = StraddleStrategy(base_params)
        assert strategy.name == 'straddle'
    
    def test_generate_signals(self, base_params):
        """Test signal generation."""
        strategy = StraddleStrategy(base_params)
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[],
        )
        
        assert len(signals) == 2
        assert signals[0].trade_type == 'STRADDLE_PUT'
        assert signals[1].trade_type == 'STRADDLE_CALL'
        
        # Both should be ATM
        assert signals[0].strike == signals[1].strike


class TestStrangleStrategy:
    """Tests for StrangleStrategy."""
    
    def test_name(self, base_params):
        """Test strategy name."""
        strategy = StrangleStrategy(base_params)
        assert strategy.name == 'strangle'
    
    def test_generate_signals(self, base_params):
        """Test signal generation."""
        strategy = StrangleStrategy(base_params)
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[],
        )
        
        assert len(signals) == 2
        assert signals[0].trade_type == 'STRANGLE_PUT'
        assert signals[1].trade_type == 'STRANGLE_CALL'
        
        # Put strike should be below call strike
        assert signals[0].strike < signals[1].strike


class TestSignal:
    """Tests for Signal dataclass."""
    
    def test_signal_creation(self):
        """Test signal creation."""
        signal = Signal(
            symbol='NVDA',
            trade_type='SELL_PUT',
            right='P',
            strike=145.0,
            expiry='20240215',
            quantity=-1,
            iv=0.25,
            delta=-0.30,
            premium=2.5,
        )
        
        assert signal.symbol == 'NVDA'
        assert signal.trade_type == 'SELL_PUT'
        assert signal.right == 'P'
        assert signal.strike == 145.0
        assert signal.quantity == -1
    
    def test_signal_defaults(self):
        """Test signal default values."""
        signal = Signal(
            symbol='NVDA',
            trade_type='SELL_PUT',
            right='P',
            strike=145.0,
            expiry='20240215',
            quantity=-1,
            iv=0.25,
            delta=-0.30,
            premium=2.5,
        )
        
        assert signal.underlying_price == 0.0
        assert signal.margin_requirement is None


class TestMLDeltaIntegration:
    """Tests for ML Delta integration in strategies."""
    
    def test_get_optimized_delta_without_ml(self, base_params):
        """Test get_optimized_delta returns default when ML disabled."""
        strategy = SellPutStrategy(base_params)
        
        delta = strategy.get_optimized_delta(
            underlying_price=150.0,
            iv=0.25,
            right='P',
        )
        
        assert delta == strategy.delta_target
    
    def test_pretrain_ml_model_disabled(self, base_params):
        """Test pretrain_ml_model when ML disabled."""
        strategy = SellPutStrategy(base_params)
        
        bars = [{'date': f'2024-01-{i:02d}', 'close': 100 + i} for i in range(1, 100)]
        
        result = strategy.pretrain_ml_model(bars)
        
        assert result['status'] == 'skipped'
        assert result['reason'] == 'ml_not_enabled'
    
    def test_pretrain_ml_model_insufficient_data(self, ml_params):
        """Test pretrain_ml_model with insufficient data."""
        try:
            strategy = SellPutStrategy(ml_params)
            
            if not strategy.ml_delta_optimization:
                pytest.skip("ML not available")
            
            bars = [{'date': f'2024-01-{i:02d}', 'close': 100} for i in range(1, 30)]
            
            result = strategy.pretrain_ml_model(bars)
            
            assert result['status'] == 'skipped'
            assert result['reason'] == 'insufficient_data'
        except ImportError:
            pytest.skip("ML dependencies not available")
    
    def test_pretrain_ml_model_success(self, ml_params):
        """Test successful pretrain_ml_model."""
        try:
            strategy = SellPutStrategy(ml_params)
            
            if not strategy.ml_delta_optimization:
                pytest.skip("ML not available")
            
            # Generate sample data
            np.random.seed(42)
            prices = 150 * np.cumprod(1 + np.random.normal(0, 0.02, 150))
            bars = [{'date': f'2024-{i//30+1:02d}-{i%30+1:02d}', 'close': p} for i, p in enumerate(prices)]
            
            result = strategy.pretrain_ml_model(bars, iv_estimate=0.25)
            
            assert result['status'] == 'success'
            assert result['put_simulations'] > 0
            assert result['call_simulations'] > 0
        except ImportError:
            pytest.skip("ML dependencies not available")


class TestWheelStrategy:
    """Tests for WheelStrategy."""
    
    def test_name(self, base_params):
        """Test strategy name."""
        strategy = WheelStrategy(base_params)
        assert strategy.name == 'wheel'
    
    def test_initial_phase_is_sp(self, base_params):
        """Test that initial phase is Sell Put."""
        strategy = WheelStrategy(base_params)
        assert strategy.phase == 'SP'
    
    def test_generate_signals_sp_phase(self, base_params):
        """Test signal generation in SP phase."""
        strategy = WheelStrategy(base_params)
        position_mgr = PositionManager(initial_capital=100000)
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[],
            position_mgr=position_mgr,
        )
        
        assert len(signals) == 1
        assert signals[0].trade_type == 'WHEEL_PUT'
        assert signals[0].right == 'P'
        assert signals[0].quantity < 0
    
    def test_put_assignment_transitions_to_cc(self, base_params):
        """Test that Put assignment transitions to CC phase."""
        strategy = WheelStrategy(base_params)
        
        # Simulate Put assignment
        trade = {
            'exit_reason': 'ASSIGNMENT',
            'trade_type': 'WHEEL_PUT',
            'strike': 145.0,
            'quantity': -1,
            'entry_price': 2.5,
        }
        
        strategy.on_trade_closed(trade)
        
        assert strategy.phase == 'CC'
        assert strategy.stock_holding.shares == 100
        assert strategy.stock_holding.cost_basis == 145.0
    
    def test_cc_signal_only_one_contract_per_call(self, base_params):
        """Test that CC phase generates only 1 contract per signal."""
        strategy = WheelStrategy(base_params)
        strategy.phase = 'CC'
        strategy.stock_holding.shares = 500  # 5 contracts worth
        strategy.stock_holding.cost_basis = 150.0
        position_mgr = PositionManager(initial_capital=100000)
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[],
            position_mgr=position_mgr,
        )
        
        # Should only sell 1 contract per signal (not 5)
        assert len(signals) == 1
        assert abs(signals[0].quantity) == 1
    
    def test_no_cc_signal_when_shares_covered(self, base_params):
        """Test that no CC signal when all shares already covered by existing Calls."""
        strategy = WheelStrategy(base_params)
        strategy.phase = 'CC'
        strategy.stock_holding.shares = 200  # 2 contracts worth
        strategy.stock_holding.cost_basis = 150.0
        
        # Mock existing Call position covering all shares
        from dataclasses import dataclass
        @dataclass
        class MockPosition:
            trade_type = 'WHEEL_CALL'
            quantity = -2  # 2 contracts sold, covering 200 shares
        
        signals = strategy.generate_signals(
            current_date='2024-01-15',
            underlying_price=150.0,
            iv=0.25,
            open_positions=[MockPosition()],
        )
        
        # No new signal should be generated
        assert len(signals) == 0
    
    def test_call_assignment_returns_correct_stock_pnl(self, base_params):
        """Test that Call assignment returns correct stock PnL."""
        strategy = WheelStrategy(base_params)
        strategy.phase = 'CC'
        strategy.stock_holding.shares = 100
        strategy.stock_holding.cost_basis = 145.0  # Bought at 145
        
        # Simulate Call assignment at 155
        trade = {
            'exit_reason': 'ASSIGNMENT',
            'trade_type': 'WHEEL_CALL',
            'strike': 155.0,
            'quantity': -1,
            'entry_price': 3.0,
            'pnl': -200.0,  # Option PnL
        }
        
        result = strategy.on_trade_closed(trade)
        
        # Stock PnL = (155 - 145) * 100 = +1000
        assert result == 1000.0
        assert strategy.stock_holding.shares == 0
    
    def test_call_assignment_with_no_shares_returns_zero(self, base_params):
        """Test that Call assignment with no shares returns 0 (defensive check)."""
        strategy = WheelStrategy(base_params)
        strategy.phase = 'CC'
        strategy.stock_holding.shares = 0  # No shares!
        strategy.stock_holding.cost_basis = 0.0
        
        trade = {
            'exit_reason': 'ASSIGNMENT',
            'trade_type': 'WHEEL_CALL',
            'strike': 155.0,
            'quantity': -1,
            'entry_price': 3.0,
            'pnl': -200.0,
        }
        
        result = strategy.on_trade_closed(trade)
        
        # Should return 0 to avoid incorrect PnL
        assert result == 0.0
    
    def test_call_assignment_partial_shares(self, base_params):
        """Test Call assignment when shares_sold exceeds shares held."""
        strategy = WheelStrategy(base_params)
        strategy.phase = 'CC'
        strategy.stock_holding.shares = 100  # Only 100 shares
        strategy.stock_holding.cost_basis = 150.0
        
        # Try to assign 2 contracts (200 shares)
        trade = {
            'exit_reason': 'ASSIGNMENT',
            'trade_type': 'WHEEL_CALL',
            'strike': 160.0,
            'quantity': -2,  # 2 contracts = 200 shares
            'entry_price': 3.0,
            'pnl': -200.0,
        }
        
        result = strategy.on_trade_closed(trade)
        
        # Should only calculate PnL for 100 shares (actual held)
        # Stock PnL = (160 - 150) * 100 = +1000
        assert result == 1000.0
    
    def test_phase_transition_back_to_sp(self, base_params):
        """Test transition back to SP after all shares sold."""
        strategy = WheelStrategy(base_params)
        strategy.phase = 'CC'
        strategy.stock_holding.shares = 100
        strategy.stock_holding.cost_basis = 150.0
        
        trade = {
            'exit_reason': 'ASSIGNMENT',
            'trade_type': 'WHEEL_CALL',
            'strike': 155.0,
            'quantity': -1,
            'entry_price': 3.0,
            'pnl': -200.0,
        }
        
        strategy.on_trade_closed(trade)
        
        # Should transition back to SP
        assert strategy.phase == 'SP'
        assert strategy.stock_holding.cost_basis == 0.0
    
    def test_multiple_put_assignments_accumulate_shares(self, base_params):
        """Test that multiple Put assignments accumulate shares correctly."""
        strategy = WheelStrategy(base_params)
        
        # First Put assignment
        trade1 = {
            'exit_reason': 'ASSIGNMENT',
            'trade_type': 'WHEEL_PUT',
            'strike': 150.0,
            'quantity': -1,
            'entry_price': 2.5,
        }
        strategy.on_trade_closed(trade1)
        
        # Second Put assignment (before CC phase sold shares)
        trade2 = {
            'exit_reason': 'ASSIGNMENT',
            'trade_type': 'WHEEL_PUT',
            'strike': 145.0,
            'quantity': -2,
            'entry_price': 2.0,
        }
        strategy.on_trade_closed(trade2)
        
        # Should have 300 shares total (100 + 200)
        assert strategy.stock_holding.shares == 300
        # Cost basis should be weighted average: (100*150 + 200*145) / 300 = 146.67
        assert strategy.stock_holding.cost_basis == pytest.approx(146.67, rel=0.01)


class TestBinbinGodStrategy:
    """Tests for BinbinGodStrategy."""
    
    def test_name(self, base_params):
        """Test strategy name."""
        params = base_params.copy()
        params['symbol'] = 'MAG7_AUTO'
        strategy = BinbinGodStrategy(params)
        assert strategy.name == 'binbin_god'
    
    def test_initial_phase_is_sp(self, base_params):
        """Test that initial phase is Sell Put."""
        params = base_params.copy()
        params['symbol'] = 'MAG7_AUTO'
        strategy = BinbinGodStrategy(params)
        assert strategy.phase == 'SP'
    
    def test_strike_selection_normal_range(self, base_params):
        """Test strike selection returns reasonable value within normal range."""
        params = base_params.copy()
        params['symbol'] = 'NVDA'
        strategy = BinbinGodStrategy(params)
        
        # Normal case: strike should be OTM for call
        underlying_price = 180.0
        iv = 0.40
        T = 30 / 365.0
        
        strike = strategy.select_strike_with_constraints(
            underlying_price, iv, T, 'C', {}
        )
        
        # Strike should be within reasonable range (80%-120% of underlying)
        assert underlying_price * 0.8 <= strike <= underlying_price * 1.3
    
    def test_strike_selection_with_impossible_min_strike(self, base_params):
        """Test that min_strike constraint is relaxed when impossible.
        
        This is the critical fix for the bug where strike=600+ when price=180.
        When stock price drops far below cost basis, min_strike constraint
        should be relaxed to allow valid strike selection.
        """
        params = base_params.copy()
        params['symbol'] = 'NVDA'
        strategy = BinbinGodStrategy(params)
        
        # Scenario: cost_basis=600 but price dropped to 180
        underlying_price = 180.0
        cost_basis = 600.0
        iv = 0.40
        T = 30 / 365.0
        
        # min_strike would be ~588 (cost_basis * 0.98)
        # but high = 180 * 1.2 = 216
        # So min_strike > high, constraint should be relaxed
        min_strike = cost_basis * 0.98
        constraints = {"min_strike": min_strike}
        
        strike = strategy.select_strike_with_constraints(
            underlying_price, iv, T, 'C', constraints
        )
        
        # Strike should be within reasonable range relative to CURRENT price
        # NOT relative to cost_basis
        assert underlying_price * 0.8 <= strike <= underlying_price * 1.3
        # Strike should NOT be near cost_basis
        assert strike < cost_basis * 0.5
    
    def test_strike_selection_with_valid_min_strike(self, base_params):
        """Test that min_strike constraint is applied when valid."""
        params = base_params.copy()
        params['symbol'] = 'NVDA'
        params['call_delta'] = 0.20
        strategy = BinbinGodStrategy(params)
        
        # Scenario: price=180, cost_basis=150, min_strike=147
        underlying_price = 180.0
        iv = 0.40
        T = 30 / 365.0
        min_strike = 147.0  # Valid: within search range
        
        constraints = {"min_strike": min_strike}
        strike = strategy.select_strike_with_constraints(
            underlying_price, iv, T, 'C', constraints
        )
        
        # Strike should be >= min_strike
        assert strike >= min_strike
        # And within reasonable range
        assert strike <= underlying_price * 1.3