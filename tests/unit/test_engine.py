"""
Unit tests for Backtest Engine.

Tests cover:
- Engine initialization
- Historical data handling
- Trade simulation
- Performance metrics
- ML pretraining integration
"""

import pytest
import numpy as np
from datetime import datetime

from core.backtesting.engine import BacktestEngine, STRATEGY_MAP
from core.backtesting.metrics import PerformanceMetrics
from core.backtesting.pricing import OptionsPricer


@pytest.fixture
def engine():
    """Create backtest engine instance."""
    return BacktestEngine()


@pytest.fixture
def basic_params():
    """Basic backtest parameters."""
    return {
        'strategy': 'sell_put',
        'symbol': 'NVDA',
        'start_date': '2024-01-01',
        'end_date': '2024-03-31',
        'initial_capital': 100000,
        'dte_min': 30,
        'dte_max': 45,
        'delta_target': 0.30,
        'profit_target_pct': 50,
        'stop_loss_pct': 200,
        'max_positions': 5,
        'use_synthetic_data': True,  # Use synthetic data for testing
    }


class TestBacktestEngine:
    """Tests for BacktestEngine class."""
    
    def test_strategy_map_contains_all_strategies(self):
        """Test that all strategies are registered."""
        expected_strategies = [
            'sell_put',
            'covered_call',
            'iron_condor',
            'bull_put_spread',
            'bear_call_spread',
            'straddle',
            'strangle',
            'wheel',
            'binbin_god',
        ]
        
        for strategy in expected_strategies:
            assert strategy in STRATEGY_MAP, f"Missing strategy: {strategy}"
    
    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine._client is None
        assert engine._vol_predictor is None
    
    def test_run_with_synthetic_data(self, engine, basic_params):
        """Test running backtest with synthetic data."""
        result = engine.run(basic_params)
        
        assert 'metrics' in result
        assert 'trades' in result
        assert 'daily_pnl' in result
    
    def test_run_returns_metrics(self, engine, basic_params):
        """Test that backtest returns performance metrics."""
        result = engine.run(basic_params)
        metrics = result['metrics']
        
        assert 'total_return_pct' in metrics
        assert 'win_rate' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown_pct' in metrics
        assert 'total_trades' in metrics
    
    def test_run_sell_put_strategy(self, engine, basic_params):
        """Test sell put strategy backtest."""
        basic_params['strategy'] = 'sell_put'
        result = engine.run(basic_params)
        
        assert result is not None
        assert len(result.get('trades', [])) >= 0
    
    def test_run_covered_call_strategy(self, engine, basic_params):
        """Test covered call strategy backtest."""
        basic_params['strategy'] = 'covered_call'
        result = engine.run(basic_params)
        
        assert result is not None
    
    def test_run_iron_condor_strategy(self, engine, basic_params):
        """Test iron condor strategy backtest."""
        basic_params['strategy'] = 'iron_condor'
        basic_params['profit_target_pct'] = 50
        basic_params['stop_loss_pct'] = 200
        result = engine.run(basic_params)
        
        assert result is not None
    
    def test_run_wheel_strategy(self, engine, basic_params):
        """Test wheel strategy backtest."""
        basic_params['strategy'] = 'wheel'
        basic_params['put_delta'] = 0.30
        basic_params['call_delta'] = 0.30
        result = engine.run(basic_params)
        
        assert result is not None
    
    def test_invalid_strategy(self, engine, basic_params):
        """Test handling of invalid strategy."""
        basic_params['strategy'] = 'invalid_strategy'
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            engine.run(basic_params)
    
    def test_ml_delta_optimization_enabled(self, engine, basic_params):
        """Test ML Delta optimization in backtest."""
        basic_params['ml_delta_optimization'] = True
        basic_params['ml_adoption_rate'] = 0.5
        
        try:
            result = engine.run(basic_params)
            assert result is not None
        except ImportError:
            pytest.skip("ML dependencies not available")
    
    def test_max_positions_parameter(self, engine, basic_params):
        """Test max positions parameter."""
        basic_params['max_positions'] = 2
        result = engine.run(basic_params)
        
        # Count max concurrent positions
        trades = result.get('trades', [])
        # This is a simple check - more detailed position tracking would be better
        assert result is not None
    
    def test_profit_target_disabled(self, engine, basic_params):
        """Test profit target disabled (special value 999999)."""
        basic_params['profit_target_pct'] = 999999
        result = engine.run(basic_params)
        
        assert result is not None
    
    def test_stop_loss_disabled(self, engine, basic_params):
        """Test stop loss disabled (special value 999999)."""
        basic_params['stop_loss_pct'] = 999999
        result = engine.run(basic_params)
        
        assert result is not None


class TestOptionsPricer:
    """Tests for OptionsPricer."""
    
    def test_put_price(self):
        """Test put option pricing."""
        price = OptionsPricer.put_price(
            S=150.0,  # Stock price
            K=145.0,  # Strike
            T=30/365,  # Time to expiry
            sigma=0.25,  # Volatility
            r=0.05,  # Risk-free rate
        )
        
        assert price > 0
        assert price < 10  # Reasonable range for OTM put
    
    def test_call_price(self):
        """Test call option pricing."""
        price = OptionsPricer.call_price(
            S=150.0,
            K=155.0,
            T=30/365,
            sigma=0.25,
            r=0.05,
        )
        
        assert price > 0
        assert price < 10  # Reasonable range for OTM call
    
    def test_put_delta(self):
        """Test put delta calculation."""
        delta = OptionsPricer.delta(
            S=150.0,
            K=145.0,
            T=30/365,
            sigma=0.25,
            right='P',
        )
        
        assert -1 < delta < 0  # Put delta is negative
    
    def test_call_delta(self):
        """Test call delta calculation."""
        delta = OptionsPricer.delta(
            S=150.0,
            K=155.0,
            T=30/365,
            sigma=0.25,
            right='C',
        )
        
        assert 0 < delta < 1  # Call delta is positive
    
    def test_atm_put_price(self):
        """Test ATM put pricing."""
        S = 150.0
        price = OptionsPricer.put_price(S, K=S, T=30/365, sigma=0.25)
        
        assert price > 0
    
    def test_itm_put_price_vs_otm(self):
        """Test ITM put is more expensive than OTM."""
        S = 150.0
        itm_price = OptionsPricer.put_price(S, K=160.0, T=30/365, sigma=0.25)
        otm_price = OptionsPricer.put_price(S, K=140.0, T=30/365, sigma=0.25)
        
        assert itm_price > otm_price
    
    def test_higher_vol_higher_price(self):
        """Test higher volatility leads to higher option price."""
        S = 150.0
        K = 145.0
        T = 30/365
        
        low_vol_price = OptionsPricer.put_price(S, K, T, sigma=0.20)
        high_vol_price = OptionsPricer.put_price(S, K, T, sigma=0.40)
        
        assert high_vol_price > low_vol_price


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""
    
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation."""
        daily_pnl = [
            {'date': '2024-01-01', 'cumulative_pnl': 0},
            {'date': '2024-01-02', 'cumulative_pnl': 100},
            {'date': '2024-01-03', 'cumulative_pnl': 150},
            {'date': '2024-01-04', 'cumulative_pnl': 80},
            {'date': '2024-01-05', 'cumulative_pnl': 200},
        ]
        
        trades = [
            {'pnl': 100, 'entry_date': '2024-01-01', 'exit_date': '2024-01-02'},
            {'pnl': 50, 'entry_date': '2024-01-02', 'exit_date': '2024-01-03'},
            {'pnl': -70, 'entry_date': '2024-01-03', 'exit_date': '2024-01-04'},
            {'pnl': 120, 'entry_date': '2024-01-04', 'exit_date': '2024-01-05'},
        ]
        
        metrics = PerformanceMetrics.calculate(trades, daily_pnl, initial_capital=100000)
        
        assert 'total_return_pct' in metrics
        assert 'win_rate' in metrics
        assert 'total_trades' in metrics
        assert metrics['total_trades'] == 4
    
    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        trades = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 75},
            {'pnl': -25},
        ]
        
        metrics = PerformanceMetrics.calculate(trades, [], initial_capital=100000)
        
        assert metrics['win_rate'] == 50.0  # 2 wins out of 4
    
    def test_empty_trades(self):
        """Test metrics with no trades."""
        metrics = PerformanceMetrics.calculate([], [], initial_capital=100000)
        
        assert metrics['total_trades'] == 0
        assert metrics['win_rate'] == 0


class TestSyntheticData:
    """Tests for synthetic data generation."""
    
    def test_synthetic_data_generation(self, engine):
        """Test synthetic data is generated correctly."""
        params = {
            'strategy': 'sell_put',
            'symbol': 'TEST',
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'use_synthetic_data': True,
            'initial_capital': 100000,
            'dte_min': 30,
            'dte_max': 45,
            'delta_target': 0.30,
        }
        
        result = engine.run(params)
        
        # Should complete without error
        assert result is not None
    
    def test_synthetic_data_has_required_fields(self, engine):
        """Test synthetic data has required OHLC fields."""
        # The engine should handle synthetic data generation internally
        bars = engine._get_historical_data(
            symbol='TEST',
            start_date='2024-01-01',
            end_date='2024-01-31',
            use_synthetic=True,
        )
        
        assert bars is not None
        assert len(bars) > 0
        
        required_fields = ['date', 'close']
        for field in required_fields:
            assert field in bars[0]


class TestPositionManager:
    """Tests for PositionManager."""
    
    def test_position_manager_import(self):
        """Test position manager can be imported."""
        from core.backtesting.position_manager import PositionManager
        
        pm = PositionManager(
            initial_capital=100000,
            max_leverage=1.0,
        )
        
        assert pm.initial_capital == 100000
    
    def test_margin_allocation(self):
        """Test margin allocation."""
        from core.backtesting.position_manager import PositionManager
        
        pm = PositionManager(initial_capital=100000)
        
        # Allocate margin for a position
        result = pm.allocate_margin(
            position_id='TEST_1',
            strategy='sell_put',
            symbol='TEST',
            entry_date='2024-01-01',
            margin_amount=20000,
        )
        
        assert result is True
        assert pm.total_margin_used == 20000
    
    def test_margin_release(self):
        """Test margin release."""
        from core.backtesting.position_manager import PositionManager
        
        pm = PositionManager(initial_capital=100000)
        pm.allocate_margin(
            position_id='TEST_1',
            strategy='sell_put',
            symbol='TEST',
            entry_date='2024-01-01',
            margin_amount=20000,
        )
        pm.release_margin('TEST_1', pnl=1000)
        
        assert pm.total_margin_used == 0
    
    def test_position_size_calculation(self):
        """Test position size calculation."""
        from core.backtesting.position_manager import PositionManager
        
        pm = PositionManager(initial_capital=100000)
        
        # Calculate how many contracts we can afford
        num_contracts = pm.calculate_position_size(
            margin_per_contract=20000,
            max_positions=5,
        )
        
        assert num_contracts > 0
        assert num_contracts <= 5
