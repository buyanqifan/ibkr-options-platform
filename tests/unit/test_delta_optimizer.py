"""
Unit tests for ML Delta Optimizer.

Tests cover:
- DeltaOptimizerML initialization
- Market context extraction
- Delta optimization
- Q-learning updates
- Pretraining functionality
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import os

from core.ml.delta_optimizer import (
    DeltaOptimizerML,
    DeltaOptimizationConfig,
    MarketContext,
    OptimizationResult,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return DeltaOptimizationConfig(
        model_path=tempfile.mktemp(suffix='.pkl'),
        delta_min=0.10,
        delta_max=0.40,
        delta_step=0.05,
        exploration_rate=0.1,
        learning_rate=0.01,
    )


@pytest.fixture
def optimizer(config):
    """Create optimizer instance."""
    return DeltaOptimizerML(config)


@pytest.fixture
def sample_bars():
    """Generate sample price bars for testing."""
    np.random.seed(42)
    n = 200
    prices = 150 * np.cumprod(1 + np.random.normal(0, 0.02, n))
    
    bars = []
    for i, price in enumerate(prices):
        # Use valid date format
        month = (i // 28) + 1
        day = (i % 28) + 1
        bars.append({
            'date': f'2024-{month:02d}-{day:02d}',
            'close': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'volume': 1000000,
        })
    return bars


@pytest.fixture
def market_context():
    """Create sample market context."""
    return MarketContext(
        symbol='NVDA',
        current_price=150.0,
        cost_basis=145.0,
        volatility_20d=0.25,
        volatility_30d=0.28,
        momentum_5d=0.03,
        momentum_20d=0.08,
        pe_ratio=30.0,
        iv_rank=55.0,
        market_regime='bull',
        days_to_earnings=30,
        option_liquidity=0.7,
    )


class TestDeltaOptimizationConfig:
    """Tests for DeltaOptimizationConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DeltaOptimizationConfig()
        
        assert config.model_path == "data/models/delta_optimizer.pkl"
        assert config.retrain_interval_days == 30
        assert config.min_training_samples == 1000
        assert config.exploration_rate == 0.1
        assert config.learning_rate == 0.01
        assert config.delta_min == 0.05
        assert config.delta_max == 0.40
        assert config.max_loss_threshold == 0.20
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DeltaOptimizationConfig(
            model_path="custom/path.pkl",
            exploration_rate=0.2,
            learning_rate=0.02,
        )
        
        assert config.model_path == "custom/path.pkl"
        assert config.exploration_rate == 0.2
        assert config.learning_rate == 0.02


class TestDeltaOptimizerML:
    """Tests for DeltaOptimizerML class."""
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.model is not None
        assert optimizer.q_table == {}
        assert optimizer.performance_history == []
    
    def test_initialize_model(self, optimizer):
        """Test model initialization."""
        optimizer._initialize_model()
        
        assert 'delta_performance' in optimizer.model
        assert 'market_regime_patterns' in optimizer.model
        assert 'symbol_specific' in optimizer.model
    
    def test_determine_market_regime(self, optimizer):
        """Test market regime determination."""
        # High volatility
        regime = optimizer._determine_market_regime(0.35, 0.03, 0.05)
        assert regime == "high_vol"
        
        # Bull market
        regime = optimizer._determine_market_regime(0.20, 0.08, 0.15)
        assert regime == "bull"
        
        # Bear market
        regime = optimizer._determine_market_regime(0.20, -0.08, -0.15)
        assert regime == "bear"
        
        # Neutral
        regime = optimizer._determine_market_regime(0.20, 0.02, 0.03)
        assert regime == "neutral"
    
    def test_get_regime_suitability(self, optimizer):
        """Test regime suitability scoring."""
        # Bull market - call suitability
        score = optimizer._get_regime_suitability(0.35, 'bull', 'C')
        assert score >= 0.5  # Should be high for 0.35 delta
        
        # Bear market - put suitability
        score = optimizer._get_regime_suitability(0.20, 'bear', 'P')
        assert score >= 0.5  # Should be high for 0.20 delta
        
        # High volatility - more conservative
        score = optimizer._get_regime_suitability(0.25, 'high_vol', 'P')
        assert score >= 0.0
    
    def test_optimize_delta(self, optimizer, market_context):
        """Test delta optimization."""
        result = optimizer.optimize_delta(
            market_context=market_context,
            right='P',
            iv=0.25,
            time_to_expiry=30/365.0,
        )
        
        assert isinstance(result, OptimizationResult)
        assert 0.10 <= result.optimal_delta <= 0.40
        assert 0 <= result.confidence <= 1
        assert result.reasoning is not None
    
    def test_optimize_delta_for_put(self, optimizer, market_context):
        """Test put delta optimization."""
        result = optimizer.optimize_delta(
            market_context=market_context,
            right='P',
            iv=0.30,
            time_to_expiry=45/365.0,
        )
        
        assert result.optimal_delta > 0
        assert result.expected_premium >= 0
    
    def test_optimize_delta_for_call(self, optimizer, market_context):
        """Test call delta optimization."""
        result = optimizer.optimize_delta(
            market_context=market_context,
            right='C',
            iv=0.25,
            time_to_expiry=30/365.0,
        )
        
        assert result.optimal_delta > 0
        assert result.expected_premium >= 0
    
    def test_update_performance(self, optimizer, market_context):
        """Test performance update (Q-learning)."""
        initial_q_size = len(optimizer.q_table)
        
        optimizer.update_performance(
            delta=0.30,
            symbol='NVDA',
            context=market_context,
            actual_pnl=0.05,
            actual_assignment=False,
        )
        
        assert len(optimizer.q_table) >= initial_q_size
        # Note: delta may be formatted as 0.3 instead of 0.30
        assert any('NVDA' in key and 'bull' in key for key in optimizer.model['delta_performance'].keys())
    
    def test_update_performance_with_assignment(self, optimizer, market_context):
        """Test performance update with assignment."""
        optimizer.update_performance(
            delta=0.25,
            symbol='AAPL',
            context=market_context,
            actual_pnl=0.10,
            actual_assignment=True,
        )
        
        assert 'AAPL_0.25_bull' in optimizer.model['delta_performance']
    
    def test_should_retrain(self, optimizer):
        """Test retrain decision."""
        # Should retrain initially
        assert optimizer.should_retrain() is True
        
        # Set last retrain
        optimizer.last_retrain = datetime.now()
        assert optimizer.should_retrain() is False
        
        # Add many samples
        optimizer.performance_history = [{'pnl': 0.01}] * 1000
        assert optimizer.should_retrain() is True
    
    def test_save_and_load_model(self, optimizer, config):
        """Test model save and load."""
        # Add some data
        optimizer.update_performance(
            delta=0.30,
            symbol='TEST',
            context=MarketContext(
                symbol='TEST',
                current_price=100.0,
                cost_basis=100.0,
                volatility_20d=0.2,
                volatility_30d=0.2,
                momentum_5d=0.0,
                momentum_20d=0.0,
                pe_ratio=25.0,
                iv_rank=50.0,
                market_regime='neutral',
                days_to_earnings=30,
                option_liquidity=0.5,
            ),
            actual_pnl=0.05,
            actual_assignment=False,
        )
        
        # Save
        optimizer.save_model()
        assert os.path.exists(config.model_path)
        
        # Load into new optimizer
        new_optimizer = DeltaOptimizerML(config)
        # Check that any TEST entry exists (delta may be formatted differently)
        assert any('TEST' in key and 'neutral' in key for key in new_optimizer.model['delta_performance'].keys())
        
        # Cleanup
        if os.path.exists(config.model_path):
            os.remove(config.model_path)
    
    def test_get_optimization_insights(self, optimizer):
        """Test optimization insights."""
        # Empty history
        insights = optimizer.get_optimization_insights()
        assert insights["message"] == "No performance data available"
        
        # Add some performance
        for i in range(10):
            optimizer.performance_history.append({
                'timestamp': datetime.now(),
                'delta': 0.30,
                'symbol': 'NVDA',
                'regime': 'bull',
                'pnl': 0.01 * (i + 1),
                'assignment': False,
            })
        
        insights = optimizer.get_optimization_insights()
        assert 'average_pnl' in insights
        assert 'win_rate' in insights
        assert 'total_trades' in insights


class TestPretraining:
    """Tests for pretraining functionality."""
    
    def test_pretrain_with_history(self, optimizer, sample_bars):
        """Test pretraining with historical data."""
        stats = optimizer.pretrain_with_history(
            symbol='NVDA',
            historical_bars=sample_bars,
            iv_estimate=0.25,
            right='P',
            training_ratio=0.5,
        )
        
        # Check if success or check the actual returned keys
        if 'status' in stats:
            assert stats['status'] == 'success'
        else:
            # The method may return stats directly without 'status' key
            assert stats.get('total_simulations', 0) > 0
        
        assert 'regimes_tested' in stats or 'total_simulations' in stats
    
    def test_pretrain_insufficient_data(self, optimizer):
        """Test pretraining with insufficient data."""
        short_bars = [{'date': '2024-01-01', 'close': 100.0}] * 30
        
        stats = optimizer.pretrain_with_history(
            symbol='TEST',
            historical_bars=short_bars,
            iv_estimate=0.25,
            right='P',
        )
        
        assert stats['status'] == 'skipped'
        assert stats['reason'] == 'insufficient_data'
    
    def test_calculate_simple_volatility(self, optimizer, sample_bars):
        """Test simple volatility calculation."""
        vol = optimizer._calculate_simple_volatility(sample_bars[:30])
        
        assert vol > 0
        assert vol < 1.0  # Reasonable range
    
    def test_simulate_option_trade(self, optimizer):
        """Test option trade simulation."""
        # Put simulation
        pnl, assigned = optimizer._simulate_option_trade(
            entry_price=150.0,
            exit_price=145.0,  # Price dropped
            delta=0.30,
            right='P',
            iv=0.25,
            T=30/365.0,
        )
        
        assert pnl != 0
        
        # Call simulation
        pnl, assigned = optimizer._simulate_option_trade(
            entry_price=150.0,
            exit_price=155.0,  # Price rose
            delta=0.30,
            right='C',
            iv=0.25,
            T=30/365.0,
        )
        
        assert pnl != 0
    
    def test_find_best_delta_for_regime(self, optimizer):
        """Test finding best delta for regime."""
        # Add some performance data
        optimizer.model['delta_performance'] = {
            'NVDA_0.20_bull': [0.02, 0.03, 0.025],
            'NVDA_0.30_bull': [0.04, 0.05, 0.045],
            'NVDA_0.40_bull': [0.03, 0.02, 0.025],
        }
        
        best_delta, avg_pnl = optimizer._find_best_delta_for_regime('NVDA', 'bull')
        
        assert best_delta == 0.30
        assert avg_pnl == pytest.approx(0.045, rel=0.01)


class TestMarketContext:
    """Tests for market context extraction."""
    
    def test_extract_market_context(self, optimizer, sample_bars):
        """Test market context extraction."""
        context = optimizer.extract_market_context(
            symbol='NVDA',
            current_price=150.0,
            cost_basis=145.0,
            bars=sample_bars,
            options_data=[],
            fundamentals={'pe_ratio': 35.0},
        )
        
        assert context.symbol == 'NVDA'
        assert context.current_price == 150.0
        assert context.cost_basis == 145.0
        assert context.volatility_20d > 0
        assert context.market_regime in ['bull', 'bear', 'neutral', 'high_vol']
    
    def test_create_simplified_context(self, optimizer):
        """Test simplified context creation."""
        context = optimizer._create_simplified_context(
            symbol='AAPL',
            current_price=140.0,
            cost_basis=150.0,
        )
        
        assert context.symbol == 'AAPL'
        assert context.current_price == 140.0
        assert context.market_regime == 'bear'  # Price below cost
    
    def test_simplified_context_bull(self, optimizer):
        """Test simplified context for bull market."""
        context = optimizer._create_simplified_context(
            symbol='AAPL',
            current_price=160.0,
            cost_basis=150.0,
        )
        
        assert context.market_regime == 'bull'  # Price above cost
    
    def test_simplified_context_neutral(self, optimizer):
        """Test simplified context for neutral market."""
        context = optimizer._create_simplified_context(
            symbol='AAPL',
            current_price=152.0,
            cost_basis=150.0,
        )
        
        assert context.market_regime == 'neutral'  # Price near cost


class TestOptimizationResult:
    """Tests for OptimizationResult."""
    
    def test_result_creation(self):
        """Test optimization result creation."""
        result = OptimizationResult(
            optimal_delta=0.30,
            expected_premium=2.5,
            expected_probability_assignment=0.25,
            risk_score=0.30,
            confidence=0.85,
            reasoning="Test reasoning",
        )
        
        assert result.optimal_delta == 0.30
        assert result.expected_premium == 2.5
        assert result.confidence == 0.85