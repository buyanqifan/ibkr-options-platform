"""
Unit tests for Backtest Storage.

Tests cover:
- Save backtest results
- Retrieve backtest by ID
- List backtests with filtering
- Delete backtest
- Summary statistics
"""

import pytest
import json
import os
import tempfile
from datetime import datetime

# Setup test database path before imports
@pytest.fixture(scope='module', autouse=True)
def setup_test_db():
    """Setup test database for all tests."""
    # Create temp database
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.environ['DB_PATH'] = db_path
    
    # Import after setting env var
    from models.base import Base, engine, SessionLocal
    from core.backtesting.storage import BacktestStorage, _storage_instance
    
    # Reset singleton
    import core.backtesting.storage as storage_module
    storage_module._storage_instance = None
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    yield db_path
    
    # Cleanup
    os.close(fd)
    os.unlink(db_path)


@pytest.fixture
def storage():
    """Create BacktestStorage instance."""
    from core.backtesting.storage import BacktestStorage
    return BacktestStorage()


@pytest.fixture
def sample_params():
    """Sample backtest parameters."""
    return {
        "strategy": "sell_put",
        "symbol": "NVDA",
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "initial_capital": 100000,
        "dte_min": 30,
        "dte_max": 45,
        "delta_target": 0.30,
        "profit_target_pct": 50,
        "stop_loss_pct": 200,
        "max_positions": 5,
    }


@pytest.fixture
def sample_result():
    """Sample backtest result."""
    return {
        "metrics": {
            "total_return_pct": 15.5,
            "annualized_return_pct": 62.0,
            "max_drawdown_pct": 8.2,
            "sharpe_ratio": 1.85,
            "sortino_ratio": 2.45,
            "win_rate": 75.0,
            "total_trades": 20,
            "avg_profit": 500.0,
            "avg_loss": -200.0,
            "profit_factor": 2.5,
        },
        "trades": [
            {
                "symbol": "NVDA",
                "trade_type": "SELL_PUT",
                "entry_date": "2024-01-02",
                "exit_date": "2024-01-15",
                "expiry": "2024-02-16",
                "strike": 450.0,
                "entry_price": 5.20,
                "exit_price": 2.60,
                "quantity": 1,
                "pnl": 260.0,
                "pnl_pct": 50.0,
                "exit_reason": "PROFIT_TARGET",
                "underlying_entry": 480.0,
                "underlying_exit": 495.0,
                "iv_at_entry": 0.28,
                "delta_at_entry": -0.30,
                "capital_at_entry": 100000,
                "capital_at_exit": 100260,
            },
            {
                "symbol": "NVDA",
                "trade_type": "SELL_PUT",
                "entry_date": "2024-01-16",
                "exit_date": "2024-02-10",
                "expiry": "2024-03-15",
                "strike": 460.0,
                "entry_price": 6.50,
                "exit_price": 0.0,
                "quantity": 1,
                "pnl": 650.0,
                "pnl_pct": 100.0,
                "exit_reason": "EXPIRY",
                "underlying_entry": 490.0,
                "underlying_exit": 520.0,
                "iv_at_entry": 0.26,
                "delta_at_entry": -0.28,
                "capital_at_entry": 100260,
                "capital_at_exit": 100910,
            },
        ],
        "daily_pnl": [
            {"date": "2024-01-02", "cumulative_pnl": 0, "portfolio_value": 100000},
            {"date": "2024-01-15", "cumulative_pnl": 260, "portfolio_value": 100260},
            {"date": "2024-02-10", "cumulative_pnl": 910, "portfolio_value": 100910},
        ],
        "strategy_performance": {},
    }


class TestBacktestStorage:
    """Tests for BacktestStorage class."""
    
    def test_save_backtest(self, storage, sample_params, sample_result):
        """Test saving a backtest result."""
        backtest_id = storage.save_backtest(sample_params, sample_result)
        
        assert backtest_id is not None
        assert isinstance(backtest_id, int)
        assert backtest_id > 0
    
    def test_save_and_get_backtest(self, storage, sample_params, sample_result):
        """Test saving and retrieving a backtest."""
        backtest_id = storage.save_backtest(sample_params, sample_result)
        
        retrieved = storage.get_backtest(backtest_id)
        
        assert retrieved is not None
        assert retrieved["id"] == backtest_id
        assert retrieved["strategy_name"] == "sell_put"
        assert retrieved["symbol"] == "NVDA"
        assert retrieved["start_date"] == "2024-01-01"
        assert retrieved["end_date"] == "2024-03-31"
        assert retrieved["initial_capital"] == 100000
        assert len(retrieved["trades"]) == 2
        assert retrieved["trades"][0]["exit_reason"] == "PROFIT_TARGET"
    
    def test_get_nonexistent_backtest(self, storage):
        """Test retrieving a non-existent backtest."""
        result = storage.get_backtest(99999)
        
        assert result is None
    
    def test_list_backtests(self, storage, sample_params, sample_result):
        """Test listing backtests."""
        # Save a few backtests
        storage.save_backtest(sample_params, sample_result)
        
        sample_params["symbol"] = "AAPL"
        sample_result["metrics"]["total_return_pct"] = 10.0
        storage.save_backtest(sample_params, sample_result)
        
        # List all
        results = storage.list_backtests()
        
        assert len(results) >= 2
    
    def test_list_backtests_filter_by_strategy(self, storage, sample_params, sample_result):
        """Test filtering backtests by strategy."""
        storage.save_backtest(sample_params, sample_result)
        
        results = storage.list_backtests(strategy="sell_put")
        
        for r in results:
            assert r["strategy_name"] == "sell_put"
    
    def test_list_backtests_filter_by_symbol(self, storage, sample_params, sample_result):
        """Test filtering backtests by symbol."""
        sample_params["symbol"] = "TSLA"
        storage.save_backtest(sample_params, sample_result)
        
        results = storage.list_backtests(symbol="TSLA")
        
        assert len(results) >= 1
        for r in results:
            assert r["symbol"] == "TSLA"
    
    def test_list_backtests_limit(self, storage, sample_params, sample_result):
        """Test limiting results."""
        # Save multiple backtests
        for i in range(5):
            sample_params["symbol"] = f"TEST{i}"
            storage.save_backtest(sample_params, sample_result)
        
        results = storage.list_backtests(limit=3)
        
        assert len(results) == 3
    
    def test_delete_backtest(self, storage, sample_params, sample_result):
        """Test deleting a backtest."""
        backtest_id = storage.save_backtest(sample_params, sample_result)
        
        # Verify it exists
        retrieved = storage.get_backtest(backtest_id)
        assert retrieved is not None
        
        # Delete it
        deleted = storage.delete_backtest(backtest_id)
        assert deleted is True
        
        # Verify it's gone
        retrieved = storage.get_backtest(backtest_id)
        assert retrieved is None
    
    def test_delete_nonexistent_backtest(self, storage):
        """Test deleting a non-existent backtest."""
        deleted = storage.delete_backtest(99999)
        assert deleted is False
    
    def test_get_summary_stats(self, storage, sample_params, sample_result):
        """Test getting summary statistics."""
        # Save some backtests with different strategies
        sample_params["strategy"] = "sell_put"
        sample_params["symbol"] = "NVDA"
        storage.save_backtest(sample_params, sample_result)
        
        sample_params["strategy"] = "wheel"
        sample_params["symbol"] = "AAPL"
        storage.save_backtest(sample_params, sample_result)
        
        stats = storage.get_summary_stats()
        
        assert "total_backtests" in stats
        assert stats["total_backtests"] >= 2
        assert "by_strategy" in stats
        assert "by_symbol" in stats
    
    def test_get_summary_stats_empty(self, storage):
        """Test summary stats structure with no data in filter."""
        # Just verify the structure is correct
        stats = storage.get_summary_stats()
        
        # Structure should be correct regardless of data
        assert "total_backtests" in stats
        assert "by_strategy" in stats
        assert "by_symbol" in stats
        assert isinstance(stats["by_strategy"], dict)
        assert isinstance(stats["by_symbol"], dict)
    
    def test_save_backtest_with_assignment(self, storage, sample_params, sample_result):
        """Test saving backtest with assigned trades."""
        # Add an assignment trade
        sample_result["trades"].append({
            "symbol": "NVDA",
            "trade_type": "SELL_PUT",
            "entry_date": "2024-02-15",
            "exit_date": "2024-03-15",
            "expiry": "2024-03-15",
            "strike": 500.0,
            "entry_price": 4.0,
            "exit_price": 0.0,
            "quantity": 1,
            "pnl": -500.0,
            "pnl_pct": -125.0,
            "exit_reason": "ASSIGNMENT",
            "underlying_entry": 510.0,
            "underlying_exit": 480.0,
            "iv_at_entry": 0.30,
            "delta_at_entry": -0.35,
        })
        
        backtest_id = storage.save_backtest(sample_params, sample_result)
        
        retrieved = storage.get_backtest(backtest_id)
        assert retrieved["assignment_count"] == 1
    
    def test_save_backtest_empty_trades(self, storage, sample_params):
        """Test saving backtest with no trades."""
        empty_result = {
            "metrics": {
                "total_return_pct": 0,
                "annualized_return_pct": 0,
                "max_drawdown_pct": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "win_rate": 0,
                "total_trades": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "profit_factor": 0,
            },
            "trades": [],
            "daily_pnl": [],
            "strategy_performance": {},
        }
        
        backtest_id = storage.save_backtest(sample_params, empty_result)
        
        retrieved = storage.get_backtest(backtest_id)
        assert retrieved["total_trades"] == 0
        assert retrieved["trades"] == []
    
    def test_daily_pnl_serialization(self, storage, sample_params, sample_result):
        """Test that daily P&L is properly serialized and deserialized."""
        backtest_id = storage.save_backtest(sample_params, sample_result)
        
        retrieved = storage.get_backtest(backtest_id)
        
        assert retrieved["daily_pnl"] is not None
        assert len(retrieved["daily_pnl"]) == 3
        assert retrieved["daily_pnl"][0]["date"] == "2024-01-02"
        assert retrieved["daily_pnl"][-1]["cumulative_pnl"] == 910


class TestBacktestStorageSingleton:
    """Tests for singleton pattern."""
    
    def test_get_backtest_storage_singleton(self):
        """Test that get_backtest_storage returns singleton."""
        from core.backtesting.storage import get_backtest_storage
        
        storage1 = get_backtest_storage()
        storage2 = get_backtest_storage()
        
        assert storage1 is storage2