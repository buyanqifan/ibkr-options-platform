"""Tests that Binbin God always routes through the QC replay engine path."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import core.backtesting.engine as engine_module
from core.backtesting.engine import BacktestEngine
from core.backtesting.strategies.binbin_god import BinbinGodStrategy


def test_binbin_god_run_always_uses_qc_parity_runner(monkeypatch):
    engine = BacktestEngine()
    called = {}

    bars = [
        {
            "date": "2024-01-01",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
        },
        {
            "date": "2024-01-02",
            "open": 101.0,
            "high": 102.0,
            "low": 100.0,
            "close": 101.0,
            "volume": 1_000_000,
        },
    ]

    monkeypatch.setattr(engine, "_get_historical_data", lambda *args, **kwargs: bars)
    monkeypatch.setattr(engine, "_rolling_hv", lambda prices, window=20: [0.25] * len(prices))

    def fake_qc_runner(*, strategy, params, bars, hv, pool_data):
        called["strategy"] = strategy
        called["params"] = params
        called["bars"] = bars
        called["hv"] = hv
        called["pool_data"] = pool_data
        return {"metrics": {}, "trades": [], "daily_pnl": []}

    monkeypatch.setattr(engine, "_run_binbin_god_qc_parity", fake_qc_runner)

    result = engine.run(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": 300000,
            "ml_enabled": False,
        }
    )

    assert result == {"metrics": {}, "trades": [], "daily_pnl": []}
    assert called["params"]["parity_mode"] == "qc"
    assert called["params"]["contract_universe_mode"] == "qc_emulated_lattice"


def _make_bars(days: int) -> list[dict]:
    bars = []
    start = datetime(2024, 1, 1)
    for offset in range(days):
        date_value = (start + timedelta(days=offset)).strftime("%Y-%m-%d")
        bars.append(
            {
                "date": date_value,
                "open": 100.0 + offset,
                "high": 101.0 + offset,
                "low": 99.0 + offset,
                "close": 100.0 + offset,
                "volume": 1_000_000,
            }
        )
    return bars


def test_qc_replay_does_not_open_positions_during_warmup(monkeypatch):
    engine = BacktestEngine()
    bars = _make_bars(59)
    opened_dates = []

    monkeypatch.setattr(engine, "_get_historical_data", lambda *args, **kwargs: bars)
    monkeypatch.setattr(engine, "_rolling_hv", lambda prices, window=20: [0.25] * len(prices))
    monkeypatch.setattr(
        BinbinGodStrategy,
        "generate_signals",
        lambda self, current_date, underlying_price, iv, open_positions, position_mgr=None: [SimpleNamespace(symbol="NVDA")],
    )
    monkeypatch.setattr(BinbinGodStrategy, "generate_immediate_cc_signal", lambda *args, **kwargs: None)

    def fake_open_signal_position(*, bar_date, **kwargs):
        opened_dates.append(bar_date)
        return False

    monkeypatch.setattr(engine, "_open_signal_position", fake_open_signal_position)

    engine.run(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-02-28",
            "initial_capital": 300000,
            "ml_enabled": False,
        }
    )

    assert opened_dates == []


def test_qc_replay_only_opens_positions_after_warmup(monkeypatch):
    engine = BacktestEngine()
    bars = _make_bars(61)
    opened_dates = []

    monkeypatch.setattr(engine, "_get_historical_data", lambda *args, **kwargs: bars)
    monkeypatch.setattr(engine, "_rolling_hv", lambda prices, window=20: [0.25] * len(prices))
    monkeypatch.setattr(
        BinbinGodStrategy,
        "generate_signals",
        lambda self, current_date, underlying_price, iv, open_positions, position_mgr=None: [SimpleNamespace(symbol="NVDA")],
    )
    monkeypatch.setattr(BinbinGodStrategy, "generate_immediate_cc_signal", lambda *args, **kwargs: None)

    def fake_open_signal_position(*, bar_date, **kwargs):
        opened_dates.append(bar_date)
        return False

    monkeypatch.setattr(engine, "_open_signal_position", fake_open_signal_position)

    engine.run(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "initial_capital": 300000,
            "ml_enabled": False,
        }
    )

    assert opened_dates
    assert min(opened_dates) == bars[60]["date"]


def test_qc_replay_warmup_pretrains_ml_once(monkeypatch):
    engine = BacktestEngine()
    bars = _make_bars(100)
    pretrain_calls = []

    monkeypatch.setattr(engine, "_get_historical_data", lambda *args, **kwargs: bars)
    monkeypatch.setattr(engine, "_rolling_hv", lambda prices, window=20: [0.25] * len(prices))
    monkeypatch.setattr(BinbinGodStrategy, "generate_signals", lambda *args, **kwargs: [])
    monkeypatch.setattr(BinbinGodStrategy, "generate_immediate_cc_signal", lambda *args, **kwargs: None)

    def fake_pretrain(self, historical_bars, iv_estimate=0.25):
        pretrain_calls.append((len(historical_bars), iv_estimate))
        return {"status": "ok"}

    monkeypatch.setattr(BinbinGodStrategy, "pretrain_ml_model", fake_pretrain)

    engine.run(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-04-30",
            "initial_capital": 300000,
            "ml_enabled": True,
            "ml_delta_optimization": True,
        }
    )

    assert len(pretrain_calls) == 1
    assert pretrain_calls[0][0] >= 60


def test_qc_replay_passes_portfolio_value_into_dynamic_capacity(monkeypatch):
    engine = BacktestEngine()
    bars = _make_bars(61)
    portfolio_values = []

    monkeypatch.setattr(engine, "_get_historical_data", lambda *args, **kwargs: bars)
    monkeypatch.setattr(engine, "_rolling_hv", lambda prices, window=20: [0.25] * len(prices))
    monkeypatch.setattr(BinbinGodStrategy, "generate_signals", lambda *args, **kwargs: [])
    monkeypatch.setattr(BinbinGodStrategy, "generate_immediate_cc_signal", lambda *args, **kwargs: None)

    original_calc = engine_module.calculate_dynamic_max_positions_from_prices

    def capture_dynamic_max_positions(prices, config, portfolio_value=None):
        portfolio_values.append(portfolio_value)
        return original_calc(prices, config, portfolio_value=portfolio_value)

    monkeypatch.setattr(engine_module, "calculate_dynamic_max_positions_from_prices", capture_dynamic_max_positions)

    engine.run(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "initial_capital": 300000,
            "ml_enabled": False,
        }
    )

    assert portfolio_values
    assert all(value is not None for value in portfolio_values)
