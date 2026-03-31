"""Tests that Binbin God always routes through the QC replay engine path."""

from __future__ import annotations

from core.backtesting.engine import BacktestEngine


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
