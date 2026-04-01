"""Focused QC parity tests for phase-aware pricing."""

from types import SimpleNamespace

from core.backtesting.engine import BacktestEngine
from core.backtesting.strategies.binbin_god import BinbinGodStrategy
from core.backtesting.simulator import TradeSimulator


def _make_bars(days: int) -> list[dict]:
    bars = []
    for offset in range(days):
        bars.append(
            {
                "date": f"2024-03-{offset + 1:02d}",
                "open": 100.0 + offset,
                "high": 101.0 + offset,
                "low": 99.0 + offset,
                "close": 110.0 + offset,
                "volume": 1_000_000,
            }
        )
    return bars


def test_get_price_for_symbol_supports_phase_price_fields():
    engine = BacktestEngine()
    pool_data = {
        "NVDA": [
            {
                "date": "2024-03-01",
                "open": 101.0,
                "high": 105.0,
                "low": 99.0,
                "close": 110.0,
            }
        ]
    }

    assert engine._get_price_for_symbol("NVDA", "2024-03-01", pool_data, price_field="open") == 101.0
    assert engine._get_price_for_symbol("NVDA", "2024-03-01", pool_data, price_field="close") == 110.0


def test_qc_parity_uses_open_for_rebalance_and_close_for_immediate_cc(monkeypatch):
    engine = BacktestEngine()
    bars = _make_bars(61)
    signal_prices: list[float] = []
    immediate_cc_prices: list[float] = []
    managed_price_lookups: list[dict] = []
    expiry_price_lookups: list[dict] = []

    monkeypatch.setattr(engine, "_get_historical_data", lambda *args, **kwargs: bars)
    monkeypatch.setattr(engine, "_rolling_hv", lambda prices, window=20: [0.25] * len(prices))
    monkeypatch.setattr(TradeSimulator, "check_position_management", lambda self, **kwargs: (managed_price_lookups.append(kwargs["price_lookup"]) or ([], self.open_positions)))
    monkeypatch.setattr(TradeSimulator, "check_expiries", lambda self, **kwargs: (expiry_price_lookups.append(kwargs["price_lookup"]) or ([], self.open_positions)))

    def fake_process_closed_trades_parity(self, **kwargs):
        if kwargs["allow_roll"]:
            return []
        return ["NVDA"]

    monkeypatch.setattr(BacktestEngine, "_process_closed_trades_parity", fake_process_closed_trades_parity)
    monkeypatch.setattr(BacktestEngine, "_open_signal_position", lambda *args, **kwargs: False)

    def fake_generate_signals(self, current_date, underlying_price, iv, open_positions, position_mgr=None):
        signal_prices.append(underlying_price)
        return []

    def fake_immediate_cc(self, symbol, current_date, underlying_price, iv, open_positions, position_mgr=None):
        immediate_cc_prices.append(underlying_price)
        return None

    monkeypatch.setattr(BinbinGodStrategy, "generate_signals", fake_generate_signals)
    monkeypatch.setattr(BinbinGodStrategy, "generate_immediate_cc_signal", fake_immediate_cc)

    engine.run(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA"],
            "start_date": "2024-03-01",
            "end_date": "2024-05-31",
            "initial_capital": 300000,
            "ml_enabled": False,
        }
    )

    actionable_bar = bars[60]
    assert signal_prices == [actionable_bar["open"]]
    assert immediate_cc_prices == [actionable_bar["close"]]
    assert managed_price_lookups == [{"NVDA": actionable_bar["open"]}]
    assert expiry_price_lookups == [{"NVDA": actionable_bar["close"]}]
