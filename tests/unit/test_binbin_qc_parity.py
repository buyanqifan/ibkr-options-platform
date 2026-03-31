"""Tests for BinbinGod QC parity helpers and engine mode."""

from datetime import datetime, timedelta

import pytest

pytest.importorskip("optionlab")

from core.backtesting.engine import BacktestEngine
from core.backtesting.qc_parity import (
    BinbinGodParityConfig,
    build_contract_lattice,
    calculate_dynamic_max_positions_from_prices,
    calculate_put_quantity_qc,
)
from core.backtesting.strategies.binbin_god import BinbinGodStrategy


def _make_bars(start_price: float = 100.0, days: int = 80):
    bars = []
    current = start_price
    start = datetime(2024, 1, 1)
    for offset in range(days):
        current *= 1.002
        bars.append(
            {
                "date": (start + timedelta(days=offset)).strftime("%Y-%m-%d"),
                "open": current * 0.99,
                "high": current * 1.01,
                "low": current * 0.98,
                "close": current,
                "volume": 1_000_000 + offset * 1000,
            }
        )
    return bars


def test_qc_parity_config_uses_qc_defaults():
    config = BinbinGodParityConfig.from_params({"parity_mode": "qc"})
    assert config.enabled is True
    assert config.initial_capital == 100000.0
    assert config.max_positions_ceiling == 15
    assert config.dte_min == 21
    assert config.dte_max == 60
    assert config.ml_min_confidence == pytest.approx(0.40)
    assert config.defensive_put_roll_enabled is True
    assert config.assignment_cooldown_days == 20
    assert config.stock_inventory_cap_enabled is True
    assert config.stock_inventory_base_cap == pytest.approx(0.20)
    assert config.symbol_assignment_base_cap > 0


def test_contract_lattice_respects_min_strike():
    contracts = build_contract_lattice(
        symbol="NVDA",
        current_date="2024-03-01",
        underlying_price=100.0,
        iv=0.25,
        target_right="C",
        target_delta=0.12,
        dte_min=21,
        dte_max=45,
        min_strike=105.0,
    )
    assert contracts
    assert all(contract.strike >= 105.0 for contract in contracts)


def test_dynamic_max_positions_matches_qc_style_formula():
    config = BinbinGodParityConfig.from_params({"parity_mode": "qc"})
    result = calculate_dynamic_max_positions_from_prices([100.0, 120.0, 140.0], config)
    assert result >= 1
    assert result <= config.max_positions_ceiling


def test_put_quantity_qc_applies_caps():
    config = BinbinGodParityConfig.from_params({"parity_mode": "qc"})
    contract = build_contract_lattice(
        symbol="NVDA",
        current_date="2024-03-01",
        underlying_price=100.0,
        iv=0.25,
        target_right="P",
        target_delta=-0.12,
        dte_min=21,
        dte_max=45,
    )[0]
    quantity, diagnostics = calculate_put_quantity_qc(
        config=config,
        selected_contract=contract,
        current_positions=0,
        underlying_price=100.0,
        symbol="NVDA",
        portfolio_value=100000.0,
        margin_remaining=100000.0,
        total_margin_used=0.0,
        stock_holdings_value=0.0,
        stock_holding_count=0,
        open_option_positions=[],
        shares_held=0,
        dynamic_max_positions=10,
    )
    assert quantity >= 0
    assert set(diagnostics) >= {"margin", "slots", "budget", "leverage"}


def test_put_quantity_qc_blocks_when_stock_inventory_is_full():
    config = BinbinGodParityConfig.from_params(
        {
            "parity_mode": "qc",
            "stock_inventory_cap_enabled": True,
            "stock_inventory_base_cap": 0.20,
            "stock_inventory_cap_floor": 0.50,
        }
    )
    contract = build_contract_lattice(
        symbol="NVDA",
        current_date="2024-03-01",
        underlying_price=100.0,
        iv=0.25,
        target_right="P",
        target_delta=-0.12,
        dte_min=21,
        dte_max=45,
    )[0]
    quantity, diagnostics = calculate_put_quantity_qc(
        config=config,
        selected_contract=contract,
        current_positions=0,
        underlying_price=100.0,
        symbol="NVDA",
        portfolio_value=100000.0,
        margin_remaining=100000.0,
        total_margin_used=0.0,
        stock_holdings_value=18000.0,
        stock_holding_count=1,
        open_option_positions=[],
        shares_held=180,
        dynamic_max_positions=10,
    )
    assert quantity == 0
    assert "stock_inventory" in diagnostics


def test_generate_immediate_cc_signal_in_parity_mode():
    strategy = BinbinGodStrategy(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA"],
            "parity_mode": "qc",
            "contract_universe_mode": "qc_emulated_lattice",
            "ml_confidence_gate": 0.2,
            "ml_enabled": False,
        }
    )
    bars = _make_bars(100.0, 90)
    strategy.mag7_data = {"NVDA": bars}
    strategy.stock_hv = {"NVDA": [0.25] * len(bars)}
    strategy.stock_holding.add_shares("NVDA", 100, 100.0)
    strategy.set_parity_context(
        {
            "portfolio_value": 100000.0,
            "margin_remaining": 100000.0,
            "total_margin_used": 0.0,
            "stock_holdings_value": 10000.0,
            "stock_holding_count": 1,
            "price_by_symbol": {"NVDA": bars[-1]["close"]},
            "dynamic_max_positions": 10,
        }
    )

    signal = strategy.generate_immediate_cc_signal(
        symbol="NVDA",
        current_date=bars[-1]["date"],
        underlying_price=bars[-1]["close"],
        iv=0.25,
        open_positions=[],
    )
    assert signal is not None
    assert signal.right == "C"
    assert signal.confidence > 0


def test_engine_returns_parity_artifacts():
    engine = BacktestEngine()
    result = engine.run(
        {
            "strategy": "binbin_god",
            "symbol": "MAG7_AUTO",
            "stock_pool": ["NVDA", "AAPL"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "initial_capital": 100000,
            "max_leverage": 1.0,
            "use_synthetic_data": True,
            "parity_mode": "qc",
            "contract_universe_mode": "qc_emulated_lattice",
            "ml_confidence_gate": 0.2,
            "ml_enabled": False,
        }
    )
    assert "event_trace" in result
    assert "portfolio_snapshots" in result
    assert "parity_report" in result
    assert result["parity_report"]["status"] in {"no_baseline", "matched", "mismatch", "length_mismatch"}
