"""Tests for simplified BinbinGod QC parity helpers."""

from __future__ import annotations

from datetime import datetime

import pytest

from core.backtesting.qc_parity import (
    BinbinGodParityConfig,
    QC_PARAMETER_DEFAULTS,
    _QC_PARAMETER_FALLBACKS,
    _extract_strategy_init_parameter_defaults,
    build_cc_selection_tiers_qc,
    build_contract_lattice,
    calculate_dynamic_max_positions_from_prices,
    calculate_put_quantity_qc,
)
from core.backtesting.strategies.base import Signal
from core.backtesting.strategies.binbin_god import BinbinGodStrategy


class _Position:
    def __init__(self, symbol: str, right: str, quantity: int, strike: float):
        self.symbol = symbol
        self.right = right
        self.quantity = quantity
        self.strike = strike


def test_qc_parity_config_uses_simplified_qc_defaults():
    config = BinbinGodParityConfig.from_params({"parity_mode": "qc"})
    assert config.enabled is True
    assert config.initial_capital == 300000.0
    assert config.max_positions_ceiling == 20
    assert config.target_margin_utilization == pytest.approx(0.65)
    assert config.symbol_assignment_base_cap == pytest.approx(0.45)
    assert config.max_assignment_risk_per_trade == pytest.approx(0.20)
    assert config.roll_threshold_pct == pytest.approx(78.0)
    assert config.min_dte_for_roll == 7
    assert config.cc_target_delta == pytest.approx(0.25)
    assert config.cc_target_dte_min == 10
    assert config.cc_target_dte_max == 28
    assert config.cc_max_discount_to_cost == pytest.approx(0.03)
    assert config.assigned_stock_fail_safe_enabled is True
    assert config.max_new_puts_per_day == 3
    assert config.ml_min_confidence == pytest.approx(0.45)


def test_extract_strategy_init_parameter_defaults_reads_simplified_qc_defaults():
    defaults = _extract_strategy_init_parameter_defaults()
    assert defaults["max_positions_ceiling"] == 20
    assert defaults["target_margin_utilization"] == pytest.approx(0.65)
    assert defaults["symbol_assignment_base_cap"] == pytest.approx(0.45)
    assert defaults["max_assignment_risk_per_trade"] == pytest.approx(0.20)
    assert defaults["roll_threshold_pct"] == pytest.approx(78.0)
    assert defaults["cc_target_delta"] == pytest.approx(0.25)
    assert defaults["assigned_stock_drawdown_pct"] == pytest.approx(0.12)


def test_qc_parameter_defaults_merge_config_and_strategy_init_sources():
    assert QC_PARAMETER_DEFAULTS["initial_capital"] == 300000.0
    assert QC_PARAMETER_DEFAULTS["target_margin_utilization"] == pytest.approx(0.65)
    assert QC_PARAMETER_DEFAULTS["symbol_assignment_base_cap"] == pytest.approx(0.45)
    assert QC_PARAMETER_DEFAULTS["roll_target_dte_max"] == 45
    assert QC_PARAMETER_DEFAULTS["cc_target_dte_max"] == 28
    assert QC_PARAMETER_DEFAULTS["ml_min_confidence"] == pytest.approx(0.45)


def test_qc_parameter_fallbacks_track_runtime_defaults():
    assert _QC_PARAMETER_FALLBACKS["target_margin_utilization"] == pytest.approx(0.65)
    assert _QC_PARAMETER_FALLBACKS["symbol_assignment_base_cap"] == pytest.approx(0.45)
    assert _QC_PARAMETER_FALLBACKS["max_assignment_risk_per_trade"] == pytest.approx(0.20)
    assert _QC_PARAMETER_FALLBACKS["roll_threshold_pct"] == pytest.approx(78.0)
    assert _QC_PARAMETER_FALLBACKS["cc_target_delta"] == pytest.approx(0.25)


def test_build_cc_selection_tiers_qc_only_returns_primary_and_relaxed():
    config = BinbinGodParityConfig.from_params({})
    tiers = build_cc_selection_tiers_qc(
        config=config,
        underlying_price=120.0,
        cost_basis=150.0,
        primary_dte_min=10,
        primary_dte_max=28,
        primary_delta_tolerance=0.08,
        primary_min_strike=145.5,
    )
    assert [tier["label"] for tier in tiers] == ["primary", "delta_relaxed"]
    assert tiers[1]["delta_tolerance"] == pytest.approx(0.16)
    assert tiers[1]["min_strike"] == pytest.approx(145.5)


def test_calculate_dynamic_max_positions_from_prices_uses_portfolio_value_budget():
    config = BinbinGodParityConfig.from_params({})
    result_small = calculate_dynamic_max_positions_from_prices([250.0, 250.0, 250.0], config, portfolio_value=100000.0)
    result_large = calculate_dynamic_max_positions_from_prices([250.0, 250.0, 250.0], config, portfolio_value=160000.0)
    assert result_small < result_large <= config.max_positions_ceiling


def test_calculate_put_quantity_qc_only_uses_three_caps_and_slot_limit():
    config = BinbinGodParityConfig.from_params({})
    contract = build_contract_lattice(
        symbol="NVDA",
        current_date="2024-01-15",
        underlying_price=100.0,
        iv=0.25,
        target_right="P",
        target_delta=-0.30,
        dte_min=21,
        dte_max=45,
        delta_tolerance=0.20,
    )[0]
    quantity, diagnostics = calculate_put_quantity_qc(
        config=config,
        selected_contract=contract,
        current_positions=0,
        underlying_price=100.0,
        symbol="NVDA",
        portfolio_value=300000.0,
        margin_remaining=300000.0,
        total_margin_used=0.0,
        stock_holdings_value=0.0,
        stock_holding_count=0,
        open_option_positions=[],
        shares_held=0,
        dynamic_max_positions=10,
        symbol_history_bars=[],
        pool_history_bars={},
    )
    assert quantity > 0
    assert set(diagnostics).issuperset({
        "portfolio_margin_capacity",
        "symbol_assignment_capacity",
        "trade_assignment_capacity",
        "position_slot_capacity",
        "block_reason",
    })
    assert diagnostics["block_reason"] == ""


def test_calculate_put_quantity_qc_reports_single_block_reason():
    config = BinbinGodParityConfig.from_params({})
    contract = build_contract_lattice(
        symbol="NVDA",
        current_date="2024-01-15",
        underlying_price=100.0,
        iv=0.25,
        target_right="P",
        target_delta=-0.30,
        dte_min=21,
        dte_max=45,
        delta_tolerance=0.20,
    )[0]
    quantity, diagnostics = calculate_put_quantity_qc(
        config=config,
        selected_contract=contract,
        current_positions=10,
        underlying_price=100.0,
        symbol="NVDA",
        portfolio_value=300000.0,
        margin_remaining=300000.0,
        total_margin_used=0.0,
        stock_holdings_value=0.0,
        stock_holding_count=0,
        open_option_positions=[],
        shares_held=0,
        dynamic_max_positions=10,
        symbol_history_bars=[],
        pool_history_bars={},
    )
    assert quantity == 0
    assert diagnostics["block_reason"] == "position_slots"


def test_binbin_god_cc_below_cost_uses_cost_floor_and_two_tiers(monkeypatch):
    strategy = BinbinGodStrategy({"symbol": "NVDA", "stock_pool": ["NVDA"]})
    strategy.stock_holding.add_shares("NVDA", 100, 150.0)
    captured = {}

    def fake_select_contract_from_lattice(**kwargs):
        captured.update(kwargs)
        return type(
            "SelectedContract",
            (),
            {
                "strike": 146.0,
                "expiry": datetime(2024, 2, 9),
                "dte": 25,
                "premium": 1.75,
                "delta": 0.24,
                "to_dict": lambda self: {"selection_tier": "delta_relaxed", "strike": 146.0},
            },
        )()

    monkeypatch.setattr("core.backtesting.strategies.binbin_god.select_contract_from_lattice", fake_select_contract_from_lattice)
    signals = strategy._generate_backtest_call_signal(
        symbol="NVDA",
        current_date="2024-01-15",
        underlying_price=120.0,
        iv=0.25,
        shares_available=100,
        cost_basis=150.0,
    )
    assert len(signals) == 1
    assert signals[0].metadata["selection_tier"] == "delta_relaxed"
    assert captured["min_strike"] == pytest.approx(max(120.0 * 1.01, 150.0 * 0.97))
    assert [tier["label"] for tier in captured["selection_tiers"]] == ["primary", "delta_relaxed"]


def test_binbin_god_roll_rule_only_uses_threshold_and_dte():
    strategy = BinbinGodStrategy({"symbol": "NVDA"})
    should_exit, reason = strategy.should_exit_position(
        position={"right": "P", "expiry": "20240216"},
        current_price=0.3,
        entry_price=2.0,
        current_dt=datetime(2024, 1, 15),
        market_data={"price": 90.0},
    )
    assert should_exit is True
    assert reason == "ROLL_FORWARD"
