"""Tests for the Binbin God page QC-aligned parameter UI."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import dash
from dash import no_update


def _load_binbin_god_page(monkeypatch):
    monkeypatch.setattr(dash, "register_page", lambda *args, **kwargs: None)
    sys.modules.pop("app.pages.binbin_god", None)
    return importlib.import_module("app.pages.binbin_god")


def _collect_ids(node: Any) -> set[str]:
    ids: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        if isinstance(current, (list, tuple)):
            stack.extend(current)
            continue
        if isinstance(current, (str, int, float, bool)):
            continue
        current_id = getattr(current, "id", None)
        if isinstance(current_id, str):
            ids.add(current_id)
        children = getattr(current, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return ids


def _find_component(node: Any, target_id: str):
    stack = [node]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        if isinstance(current, (list, tuple)):
            stack.extend(current)
            continue
        if isinstance(current, (str, int, float, bool)):
            continue
        if getattr(current, "id", None) == target_id:
            return current
        children = getattr(current, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return None


def _count_text(node: Any, needle: str) -> int:
    count = 0
    stack = [node]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        if isinstance(current, (list, tuple)):
            stack.extend(current)
            continue
        if isinstance(current, str):
            count += current.count(needle)
            continue
        if isinstance(current, (int, float, bool)):
            continue
        children = getattr(current, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return count


def _get_layout(page):
    layout = page.layout
    return layout() if callable(layout) else layout


def _default_form_inputs() -> dict[str, Any]:
    return {
        "start_date": "2024-01-01",
        "end_date": "2025-12-31",
        "initial_capital": 300000,
        "stock_pool_text": "MSFT,AAPL,NVDA,GOOGL,AMZN,META,TSLA",
        "max_positions_ceiling": 20,
        "max_leverage": 1.0,
        "target_margin_utilization": 0.45,
        "position_aggressiveness": 1.2,
        "profit_target_pct": 70,
        "stop_loss_pct": 999999,
        "margin_buffer_pct": 0.40,
        "margin_rate_per_contract": 0.25,
        "max_risk_per_trade": 0.03,
        "dte_min": 21,
        "dte_max": 60,
        "put_delta": 0.30,
        "call_delta": 0.30,
        "ml_enabled": True,
        "ml_adoption_rate": 0.5,
        "ml_min_confidence": 0.4,
        "ml_exploration_rate": 0.1,
        "ml_learning_rate": 0.01,
        "cc_optimization_enabled": True,
        "cc_min_delta_cost": 0.15,
        "cc_cost_basis_threshold": 0.05,
        "cc_min_strike_premium": 0.02,
        "repair_call_threshold_pct": 0.08,
        "repair_call_delta": 0.35,
        "repair_call_dte_min": 7,
        "repair_call_dte_max": 21,
        "repair_call_max_discount_pct": 0.08,
        "defensive_put_roll_enabled": True,
        "defensive_put_roll_loss_pct": 70,
        "defensive_put_roll_itm_buffer_pct": 0.03,
        "defensive_put_roll_min_dte": 7,
        "defensive_put_roll_max_dte": 21,
        "defensive_put_roll_dte_min": 21,
        "defensive_put_roll_dte_max": 60,
        "defensive_put_roll_delta": 0.20,
        "assignment_cooldown_days": 20,
        "large_loss_cooldown_days": 15,
        "large_loss_cooldown_pct": 100,
        "volatility_cap_floor": 0.35,
        "volatility_cap_ceiling": 1.0,
        "volatility_lookback": 20,
        "dynamic_symbol_risk_enabled": True,
        "symbol_state_cap_floor": 0.20,
        "symbol_state_cap_ceiling": 1.0,
        "symbol_drawdown_lookback": 60,
        "symbol_drawdown_sensitivity": 1.20,
        "symbol_downtrend_sensitivity": 1.50,
        "symbol_volatility_sensitivity": 0.75,
        "symbol_exposure_sensitivity": 1.25,
        "symbol_assignment_base_cap": 0.20,
        "stock_inventory_cap_enabled": True,
        "stock_inventory_base_cap": 0.12,
        "stock_inventory_cap_floor": 0.50,
        "stock_inventory_block_threshold": 0.75,
    }


class StubEngine:
    def run(self, params):
        return {
            "metrics": {
                "total_return_pct": 1.0,
                "annualized_return_pct": 1.0,
                "win_rate": 50.0,
                "sharpe_ratio": 1.0,
                "max_drawdown_pct": -5.0,
                "total_trades": 1,
                "avg_profit": 100.0,
                "avg_loss": -50.0,
                "profit_factor": 2.0,
                "sortino_ratio": 1.5,
                "monthly_returns": {},
            },
            "trades": [],
            "daily_pnl": [],
            "benchmark_data": {},
            "underlying_prices": [],
            "multi_stock_prices": {},
            "strategy_performance": {},
        }


def test_layout_contains_qc_fields_and_omits_legacy_controls(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)

    ids = _collect_ids(_get_layout(page))

    expected_ids = {
        "bbg-stock-pool-text",
        "bbg-max-positions-ceiling",
        "bbg-target-margin-utilization",
        "bbg-position-aggressiveness",
        "bbg-ml-enabled",
        "bbg-ml-min-confidence",
        "bbg-defensive-put-roll-enabled",
        "bbg-symbol-assignment-base-cap",
        "bbg-stock-inventory-cap-enabled",
    }
    removed_ids = {
        "bbg-run-mode",
        "bbg-use-synthetic",
        "bbg-stock-pool",
        "bbg-custom-stocks",
        "bbg-ml-optimization",
        "bbg-ml-dte-optimization",
        "bbg-ml-roll-optimization",
        "bbg-ml-position-optimization",
        "bbg-disable-profit-target",
        "bbg-disable-stop-loss",
        "bbg-rebalance-threshold",
    }

    assert expected_ids.issubset(ids)
    assert ids.isdisjoint(removed_ids)


def test_build_binbin_backtest_params_uses_qc_defaults(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)

    params = page.build_binbin_backtest_params(_default_form_inputs())

    assert params["strategy"] == "binbin_god"
    assert params["parity_mode"] == "qc"
    assert params["contract_universe_mode"] == "qc_emulated_lattice"
    assert params["ml_confidence_gate"] == 0.4
    assert params["symbol"] == "MAG7_AUTO"
    assert params["stock_pool"] == ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
    assert params["initial_capital"] == 300000
    assert params["end_date"] == "2025-12-31"
    assert params["max_positions_ceiling"] == 20
    assert params["max_positions"] == 20
    assert params["position_aggressiveness"] == 1.2
    assert params["profit_target_pct"] == 70
    assert params["margin_buffer_pct"] == 0.40
    assert params["target_margin_utilization"] == 0.45
    assert params["max_risk_per_trade"] == 0.03
    assert params["ml_delta_optimization"] is True
    assert params["ml_dte_optimization"] is True
    assert params["ml_roll_optimization"] is True
    assert params["ml_position_optimization"] is True
    assert params["stop_loss_pct"] == 999999
    assert params["symbol_assignment_base_cap"] == 0.20
    assert params["stock_inventory_base_cap"] == 0.12
    assert params["stock_inventory_block_threshold"] == 0.75
    assert "run_mode" not in params


def test_build_binbin_backtest_params_normalizes_stock_pool(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)
    form_data = _default_form_inputs()
    form_data["stock_pool_text"] = " msft, aapl , nvda , amd "
    form_data["ml_enabled"] = False

    params = page.build_binbin_backtest_params(form_data)

    assert params["stock_pool"] == ["MSFT", "AAPL", "NVDA", "AMD"]
    assert params["symbol"] == "CUSTOM_MSFT_AAPL_NVDA"
    assert params["ml_delta_optimization"] is False
    assert params["ml_dte_optimization"] is False
    assert params["ml_roll_optimization"] is False
    assert params["ml_position_optimization"] is False


def test_run_binbin_backtest_submits_qc_payload_and_returns_results(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)

    captured = {}

    class CapturingEngine(StubEngine):
        def run(self, params):
            captured["params"] = params
            return super().run(params)

    monkeypatch.setattr(page, "get_services_cached", lambda: {"backtest_engine": CapturingEngine()})

    result, params, _, content, loading_style, export_class = page.run_binbin_backtest(
        n_clicks=1,
        **_default_form_inputs(),
    )

    assert result["metrics"]["total_return_pct"] == 1.0
    assert params["parity_mode"] == "qc"
    assert captured["params"]["ml_roll_optimization"] is True
    assert content is not no_update
    assert loading_style == {"display": "none"}
    assert export_class == "d-block mt-2"


def test_run_binbin_backtest_saves_last_result(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)
    saved = {}

    monkeypatch.setattr(
        page,
        "save_last_binbin_god_result",
        lambda params, result: saved.update({"params": params, "result": result}),
    )
    monkeypatch.setattr(page, "get_services_cached", lambda: {"backtest_engine": StubEngine()})

    result, params, _, _, _, _ = page.run_binbin_backtest(
        n_clicks=1,
        **_default_form_inputs(),
    )

    assert saved["params"] == params
    assert saved["result"] == result
    assert saved["params"]["strategy"] == "binbin_god"


def test_layout_uses_last_result_as_initial_state(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)
    params = page.build_binbin_backtest_params(_default_form_inputs())
    result = StubEngine().run(params)

    monkeypatch.setattr(
        page,
        "load_last_binbin_god_result",
        lambda: {"params": params, "result": result},
    )

    layout = _get_layout(page)
    results_store = _find_component(layout, "binbin-results-store")
    params_store = _find_component(layout, "binbin-params-store")
    export_container = _find_component(layout, "bbg-export-container")
    start_input = _find_component(layout, "bbg-start")

    assert results_store.data["metrics"]["total_return_pct"] == 1.0
    assert params_store.data == params
    assert export_container.className == "d-block mt-2"
    assert start_input.value == params["start_date"]


def test_results_view_does_not_duplicate_mag7_analysis(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)
    params = page.build_binbin_backtest_params(_default_form_inputs())
    result = StubEngine().run(params)
    result["mag7_analysis"] = {
        "best_pick": {"symbol": "NVDA", "total_score": 9.5},
        "ranked_stocks": [
            {"symbol": "NVDA", "total_score": 9.5, "momentum": 1.0, "stability": 2.0},
        ],
    }

    mag7_section, content, _, _ = page._build_binbin_results_view(result, params)

    assert _count_text(mag7_section, "MAG7 Stock Analysis") == 1
    assert _count_text(content, "MAG7 Stock Analysis") == 0
