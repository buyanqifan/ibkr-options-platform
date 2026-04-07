"""Tests for the simplified Binbin God page QC-aligned parameter UI."""

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
        "target_margin_utilization": 0.65,
        "symbol_assignment_base_cap": 0.35,
        "max_assignment_risk_per_trade": 0.20,
        "max_new_puts_per_day": 3,
        "roll_threshold_pct": 80,
        "min_dte_for_roll": 7,
        "roll_target_dte_min": 21,
        "roll_target_dte_max": 45,
        "cc_below_cost_enabled": True,
        "cc_target_delta": 0.25,
        "cc_target_dte_min": 10,
        "cc_target_dte_max": 28,
        "cc_max_discount_to_cost": 0.03,
        "assigned_stock_fail_safe_enabled": True,
        "assigned_stock_min_days_held": 5,
        "assigned_stock_drawdown_pct": 0.12,
        "assigned_stock_force_exit_pct": 1.0,
        "ml_enabled": True,
        "ml_adoption_rate": 0.5,
        "ml_min_confidence": 0.45,
        "ml_exploration_rate": 0.1,
        "ml_learning_rate": 0.01,
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


def test_layout_contains_simplified_fields(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)

    ids = _collect_ids(_get_layout(page))

    expected_ids = {
        "bbg-stock-pool-text",
        "bbg-max-positions-ceiling",
        "bbg-target-margin-utilization",
        "bbg-symbol-assignment-base-cap",
        "bbg-roll-threshold-pct",
        "bbg-cc-target-delta",
        "bbg-assigned-stock-fail-safe-enabled",
        "bbg-ml-enabled",
    }
    removed_ids = {
        "bbg-max-leverage",
        "bbg-position-aggressiveness",
        "bbg-max-risk-per-trade",
        "bbg-profit-target-pct",
        "bbg-stop-loss-pct",
        "bbg-defensive-put-roll-enabled",
        "bbg-stock-inventory-cap-enabled",
    }

    assert expected_ids.issubset(ids)
    assert ids.isdisjoint(removed_ids)


def test_build_binbin_backtest_params_uses_simplified_payload(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)

    params = page.build_binbin_backtest_params(_default_form_inputs())

    assert params["strategy"] == "binbin_god"
    assert params["parity_mode"] == "qc"
    assert params["contract_universe_mode"] == "qc_emulated_lattice"
    assert params["symbol"] == "MAG7_AUTO"
    assert params["stock_pool"] == ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
    assert params["target_margin_utilization"] == 0.65
    assert params["symbol_assignment_base_cap"] == 0.35
    assert params["max_assignment_risk_per_trade"] == 0.20
    assert params["cc_target_delta"] == 0.25
    assert params["assigned_stock_fail_safe_enabled"] is True
    assert "profit_target_pct" not in params
    assert "defensive_put_roll_enabled" not in params
    assert "stock_inventory_cap_enabled" not in params


def test_build_binbin_backtest_params_normalizes_stock_pool(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)
    form_data = _default_form_inputs()
    form_data["stock_pool_text"] = " msft, aapl , nvda , amd "
    form_data["ml_enabled"] = False

    params = page.build_binbin_backtest_params(form_data)

    assert params["stock_pool"] == ["MSFT", "AAPL", "NVDA", "AMD"]
    assert params["symbol"] == "CUSTOM_MSFT_AAPL_NVDA"
    assert params["ml_enabled"] is False


def test_run_binbin_backtest_submits_simplified_payload_and_returns_results(monkeypatch):
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
    assert captured["params"]["roll_threshold_pct"] == 80
    assert captured["params"]["cc_target_dte_max"] == 28
    assert content is not no_update
    assert loading_style == {"display": "none"}
    assert export_class == "d-block mt-2"


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

    assert results_store.data == result
    assert params_store.data == params
    assert export_container.className == "d-block mt-2"


def test_load_last_result_callback_returns_one_value_per_output_when_missing_payload(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)
    monkeypatch.setattr(page, "load_last_binbin_god_result", lambda: None)

    response = page.load_last_result_callback(1)

    assert len(response) == 32
    assert all(value is no_update for value in response)


def test_layout_avoids_startup_loading_overlay_that_blocks_interaction(monkeypatch):
    page = _load_binbin_god_page(monkeypatch)

    layout = _get_layout(page)
    loading = _find_component(layout, "binbin-loading")
    props = loading.to_plotly_json()["props"]

    assert loading is not None
    assert props.get("show_initially") is False
    assert "visibility" not in props.get("overlay_style", {})
