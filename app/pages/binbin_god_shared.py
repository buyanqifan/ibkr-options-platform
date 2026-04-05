"""Shared constants and utilities for Binbin God pages."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from core.backtesting.qc_parity import QC_BINBIN_DEFAULTS, QC_PARAMETER_DEFAULTS


def _parse_default_stock_pool(raw_value) -> list[str]:
    if isinstance(raw_value, str):
        symbols = [symbol.strip().upper() for symbol in raw_value.split(",") if symbol.strip()]
        if symbols:
            return symbols
    if isinstance(raw_value, (list, tuple)):
        symbols = [str(symbol).strip().upper() for symbol in raw_value if str(symbol).strip()]
        if symbols:
            return symbols
    return ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]


DEFAULT_STOCK_POOL = _parse_default_stock_pool(QC_PARAMETER_DEFAULTS.get("stock_pool"))

QC_UI_DEFAULTS = {
    "start_date": str(QC_PARAMETER_DEFAULTS.get("start_date", "2024-01-01")),
    "end_date": str(QC_PARAMETER_DEFAULTS.get("end_date", "2024-12-31")),
    "initial_capital": QC_BINBIN_DEFAULTS["initial_capital"],
    "stock_pool_text": ",".join(DEFAULT_STOCK_POOL),
    "max_positions_ceiling": QC_BINBIN_DEFAULTS["max_positions_ceiling"],
    "target_margin_utilization": QC_BINBIN_DEFAULTS["target_margin_utilization"],
    "symbol_assignment_base_cap": QC_BINBIN_DEFAULTS["symbol_assignment_base_cap"],
    "max_assignment_risk_per_trade": QC_BINBIN_DEFAULTS["max_assignment_risk_per_trade"],
    "roll_threshold_pct": QC_BINBIN_DEFAULTS["roll_threshold_pct"],
    "min_dte_for_roll": QC_BINBIN_DEFAULTS["min_dte_for_roll"],
    "roll_target_dte_min": QC_BINBIN_DEFAULTS["roll_target_dte_min"],
    "roll_target_dte_max": QC_BINBIN_DEFAULTS["roll_target_dte_max"],
    "cc_below_cost_enabled": QC_BINBIN_DEFAULTS["cc_below_cost_enabled"],
    "cc_target_delta": QC_BINBIN_DEFAULTS["cc_target_delta"],
    "cc_target_dte_min": QC_BINBIN_DEFAULTS["cc_target_dte_min"],
    "cc_target_dte_max": QC_BINBIN_DEFAULTS["cc_target_dte_max"],
    "cc_max_discount_to_cost": QC_BINBIN_DEFAULTS["cc_max_discount_to_cost"],
    "assigned_stock_fail_safe_enabled": QC_BINBIN_DEFAULTS["assigned_stock_fail_safe_enabled"],
    "assigned_stock_min_days_held": QC_BINBIN_DEFAULTS["assigned_stock_min_days_held"],
    "assigned_stock_drawdown_pct": QC_BINBIN_DEFAULTS["assigned_stock_drawdown_pct"],
    "assigned_stock_force_exit_pct": QC_BINBIN_DEFAULTS["assigned_stock_force_exit_pct"],
    "max_new_puts_per_day": QC_BINBIN_DEFAULTS["max_new_puts_per_day"],
    "ml_enabled": QC_BINBIN_DEFAULTS["ml_enabled"],
    "ml_min_confidence": QC_BINBIN_DEFAULTS["ml_min_confidence"],
    "ml_adoption_rate": QC_BINBIN_DEFAULTS["ml_adoption_rate"],
    "ml_exploration_rate": QC_BINBIN_DEFAULTS["ml_exploration_rate"],
    "ml_learning_rate": QC_BINBIN_DEFAULTS["ml_learning_rate"],
}

RUN_SETUP_FIELDS = [
    {"id": "bbg-start", "label": "Start Date", "type": "date", "default": "start_date"},
    {"id": "bbg-end", "label": "End Date", "type": "date", "default": "end_date"},
    {"id": "bbg-initial-capital", "label": "Initial Capital ($)", "type": "number", "default": "initial_capital", "step": 1000, "min": 0},
    {"id": "bbg-stock-pool-text", "label": "Stock Pool (comma separated)", "type": "text", "default": "stock_pool_text", "width": 12},
]

SP_RISK_FIELDS = [
    {"id": "bbg-max-positions-ceiling", "label": "Max Positions Ceiling", "type": "number", "default": "max_positions_ceiling", "step": 1, "min": 1},
    {"id": "bbg-target-margin-utilization", "label": "Target Margin Utilization", "type": "number", "default": "target_margin_utilization", "step": 0.05, "min": 0, "max": 1.5},
    {"id": "bbg-symbol-assignment-base-cap", "label": "Symbol Assignment Base Cap", "type": "number", "default": "symbol_assignment_base_cap", "step": 0.05, "min": 0, "max": 1.5},
    {"id": "bbg-max-assignment-risk-per-trade", "label": "Max Assignment Risk / Trade", "type": "number", "default": "max_assignment_risk_per_trade", "step": 0.05, "min": 0, "max": 1.0},
    {"id": "bbg-max-new-puts-per-day", "label": "Max New Puts / Day", "type": "number", "default": "max_new_puts_per_day", "step": 1, "min": 1},
]

SP_ROLL_FIELDS = [
    {"id": "bbg-roll-threshold-pct", "label": "Roll Threshold (% premium captured)", "type": "number", "default": "roll_threshold_pct", "step": 5, "min": 0},
    {"id": "bbg-min-dte-for-roll", "label": "Min DTE For Roll", "type": "number", "default": "min_dte_for_roll", "step": 1, "min": 0},
    {"id": "bbg-roll-target-dte-min", "label": "Roll Target DTE Min", "type": "number", "default": "roll_target_dte_min", "step": 1, "min": 1},
    {"id": "bbg-roll-target-dte-max", "label": "Roll Target DTE Max", "type": "number", "default": "roll_target_dte_max", "step": 1, "min": 1},
]

CC_FIELDS = [
    {"id": "bbg-cc-below-cost-enabled", "label": "Allow Below-Cost CC", "type": "switch", "default": "cc_below_cost_enabled"},
    {"id": "bbg-cc-target-delta", "label": "CC Target Delta", "type": "number", "default": "cc_target_delta", "step": 0.01, "min": 0.1, "max": 0.6},
    {"id": "bbg-cc-target-dte-min", "label": "CC Target DTE Min", "type": "number", "default": "cc_target_dte_min", "step": 1, "min": 1},
    {"id": "bbg-cc-target-dte-max", "label": "CC Target DTE Max", "type": "number", "default": "cc_target_dte_max", "step": 1, "min": 1},
    {"id": "bbg-cc-max-discount-to-cost", "label": "CC Max Discount To Cost", "type": "number", "default": "cc_max_discount_to_cost", "step": 0.01, "min": 0, "max": 0.3},
]

ASSIGNED_STOCK_FIELDS = [
    {"id": "bbg-assigned-stock-fail-safe-enabled", "label": "Enable Assigned Stock Fail-Safe", "type": "switch", "default": "assigned_stock_fail_safe_enabled"},
    {"id": "bbg-assigned-stock-min-days-held", "label": "Assigned Stock Min Days Held", "type": "number", "default": "assigned_stock_min_days_held", "step": 1, "min": 0},
    {"id": "bbg-assigned-stock-drawdown-pct", "label": "Assigned Stock Drawdown", "type": "number", "default": "assigned_stock_drawdown_pct", "step": 0.01, "min": 0, "max": 1},
    {"id": "bbg-assigned-stock-force-exit-pct", "label": "Assigned Stock Force Exit %", "type": "number", "default": "assigned_stock_force_exit_pct", "step": 0.05, "min": 0, "max": 1},
]

ML_FIELDS = [
    {"id": "bbg-ml-enabled", "label": "Enable ML", "type": "switch", "default": "ml_enabled"},
    {"id": "bbg-ml-min-confidence", "label": "ML Min Confidence", "type": "number", "default": "ml_min_confidence", "step": 0.05, "min": 0, "max": 1},
    {"id": "bbg-ml-adoption-rate", "label": "ML Adoption Rate", "type": "number", "default": "ml_adoption_rate", "step": 0.1, "min": 0, "max": 1},
    {"id": "bbg-ml-exploration-rate", "label": "ML Exploration Rate", "type": "number", "default": "ml_exploration_rate", "step": 0.01, "min": 0, "max": 1},
    {"id": "bbg-ml-learning-rate", "label": "ML Learning Rate", "type": "number", "default": "ml_learning_rate", "step": 0.001, "min": 0, "max": 1},
]

ALL_FORM_FIELDS = RUN_SETUP_FIELDS + SP_RISK_FIELDS + SP_ROLL_FIELDS + CC_FIELDS + ASSIGNED_STOCK_FIELDS + ML_FIELDS


def _control_from_field(field, defaults=None):
    defaults = defaults or QC_UI_DEFAULTS
    default_value = defaults[field["default"]]
    width = field.get("width", 6)
    if field["type"] == "switch":
        control = html.Div([dbc.Switch(id=field["id"], value=default_value, label=field["label"], className="mb-1")], className="mb-3")
    else:
        input_type = "text" if field["type"] == "text" else field["type"]
        control = html.Div([
            dbc.Label(field["label"], className="fw-semibold"),
            dbc.Input(id=field["id"], type=input_type, value=default_value, step=field.get("step"), min=field.get("min"), max=field.get("max")),
        ], className="mb-3")
    return dbc.Col(control, md=width)


def _build_rows(fields, defaults=None):
    current_row = []
    current_width = 0
    rows = []
    for field in fields:
        width = field.get("width", 6)
        if current_row and current_width + width > 12:
            rows.append(dbc.Row(current_row, className="g-2"))
            current_row = []
            current_width = 0
        current_row.append(_control_from_field(field, defaults))
        current_width += width
        if current_width == 12:
            rows.append(dbc.Row(current_row, className="g-2"))
            current_row = []
            current_width = 0
    if current_row:
        rows.append(dbc.Row(current_row, className="g-2"))
    return rows


def _section_card(title: str, children, color: str = "secondary"):
    return dbc.Card([dbc.CardHeader(title, className=f"bg-{color} text-white fw-semibold"), dbc.CardBody(children)], className="mb-3")


def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _to_bool(value, default):
    if value is None:
        return bool(default)
    return bool(value)


def _normalize_stock_pool(stock_pool_text: str | None) -> list[str]:
    raw = stock_pool_text if stock_pool_text is not None else QC_UI_DEFAULTS["stock_pool_text"]
    symbols = [symbol.strip().upper() for symbol in str(raw).split(",") if symbol.strip()]
    return symbols or DEFAULT_STOCK_POOL.copy()


def _derive_symbol(stock_pool: list[str]) -> str:
    if stock_pool == DEFAULT_STOCK_POOL:
        return "MAG7_AUTO"
    return f"CUSTOM_{'_'.join(stock_pool[:3])}" if stock_pool else "MAG7_AUTO"


def _form_values_from_params(params: dict | None):
    params = params or {}
    merged = dict(QC_UI_DEFAULTS)
    merged.update(params)
    stock_pool = params.get("stock_pool") if isinstance(params, dict) else None
    merged["stock_pool_text"] = ",".join(stock_pool) if stock_pool else QC_UI_DEFAULTS["stock_pool_text"]
    return tuple(merged.get(field["default"], QC_UI_DEFAULTS[field["default"]]) for field in ALL_FORM_FIELDS)


def build_binbin_backtest_params(form_data):
    """Normalize simplified UI inputs into engine parameters."""
    merged = {**QC_UI_DEFAULTS, **(form_data or {})}
    stock_pool = _normalize_stock_pool(merged.get("stock_pool_text"))
    return {
        "strategy": "binbin_god",
        "start_date": merged.get("start_date", QC_UI_DEFAULTS["start_date"]),
        "end_date": merged.get("end_date", QC_UI_DEFAULTS["end_date"]),
        "initial_capital": _to_float(merged.get("initial_capital"), QC_UI_DEFAULTS["initial_capital"]),
        "symbol": _derive_symbol(stock_pool),
        "stock_pool": stock_pool,
        "max_positions_ceiling": _to_int(merged.get("max_positions_ceiling"), QC_UI_DEFAULTS["max_positions_ceiling"]),
        "max_positions": _to_int(merged.get("max_positions_ceiling"), QC_UI_DEFAULTS["max_positions_ceiling"]),
        "target_margin_utilization": _to_float(merged.get("target_margin_utilization"), QC_UI_DEFAULTS["target_margin_utilization"]),
        "symbol_assignment_base_cap": _to_float(merged.get("symbol_assignment_base_cap"), QC_UI_DEFAULTS["symbol_assignment_base_cap"]),
        "max_assignment_risk_per_trade": _to_float(merged.get("max_assignment_risk_per_trade"), QC_UI_DEFAULTS["max_assignment_risk_per_trade"]),
        "roll_threshold_pct": _to_float(merged.get("roll_threshold_pct"), QC_UI_DEFAULTS["roll_threshold_pct"]),
        "min_dte_for_roll": _to_int(merged.get("min_dte_for_roll"), QC_UI_DEFAULTS["min_dte_for_roll"]),
        "roll_target_dte_min": _to_int(merged.get("roll_target_dte_min"), QC_UI_DEFAULTS["roll_target_dte_min"]),
        "roll_target_dte_max": _to_int(merged.get("roll_target_dte_max"), QC_UI_DEFAULTS["roll_target_dte_max"]),
        "cc_below_cost_enabled": _to_bool(merged.get("cc_below_cost_enabled"), QC_UI_DEFAULTS["cc_below_cost_enabled"]),
        "cc_target_delta": _to_float(merged.get("cc_target_delta"), QC_UI_DEFAULTS["cc_target_delta"]),
        "cc_target_dte_min": _to_int(merged.get("cc_target_dte_min"), QC_UI_DEFAULTS["cc_target_dte_min"]),
        "cc_target_dte_max": _to_int(merged.get("cc_target_dte_max"), QC_UI_DEFAULTS["cc_target_dte_max"]),
        "cc_max_discount_to_cost": _to_float(merged.get("cc_max_discount_to_cost"), QC_UI_DEFAULTS["cc_max_discount_to_cost"]),
        "assigned_stock_fail_safe_enabled": _to_bool(merged.get("assigned_stock_fail_safe_enabled"), QC_UI_DEFAULTS["assigned_stock_fail_safe_enabled"]),
        "assigned_stock_min_days_held": _to_int(merged.get("assigned_stock_min_days_held"), QC_UI_DEFAULTS["assigned_stock_min_days_held"]),
        "assigned_stock_drawdown_pct": _to_float(merged.get("assigned_stock_drawdown_pct"), QC_UI_DEFAULTS["assigned_stock_drawdown_pct"]),
        "assigned_stock_force_exit_pct": _to_float(merged.get("assigned_stock_force_exit_pct"), QC_UI_DEFAULTS["assigned_stock_force_exit_pct"]),
        "max_new_puts_per_day": _to_int(merged.get("max_new_puts_per_day"), QC_UI_DEFAULTS["max_new_puts_per_day"]),
        "ml_enabled": _to_bool(merged.get("ml_enabled"), QC_UI_DEFAULTS["ml_enabled"]),
        "ml_min_confidence": _to_float(merged.get("ml_min_confidence"), QC_UI_DEFAULTS["ml_min_confidence"]),
        "ml_confidence_gate": _to_float(merged.get("ml_min_confidence"), QC_UI_DEFAULTS["ml_min_confidence"]),
        "ml_adoption_rate": _to_float(merged.get("ml_adoption_rate"), QC_UI_DEFAULTS["ml_adoption_rate"]),
        "ml_exploration_rate": _to_float(merged.get("ml_exploration_rate"), QC_UI_DEFAULTS["ml_exploration_rate"]),
        "ml_learning_rate": _to_float(merged.get("ml_learning_rate"), QC_UI_DEFAULTS["ml_learning_rate"]),
        "parity_mode": "qc",
        "contract_universe_mode": "qc_emulated_lattice",
    }
