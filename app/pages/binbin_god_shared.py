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
    "max_leverage": 1.0,
    "target_margin_utilization": QC_BINBIN_DEFAULTS["target_margin_utilization"],
    "position_aggressiveness": QC_BINBIN_DEFAULTS["position_aggressiveness"],
    "profit_target_pct": QC_BINBIN_DEFAULTS["profit_target_pct"],
    "stop_loss_pct": QC_BINBIN_DEFAULTS["stop_loss_pct"],
    "margin_buffer_pct": QC_BINBIN_DEFAULTS["margin_buffer_pct"],
    "margin_rate_per_contract": QC_BINBIN_DEFAULTS["margin_rate_per_contract"],
    "max_risk_per_trade": QC_BINBIN_DEFAULTS["max_risk_per_trade"],
    "dte_min": QC_BINBIN_DEFAULTS["dte_min"],
    "dte_max": QC_BINBIN_DEFAULTS["dte_max"],
    "put_delta": QC_BINBIN_DEFAULTS["put_delta"],
    "call_delta": QC_BINBIN_DEFAULTS["call_delta"],
    "ml_enabled": QC_BINBIN_DEFAULTS["ml_enabled"],
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
    "defensive_put_roll_loss_pct": QC_BINBIN_DEFAULTS["defensive_put_roll_loss_pct"],
    "defensive_put_roll_itm_buffer_pct": QC_BINBIN_DEFAULTS["defensive_put_roll_itm_buffer_pct"],
    "defensive_put_roll_min_dte": 7,
    "defensive_put_roll_max_dte": 14,
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
    "symbol_assignment_base_cap": QC_BINBIN_DEFAULTS["symbol_assignment_base_cap"],
    "stock_inventory_cap_enabled": True,
    "stock_inventory_base_cap": QC_BINBIN_DEFAULTS["stock_inventory_base_cap"],
    "stock_inventory_cap_floor": 0.50,
    "stock_inventory_block_threshold": QC_BINBIN_DEFAULTS["stock_inventory_block_threshold"],
}

RUN_SETUP_FIELDS = [
    {"id": "bbg-start", "label": "Start Date", "type": "date", "default": "start_date"},
    {"id": "bbg-end", "label": "End Date", "type": "date", "default": "end_date"},
    {
        "id": "bbg-initial-capital",
        "label": "Initial Capital ($)",
        "type": "number",
        "default": "initial_capital",
        "step": 1000,
        "min": 0,
    },
    {
        "id": "bbg-stock-pool-text",
        "label": "Stock Pool (comma separated)",
        "type": "text",
        "default": "stock_pool_text",
        "width": 12,
    },
]

CORE_WHEEL_FIELDS = [
    {"id": "bbg-max-positions-ceiling", "label": "Max Positions Ceiling", "type": "number", "default": "max_positions_ceiling", "step": 1, "min": 1},
    {"id": "bbg-max-leverage", "label": "Max Leverage", "type": "number", "default": "max_leverage", "step": 0.1, "min": 0},
    {
        "id": "bbg-target-margin-utilization",
        "label": "Target Margin Utilization",
        "type": "number",
        "default": "target_margin_utilization",
        "step": 0.05,
        "min": 0,
        "max": 1.5,
    },
    {
        "id": "bbg-position-aggressiveness",
        "label": "Position Aggressiveness",
        "type": "number",
        "default": "position_aggressiveness",
        "step": 0.1,
        "min": 0.3,
        "max": 2.0,
    },
    {
        "id": "bbg-max-risk-per-trade",
        "label": "Max Risk Per Trade",
        "type": "number",
        "default": "max_risk_per_trade",
        "step": 0.01,
        "min": 0,
        "max": 1,
    },
    {"id": "bbg-profit-target-pct", "label": "Profit Target (% of premium)", "type": "number", "default": "profit_target_pct", "step": 5, "min": 0},
    {
        "id": "bbg-stop-loss-pct",
        "label": "Stop Loss (% of premium)",
        "type": "number",
        "default": "stop_loss_pct",
        "step": 50,
        "min": 0,
        "help_text": "Use 999999 to disable the traditional stop loss, matching QC.",
    },
    {"id": "bbg-margin-buffer-pct", "label": "Margin Buffer", "type": "number", "default": "margin_buffer_pct", "step": 0.05, "min": 0, "max": 1.5},
    {
        "id": "bbg-margin-rate-per-contract",
        "label": "Margin Rate Per Contract",
        "type": "number",
        "default": "margin_rate_per_contract",
        "step": 0.05,
        "min": 0,
        "max": 1.0,
    },
    {"id": "bbg-dte-min", "label": "DTE Min", "type": "number", "default": "dte_min", "step": 1, "min": 1},
    {"id": "bbg-dte-max", "label": "DTE Max", "type": "number", "default": "dte_max", "step": 1, "min": 1},
    {"id": "bbg-put-delta", "label": "Put Delta", "type": "number", "default": "put_delta", "step": 0.01, "min": 0, "max": 1},
    {"id": "bbg-call-delta", "label": "Call Delta", "type": "number", "default": "call_delta", "step": 0.01, "min": 0, "max": 1},
]

ML_FIELDS = [
    {"id": "bbg-ml-enabled", "label": "Enable ML", "type": "switch", "default": "ml_enabled", "help_text": "One toggle controls delta, DTE, roll, and position sizing ML paths."},
    {"id": "bbg-ml-adoption-rate", "label": "ML Adoption Rate", "type": "number", "default": "ml_adoption_rate", "step": 0.1, "min": 0, "max": 1},
    {"id": "bbg-ml-min-confidence", "label": "ML Min Confidence", "type": "number", "default": "ml_min_confidence", "step": 0.05, "min": 0, "max": 1},
    {"id": "bbg-ml-exploration-rate", "label": "ML Exploration Rate", "type": "number", "default": "ml_exploration_rate", "step": 0.01, "min": 0, "max": 1},
    {"id": "bbg-ml-learning-rate", "label": "ML Learning Rate", "type": "number", "default": "ml_learning_rate", "step": 0.001, "min": 0, "max": 1},
]

CC_REPAIR_FIELDS = [
    {"id": "bbg-cc-optimization-enabled", "label": "Enable CC Optimization", "type": "switch", "default": "cc_optimization_enabled"},
    {"id": "bbg-cc-min-delta-cost", "label": "CC Min Delta Cost", "type": "number", "default": "cc_min_delta_cost", "step": 0.01, "min": 0, "max": 1},
    {
        "id": "bbg-cc-cost-basis-threshold",
        "label": "CC Cost Basis Threshold",
        "type": "number",
        "default": "cc_cost_basis_threshold",
        "step": 0.01,
        "min": 0,
        "max": 1,
    },
    {
        "id": "bbg-cc-min-strike-premium",
        "label": "CC Min Strike Premium",
        "type": "number",
        "default": "cc_min_strike_premium",
        "step": 0.01,
        "min": 0,
        "max": 1,
    },
    {
        "id": "bbg-repair-call-threshold-pct",
        "label": "Repair Call Threshold",
        "type": "number",
        "default": "repair_call_threshold_pct",
        "step": 0.01,
        "min": 0,
        "max": 1,
    },
    {"id": "bbg-repair-call-delta", "label": "Repair Call Delta", "type": "number", "default": "repair_call_delta", "step": 0.01, "min": 0, "max": 1},
    {"id": "bbg-repair-call-dte-min", "label": "Repair Call DTE Min", "type": "number", "default": "repair_call_dte_min", "step": 1, "min": 1},
    {"id": "bbg-repair-call-dte-max", "label": "Repair Call DTE Max", "type": "number", "default": "repair_call_dte_max", "step": 1, "min": 1},
    {
        "id": "bbg-repair-call-max-discount-pct",
        "label": "Repair Call Max Discount",
        "type": "number",
        "default": "repair_call_max_discount_pct",
        "step": 0.01,
        "min": 0,
        "max": 1,
    },
]

DEFENSIVE_FIELDS = [
    {"id": "bbg-defensive-put-roll-enabled", "label": "Enable Defensive Put Roll", "type": "switch", "default": "defensive_put_roll_enabled"},
    {
        "id": "bbg-defensive-put-roll-loss-pct",
        "label": "Defensive Roll Loss %",
        "type": "number",
        "default": "defensive_put_roll_loss_pct",
        "step": 10,
        "min": 0,
    },
    {
        "id": "bbg-defensive-put-roll-itm-buffer-pct",
        "label": "Defensive Roll ITM Buffer",
        "type": "number",
        "default": "defensive_put_roll_itm_buffer_pct",
        "step": 0.01,
        "min": 0,
        "max": 1,
    },
    {"id": "bbg-defensive-put-roll-min-dte", "label": "Defensive Roll Min DTE", "type": "number", "default": "defensive_put_roll_min_dte", "step": 1, "min": 1},
    {"id": "bbg-defensive-put-roll-max-dte", "label": "Defensive Roll Max DTE", "type": "number", "default": "defensive_put_roll_max_dte", "step": 1, "min": 1},
    {"id": "bbg-defensive-put-roll-dte-min", "label": "New Put DTE Min", "type": "number", "default": "defensive_put_roll_dte_min", "step": 1, "min": 1},
    {"id": "bbg-defensive-put-roll-dte-max", "label": "New Put DTE Max", "type": "number", "default": "defensive_put_roll_dte_max", "step": 1, "min": 1},
    {"id": "bbg-defensive-put-roll-delta", "label": "Defensive Roll Delta", "type": "number", "default": "defensive_put_roll_delta", "step": 0.01, "min": 0, "max": 1},
    {"id": "bbg-assignment-cooldown-days", "label": "Assignment Cooldown Days", "type": "number", "default": "assignment_cooldown_days", "step": 1, "min": 0},
    {"id": "bbg-large-loss-cooldown-days", "label": "Large Loss Cooldown Days", "type": "number", "default": "large_loss_cooldown_days", "step": 1, "min": 0},
    {"id": "bbg-large-loss-cooldown-pct", "label": "Large Loss Cooldown %", "type": "number", "default": "large_loss_cooldown_pct", "step": 10, "min": 0},
]

SYMBOL_RISK_FIELDS = [
    {"id": "bbg-volatility-cap-floor", "label": "Volatility Cap Floor", "type": "number", "default": "volatility_cap_floor", "step": 0.01, "min": 0, "max": 1},
    {"id": "bbg-volatility-cap-ceiling", "label": "Volatility Cap Ceiling", "type": "number", "default": "volatility_cap_ceiling", "step": 0.05, "min": 0.5, "max": 3},
    {"id": "bbg-volatility-lookback", "label": "Volatility Lookback", "type": "number", "default": "volatility_lookback", "step": 1, "min": 1},
    {"id": "bbg-dynamic-symbol-risk-enabled", "label": "Enable Dynamic Symbol Risk", "type": "switch", "default": "dynamic_symbol_risk_enabled"},
    {"id": "bbg-symbol-state-cap-floor", "label": "Symbol State Cap Floor", "type": "number", "default": "symbol_state_cap_floor", "step": 0.01, "min": 0, "max": 1},
    {"id": "bbg-symbol-state-cap-ceiling", "label": "Symbol State Cap Ceiling", "type": "number", "default": "symbol_state_cap_ceiling", "step": 0.01, "min": 0, "max": 1},
    {"id": "bbg-symbol-drawdown-lookback", "label": "Symbol Drawdown Lookback", "type": "number", "default": "symbol_drawdown_lookback", "step": 1, "min": 1},
    {"id": "bbg-symbol-drawdown-sensitivity", "label": "Drawdown Sensitivity", "type": "number", "default": "symbol_drawdown_sensitivity", "step": 0.05, "min": 0},
    {"id": "bbg-symbol-downtrend-sensitivity", "label": "Downtrend Sensitivity", "type": "number", "default": "symbol_downtrend_sensitivity", "step": 0.05, "min": 0},
    {"id": "bbg-symbol-volatility-sensitivity", "label": "Volatility Sensitivity", "type": "number", "default": "symbol_volatility_sensitivity", "step": 0.05, "min": 0},
    {"id": "bbg-symbol-exposure-sensitivity", "label": "Exposure Sensitivity", "type": "number", "default": "symbol_exposure_sensitivity", "step": 0.05, "min": 0},
    {"id": "bbg-symbol-assignment-base-cap", "label": "Symbol Assignment Base Cap", "type": "number", "default": "symbol_assignment_base_cap", "step": 0.05, "min": 0},
    {"id": "bbg-stock-inventory-cap-enabled", "label": "Enable Stock Inventory Cap", "type": "switch", "default": "stock_inventory_cap_enabled"},
    {"id": "bbg-stock-inventory-base-cap", "label": "Stock Inventory Base Cap", "type": "number", "default": "stock_inventory_base_cap", "step": 0.01, "min": 0},
    {"id": "bbg-stock-inventory-cap-floor", "label": "Stock Inventory Cap Floor", "type": "number", "default": "stock_inventory_cap_floor", "step": 0.01, "min": 0},
    {"id": "bbg-stock-inventory-block-threshold", "label": "Stock Inventory Block Threshold", "type": "number", "default": "stock_inventory_block_threshold", "step": 0.01, "min": 0},
]

ALL_FORM_FIELDS = (
    RUN_SETUP_FIELDS
    + CORE_WHEEL_FIELDS
    + ML_FIELDS
    + CC_REPAIR_FIELDS
    + DEFENSIVE_FIELDS
    + SYMBOL_RISK_FIELDS
)


def _control_from_field(field, defaults=None):
    defaults = defaults or QC_UI_DEFAULTS
    default_value = defaults[field["default"]]
    width = field.get("width", 6)
    help_text = field.get("help_text")
    if field["type"] == "switch":
        control = html.Div(
            [
                dbc.Switch(
                    id=field["id"],
                    value=default_value,
                    label=field["label"],
                    className="mb-1",
                ),
                html.Small(help_text, className="text-muted") if help_text else None,
            ],
            className="mb-3",
        )
    elif field["type"] == "select":
        control = html.Div(
            [
                dbc.Label(field["label"], className="fw-semibold"),
                dbc.Select(id=field["id"], options=field["options"], value=default_value),
                html.Small(help_text, className="text-muted") if help_text else None,
            ],
            className="mb-3",
        )
    else:
        input_type = "text" if field["type"] == "text" else field["type"]
        control = html.Div(
            [
                dbc.Label(field["label"], className="fw-semibold"),
                dbc.Input(
                    id=field["id"],
                    type=input_type,
                    value=default_value,
                    step=field.get("step"),
                    min=field.get("min"),
                    max=field.get("max"),
                ),
                html.Small(help_text, className="text-muted") if help_text else None,
            ],
            className="mb-3",
        )
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
    return dbc.Card(
        [
            dbc.CardHeader(title, className=f"bg-{color} text-white fw-semibold"),
            dbc.CardBody(children),
        ],
        className="mb-3",
    )


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
    if not params:
        params = {}

    merged = dict(QC_UI_DEFAULTS)
    merged.update(params)
    stock_pool = params.get("stock_pool") if isinstance(params, dict) else None
    merged["stock_pool_text"] = ",".join(stock_pool) if stock_pool else QC_UI_DEFAULTS["stock_pool_text"]
    return tuple(merged.get(field["default"], QC_UI_DEFAULTS[field["default"]]) for field in ALL_FORM_FIELDS)


def build_binbin_backtest_params(form_data):
    """Normalize QC-aligned UI inputs into engine parameters."""
    merged = {**QC_UI_DEFAULTS, **(form_data or {})}
    stock_pool = _normalize_stock_pool(merged.get("stock_pool_text"))
    ml_enabled = _to_bool(merged.get("ml_enabled"), QC_UI_DEFAULTS["ml_enabled"])

    params = {
        "strategy": "binbin_god",
        "start_date": merged.get("start_date", QC_UI_DEFAULTS["start_date"]),
        "end_date": merged.get("end_date", QC_UI_DEFAULTS["end_date"]),
        "initial_capital": _to_float(merged.get("initial_capital"), QC_UI_DEFAULTS["initial_capital"]),
        "symbol": _derive_symbol(stock_pool),
        "stock_pool": stock_pool,
        "max_positions_ceiling": _to_int(merged.get("max_positions_ceiling"), QC_UI_DEFAULTS["max_positions_ceiling"]),
        "max_positions": _to_int(merged.get("max_positions_ceiling"), QC_UI_DEFAULTS["max_positions_ceiling"]),
        "max_leverage": _to_float(merged.get("max_leverage"), QC_UI_DEFAULTS["max_leverage"]),
        "target_margin_utilization": _to_float(merged.get("target_margin_utilization"), QC_UI_DEFAULTS["target_margin_utilization"]),
        "position_aggressiveness": _to_float(merged.get("position_aggressiveness"), QC_UI_DEFAULTS["position_aggressiveness"]),
        "max_risk_per_trade": _to_float(merged.get("max_risk_per_trade"), QC_UI_DEFAULTS["max_risk_per_trade"]),
        "profit_target_pct": _to_float(merged.get("profit_target_pct"), QC_UI_DEFAULTS["profit_target_pct"]),
        "stop_loss_pct": _to_float(merged.get("stop_loss_pct"), QC_UI_DEFAULTS["stop_loss_pct"]),
        "margin_buffer_pct": _to_float(merged.get("margin_buffer_pct"), QC_UI_DEFAULTS["margin_buffer_pct"]),
        "margin_rate_per_contract": _to_float(merged.get("margin_rate_per_contract"), QC_UI_DEFAULTS["margin_rate_per_contract"]),
        "dte_min": _to_int(merged.get("dte_min"), QC_UI_DEFAULTS["dte_min"]),
        "dte_max": _to_int(merged.get("dte_max"), QC_UI_DEFAULTS["dte_max"]),
        "delta_target": _to_float(merged.get("put_delta"), QC_UI_DEFAULTS["put_delta"]),
        "put_delta": _to_float(merged.get("put_delta"), QC_UI_DEFAULTS["put_delta"]),
        "call_delta": _to_float(merged.get("call_delta"), QC_UI_DEFAULTS["call_delta"]),
        "ml_enabled": ml_enabled,
        "ml_adoption_rate": _to_float(merged.get("ml_adoption_rate"), QC_UI_DEFAULTS["ml_adoption_rate"]),
        "ml_min_confidence": _to_float(merged.get("ml_min_confidence"), QC_UI_DEFAULTS["ml_min_confidence"]),
        "ml_confidence_gate": _to_float(merged.get("ml_min_confidence"), QC_UI_DEFAULTS["ml_min_confidence"]),
        "ml_exploration_rate": _to_float(merged.get("ml_exploration_rate"), QC_UI_DEFAULTS["ml_exploration_rate"]),
        "ml_learning_rate": _to_float(merged.get("ml_learning_rate"), QC_UI_DEFAULTS["ml_learning_rate"]),
        "ml_delta_optimization": ml_enabled,
        "ml_dte_optimization": ml_enabled,
        "ml_roll_optimization": ml_enabled,
        "ml_position_optimization": ml_enabled,
        "cc_optimization_enabled": _to_bool(merged.get("cc_optimization_enabled"), QC_UI_DEFAULTS["cc_optimization_enabled"]),
        "cc_min_delta_cost": _to_float(merged.get("cc_min_delta_cost"), QC_UI_DEFAULTS["cc_min_delta_cost"]),
        "cc_cost_basis_threshold": _to_float(merged.get("cc_cost_basis_threshold"), QC_UI_DEFAULTS["cc_cost_basis_threshold"]),
        "cc_min_strike_premium": _to_float(merged.get("cc_min_strike_premium"), QC_UI_DEFAULTS["cc_min_strike_premium"]),
        "repair_call_threshold_pct": _to_float(merged.get("repair_call_threshold_pct"), QC_UI_DEFAULTS["repair_call_threshold_pct"]),
        "repair_call_delta": _to_float(merged.get("repair_call_delta"), QC_UI_DEFAULTS["repair_call_delta"]),
        "repair_call_dte_min": _to_int(merged.get("repair_call_dte_min"), QC_UI_DEFAULTS["repair_call_dte_min"]),
        "repair_call_dte_max": _to_int(merged.get("repair_call_dte_max"), QC_UI_DEFAULTS["repair_call_dte_max"]),
        "repair_call_max_discount_pct": _to_float(merged.get("repair_call_max_discount_pct"), QC_UI_DEFAULTS["repair_call_max_discount_pct"]),
        "defensive_put_roll_enabled": _to_bool(merged.get("defensive_put_roll_enabled"), QC_UI_DEFAULTS["defensive_put_roll_enabled"]),
        "defensive_put_roll_loss_pct": _to_float(merged.get("defensive_put_roll_loss_pct"), QC_UI_DEFAULTS["defensive_put_roll_loss_pct"]),
        "defensive_put_roll_itm_buffer_pct": _to_float(merged.get("defensive_put_roll_itm_buffer_pct"), QC_UI_DEFAULTS["defensive_put_roll_itm_buffer_pct"]),
        "defensive_put_roll_min_dte": _to_int(merged.get("defensive_put_roll_min_dte"), QC_UI_DEFAULTS["defensive_put_roll_min_dte"]),
        "defensive_put_roll_max_dte": _to_int(merged.get("defensive_put_roll_max_dte"), QC_UI_DEFAULTS["defensive_put_roll_max_dte"]),
        "defensive_put_roll_dte_min": _to_int(merged.get("defensive_put_roll_dte_min"), QC_UI_DEFAULTS["defensive_put_roll_dte_min"]),
        "defensive_put_roll_dte_max": _to_int(merged.get("defensive_put_roll_dte_max"), QC_UI_DEFAULTS["defensive_put_roll_dte_max"]),
        "defensive_put_roll_delta": _to_float(merged.get("defensive_put_roll_delta"), QC_UI_DEFAULTS["defensive_put_roll_delta"]),
        "assignment_cooldown_days": _to_int(merged.get("assignment_cooldown_days"), QC_UI_DEFAULTS["assignment_cooldown_days"]),
        "large_loss_cooldown_days": _to_int(merged.get("large_loss_cooldown_days"), QC_UI_DEFAULTS["large_loss_cooldown_days"]),
        "large_loss_cooldown_pct": _to_float(merged.get("large_loss_cooldown_pct"), QC_UI_DEFAULTS["large_loss_cooldown_pct"]),
        "volatility_cap_floor": _to_float(merged.get("volatility_cap_floor"), QC_UI_DEFAULTS["volatility_cap_floor"]),
        "volatility_cap_ceiling": _to_float(merged.get("volatility_cap_ceiling"), QC_UI_DEFAULTS["volatility_cap_ceiling"]),
        "volatility_lookback": _to_int(merged.get("volatility_lookback"), QC_UI_DEFAULTS["volatility_lookback"]),
        "dynamic_symbol_risk_enabled": _to_bool(merged.get("dynamic_symbol_risk_enabled"), QC_UI_DEFAULTS["dynamic_symbol_risk_enabled"]),
        "symbol_state_cap_floor": _to_float(merged.get("symbol_state_cap_floor"), QC_UI_DEFAULTS["symbol_state_cap_floor"]),
        "symbol_state_cap_ceiling": _to_float(merged.get("symbol_state_cap_ceiling"), QC_UI_DEFAULTS["symbol_state_cap_ceiling"]),
        "symbol_drawdown_lookback": _to_int(merged.get("symbol_drawdown_lookback"), QC_UI_DEFAULTS["symbol_drawdown_lookback"]),
        "symbol_drawdown_sensitivity": _to_float(merged.get("symbol_drawdown_sensitivity"), QC_UI_DEFAULTS["symbol_drawdown_sensitivity"]),
        "symbol_downtrend_sensitivity": _to_float(merged.get("symbol_downtrend_sensitivity"), QC_UI_DEFAULTS["symbol_downtrend_sensitivity"]),
        "symbol_volatility_sensitivity": _to_float(merged.get("symbol_volatility_sensitivity"), QC_UI_DEFAULTS["symbol_volatility_sensitivity"]),
        "symbol_exposure_sensitivity": _to_float(merged.get("symbol_exposure_sensitivity"), QC_UI_DEFAULTS["symbol_exposure_sensitivity"]),
        "symbol_assignment_base_cap": _to_float(merged.get("symbol_assignment_base_cap"), QC_UI_DEFAULTS["symbol_assignment_base_cap"]),
        "stock_inventory_cap_enabled": _to_bool(merged.get("stock_inventory_cap_enabled"), QC_UI_DEFAULTS["stock_inventory_cap_enabled"]),
        "stock_inventory_base_cap": _to_float(merged.get("stock_inventory_base_cap"), QC_UI_DEFAULTS["stock_inventory_base_cap"]),
        "stock_inventory_cap_floor": _to_float(merged.get("stock_inventory_cap_floor"), QC_UI_DEFAULTS["stock_inventory_cap_floor"]),
        "stock_inventory_block_threshold": _to_float(merged.get("stock_inventory_block_threshold"), QC_UI_DEFAULTS["stock_inventory_block_threshold"]),
        "parity_mode": "qc",
        "contract_universe_mode": "qc_emulated_lattice",
    }
    return params
