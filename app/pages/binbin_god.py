"""Binbin God strategy page with QC-aligned configuration controls."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

from app.components.charts import (
    create_monthly_heatmap,
    create_pnl_chart,
    create_trade_timeline_chart,
)
from app.components.monitoring import (
    create_holdings_card,
    create_monitoring_dashboard,
    create_phase_transition_log,
    create_trade_history_table,
)
from app.components.tables import create_data_table, metric_card
from app.services import get_services
from core.backtesting.last_result_store import (
    load_last_binbin_god_result,
    save_last_binbin_god_result,
)
from core.backtesting.qc_parity import QC_BINBIN_DEFAULTS, QC_PARAMETER_DEFAULTS

_services = None


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
    "defensive_put_roll_loss_pct": 200,
    "defensive_put_roll_itm_buffer_pct": 0.10,
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
    "stock_inventory_base_cap": 0.20,
    "stock_inventory_cap_floor": 0.50,
    "stock_inventory_block_threshold": 0.90,
}

dash.register_page(__name__, path="/binbin-god", name="Binbin God", icon="bi bi-robot")

STRATEGY_INFO = {
    "name": "Binbin God Strategy",
    "version": "0.2.0",
    "description": "QC-aligned Binbin God backtester with parity-aware configuration and risk controls.",
    "universe": DEFAULT_STOCK_POOL,
    "selection_criteria": {
        "ML + Wheel": "QC-style wheel parameters with ML-assisted signal selection",
        "Repair Calls": "Adaptive covered calls when assigned stock is underwater",
        "Defensive Rolls": "Protective short put roll rules before expiry",
        "Symbol Risk": "Cooldowns, dynamic risk caps, and stock inventory limits",
    },
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


def get_services_cached():
    """Get services with caching to avoid repeated initialization."""
    global _services
    if _services is None:
        _services = get_services()
    return _services


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


def create_strategy_info_card():
    """Create the static strategy overview card."""
    return dbc.Card(
        [
            dbc.CardHeader(
                [html.I(className="bi bi-info-circle me-2"), "Strategy Information"],
                className="bg-primary text-white",
            ),
            dbc.CardBody(
                [
                    html.H5(f"{STRATEGY_INFO['name']} v{STRATEGY_INFO['version']}", className="card-title"),
                    html.P(STRATEGY_INFO["description"], className="card-text"),
                    html.Hr(),
                    html.H6("Universe", className="fw-bold"),
                    html.Div([dbc.Badge(symbol, color="info", className="me-1 mb-1") for symbol in STRATEGY_INFO["universe"]]),
                    html.H6("Focus Areas", className="fw-bold mt-3"),
                    html.Ul(
                        [
                            html.Li(dbc.Badge(f"{name}: {detail}", color="secondary", className="me-1"))
                            for name, detail in STRATEGY_INFO["selection_criteria"].items()
                        ],
                        className="list-unstyled",
                    ),
                ]
            ),
        ],
        className="mb-4",
    )


def create_mag7_analysis_placeholder(initial_children=None):
    """Placeholder for MAG7 analysis display."""
    return dbc.Card(
        [
            dbc.CardHeader(
                [html.I(className="bi bi-bar-chart me-2"), "MAG7 Stock Analysis"],
                className="bg-info text-white",
            ),
            dbc.CardBody(
                [
                    html.Div(
                        id="binbin-mag7-analysis",
                        children=initial_children or [html.P("Run backtest to see MAG7 stock rankings and selection", className="text-muted")],
                    ),
                ]
            ),
        ],
        className="mb-4",
    )


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


def _build_binbin_results_view(result, params):
    metrics = result.get("metrics", {})
    trades = result.get("trades", [])
    daily_pnl = result.get("daily_pnl", [])

    total_ret = metrics.get("total_return_pct", 0)
    ret_color = "success" if total_ret >= 0 else "danger"
    metrics_row = dbc.Row(
        [
            dbc.Col(metric_card("Total Return", f"{total_ret:+.2f}%", ret_color), md=2),
            dbc.Col(metric_card("Annual Return", f"{metrics.get('annualized_return_pct', 0):+.2f}%", ret_color), md=2),
            dbc.Col(metric_card("Win Rate", f"{metrics.get('win_rate', 0):.1f}%", "info"), md=2),
            dbc.Col(metric_card("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}", "primary"), md=2),
            dbc.Col(metric_card("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%", "danger"), md=2),
            dbc.Col(metric_card("Total Trades", f"{metrics.get('total_trades', 0)}", "secondary"), md=2),
        ],
        className="mb-4 g-3",
    )

    pnl_dates = [item["date"] for item in daily_pnl] if daily_pnl else []
    pnl_values = [item["cumulative_pnl"] for item in daily_pnl] if daily_pnl else []
    benchmark_data = result.get("benchmark_data", {})
    pnl_chart = dcc.Graph(
        figure=create_pnl_chart(
            pnl_dates,
            pnl_values,
            benchmark_data=benchmark_data,
            initial_capital=params.get("initial_capital", QC_UI_DEFAULTS["initial_capital"]),
        )
    )

    monthly = metrics.get("monthly_returns", {})
    monthly_tupled = {(int(key.split("-")[0]), int(key.split("-")[1])): value for key, value in monthly.items()} if monthly else {}
    heatmap = dcc.Graph(figure=create_monthly_heatmap(monthly_tupled)) if monthly_tupled else html.Div()

    trade_timeline = dcc.Graph(
        figure=create_trade_timeline_chart(
            trades=trades,
            daily_pnl=daily_pnl,
            underlying_prices=result.get("underlying_prices", []),
            multi_stock_prices=result.get("multi_stock_prices", {}),
            title="Binbin God Trade Timeline: Entry/Exit Points & Performance",
        )
    )

    trade_columns = [
        {"headerName": "Entry", "field": "entry_date", "width": 100, "sort": "desc"},
        {"headerName": "Exit", "field": "exit_date", "width": 100},
        {"headerName": "Option Contract", "field": "contract_name", "width": 220},
        {"headerName": "Type", "field": "trade_type", "width": 120},
        {"headerName": "Strike", "field": "strike", "width": 80, "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Expiry", "field": "expiry", "width": 100},
        {"headerName": "Right", "field": "right", "width": 80},
        {"headerName": "Qty", "field": "quantity", "width": 60},
        {"headerName": "Stock Entry $", "field": "underlying_entry", "width": 100, "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Stock Exit $", "field": "underlying_exit", "width": 100, "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Option Entry $", "field": "entry_price", "width": 90, "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Option Exit $", "field": "exit_price", "width": 90, "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {
            "headerName": "P&L",
            "field": "pnl",
            "width": 100,
            "valueFormatter": {"function": "d3.format(',.2f')(params.value)", "cellStyle": {"function": "params.value >= 0 ? {'color': '#26a69a'} : {'color': '#ef5350'}"}},
        },
        {"headerName": "Reason", "field": "exit_reason", "width": 120},
    ]
    trades_table = create_data_table(trades, trade_columns, "bbg-trades-table", height=450) if trades else html.P("No trades", className="text-muted")

    extra_row = dbc.Row(
        [
            dbc.Col(metric_card("Avg Profit", f"${metrics.get('avg_profit', 0):,.2f}", "success"), md=3),
            dbc.Col(metric_card("Avg Loss", f"${metrics.get('avg_loss', 0):,.2f}", "danger"), md=3),
            dbc.Col(metric_card("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}", "primary"), md=3),
            dbc.Col(metric_card("Sortino", f"{metrics.get('sortino_ratio', 0):.2f}", "info"), md=3),
        ],
        className="mb-3 g-3",
    )

    holdings_card = html.Div()
    strategy_performance = result.get("strategy_performance", {})
    if strategy_performance:
        current_state = strategy_performance.get("current_state", {})
        holdings_card = create_holdings_card(
            {
                "shares_held": strategy_performance.get("shares_held") or current_state.get("shares_held", 0),
                "cost_basis": strategy_performance.get("cost_basis") or current_state.get("cost_basis", 0),
                "options_held": strategy_performance.get("open_positions", []),
            }
        )

    monitoring_section = html.Div()
    if strategy_performance:
        monitoring_section = create_monitoring_dashboard({**strategy_performance, "metrics": metrics})

    mag7_analysis = result.get("mag7_analysis", {})
    if mag7_analysis and "ranked_stocks" in mag7_analysis:
        ranked = mag7_analysis["ranked_stocks"]
        best = mag7_analysis.get("best_pick", {})
        mag7_section = html.Div(
            [
                html.H5("MAG7 Stock Analysis", className="mt-4 mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H6("Best Pick", className="fw-bold text-success"),
                                html.H4(best.get("symbol", "N/A"), className="text-success"),
                                html.Small(f"Score: {best.get('total_score', 0):.1f}", className="text-muted"),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.H6("Total Stocks Analyzed", className="fw-bold"),
                                html.H4(len(ranked), className="text-primary"),
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Table.from_dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Rank": idx + 1,
                                "Symbol": stock["symbol"],
                                "Score": stock["total_score"],
                                "PE": stock.get("pe_ratio", "N/A"),
                                "IV Rank": stock.get("iv_rank", "N/A"),
                                "Momentum": f"{stock.get('momentum', 0):.1f}",
                                "Stability": f"{stock.get('stability', 0):.1f}",
                            }
                            for idx, stock in enumerate(ranked)
                        ]
                    ),
                    striped=True,
                    hover=True,
                    bordered=True,
                    className="table-sm",
                ),
            ]
        )
    else:
        mag7_section = html.Div()

    content = html.Div(
        [
            html.H5("Performance Summary", className="mb-3"),
            metrics_row,
            pnl_chart,
            html.H5("Additional Metrics", className="mt-4 mb-3"),
            extra_row,
            heatmap,
            holdings_card,
            monitoring_section,
            html.H5("Trade Timeline Analysis", className="mt-4 mb-3"),
            trade_timeline,
            html.H5("Trade Log", className="mt-4 mb-3"),
            trades_table,
            mag7_section,
        ]
    )

    if strategy_performance:
        trade_history = strategy_performance.get("trade_history", [])
        phase_history = strategy_performance.get("phase_history", [])
        if trade_history:
            content.children.append(
                html.Div([html.H5("Recent Trade History", className="mt-4 mb-3"), create_trade_history_table(trade_history)])
            )
        if phase_history:
            content.children.append(
                html.Div([html.H5("Phase Transition Log", className="mt-4 mb-3"), create_phase_transition_log(phase_history)])
            )

    return mag7_section, content, {"display": "none"}, "d-block mt-2"


def _get_initial_layout_state():
    """Build initial server-rendered state from the persisted last result."""
    payload = load_last_binbin_god_result() or {}
    params = payload.get("params") if isinstance(payload, dict) else None
    result = payload.get("result") if isinstance(payload, dict) else None

    defaults = dict(QC_UI_DEFAULTS)
    if params:
        field_values = _form_values_from_params(params)
        defaults = {field["default"]: value for field, value in zip(ALL_FORM_FIELDS, field_values)}

    if params and result:
        mag7_section, content, _, export_class = _build_binbin_results_view(result, params)
        return {
            "defaults": defaults,
            "result": result,
            "params": params,
            "mag7_children": mag7_section,
            "results_children": content,
            "export_class": export_class,
        }

    placeholder = [
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.Div(
                            [html.I(className="bi bi-graph-up me-2"), "Run a backtest to see results"],
                            className="text-center text-muted py-5",
                        )
                    ]
                )
            ]
        )
    ]
    return {
        "defaults": defaults,
        "result": {},
        "params": {},
        "mag7_children": None,
        "results_children": placeholder,
        "export_class": "d-none",
    }


def _build_configuration_panel(defaults):
    return html.Div(
        [
            create_strategy_info_card(),
            _section_card("Run Setup", _build_rows(RUN_SETUP_FIELDS, defaults), color="success"),
            _section_card("Core Wheel", _build_rows(CORE_WHEEL_FIELDS, defaults), color="secondary"),
            _section_card("ML", _build_rows(ML_FIELDS, defaults), color="primary"),
            _section_card("Covered Call / Repair", _build_rows(CC_REPAIR_FIELDS, defaults), color="warning"),
            _section_card("Defensive Put / Cooldown", _build_rows(DEFENSIVE_FIELDS, defaults), color="danger"),
            _section_card("Symbol Risk / Inventory", _build_rows(SYMBOL_RISK_FIELDS, defaults), color="dark"),
            dbc.Button(
                [html.I(className="bi bi-play-fill me-2"), "Run Backtest"],
                id="bbg-run-btn",
                color="primary",
                className="w-100",
                size="lg",
            ),
            html.Div(
                id="bbg-export-container",
                className=defaults.get("export_class", "d-none"),
                children=[
                    dbc.Button("📤 Export for AI Analysis", id="bbg-export-btn", color="info", className="w-100 mt-2", n_clicks=0),
                ],
            ),
            dcc.Download(id="bbg-download"),
        ]
    )


def layout():
    initial_state = _get_initial_layout_state()
    defaults = dict(initial_state["defaults"])
    defaults["export_class"] = initial_state["export_class"]
    return dbc.Container(
        [
        html.Div(
            [
                html.H1([html.I(className="bi bi-robot me-2"), "Binbin God Strategy Backtester"], className="mb-2"),
                html.P("QC-aligned Binbin God configuration with parity-aware backtesting controls", className="lead text-muted"),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(_build_configuration_panel(defaults), md=4),
                dbc.Col(
                    [
                        create_mag7_analysis_placeholder(initial_state["mag7_children"]),
                        html.Div(
                            id="bbg-loading-indicator",
                            style={"display": "none"},
                            className="mb-3",
                            children=[
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [
                                                        html.I(className="bi bi-hourglass-split me-2", style={"fontSize": "2rem"}),
                                                        html.H4("Running Binbin God Backtest...", className="mt-2"),
                                                        html.P("Generating results with the selected QC-style configuration", className="text-muted"),
                                                        dbc.Progress(value=100, striped=True, animated=True, className="mt-3"),
                                                    ],
                                                    className="text-center py-4",
                                                )
                                            ]
                                        )
                                    ],
                                    className="bg-dark border-success",
                                )
                            ],
                        ),
                        dcc.Loading(
                            id="binbin-loading",
                            type="circle",
                            children=html.Div(
                                id="binbin-results-container",
                                children=initial_state["results_children"],
                            ),
                            overlay_style={"visibility": "visible", "opacity": 0.9, "backgroundColor": "#1a1a2e"},
                        ),
                        dcc.Store(id="binbin-results-store", data=initial_state["result"]),
                        dcc.Store(id="binbin-params-store", data=initial_state["params"]),
                    ],
                    md=8,
                ),
            ]
        ),
        dcc.Markdown(
            """
<style>
.dash-loading {
    background-color: #1a1a2e !important;
}
.dash-spinner circle {
    stroke: #4CAF50 !important;
    stroke-width: 4;
}
</style>
            """,
            dangerously_allow_html=True,
        ),
        ],
        fluid=True,
    )


@callback(
    Output("binbin-results-store", "data"),
    Output("binbin-params-store", "data"),
    Output("binbin-mag7-analysis", "children"),
    Output("binbin-results-container", "children"),
    Output("bbg-loading-indicator", "style"),
    Output("bbg-export-container", "className"),
    Input("bbg-run-btn", "n_clicks"),
    State("bbg-start", "value"),
    State("bbg-end", "value"),
    State("bbg-initial-capital", "value"),
    State("bbg-stock-pool-text", "value"),
    State("bbg-max-positions-ceiling", "value"),
    State("bbg-max-leverage", "value"),
    State("bbg-target-margin-utilization", "value"),
    State("bbg-position-aggressiveness", "value"),
    State("bbg-profit-target-pct", "value"),
    State("bbg-stop-loss-pct", "value"),
    State("bbg-margin-buffer-pct", "value"),
    State("bbg-margin-rate-per-contract", "value"),
    State("bbg-dte-min", "value"),
    State("bbg-dte-max", "value"),
    State("bbg-put-delta", "value"),
    State("bbg-call-delta", "value"),
    State("bbg-ml-enabled", "value"),
    State("bbg-ml-adoption-rate", "value"),
    State("bbg-ml-min-confidence", "value"),
    State("bbg-ml-exploration-rate", "value"),
    State("bbg-ml-learning-rate", "value"),
    State("bbg-cc-optimization-enabled", "value"),
    State("bbg-cc-min-delta-cost", "value"),
    State("bbg-cc-cost-basis-threshold", "value"),
    State("bbg-cc-min-strike-premium", "value"),
    State("bbg-repair-call-threshold-pct", "value"),
    State("bbg-repair-call-delta", "value"),
    State("bbg-repair-call-dte-min", "value"),
    State("bbg-repair-call-dte-max", "value"),
    State("bbg-repair-call-max-discount-pct", "value"),
    State("bbg-defensive-put-roll-enabled", "value"),
    State("bbg-defensive-put-roll-loss-pct", "value"),
    State("bbg-defensive-put-roll-itm-buffer-pct", "value"),
    State("bbg-defensive-put-roll-min-dte", "value"),
    State("bbg-defensive-put-roll-max-dte", "value"),
    State("bbg-defensive-put-roll-dte-min", "value"),
    State("bbg-defensive-put-roll-dte-max", "value"),
    State("bbg-defensive-put-roll-delta", "value"),
    State("bbg-assignment-cooldown-days", "value"),
    State("bbg-large-loss-cooldown-days", "value"),
    State("bbg-large-loss-cooldown-pct", "value"),
    State("bbg-volatility-cap-floor", "value"),
    State("bbg-volatility-cap-ceiling", "value"),
    State("bbg-volatility-lookback", "value"),
    State("bbg-dynamic-symbol-risk-enabled", "value"),
    State("bbg-symbol-state-cap-floor", "value"),
    State("bbg-symbol-state-cap-ceiling", "value"),
    State("bbg-symbol-drawdown-lookback", "value"),
    State("bbg-symbol-drawdown-sensitivity", "value"),
    State("bbg-symbol-downtrend-sensitivity", "value"),
    State("bbg-symbol-volatility-sensitivity", "value"),
    State("bbg-symbol-exposure-sensitivity", "value"),
    State("bbg-symbol-assignment-base-cap", "value"),
    State("bbg-stock-inventory-cap-enabled", "value"),
    State("bbg-stock-inventory-base-cap", "value"),
    State("bbg-stock-inventory-cap-floor", "value"),
    State("bbg-stock-inventory-block-threshold", "value"),
    prevent_initial_call=True,
)
def run_binbin_backtest(
    n_clicks,
    start_date,
    end_date,
    initial_capital,
    stock_pool_text,
    max_positions_ceiling,
    max_leverage,
    target_margin_utilization,
    position_aggressiveness,
    profit_target_pct,
    stop_loss_pct,
    margin_buffer_pct,
    margin_rate_per_contract,
    dte_min,
    dte_max,
    put_delta,
    call_delta,
    ml_enabled,
    ml_adoption_rate,
    ml_min_confidence,
    ml_exploration_rate,
    ml_learning_rate,
    cc_optimization_enabled,
    cc_min_delta_cost,
    cc_cost_basis_threshold,
    cc_min_strike_premium,
    repair_call_threshold_pct,
    repair_call_delta,
    repair_call_dte_min,
    repair_call_dte_max,
    repair_call_max_discount_pct,
    defensive_put_roll_enabled,
    defensive_put_roll_loss_pct,
    defensive_put_roll_itm_buffer_pct,
    defensive_put_roll_min_dte,
    defensive_put_roll_max_dte,
    defensive_put_roll_dte_min,
    defensive_put_roll_dte_max,
    defensive_put_roll_delta,
    assignment_cooldown_days,
    large_loss_cooldown_days,
    large_loss_cooldown_pct,
    volatility_cap_floor,
    volatility_cap_ceiling,
    volatility_lookback,
    dynamic_symbol_risk_enabled,
    symbol_state_cap_floor,
    symbol_state_cap_ceiling,
    symbol_drawdown_lookback,
    symbol_drawdown_sensitivity,
    symbol_downtrend_sensitivity,
    symbol_volatility_sensitivity,
    symbol_exposure_sensitivity,
    symbol_assignment_base_cap,
    stock_inventory_cap_enabled,
    stock_inventory_base_cap,
    stock_inventory_cap_floor,
    stock_inventory_block_threshold,
):
    """Run the Binbin God backtest using QC-aligned parameters."""
    if not n_clicks or not start_date or not end_date:
        return no_update, no_update, no_update, no_update, no_update, no_update

    params = build_binbin_backtest_params(
        {
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "stock_pool_text": stock_pool_text,
            "max_positions_ceiling": max_positions_ceiling,
            "max_leverage": max_leverage,
            "target_margin_utilization": target_margin_utilization,
            "position_aggressiveness": position_aggressiveness,
            "profit_target_pct": profit_target_pct,
            "stop_loss_pct": stop_loss_pct,
            "margin_buffer_pct": margin_buffer_pct,
            "margin_rate_per_contract": margin_rate_per_contract,
            "dte_min": dte_min,
            "dte_max": dte_max,
            "put_delta": put_delta,
            "call_delta": call_delta,
            "ml_enabled": ml_enabled,
            "ml_adoption_rate": ml_adoption_rate,
            "ml_min_confidence": ml_min_confidence,
            "ml_exploration_rate": ml_exploration_rate,
            "ml_learning_rate": ml_learning_rate,
            "cc_optimization_enabled": cc_optimization_enabled,
            "cc_min_delta_cost": cc_min_delta_cost,
            "cc_cost_basis_threshold": cc_cost_basis_threshold,
            "cc_min_strike_premium": cc_min_strike_premium,
            "repair_call_threshold_pct": repair_call_threshold_pct,
            "repair_call_delta": repair_call_delta,
            "repair_call_dte_min": repair_call_dte_min,
            "repair_call_dte_max": repair_call_dte_max,
            "repair_call_max_discount_pct": repair_call_max_discount_pct,
            "defensive_put_roll_enabled": defensive_put_roll_enabled,
            "defensive_put_roll_loss_pct": defensive_put_roll_loss_pct,
            "defensive_put_roll_itm_buffer_pct": defensive_put_roll_itm_buffer_pct,
            "defensive_put_roll_min_dte": defensive_put_roll_min_dte,
            "defensive_put_roll_max_dte": defensive_put_roll_max_dte,
            "defensive_put_roll_dte_min": defensive_put_roll_dte_min,
            "defensive_put_roll_dte_max": defensive_put_roll_dte_max,
            "defensive_put_roll_delta": defensive_put_roll_delta,
            "assignment_cooldown_days": assignment_cooldown_days,
            "large_loss_cooldown_days": large_loss_cooldown_days,
            "large_loss_cooldown_pct": large_loss_cooldown_pct,
            "volatility_cap_floor": volatility_cap_floor,
            "volatility_cap_ceiling": volatility_cap_ceiling,
            "volatility_lookback": volatility_lookback,
            "dynamic_symbol_risk_enabled": dynamic_symbol_risk_enabled,
            "symbol_state_cap_floor": symbol_state_cap_floor,
            "symbol_state_cap_ceiling": symbol_state_cap_ceiling,
            "symbol_drawdown_lookback": symbol_drawdown_lookback,
            "symbol_drawdown_sensitivity": symbol_drawdown_sensitivity,
            "symbol_downtrend_sensitivity": symbol_downtrend_sensitivity,
            "symbol_volatility_sensitivity": symbol_volatility_sensitivity,
            "symbol_exposure_sensitivity": symbol_exposure_sensitivity,
            "symbol_assignment_base_cap": symbol_assignment_base_cap,
            "stock_inventory_cap_enabled": stock_inventory_cap_enabled,
            "stock_inventory_base_cap": stock_inventory_base_cap,
            "stock_inventory_cap_floor": stock_inventory_cap_floor,
            "stock_inventory_block_threshold": stock_inventory_block_threshold,
        }
    )

    services = get_services_cached()
    if not services:
        return {}, params, html.Div(), html.P("Services not initialized", className="text-warning"), {"display": "none"}, "d-none"

    engine = services["backtest_engine"]
    try:
        result = engine.run(params)
    except Exception as exc:
        return {}, params, html.Div(), html.P(f"Backtest error: {exc}", className="text-danger"), {"display": "none"}, "d-none"

    if not result:
        return {}, params, html.Div(), html.P("No results generated", className="text-muted"), {"display": "none"}, "d-none"
    save_last_binbin_god_result(params, result)
    mag7_section, content, loading_style, export_class = _build_binbin_results_view(result, params)
    return result, params, mag7_section, content, loading_style, export_class


@callback(
    Output("bbg-download", "data"),
    Input("bbg-export-btn", "n_clicks"),
    State("binbin-results-store", "data"),
    State("binbin-params-store", "data"),
    prevent_initial_call=True,
)
def export_binbin_backtest_result(n_clicks, result, params):
    """Export the current Binbin God backtest payload as JSON."""
    import json
    from datetime import datetime

    if not n_clicks or not result or not params:
        return no_update

    metrics = result.get("metrics", {})
    export_data = {
        "export_info": {
            "exported_at": datetime.utcnow().isoformat(),
            "export_version": "1.0",
            "purpose": "AI analysis and debugging",
        },
        "backtest_summary": {
            "strategy": params.get("strategy"),
            "symbol": params.get("symbol"),
            "stock_pool": params.get("stock_pool"),
            "period": {"start_date": params.get("start_date"), "end_date": params.get("end_date")},
            "capital": {"initial": params.get("initial_capital")},
        },
        "parameters": params,
        "performance_metrics": {
            "total_return_pct": metrics.get("total_return_pct"),
            "annualized_return_pct": metrics.get("annualized_return_pct"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "sortino_ratio": metrics.get("sortino_ratio"),
            "win_rate": metrics.get("win_rate"),
            "total_trades": metrics.get("total_trades"),
            "avg_profit": metrics.get("avg_profit"),
            "avg_loss": metrics.get("avg_loss"),
            "profit_factor": metrics.get("profit_factor"),
            "monthly_returns": metrics.get("monthly_returns"),
        },
        "trades": result.get("trades", []),
        "daily_pnl": result.get("daily_pnl", []),
        "strategy_performance": result.get("strategy_performance", {}),
    }

    filename = f"backtest_{params.get('strategy', 'unknown')}_{params.get('symbol', 'UNKNOWN')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return dict(content=json.dumps(export_data, indent=2, ensure_ascii=False, default=str), filename=filename, mime_type="application/json")
