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
from app.pages.binbin_god_shared import (
    ALL_FORM_FIELDS,
    CC_REPAIR_FIELDS,
    CORE_WHEEL_FIELDS,
    DEFAULT_STOCK_POOL,
    DEFENSIVE_FIELDS,
    ML_FIELDS,
    QC_UI_DEFAULTS,
    RUN_SETUP_FIELDS,
    SYMBOL_RISK_FIELDS,
    _build_rows,
    _form_values_from_params,
    _section_card,
    build_binbin_backtest_params,
)
from app.services import get_services
from core.backtesting.last_result_store import (
    load_last_binbin_god_result,
    save_last_binbin_god_result,
)

_services = None

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


def get_services_cached():
    """Get services with caching to avoid repeated initialization."""
    global _services
    if _services is None:
        _services = get_services()
    return _services


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
    State("bbg-max-risk-per-trade", "value"),
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
    max_risk_per_trade,
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
            "max_risk_per_trade": max_risk_per_trade,
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
