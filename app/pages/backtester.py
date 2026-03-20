"""Backtester page: strategy configuration, run backtest, results visualization."""

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc
from app.components.tables import metric_card, create_data_table
from app.components.charts import create_pnl_chart, create_monthly_heatmap, create_trade_timeline_chart
from app.components.monitoring import (
    create_monitoring_dashboard, 
    create_trade_history_table, 
    create_phase_transition_log,
    create_holdings_card,  # New component for displaying holdings
)
from app.services import get_services

dash.register_page(__name__, path="/backtester", name="Backtester", order=4)

layout = html.Div([
    html.H3("Options Strategy Backtester", className="mb-3"),

    dbc.Row([
        # Left: Config panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Backtest Configuration"),
                dbc.CardBody([
                    dbc.Label("Strategy"),
                    dbc.Select(
                        id="bt-strategy",
                        options=[
                            {"label": "Sell Put (Cash Secured)", "value": "sell_put"},
                            {"label": "Covered Call", "value": "covered_call"},
                            {"label": "Iron Condor", "value": "iron_condor"},
                            {"label": "Bull Put Spread", "value": "bull_put_spread"},
                            {"label": "Bear Call Spread", "value": "bear_call_spread"},
                            {"label": "Straddle (Short)", "value": "straddle"},
                            {"label": "Strangle (Short)", "value": "strangle"},
                            {"label": "Wheel Strategy", "value": "wheel"},
                        ],
                        value="sell_put",
                        className="mb-3",
                    ),

                    dbc.Label("Symbol"),
                    dbc.Input(id="bt-symbol", value="NVDA", className="mb-3"),

                    # Use Synthetic Data Option
                    dcc.Checklist(
                        id="bt-use-synthetic",
                        options=[{"label": " Use Random Synthetic Data (for testing without IBKR connection)", "value": True}],
                        value=[],
                        inline=True,
                        className="mb-3",
                    ),

                    dbc.Label("Date Range"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="bt-start", type="date", value="2025-01-01"), width=6),
                        dbc.Col(dbc.Input(id="bt-end", type="date", value="2026-03-05"), width=6),
                    ], className="mb-3"),

                    dbc.Label("Initial Capital ($)"),
                    dbc.Input(id="bt-capital", type="number", value=150000, className="mb-3"),  # Sufficient capital for desired positions

                    dbc.Label("Max Leverage"),
                    dbc.Input(id="bt-leverage", type="number", value=1.0, step=0.1, min=1.0, className="mb-3"),

                    html.Hr(),
                    html.H6("Strategy Parameters", className="fw-bold mb-2"),

                    dbc.Label("DTE Range (days)"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="bt-dte-min", type="number", value=30, size="sm"), width=6),  # 30-45 DTE is optimal
                        dbc.Col(dbc.Input(id="bt-dte-max", type="number", value=45, size="sm"), width=6),
                    ], className="mb-2"),

                    # Delta configuration - shown for non-Wheel strategies only
                    html.Div(id="delta-config-container", children=[
                        dbc.Label("Target Delta (absolute)"),
                        dbc.Input(id="bt-delta", type="number", value=0.30, step=0.05, size="sm", className="mb-2"),
                    ]),

                    dbc.Label("Profit Target (% of premium)"),
                    dbc.Row([
                        dbc.Col(
                            dbc.Input(id="bt-profit-target", type="number", value=50, step=10, size="sm"),
                            width=8
                        ),
                       dbc.Col(
                            dcc.Checklist(
                                id="bt-disable-profit-target",
                               options=[{"label": "Disable", "value": True}],
                               value=[],
                                className="mt-2",
                                style={
                                    "color": "#fff",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "8px"
                                }
                            ),
                            width=4
                        ),
                    ], className="mb-2"),

                   dbc.Label("Stop Loss (% of premium)"),
                   dbc.Row([
                       dbc.Col(
                           dbc.Input(id="bt-stop-loss", type="number", value=200, step=50, size="sm"),
                            width=8
                        ),
                       dbc.Col(
                            dcc.Checklist(
                                id="bt-disable-stop-loss",
                               options=[{"label": "Disable", "value": True}],
                               value=[],
                                className="mt-2",
                                style={
                                    "color": "#fff",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "8px"
                                }
                            ),
                            width=4
                        ),
                    ], className="mb-3"),

                    # Max Positions (visible for all strategies)
                    html.Hr(),
                    html.H6("Position Management", className="fw-bold mb-2"),
                    
                    dbc.Label("Max Positions"),
                    dbc.Input(id="bt-max-positions", type="number", value=5, min=1, max=50, step=1, size="sm", className="mb-3"),  # Default 5 for diversification
                    
                    # ML Delta Optimization
                    html.Hr(),
                    html.H6("ML Delta Optimization", className="fw-bold mb-2"),
                    
                    dcc.Checklist(
                        id="bt-ml-delta",
                        options=[{"label": " Enable ML Delta Optimization", "value": True}],
                        value=[],
                        inline=True,
                        className="mb-2",
                        style={"color": "#fff"},
                    ),
                    
                    html.Div(id="ml-delta-params-container", children=[
                        dbc.Label("ML Adoption Rate (0.0-1.0)"),
                        dbc.Input(id="bt-ml-adoption-rate", type="number", value=0.5, step=0.1, min=0.0, max=1.0, size="sm", className="mb-3"),
                    ], className="d-none"),
                    
                    # ML DTE Optimization
                    html.Hr(),
                    html.H6("ML DTE Optimization", className="fw-bold mb-2"),
                    
                    dcc.Checklist(
                        id="bt-ml-dte",
                        options=[{"label": " Enable ML DTE Optimization", "value": True}],
                        value=[],
                        inline=True,
                        className="mb-2",
                        style={"color": "#fff"},
                    ),
                    
                    # Benchmark Comparison
                    html.Hr(),
                    html.H6("Benchmark Comparison", className="fw-bold mb-2"),
                    
                    dbc.Label("Select Benchmarks (Hold Ctrl for multiple)"),
                    dcc.Dropdown(
                        id="bt-benchmarks",
                        options=[
                            {"label": "QQQ - Nasdaq-100 ETF", "value": "QQQ"},
                            {"label": "SPY - S&P 500 ETF", "value": "SPY"},
                            {"label": "IWM - Russell 2000 ETF", "value": "IWM"},
                            {"label": "NVDA - NVIDIA Corporation", "value": "NVDA"},
                            {"label": "TSLA - Tesla Inc", "value": "TSLA"},
                            {"label": "AAPL - Apple Inc", "value": "AAPL"},
                            {"label": "MSFT - Microsoft Corporation", "value": "MSFT"},
                            {"label": "GOOGL - Alphabet Inc", "value": "GOOGL"},
                            {"label": "AMZN - Amazon.com Inc", "value": "AMZN"},
                        ],
                        value=[],
                        multi=True,
                        className="mb-3",
                    ),
                    
                    # Wheel Strategy Specific Parameters
                    html.Div(id="wheel-params-container", children=[
                        html.Hr(),
                        html.H6("Wheel Strategy Parameters", className="fw-bold mb-2"),
                        
                        dbc.Label("Put Delta (absolute)"),
                        dbc.Input(id="bt-put-delta", type="number", value=0.30, step=0.05, size="sm", className="mb-2"),
                        
                        dbc.Label("Call Delta (absolute)"),
                        dbc.Input(id="bt-call-delta", type="number", value=0.30, step=0.05, size="sm", className="mb-2"),
                    ], className="d-none"),

                    # Progress bar (hidden by default)
                    html.Div(
                        id="bt-progress-container",
                        children=[
                            dbc.Progress(
                                id="bt-progress",
                                value=0,
                                label="Running backtest...",
                                striped=True,
                                animated=True,
                                className="mb-3",
                                style={"display": "none"},
                            ),
                        ],
                        className="mb-3",
                    ),

                    dbc.Button(
                        "Run Backtest", id="bt-run-btn",
                        color="primary", className="w-100", n_clicks=0,
                    ),
                    
                    # Save button (shown after successful backtest)
                    html.Div(
                        id="bt-save-container",
                        children=[
                            dbc.Button(
                                "💾 Save Result", 
                                id="bt-save-btn",
                                color="success", 
                                className="w-100 mt-2",
                                n_clicks=0,
                            ),
                        ],
                        className="d-none",  # Hidden by default
                    ),
                ]),
            ], className="shadow-sm"),
        ], md=3),

        # Right: Results
        dbc.Col([
            dcc.Loading(
                html.Div(id="bt-results-container", children=[
                    html.P(
                        "Configure strategy and click 'Run Backtest'",
                        className="text-muted",
                    ),
                ]),
                type="circle",
            ),
        ], md=9),
    ]),

    dcc.Store(id="bt-results-store", data={}),
    dcc.Store(id="bt-params-store", data={}),  # Store params for saving
])


@callback(
    Output("wheel-params-container", "className"),
    Input("bt-strategy", "value"),
)
def toggle_wheel_params(strategy):
    """Show/hide wheel strategy specific parameters."""
    if strategy == "wheel":
        return "d-block"
    return "d-none"


@callback(
    Output("delta-config-container", "className"),
    Input("bt-strategy", "value"),
)
def toggle_delta_config(strategy):
    """Show/hide delta configuration for non-Wheel strategies."""
    if strategy != "wheel":
        return "d-block"
    return "d-none"


@callback(
    Output("ml-delta-params-container", "className"),
    Input("bt-ml-delta", "value"),
)
def toggle_ml_delta_params(ml_delta_enabled):
    """Show/hide ML delta parameters when ML optimization is enabled."""
    if ml_delta_enabled and True in ml_delta_enabled:
        return "d-block"
    return "d-none"


@callback(
    Output("bt-ml-dte", "value"),
    Input("bt-ml-delta", "value"),
)
def toggle_ml_dte_with_delta(ml_delta_enabled):
    """Enable ML DTE optimization when ML delta optimization is enabled."""
    return ml_delta_enabled


# Callback to disable traditional parameters when ML is enabled
@callback(
    Output("bt-dte-min", "disabled"),
    Output("bt-dte-max", "disabled"),
    Input("bt-ml-dte", "value"),
)
def disable_traditional_dte_params_when_ml_enabled(ml_dte_enabled):
    """Disable traditional DTE parameters when ML DTE optimization is enabled."""
    ml_enabled = bool(ml_dte_enabled and True in ml_dte_enabled)
    return ml_enabled, ml_enabled


@callback(
    Output("bt-results-store", "data"),
    Output("bt-params-store", "data"),
    Output("bt-results-container", "children"),
    Output("bt-save-container", "className"),
    Input("bt-run-btn", "n_clicks"),
    State("bt-strategy", "value"),
    State("bt-symbol", "value"),
    State("bt-start", "value"),
    State("bt-end", "value"),
    State("bt-capital", "value"),
    State("bt-leverage", "value"),
    State("bt-dte-min", "value"),
    State("bt-dte-max", "value"),
    State("bt-delta", "value"),
    State("bt-profit-target", "value"),
    State("bt-stop-loss", "value"),
    State("bt-put-delta", "value"),
    State("bt-call-delta", "value"),
    State("bt-max-positions", "value"),
    State("bt-benchmarks", "value"),
    State("bt-disable-profit-target", "value"),
    State("bt-disable-stop-loss", "value"),
    State("bt-use-synthetic", "value"),
    State("bt-ml-delta", "value"),
    State("bt-ml-dte", "value"),
    State("bt-ml-adoption-rate", "value"),
    prevent_initial_call=True,
)
def run_backtest(
    n_clicks, strategy, symbol, start_date, end_date,
    capital, leverage, dte_min, dte_max, delta, profit_target, stop_loss,
    put_delta, call_delta, max_positions, benchmarks,
    disable_profit_target, disable_stop_loss, use_synthetic,
    ml_delta, ml_dte, ml_adoption_rate
):
    if not symbol or not start_date or not end_date:
        return no_update, no_update, no_update, no_update
    symbol = symbol.strip().upper()

    services = get_services()
    if not services:
        return {}, {}, html.P("Services not initialized", className="text-warning"), "d-none"

    engine = services["backtest_engine"]
    data_client = services.get("data_client")

    # Use special value 999999 to disable profit target/stop loss
    # Checklist returns list, check if True is in the list
    profit_target_value = 999999 if (disable_profit_target and True in disable_profit_target) else (profit_target or 50)
    stop_loss_value = 999999 if (disable_stop_loss and True in disable_stop_loss) else (stop_loss or 200)

    params = {
        "strategy": strategy,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": capital or 150000,  # Default $150k for realistic trading
        "max_leverage": leverage or 1.0,
        "dte_min": dte_min or 30,
        "dte_max": dte_max or 45,
        "delta_target": delta or 0.30,
        "profit_target_pct": profit_target_value,
        "stop_loss_pct": stop_loss_value,
        # Synthetic data option - Checklist returns list, check if True is in the list
        "use_synthetic_data": bool(use_synthetic and True in use_synthetic),
        # Wheel strategy specific parameters
        "put_delta": put_delta or 0.30,
        "call_delta": call_delta or 0.30,
        "max_positions": max_positions or 10,  # Default 10 for diversification
        # ML Delta optimization parameters
        "ml_delta_optimization": bool(ml_delta and True in ml_delta),
        "ml_dte_optimization": bool(ml_dte and True in ml_dte),
        "ml_adoption_rate": ml_adoption_rate or 0.5,
    }

    # Get benchmark data if requested
    benchmark_data = {}
    if benchmarks and data_client:
        try:
            from core.backtesting.benchmark import BenchmarkService
            benchmark_service = BenchmarkService(data_client)
            benchmark_data = benchmark_service.get_multiple_benchmarks(
                symbols=benchmarks,
                start_date=start_date,
                end_date=end_date,
                initial_capital=capital or 100000
            )
        except Exception as e:
            print(f"Warning: Could not fetch benchmark data: {e}")
            benchmark_data = {}

    try:
        result = engine.run(params)
        # Add benchmark data to result
        result["benchmark_data"] = benchmark_data
    except Exception as e:
        return {}, {}, html.P(f"Backtest error: {e}", className="text-danger"), "d-none"

    if not result:
        return {}, {}, html.P("No results generated", className="text-muted"), "d-none"

    # Build results UI
    metrics = result.get("metrics", {})
    trades = result.get("trades", [])
    daily_pnl = result.get("daily_pnl", [])
    strategy_performance = result.get("strategy_performance", {})

    # Metric cards
    total_ret = metrics.get("total_return_pct", 0)
    ret_color = "success" if total_ret >= 0 else "danger"
    metrics_row = dbc.Row([
        dbc.Col(metric_card("Total Return", f"{total_ret:+.2f}%", ret_color), md=2),
        dbc.Col(metric_card("Annual Return", f"{metrics.get('annualized_return_pct', 0):+.2f}%", ret_color), md=2),
        dbc.Col(metric_card("Win Rate", f"{metrics.get('win_rate', 0):.1f}%", "info"), md=2),
        dbc.Col(metric_card("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}", "primary"), md=2),
        dbc.Col(metric_card("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%", "danger"), md=2),
        dbc.Col(metric_card("Total Trades", f"{metrics.get('total_trades', 0)}", "secondary"), md=2),
    ], className="mb-4 g-3")

    # P&L chart
    pnl_dates = [p["date"] for p in daily_pnl] if daily_pnl else []
    pnl_values = [p["cumulative_pnl"] for p in daily_pnl] if daily_pnl else []
    initial_capital = params.get("initial_capital", 100000)
    benchmark_data = result.get("benchmark_data", {})
    pnl_chart = dcc.Graph(figure=create_pnl_chart(
        pnl_dates, 
        pnl_values, 
        benchmark_data=benchmark_data,
        initial_capital=initial_capital
    ))

    # Monthly heatmap
    monthly = metrics.get("monthly_returns", {})
    monthly_tupled = {(int(k.split("-")[0]), int(k.split("-")[1])): v for k, v in monthly.items()} if monthly else {}
    heatmap = dcc.Graph(figure=create_monthly_heatmap(monthly_tupled)) if monthly_tupled else html.Div()
    
    # Trade timeline chart
    underlying_prices = result.get("underlying_prices", [])
    selection_history = strategy_performance.get("selection_history", []) if strategy_performance else []
    trade_timeline = dcc.Graph(figure=create_trade_timeline_chart(
        trades=trades,
        daily_pnl=daily_pnl,
        underlying_prices=underlying_prices,
        selection_history=selection_history,
        title="Trade Timeline: Entry/Exit Points & Performance"
    ))

    # Trades table
    trade_columns = [
        {"headerName": "Entry", "field": "entry_date", "width": 100, "sort": "desc"},  # Sort by newest first
        {"headerName": "Exit", "field": "exit_date", "width": 100},
        {"headerName": "Option Contract", "field": "contract_name", "width": 220},  # Increased width for better visibility
        {"headerName": "Type", "field": "trade_type", "width": 120},
        {"headerName": "Strike", "field": "strike", "width": 80,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Expiry", "field": "expiry", "width": 100},
        {"headerName": "Right", "field": "right", "width": 80},
        {"headerName": "Qty", "field": "quantity", "width": 60},
        {"headerName": "Stock Entry $", "field": "underlying_entry", "width": 100,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Stock Exit $", "field": "underlying_exit", "width": 100,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Option Entry $", "field": "entry_price", "width": 90,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Option Exit $", "field": "exit_price", "width": 90,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "P&L", "field": "pnl", "width": 100,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"},
         "cellStyle": {"function": "params.value >= 0 ? {'color': '#26a69a'} : {'color': '#ef5350'}"}},
        {"headerName": "Reason", "field": "exit_reason", "width": 120},
    ]
    trades_table = create_data_table(trades, trade_columns, "bt-trades-table", height=450) if trades else html.P("No trades", className="text-muted")

    # Additional metrics
    extra_row = dbc.Row([
        dbc.Col(metric_card("Avg Profit", f"${metrics.get('avg_profit', 0):,.2f}", "success"), md=3),
        dbc.Col(metric_card("Avg Loss", f"${metrics.get('avg_loss', 0):,.2f}", "danger"), md=3),
        dbc.Col(metric_card("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}", "primary"), md=3),
        dbc.Col(metric_card("Sortino", f"{metrics.get('sortino_ratio', 0):.2f}", "info"), md=3),
    ], className="mb-3 g-3")
    
    # Add monitoring dashboard for wheel strategy
    monitoring_section = html.Div()
    if strategy_performance and params.get("strategy") == "wheel":
        # Add metrics from engine result for performance card
        monitoring_data = {**strategy_performance, "metrics": metrics}
        monitoring_section = create_monitoring_dashboard(monitoring_data)
    
    # Add holdings card (for all strategies)
    holdings_card = html.Div()
    if strategy_performance:
        # Handle both nested (Wheel) and flat (BinbinGod) data structures
        current_state = strategy_performance.get("current_state", {})
        holdings_data = {
            "shares_held": strategy_performance.get("shares_held") or current_state.get("shares_held", 0),
            "cost_basis": strategy_performance.get("cost_basis") or current_state.get("cost_basis", 0),
            "options_held": strategy_performance.get("open_positions", []),
        }
        holdings_card = create_holdings_card(holdings_data)

    content = html.Div([
        html.H5("Performance Summary", className="mb-3"),
        metrics_row,
        pnl_chart,
        html.H5("Additional Metrics", className="mt-4 mb-3"),
        extra_row,
        heatmap,
        holdings_card,  # Show current holdings
        monitoring_section,
        html.H5("Trade Timeline Analysis", className="mt-4 mb-3"),
        trade_timeline,
        html.H5("Trade Log", className="mt-4 mb-3"),
        trades_table,
    ])
    
    # Add trade history and phase transition updates if monitoring data exists
    # Skip for strategies that have dedicated monitoring dashboard (wheel, binbin_god) since it already shows these
    if strategy_performance and params.get("strategy") not in ["wheel", "binbin_god"]:
        trade_history = strategy_performance.get("trade_history", [])
        phase_history = strategy_performance.get("phase_history", [])
        
        # Update trade history display
        if trade_history:
            content.children.insert(-1, html.Div([
                html.H5("Recent Trade History", className="mt-4 mb-3"),
                create_trade_history_table(trade_history),
            ]))
        
        # Update phase transition display
        if phase_history:
            content.children.insert(-1, html.Div([
                html.H5("Phase Transition Log", className="mt-4 mb-3"),
                create_phase_transition_log(phase_history),
            ]))

    # Show save button after successful backtest
    return result, params, content, "d-block mt-2"


@callback(
    Output("bt-save-btn", "children"),
    Output("bt-save-btn", "color"),
    Output("bt-save-btn", "disabled"),
    Input("bt-save-btn", "n_clicks"),
    State("bt-results-store", "data"),
    State("bt-params-store", "data"),
    prevent_initial_call=True,
)
def save_backtest_result(n_clicks, result, params):
    """Save the backtest result to database."""
    if not n_clicks or not result or not params:
        return "💾 Save Result", "success", False
    
    try:
        from core.backtesting.storage import get_backtest_storage
        storage = get_backtest_storage()
        
        backtest_id = storage.save_backtest(params, result)
        return f"✅ Saved #{backtest_id}", "secondary", True
    except Exception as e:
        return f"❌ Error: {str(e)[:30]}", "danger", False
