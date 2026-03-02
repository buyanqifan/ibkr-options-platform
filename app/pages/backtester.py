"""Backtester page: strategy configuration, run backtest, results visualization."""

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc
from app.components.tables import metric_card, create_data_table
from app.components.charts import create_pnl_chart, create_monthly_heatmap
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
                        ],
                        value="sell_put",
                        className="mb-3",
                    ),

                    dbc.Label("Symbol"),
                    dbc.Input(id="bt-symbol", value="AAPL", className="mb-3"),

                    dbc.Label("Date Range"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="bt-start", type="date", value="2023-01-01"), width=6),
                        dbc.Col(dbc.Input(id="bt-end", type="date", value="2024-01-01"), width=6),
                    ], className="mb-3"),

                    dbc.Label("Initial Capital ($)"),
                    dbc.Input(id="bt-capital", type="number", value=100000, className="mb-3"),

                    html.Hr(),
                    html.H6("Strategy Parameters", className="fw-bold mb-2"),

                    dbc.Label("DTE Range (days)"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="bt-dte-min", type="number", value=21, size="sm"), width=6),
                        dbc.Col(dbc.Input(id="bt-dte-max", type="number", value=45, size="sm"), width=6),
                    ], className="mb-2"),

                    dbc.Label("Target Delta (absolute)"),
                    dbc.Input(id="bt-delta", type="number", value=0.30, step=0.05, size="sm", className="mb-2"),

                    dbc.Label("Profit Target (% of premium)"),
                    dbc.Input(id="bt-profit-target", type="number", value=50, size="sm", className="mb-2"),

                    dbc.Label("Stop Loss (% of premium)"),
                    dbc.Input(id="bt-stop-loss", type="number", value=200, size="sm", className="mb-3"),

                    dbc.Button(
                        "Run Backtest", id="bt-run-btn",
                        color="primary", className="w-100", n_clicks=0,
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
])


@callback(
    Output("bt-results-store", "data"),
    Output("bt-results-container", "children"),
    Input("bt-run-btn", "n_clicks"),
    State("bt-strategy", "value"),
    State("bt-symbol", "value"),
    State("bt-start", "value"),
    State("bt-end", "value"),
    State("bt-capital", "value"),
    State("bt-dte-min", "value"),
    State("bt-dte-max", "value"),
    State("bt-delta", "value"),
    State("bt-profit-target", "value"),
    State("bt-stop-loss", "value"),
    prevent_initial_call=True,
)
def run_backtest(
    n_clicks, strategy, symbol, start_date, end_date,
    capital, dte_min, dte_max, delta, profit_target, stop_loss,
):
    if not symbol or not start_date or not end_date:
        return no_update, no_update
    symbol = symbol.strip().upper()

    services = get_services()
    if not services:
        return {}, html.P("Services not initialized", className="text-warning")

    engine = services["backtest_engine"]

    params = {
        "strategy": strategy,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": capital or 100000,
        "dte_min": dte_min or 21,
        "dte_max": dte_max or 45,
        "delta_target": delta or 0.30,
        "profit_target_pct": profit_target or 50,
        "stop_loss_pct": stop_loss or 200,
    }

    try:
        result = engine.run(params)
    except Exception as e:
        return {}, html.P(f"Backtest error: {e}", className="text-danger")

    if not result:
        return {}, html.P("No results generated", className="text-muted")

    # Build results UI
    metrics = result.get("metrics", {})
    trades = result.get("trades", [])
    daily_pnl = result.get("daily_pnl", [])

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
    pnl_chart = dcc.Graph(figure=create_pnl_chart(pnl_dates, pnl_values))

    # Monthly heatmap
    monthly = metrics.get("monthly_returns", {})
    monthly_tupled = {(int(k.split("-")[0]), int(k.split("-")[1])): v for k, v in monthly.items()} if monthly else {}
    heatmap = dcc.Graph(figure=create_monthly_heatmap(monthly_tupled)) if monthly_tupled else html.Div()

    # Trades table
    trade_columns = [
        {"headerName": "Entry", "field": "entry_date", "width": 100},
        {"headerName": "Exit", "field": "exit_date", "width": 100},
        {"headerName": "Type", "field": "trade_type", "width": 120},
        {"headerName": "Strike", "field": "strike", "width": 80,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Expiry", "field": "expiry", "width": 100},
        {"headerName": "Entry $", "field": "entry_price", "width": 90,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Exit $", "field": "exit_price", "width": 90,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "P&L", "field": "pnl", "width": 100,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"},
         "cellStyle": {"function": "params.value >= 0 ? {'color': '#26a69a'} : {'color': '#ef5350'}"}},
        {"headerName": "Reason", "field": "exit_reason", "width": 120},
    ]
    trades_table = create_data_table(trades, trade_columns, "bt-trades-table", height=300) if trades else html.P("No trades", className="text-muted")

    # Additional metrics
    extra_row = dbc.Row([
        dbc.Col(metric_card("Avg Profit", f"${metrics.get('avg_profit', 0):,.2f}", "success"), md=3),
        dbc.Col(metric_card("Avg Loss", f"${metrics.get('avg_loss', 0):,.2f}", "danger"), md=3),
        dbc.Col(metric_card("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}", "primary"), md=3),
        dbc.Col(metric_card("Sortino", f"{metrics.get('sortino_ratio', 0):.2f}", "info"), md=3),
    ], className="mb-3 g-3")

    content = html.Div([
        html.H5("Performance Summary", className="mb-3"),
        metrics_row,
        pnl_chart,
        html.H5("Additional Metrics", className="mt-4 mb-3"),
        extra_row,
        heatmap,
        html.H5("Trade Log", className="mt-4 mb-3"),
        trades_table,
    ])

    return result, content
