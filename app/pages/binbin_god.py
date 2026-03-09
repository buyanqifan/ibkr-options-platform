"""Binbin God Strategy Backtester Page - MAG7 intelligent Wheel strategy."""

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc
from app.components.tables import create_trade_log_table
from app.components.charts import create_equity_curve_chart
from app.components.monitoring import create_performance_summary
from app.services import get_services

dash.register_page(__name__, path="/binbin-god", name="Binbin God", icon="bi bi-robot")

# Strategy info
STRATEGY_INFO = {
    "name": "Binbin God Strategy",
    "version": "0.1.0",
    "description": "Intelligent Wheel strategy with dynamic MAG7 stock selection based on quantitative metrics.",
    "universe": ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"],
    "selection_criteria": {
        "P/E Ratio": "20% weight - Value stocks preferred",
        "Option IV": "40% weight - Higher premium income",
        "Momentum": "20% weight - Positive trend",
        "Stability": "20% weight - Risk management",
    },
}


def create_strategy_info_card():
    """Create information card about the Binbin God strategy."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi bi-info-circle me-2"),
            "Strategy Information",
        ], className="bg-primary text-white"),
        dbc.CardBody([
            html.H5(f"{STRATEGY_INFO['name']} v{STRATEGY_INFO['version']}", className="card-title"),
            html.P(STRATEGY_INFO["description"], className="card-text"),
            
            html.Hr(),
            
            html.H6("MAG7 Universe:", className="fw-bold"),
            dbc.Badge(label, color="info", className="me-1 mb-1") 
            for label in STRATEGY_INFO["universe"],
            
            html.H6("Selection Criteria:", className="fw-bold mt-3"),
            html.Ul([
                html.Li(dbc.Badge(f"{k}: {v}", color="secondary", className="me-1"))
                for k, v in STRATEGY_INFO["selection_criteria"].items()
            ], className="list-unstyled"),
        ]),
    ], className="mb-4")


def create_mag7_analysis_placeholder():
    """Placeholder for MAG7 analysis display."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="bi bi-bar-chart me-2"),
            "MAG7 Stock Analysis",
        ], className="bg-info text-white"),
        dbc.CardBody([
            html.Div(id="binbin-mag7-analysis", children=[
                html.P("Run backtest to see MAG7 stock rankings and selection", 
                      className="text-muted"),
            ]),
        ]),
    ], className="mb-4")


layout = dbc.Container([
    # Header
    html.Div([
        html.H1([
            html.I(className="bi bi-robot me-2"),
            "Binbin God Strategy Backtester",
        ], className="mb-2"),
        html.P("Intelligent Wheel strategy with dynamic MAG7 stock selection", 
              className="lead text-muted"),
    ], className="mb-4"),
    
    dbc.Row([
        # Left column: Controls
        dbc.Col([
            create_strategy_info_card(),
            
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-sliders me-2"),
                    "Backtest Configuration",
                ], className="bg-success text-white"),
                
                dbc.CardBody([
                    # Date Range
                    dbc.Label("Date Range"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="bbg-start", type="date", 
                                        value="2025-01-01"), width=6),
                        dbc.Col(dbc.Input(id="bbg-end", type="date", 
                                        value="2026-03-05"), width=6),
                    ], className="mb-3"),
                    
                    # Initial Capital
                    dbc.Label("Initial Capital ($)"),
                    dbc.Input(id="bbg-capital", type="number", 
                             value=150000, className="mb-3"),
                    
                    # Max Leverage
                    dbc.Label("Max Leverage"),
                    dbc.Input(id="bbg-leverage", type="number", 
                             value=1.0, step=0.1, min=1.0, className="mb-3"),
                    
                    html.Hr(),
                    html.H6("Wheel Parameters", className="fw-bold mb-2"),
                    
                    # DTE Range
                    dbc.Label("DTE Range (days)"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="bbg-dte-min", type="number", 
                                        value=30, size="sm"), width=6),
                        dbc.Col(dbc.Input(id="bbg-dte-max", type="number", 
                                        value=45, size="sm"), width=6),
                    ], className="mb-2"),
                    
                    # Delta Targets
                    dbc.Label("Put Delta (absolute)"),
                    dbc.Input(id="bbg-put-delta", type="number", 
                             value=0.30, step=0.05, size="sm", className="mb-2"),
                    
                    dbc.Label("Call Delta (absolute)"),
                    dbc.Input(id="bbg-call-delta", type="number", 
                             value=0.30, step=0.05, size="sm", className="mb-2"),
                    
                    # Max Positions
                    dbc.Label("Max Positions"),
                    dbc.Input(id="bbg-max-positions", type="number", 
                             value=10, min=1, max=50, step=1, size="sm", className="mb-2"),
                    
                    # Rebalance Threshold
                    dbc.Label("Rebalance Threshold (%)"),
                    dbc.Input(id="bbg-rebalance-threshold", type="number", 
                             value=15, step=5, size="sm", 
                             className="mb-3",
                             title="Switch symbols if better opportunity is X% higher scored"),
                    
                    html.Hr(),
                    html.H6("Exit Conditions", className="fw-bold mb-2"),
                    
                    # Profit Target
                    dbc.Label("Profit Target (% of premium)"),
                    dbc.Row([
                        dbc.Col(
                            dbc.Input(id="bbg-profit-target", type="number", 
                                     value=50, step=10, size="sm"),
                            width=8
                        ),
                        dbc.Col(
                            dbc.FormCheck(
                                id="bbg-disable-profit-target",
                                label="Disable",
                                value=False,
                                className="mt-2"
                            ),
                            width=4
                        ),
                    ], className="mb-2"),
                    
                    # Stop Loss
                    dbc.Label("Stop Loss (% of premium)"),
                    dbc.Row([
                        dbc.Col(
                            dbc.Input(id="bbg-stop-loss", type="number", 
                                     value=200, step=50, size="sm"),
                            width=8
                        ),
                        dbc.Col(
                            dbc.FormCheck(
                                id="bbg-disable-stop-loss",
                                label="Disable",
                                value=False,
                                className="mt-2"
                            ),
                            width=4
                        ),
                    ], className="mb-3"),
                    
                    # Run Button
                    dbc.Button([
                        html.I(className="bi bi-play-fill me-2"),
                        "Run Backtest",
                    ], id="bbg-run-btn", color="primary", className="w-100", size="lg"),
                ]),
            ]),
        ], md=4),
        
        # Right column: Results
        dbc.Col([
            create_mag7_analysis_placeholder(),
            
            # Results Container
            html.Div(id="binbin-results-container", children=[
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="bi bi-graph-up me-2"),
                            "Run a backtest to see results",
                        ], className="text-center text-muted py-5"),
                    ]),
                ]),
            ]),
            
            # Hidden store for results data
            dcc.Store(id="binbin-results-store", data={}),
        ], md=8),
    ]),
])


@callback(
    Output("binbin-results-store", "data"),
    Output("binbin-results-container", "children"),
    Output("binbin-mag7-analysis", "children"),
    Input("bbg-run-btn", "n_clicks"),
    State("bbg-start", "value"),
    State("bbg-end", "value"),
    State("bbg-capital", "value"),
    State("bbg-leverage", "value"),
    State("bbg-dte-min", "value"),
    State("bbg-dte-max", "value"),
    State("bbg-put-delta", "value"),
    State("bbg-call-delta", "value"),
    State("bbg-max-positions", "value"),
    State("bbg-rebalance-threshold", "value"),
    State("bbg-profit-target", "value"),
    State("bbg-stop-loss", "value"),
    State("bbg-disable-profit-target", "value"),
    State("bbg-disable-stop-loss", "value"),
    prevent_initial_call=True,
)
def run_binbin_backtest(
    n_clicks, start_date, end_date, capital, leverage,
    dte_min, dte_max, put_delta, call_delta, max_positions, rebalance_threshold,
    profit_target, stop_loss, disable_profit_target, disable_stop_loss
):
    """Run Binbin God strategy backtest."""
    if not start_date or not end_date:
        return no_update, no_update, no_update
    
    services = get_services()
    if not services:
        return {}, html.P("Services not initialized", className="text-warning"), ""
    
    engine = services["backtest_engine"]
    
    # Prepare parameters
    profit_target_value = 999999 if disable_profit_target else (profit_target or 50)
    stop_loss_value = 999999 if disable_stop_loss else (stop_loss or 200)
    
    params = {
        "strategy": "binbin_god",
        "symbol": "MAG7_AUTO",  # Special symbol indicating auto-selection
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": capital or 150000,
        "max_leverage": leverage or 1.0,
        "dte_min": dte_min or 30,
        "dte_max": dte_max or 45,
        "delta_target": 0.30,  # Not used by Wheel, but required by base class
        "profit_target_pct": profit_target_value,
        "stop_loss_pct": stop_loss_value,
        "put_delta": put_delta or 0.30,
        "call_delta": call_delta or 0.30,
        "max_positions": max_positions or 10,
        "rebalance_threshold": (rebalance_threshold or 15) / 100.0,  # Convert to decimal
    }
    
    try:
        result = engine.run(params)
    except Exception as e:
        return {}, html.P(f"Backtest error: {e}", className="text-danger"), ""
    
    if not result:
        return {}, html.P("No results generated", className="text-muted"), ""
    
    # Build results UI
    metrics = result.get("metrics", {})
    trades = result.get("trades", [])
    daily_pnl = result.get("daily_pnl", [])
    
    # Performance summary
    summary_html = create_performance_summary(result)
    
    # Equity curve
    chart_html = create_equity_curve_chart(daily_pnl)
    
    # Trade log
    table_html = create_trade_log_table(trades)
    
    # MAG7 analysis
    mag7_analysis = result.get("mag7_analysis", {})
    if mag7_analysis and "ranked_stocks" in mag7_analysis:
        ranked = mag7_analysis["ranked_stocks"]
        best = mag7_analysis.get("best_pick", {})
        
        mag7_html = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6("🏆 Best Pick:", className="fw-bold text-success"),
                    html.H4(best.get("symbol", "N/A"), className="text-success"),
                    html.Small(f"Score: {best.get('total_score', 0):.1f}", className="text-muted"),
                ], width=6),
                dbc.Col([
                    html.H6("📊 Total Stocks Analyzed:", className="fw-bold"),
                    html.H4(len(ranked), className="text-primary"),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Table.from_dataframe(
                pd.DataFrame([{
                    "Rank": i+1,
                    "Symbol": s["symbol"],
                    "Score": s["total_score"],
                    "PE": s["metrics"]["pe_ratio"],
                    "IV Rank": s["metrics"]["iv_rank"],
                    "Price": f"${s['metrics']['price']:.0f}",
                } for i, s in enumerate(ranked)]),
                striped=True,
                hover=True,
                bordered=True,
                className="table-sm",
            ),
        ])
    else:
        mag7_html = html.P("MAG7 analysis not available", className="text-muted")
    
    # Results container
    results_html = html.Div([
        # Summary cards
        dbc.Row([
            dbc.Col(html.H3(f"Total Return: {metrics.get('total_return_pct', 0):+.2f}%", 
                           className="text-center p-3 border rounded bg-light"), md=3),
            dbc.Col(html.H3(f"Win Rate: {metrics.get('win_rate', 0):.1f}%", 
                           className="text-center p-3 border rounded bg-light"), md=3),
            dbc.Col(html.H3(f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}", 
                           className="text-center p-3 border rounded bg-light"), md=3),
            dbc.Col(html.H3(f"Trades: {len(trades)}", 
                           className="text-center p-3 border rounded bg-light"), md=3),
        ], className="mb-4"),
        
        # Charts
        dbc.Card([
            dbc.CardHeader("Equity Curve", className="fw-bold"),
            dbc.CardBody(chart_html),
        ], className="mb-4"),
        
        # Trade log
        dbc.Card([
            dbc.CardHeader("Trade Log", className="fw-bold"),
            dbc.CardBody(table_html),
        ], className="mb-4"),
    ])
    
    return result, results_html, mag7_html


# Import pandas at module level
import pandas as pd
