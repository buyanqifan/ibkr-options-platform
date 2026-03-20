"""Binbin God Strategy Backtester Page - MAG7 intelligent Wheel strategy."""

import dash
from dash import html, dcc, callback, Output, Input, State, no_update, clientside_callback
import dash_bootstrap_components as dbc
import pandas as pd
from app.components.tables import metric_card, create_data_table
from app.components.charts import create_pnl_chart, create_monthly_heatmap, create_trade_timeline_chart
from app.components.monitoring import (
    create_monitoring_dashboard,
    create_trade_history_table,
    create_phase_transition_log,
    create_holdings_card,
    create_performance_metrics_card,
)
from app.services import get_services

# Initialize services at module level
_services = None

def get_services_cached():
    """Get services with caching to avoid local variable issues."""
    global _services
    if _services is None:
        _services = get_services()
    return _services

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
            html.Div([
                dbc.Badge(label, color="info", className="me-1 mb-1") 
                for label in STRATEGY_INFO["universe"]
            ]),
            
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
                    
                    # Use Synthetic Data Option
                    dcc.Checklist(
                        id="bbg-use-synthetic",
                        options=[{"label": " Use Random Synthetic Data (for testing without IBKR connection)", "value": True}],
                        value=[],
                        inline=True,
                        className="mb-3",
                        style={
                            "color": "#e0e0e0",
                            "backgroundColor": "rgba(255, 255, 255, 0.08)",
                            "padding": "10px 14px",
                            "borderRadius": "4px",
                            "border": "1px solid rgba(255, 255, 255, 0.2)",
                        },
                    ),
                    
                    html.Hr(),
                    html.H6("Wheel Parameters", className="fw-bold mb-2"),
                    
                    # Stock Pool Selection
                    dbc.Label("Stock Pool"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id="bbg-stock-pool",
                                options=[
                                    {"label": "MAG7 (Default)", "value": "MAG7"},
                                    {"label": "Tech Giants (MAGAMG)", "value": "MAGAMG"},
                                    {"label": "All 7 Stocks", "value": "CUSTOM"},
                                ],
                                value="MAG7",
                                clearable=False,
                                style={
                                    "color": "#212529",
                                    "backgroundColor": "#fff",
                                    "border": "1px solid #ced4da",
                                    "borderRadius": "4px",
                                },
                            ),
                        ], width=12),
                    ], className="mb-2"),
                    
                    # Custom Stock Input (shown when CUSTOM is selected)
                    html.Div([
                        dbc.Label("Custom Stocks (comma separated)"),
                        dbc.Input(id="bbg-custom-stocks", 
                                 placeholder="e.g., MSFT,AAPL,NVDA,GOOGL,AMZN,META,TSLA",
                                 className="mb-2"),
                    ], id="bbg-custom-stocks-container", style={"display": "none"}),
                    
                    html.Hr(),
                    html.H6("Wheel Parameters", className="fw-bold mb-2"),
                    
                    # DTE Range
                    html.Div([
                        dbc.Label("DTE Range (days)"),
                        dbc.Row([
                            dbc.Col(dbc.Input(id="bbg-dte-min", type="number", 
                                            value=30, size="sm", disabled=False), width=6),
                            dbc.Col(dbc.Input(id="bbg-dte-max", type="number", 
                                            value=45, size="sm", disabled=False), width=6),
                        ], className="mb-2"),
                    ], id="bbg-dte-range-container"),
                    
                    # Delta Targets
                    html.Div([
                        dbc.Label("Put Delta (absolute)"),
                        dbc.Input(id="bbg-put-delta", type="number", 
                                 value=0.30, step=0.05, size="sm", className="mb-2", disabled=False),
                        
                        dbc.Label("Call Delta (absolute)"),
                        dbc.Input(id="bbg-call-delta", type="number", 
                                 value=0.30, step=0.05, size="sm", className="mb-2", disabled=False),
                    ], id="bbg-delta-targets-container"),
                    
                    html.Hr(),
                    html.H6("ML Delta Optimization", className="fw-bold mb-2 text-primary"),
                    
                    # ML Delta Optimization Toggle
                    dbc.Label("Enable ML Delta Optimization"),
                    dbc.Switch(
                        id="bbg-ml-optimization",
                        value=False,
                        label="Enable ML-powered adaptive delta selection",
                        className="mb-3",
                        style={
                            "display": "flex", 
                            "alignItems": "center",
                            "color": "#e0e0e0",
                        }
                    ),
                    
                    # ML DTE Optimization Toggle
                    dbc.Label("Enable ML DTE Optimization"),
                    dbc.Switch(
                        id="bbg-ml-dte-optimization",
                        value=False,
                        label="Enable ML-powered adaptive DTE (Days to Expiration) selection",
                        className="mb-3",
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "color": "#e0e0e0",
                        }
                    ),

                    # ML Roll Optimization Toggle
                    dbc.Label("Enable ML Roll Optimization"),
                    dbc.Switch(
                        id="bbg-ml-roll-optimization",
                        value=False,
                        label="Enable ML-powered intelligent roll management (Roll Forward/Out)",
                        className="mb-3",
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "color": "#e0e0e0",
                        }
                    ),

                    # ML Position Optimization Toggle
                    dbc.Label("Enable ML Position Sizing"),
                    dbc.Switch(
                        id="bbg-ml-position-optimization",
                        value=False,
                        label="Enable ML-powered dynamic position sizing based on market conditions",
                        className="mb-3",
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "color": "#e0e0e0",
                        }
                    ),

                    # ML Adoption Rate (shown when ML is enabled)
                    html.Div([
                        dbc.Label("ML Adoption Rate"),
                        html.I(className="bi bi-info-circle ms-1", 
                              title="How much to trust ML vs traditional approach (0.0 = traditional only, 1.0 = full ML)",
                              style={"cursor": "pointer"}),
                        dbc.Input(id="bbg-ml-adoption-rate", type="range", 
                                 min=0.0, max=1.0, step=0.1, value=0.6,
                                 className="form-range mb-2"),
                        dbc.Row([
                            dbc.Col(html.Small("Traditional", className="text-muted"), width=4),
                            dbc.Col(html.Small("Balanced", className="text-muted"), width=4),
                            dbc.Col(html.Small("ML", className="text-muted"), width=4),
                        ]),
                        dbc.Input(id="bbg-ml-adoption-rate-text", type="number", 
                                 min=0.0, max=1.0, step=0.1, value=0.6, size="sm",
                                 className="mb-2"),
                    ], id="bbg-ml-adoption-container", style={"display": "none"}),
                    
                    # CC Optimization Settings (shown with ML)
                    html.Div([
                        dbc.Label("CC Optimization"),
                        dbc.Switch(
                            id="bbg-cc-optimization",
                            value=True,
                            label="Enable CC optimization for loss positions",
                            className="mb-3",
                            style={"color": "#e0e0e0"}
                        ),
                    ], id="bbg-cc-optimization-container", style={"display": "none"}),
                    
                    # Max Positions
                    dbc.Label("Max Positions"),
                    dbc.Input(id="bbg-max-positions", type="number", 
                             value=10, min=1, max=50, step=1, size="sm", className="mb-2"),
                    
                    # Rebalance Threshold
                    html.Div([
                        dbc.Label("Rebalance Threshold (%)"),
                        html.I(className="bi bi-question-circle ms-1", 
                              title="Switch symbols if better opportunity is X% higher scored",
                              style={"cursor": "pointer"}),
                    ]),
                    dbc.Input(id="bbg-rebalance-threshold", type="number", 
                             value=15, step=5, size="sm", 
                             className="mb-3"),

                    html.Hr(),
                    html.H6("Exit Conditions", className="fw-bold mb-2"),
                    html.P("Note: For Wheel strategy, consider using ML Roll Optimization instead of traditional stop loss.", 
                           className="text-muted small mb-2"),

                    # Profit Target
                    dbc.Label("Profit Target (% of premium)"),
                    dbc.Row([
                        dbc.Col(
                            dbc.Input(id="bbg-profit-target", type="number",
                                     value=50, step=10, size="sm"),
                           width=8
                        ),
                       dbc.Col(
                            dcc.Checklist(
                               id="bbg-disable-profit-target",
                               options=[{"label": "Disable", "value": True}],
                               value=[],
                                className="mt-2",
                                style={
                                    "color": "#e0e0e0",
                                    "backgroundColor": "rgba(255, 255, 255, 0.08)",
                                    "padding": "8px 12px",
                                    "borderRadius": "4px",
                                    "border": "1px solid rgba(255, 255, 255, 0.2)",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "8px"
                                }
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
                            dcc.Checklist(
                               id="bbg-disable-stop-loss",
                              options=[{"label": "Disable", "value": True}],
                              value=[],
                                className="mt-2",
                                style={
                                    "color": "#e0e0e0",
                                    "backgroundColor": "rgba(255, 255, 255, 0.08)",
                                    "padding": "8px 12px",
                                    "borderRadius": "4px",
                                    "border": "1px solid rgba(255, 255, 255, 0.2)",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "8px"
                                }
                            ),
                           width=4
                        ),
                    ], className="mb-3"),
                    
                    # Loading status display
                    html.Div(id="bbg-loading-status", className="mb-3"),
                    
                    # Run Button
                    dbc.Button([
                        html.I(className="bi bi-play-fill me-2"),
                        "Run Backtest",
                    ], id="bbg-run-btn", color="primary", className="w-100", size="lg"),

                    # Export button (shown after successful backtest)
                    html.Div(
                        id="bbg-export-container",
                        children=[
                            dbc.Button(
                                "📤 Export for AI Analysis",
                                id="bbg-export-btn",
                                color="info",
                                className="w-100 mt-2",
                                n_clicks=0,
                            ),
                        ],
                        className="d-none",  # Hidden by default
                    ),

                    # Download component
                    dcc.Download(id="bbg-download"),
                ]),
            ]),
        ], md=3),
        
        # Right column: Results (wider for better readability)
        dbc.Col([
            create_mag7_analysis_placeholder(),
            
            # Loading status indicator
            html.Div(id="bbg-loading-indicator", style={"display": "none"}, children=[
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="bi bi-hourglass-split me-2", style={"fontSize": "2rem"}),
                            html.H4("Running BinbinGod Backtest...", className="mt-2"),
                            html.P("This may take 30-60 seconds for MAG7 analysis", className="text-muted"),
                            dbc.Progress(value=100, striped=True, animated=True, className="mt-3"),
                        ], className="text-center py-4"),
                    ]),
                ], className="bg-dark border-success"),
            ], className="mb-3"),
            
            # Results Container with loading animation
            dcc.Loading(
                id="binbin-loading",
                type="circle",
                children=html.Div(id="binbin-results-container", children=[
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="bi bi-graph-up me-2"),
                                "Run a backtest to see results",
                            ], className="text-center text-muted py-5"),
                        ]),
                    ]),
                ]),
                overlay_style={"visibility": "visible", "opacity": 0.9, "backgroundColor": "#1a1a2e"},
            ),
            
            # Hidden store for results data
            dcc.Store(id="binbin-results-store", data={}),
            dcc.Store(id="binbin-params-store", data={}),  # Store params for export
        ], md=9),
    ]),
    
    # Custom CSS for loading animation and dropdown visibility
    dcc.Markdown('''
    <style>
    .dash-loading {
        background-color: #1a1a2e !important;
    }
    .dash-spinner circle {
        stroke: #4CAF50 !important;
        stroke-width: 4;
    }
    /* Ensure dropdown text is visible on dark backgrounds */
    #bbg-stock-pool .Select-control,
    #bbg-stock-pool .Select-menu-outer {
        background-color: #fff !important;
        color: #212529 !important;
    }
    #bbg-stock-pool .Select-value-label,
    #bbg-stock-pool .Select-placeholder {
        color: #212529 !important;
    }
    #bbg-stock-pool .Select-option {
        background-color: #fff !important;
        color: #212529 !important;
    }
    #bbg-stock-pool .Select-option:hover {
        background-color: #f0f0f0 !important;
    }
    #bbg-stock-pool .Select-option.is-selected {
        background-color: #e3f2fd !important;
        color: #212529 !important;
    }
    
    /* Improve checklist and switch visibility in dark theme */
    #bbg-use-synthetic label,
    #bbg-disable-profit-target label,
    #bbg-disable-stop-loss label {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    #bbg-use-synthetic input[type="checkbox"],
    #bbg-disable-profit-target input[type="checkbox"],
    #bbg-disable-stop-loss input[type="checkbox"] {
        accent-color: #4CAF50;
    }
    #bbg-ml-optimization label,
    #bbg-ml-dte-optimization label,
    #bbg-cc-optimization label {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    #bbg-ml-optimization input[type="checkbox"],
    #bbg-ml-dte-optimization input[type="checkbox"],
    #bbg-cc-optimization input[type="checkbox"] {
        accent-color: #4CAF50;
    }
    </style>
    ''', dangerously_allow_html=True)
])


# Callback to show/hide custom stocks input
@callback(
    Output("bbg-custom-stocks-container", "style"),
    Input("bbg-stock-pool", "value"),
)
def toggle_custom_stocks_input(stock_pool):
    """Show custom stocks input only when CUSTOM is selected."""
    if stock_pool == "CUSTOM":
        return {"display": "block"}
    return {"display": "none"}


# Callback to show/hide ML optimization controls and hide traditional params
@callback(
    Output("bbg-ml-adoption-container", "style"),
    Output("bbg-cc-optimization-container", "style"),
    Output("bbg-dte-range-container", "style"),
    Output("bbg-delta-targets-container", "style"),
    Input("bbg-ml-optimization", "value"),
    Input("bbg-ml-dte-optimization", "value"),
    Input("bbg-ml-roll-optimization", "value"),
    Input("bbg-ml-position-optimization", "value"),
)
def toggle_ml_controls(ml_delta_enabled, ml_dte_enabled, ml_roll_enabled, ml_position_enabled):
    """Show/hide ML optimization controls and hide traditional params when ML is enabled.

    When any ML optimization is enabled:
    - Show ML adoption rate and CC optimization controls
    - Hide traditional DTE range (ML will optimize DTE)
    - Hide traditional Delta targets (ML will optimize Delta)
    """
    ml_any_enabled = ml_delta_enabled or ml_dte_enabled or ml_roll_enabled or ml_position_enabled

    if ml_any_enabled:
        # Show ML controls
        ml_style = {"display": "block"}
        # Hide traditional params that ML will handle
        traditional_style = {"display": "none"}
        return ml_style, ml_style, traditional_style, traditional_style

    # Hide ML controls
    ml_style = {"display": "none"}
    # Show traditional params
    traditional_style = {"display": "block"}
    return ml_style, ml_style, traditional_style, traditional_style


# Callback to handle ML DTE optimization
@callback(
    Output("bbg-ml-dte-optimization", "value"),
    Input("bbg-ml-optimization", "value"),
)
def toggle_ml_dte_with_delta(ml_optimization_enabled):
    """Enable ML DTE optimization when ML delta optimization is enabled."""
    return ml_optimization_enabled


# Callback to sync ML adoption rate slider and text input
@callback(
    Output("bbg-ml-adoption-rate-text", "value"),
    Input("bbg-ml-adoption-rate", "value"),
)
def update_ml_adoption_rate_text(rate):
    """Update text input when range slider changes."""
    return rate


@callback(
    Output("bbg-ml-adoption-rate", "value"),
    Input("bbg-ml-adoption-rate-text", "value"),
)
def update_ml_adoption_rate_slider(rate_text):
    """Update range slider when text input changes."""
    try:
        rate = float(rate_text)
        return max(0.0, min(1.0, rate))  # Clamp to 0-1 range
    except (ValueError, TypeError):
        return 0.6  # Default value


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
    State("bbg-capital", "value"),
    State("bbg-leverage", "value"),
    State("bbg-use-synthetic", "value"),
    State("bbg-stock-pool", "value"),
    State("bbg-custom-stocks", "value"),
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
    State("bbg-ml-optimization", "value"),
    State("bbg-ml-dte-optimization", "value"),
    State("bbg-ml-roll-optimization", "value"),
    State("bbg-ml-position-optimization", "value"),
    State("bbg-ml-adoption-rate-text", "value"),
    State("bbg-cc-optimization", "value"),
    prevent_initial_call=True,
)
def run_binbin_backtest(
    n_clicks, start_date, end_date, capital, leverage, use_synthetic,
    stock_pool, custom_stocks, dte_min, dte_max, put_delta, call_delta,
    max_positions, rebalance_threshold, profit_target, stop_loss,
    disable_profit_target, disable_stop_loss,
    ml_optimization, ml_dte_optimization, ml_roll_optimization, ml_position_optimization, ml_adoption_rate, cc_optimization
):
    """Run Binbin God strategy backtest."""
    if not start_date or not end_date:
        return no_update, no_update, no_update, no_update, no_update, no_update

    services = get_services_cached()
    if not services:
        return {}, {}, html.Div(), html.P("Services not initialized", className="text-warning"), {"display": "none"}, "d-none"

    engine = services["backtest_engine"]
    
    # Prepare parameters
    # Checklist returns list, check if True is in the list
    profit_target_value = 999999 if (disable_profit_target and True in disable_profit_target) else (profit_target or 50)
    stop_loss_value = 999999 if (disable_stop_loss and True in disable_stop_loss) else (stop_loss or 200)
    
    # Resolve stock pool
    if stock_pool == "MAG7":
        stock_symbols = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
        symbol = "MAG7_AUTO"
    elif stock_pool == "MAGAMG":
        stock_symbols = ["MSFT", "AAPL", "GOOGL", "AMZN"]  # MAGAMG: 4 major tech
        symbol = "MAGAMG_AUTO"
    elif stock_pool == "CUSTOM" and custom_stocks:
        # Parse custom stocks
        stock_symbols = [s.strip().upper() for s in custom_stocks.split(",") if s.strip()]
        if not stock_symbols:
            stock_symbols = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]  # Fallback
        symbol = f"CUSTOM_{'_'.join(stock_symbols[:3])}"  # Truncate for display
    else:
        stock_symbols = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
        symbol = "MAG7_AUTO"
    
    params = {
        "strategy": "binbin_god",
        "symbol": symbol,
        "stock_pool": stock_symbols,  # Pass actual stock list to strategy
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": capital or 150000,
        "max_leverage": leverage or 1.0,
        "use_synthetic_data": bool(use_synthetic and True in use_synthetic),
        "dte_min": dte_min or 30,
        "dte_max": dte_max or 45,
        "delta_target": 0.30,
        "profit_target_pct": profit_target_value,
        "stop_loss_pct": stop_loss_value,
        "put_delta": put_delta or 0.30,
        "call_delta": call_delta or 0.30,
        "max_positions": max_positions or 10,
        "rebalance_threshold": (rebalance_threshold or 15) / 100.0,
        # ML Delta Optimization parameters
        "ml_delta_optimization": ml_optimization or False,
        "ml_dte_optimization": ml_dte_optimization or False,
        "ml_roll_optimization": ml_roll_optimization or False,
        "ml_position_optimization": ml_position_optimization or False,
        "ml_adoption_rate": ml_adoption_rate or 0.6,
        "cc_optimization_enabled": cc_optimization if cc_optimization is not None else True,
        "cc_min_delta_cost": 0.15,
        "cc_cost_basis_threshold": 0.05,
        "cc_min_strike_premium": 0.02,
    }
    
    try:
        result = engine.run(params)
    except Exception as e:
        return {}, {}, html.Div(), html.P(f"Backtest error: {e}", className="text-danger"), {"display": "none"}, "d-none"

    if not result:
        return {}, {}, html.Div(), html.P("No results generated", className="text-muted"), {"display": "none"}, "d-none"
    
    # Build results UI (same structure as backtester.py)
    metrics = result.get("metrics", {})
    trades = result.get("trades", [])
    daily_pnl = result.get("daily_pnl", [])
    
    # Metric cards (same as backtester)
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
    
    # P&L chart with benchmark support
    pnl_dates = [p["date"] for p in daily_pnl] if daily_pnl else []
    pnl_values = [p["cumulative_pnl"] for p in daily_pnl] if daily_pnl else []
    initial_capital = params.get("initial_capital", 150000)
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
    multi_stock_prices = result.get("multi_stock_prices", {})
    trade_timeline = dcc.Graph(figure=create_trade_timeline_chart(
        trades=trades,
        daily_pnl=daily_pnl,
        underlying_prices=underlying_prices,
        multi_stock_prices=multi_stock_prices,
        title="BinbinGod Trade Timeline: Entry/Exit Points & Performance"
    ))
    
    # Trades table (same columns as backtester)
    trade_columns = [
        {"headerName": "Entry", "field": "entry_date", "width": 100, "sort": "desc"},
        {"headerName": "Exit", "field": "exit_date", "width": 100},
        {"headerName": "Option Contract", "field": "contract_name", "width": 220},
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
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)",
         "cellStyle": {"function": "params.value >= 0 ? {'color': '#26a69a'} : {'color': '#ef5350'}"}}},
        {"headerName": "Reason", "field": "exit_reason", "width": 120},
    ]
    trades_table = create_data_table(trades, trade_columns, "bbg-trades-table", height=450) if trades else html.P("No trades", className="text-muted")
    
    # Additional metrics row
    extra_row = dbc.Row([
        dbc.Col(metric_card("Avg Profit", f"${metrics.get('avg_profit', 0):,.2f}", "success"), md=3),
        dbc.Col(metric_card("Avg Loss", f"${metrics.get('avg_loss', 0):,.2f}", "danger"), md=3),
        dbc.Col(metric_card("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}", "primary"), md=3),
        dbc.Col(metric_card("Sortino", f"{metrics.get('sortino_ratio', 0):.2f}", "info"), md=3),
    ], className="mb-3 g-3")
    
    # Holdings card
    holdings_card = html.Div()
    strategy_performance = result.get("strategy_performance", {})
    if strategy_performance:
        # Handle both nested (Wheel) and flat (BinbinGod) data structures
        current_state = strategy_performance.get("current_state", {})
        shares = strategy_performance.get("shares_held") or current_state.get("shares_held", 0)
        cost = strategy_performance.get("cost_basis") or current_state.get("cost_basis", 0)
        options = strategy_performance.get("open_positions", [])
        
        # Debug logging
        from app.services import get_services
        services = get_services()
        if services:
            logger = services.get("logger")
            if logger:
                logger.info(f"BinbinGod Holdings: shares={shares}, cost_basis={cost}, options={len(options)}")
        
        holdings_data = {
            "shares_held": shares,
            "cost_basis": cost,
            "options_held": options,
        }
        holdings_card = create_holdings_card(holdings_data)
    
    # Monitoring dashboard for wheel logic
    monitoring_section = html.Div()
    if strategy_performance and params.get("strategy") == "binbin_god":
        # Add metrics from engine result for performance card
        monitoring_data = {**strategy_performance, "metrics": metrics}
        monitoring_section = create_monitoring_dashboard(monitoring_data)
    
    # MAG7 analysis section (unique to BinbinGod)
    mag7_analysis = result.get("mag7_analysis", {})
    if mag7_analysis and "ranked_stocks" in mag7_analysis:
        ranked = mag7_analysis["ranked_stocks"]
        best = mag7_analysis.get("best_pick", {})
        
        mag7_section = html.Div([
            html.H5("MAG7 Stock Analysis", className="mt-4 mb-3"),
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
                    "PE": s.get("pe_ratio", "N/A"),
                    "IV Rank": s.get("iv_rank", "N/A"),
                    "Momentum": f"{s.get('momentum', 0):.1f}",
                    "Stability": f"{s.get('stability', 0):.1f}",
                } for i, s in enumerate(ranked)]),
                striped=True,
                hover=True,
                bordered=True,
                className="table-sm",
            ),
        ])
    else:
        mag7_section = html.Div()
    
    # Build complete content
    content = html.Div([
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
    ])
    
    # Add trade history and phase transitions if available
    if strategy_performance:
        trade_history = strategy_performance.get("trade_history", [])
        phase_history = strategy_performance.get("phase_history", [])
        
        if trade_history:
            content.children.append(html.Div([
                html.H5("Recent Trade History", className="mt-4 mb-3"),
                create_trade_history_table(trade_history),
            ]))
        
        if phase_history:
            content.children.append(html.Div([
                html.H5("Phase Transition Log", className="mt-4 mb-3"),
                create_phase_transition_log(phase_history),
            ]))
    
    # Hide loading indicator when backtest completes
    # Return: result(store), params(store), mag7_analysis(display in header), content(main), loading_style, export_btn_class
    return result, params, mag7_section, content, {"display": "none"}, "d-block mt-2"


@callback(
    Output("bbg-download", "data"),
    Input("bbg-export-btn", "n_clicks"),
    State("binbin-results-store", "data"),
    State("binbin-params-store", "data"),
    prevent_initial_call=True,
)
def export_binbin_backtest_result(n_clicks, result, params):
    """Export the Binbin God backtest result as JSON for AI analysis."""
    import json
    from datetime import datetime

    if not n_clicks or not result or not params:
        return no_update

    # Build export data structure for AI analysis
    metrics = result.get("metrics", {})
    trades = result.get("trades", [])
    daily_pnl = result.get("daily_pnl", [])
    strategy_performance = result.get("strategy_performance", {})

    export_data = {
        "export_info": {
            "exported_at": datetime.utcnow().isoformat(),
            "export_version": "1.0",
            "purpose": "AI analysis and debugging"
        },
        "backtest_summary": {
            "strategy": params.get("strategy"),
            "symbol": params.get("symbol"),
            "stock_pool": params.get("stock_pool"),
            "period": {
                "start_date": params.get("start_date"),
                "end_date": params.get("end_date"),
            },
            "capital": {
                "initial": params.get("initial_capital"),
            },
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
        "trades": trades,
        "daily_pnl": daily_pnl,
        "strategy_performance": strategy_performance,
    }

    # Generate filename
    symbol = params.get("symbol", "UNKNOWN")
    strategy = params.get("strategy", "unknown")
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_{strategy}_{symbol}_{date_str}.json"

    return dict(
        content=json.dumps(export_data, indent=2, ensure_ascii=False, default=str),
        filename=filename,
        mime_type="application/json",
    )

