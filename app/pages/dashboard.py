"""Dashboard page: overview, connection status, account summary, positions."""

import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
from app.components.tables import metric_card, create_data_table
from app.components.connection_status import connection_badge
from app.services import get_services

dash.register_page(__name__, path="/", name="Dashboard", order=0)


layout = html.Div([
    html.H3("Dashboard", className="mb-3"),

    # Connection status row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("IBKR Connection"),
                dbc.CardBody([
                    html.Div(id="dashboard-conn-status"),
                    html.P(id="dashboard-conn-msg", className="text-muted mt-2 mb-0 small"),
                ]),
            ], className="shadow-sm"),
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trading Mode"),
                dbc.CardBody([
                    html.H4(id="dashboard-trading-mode", className="mb-0"),
                ]),
            ], className="shadow-sm"),
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Account"),
                dbc.CardBody([
                    html.H4(id="dashboard-account-id", className="mb-0"),
                ]),
            ], className="shadow-sm"),
        ], md=4),
    ], className="mb-4"),

    # Account metrics
    html.H5("Account Summary", className="mb-3"),
    dbc.Row(id="dashboard-metrics-row", className="mb-4 g-3"),

    # Positions table
    html.H5("Current Positions", className="mb-3"),
    html.Div(id="dashboard-positions-table"),

    # Auto-refresh
    dcc.Interval(id="dashboard-interval", interval=5000, n_intervals=0),
])


@callback(
    Output("dashboard-conn-status", "children"),
    Output("dashboard-conn-msg", "children"),
    Output("dashboard-trading-mode", "children"),
    Output("dashboard-account-id", "children"),
    Input("dashboard-interval", "n_intervals"),
)
def update_connection_info(n):
    services = get_services()
    if not services:
        return (
            connection_badge("disconnected"),
            "Services not initialized",
            "N/A",
            "N/A",
        )
    conn_mgr = services["conn_mgr"]
    status = conn_mgr.status
    from config.settings import settings
    mode = "Paper" if settings.IBKR_TRADING_MODE == "paper" else "Live"
    return (
        connection_badge(status.state.value, status.message),
        status.message,
        dbc.Badge(mode, color="info" if mode == "Paper" else "danger", className="fs-5"),
        status.account or "N/A",
    )


@callback(
    Output("dashboard-metrics-row", "children"),
    Input("dashboard-interval", "n_intervals"),
)
def update_account_metrics(n):
    services = get_services()
    if not services:
        return []
    data_client = services["data_client"]
    conn_mgr = services["conn_mgr"]
    if not conn_mgr.is_connected:
        return [dbc.Col(html.P("Connect to IBKR to view account data", className="text-muted"))]

    summary = data_client.get_account_summary()
    if not summary:
        return [dbc.Col(html.P("Loading account data...", className="text-muted"))]

    cards = [
        dbc.Col(metric_card("Net Liquidation", f"${summary.get('NetLiquidation', 0):,.2f}", "primary", "bi-wallet2"), md=2),
        dbc.Col(metric_card("Cash", f"${summary.get('TotalCashValue', 0):,.2f}", "success", "bi-cash-stack"), md=2),
        dbc.Col(metric_card("Buying Power", f"${summary.get('BuyingPower', 0):,.2f}", "info", "bi-lightning"), md=2),
        dbc.Col(metric_card("Position Value", f"${summary.get('GrossPositionValue', 0):,.2f}", "warning", "bi-pie-chart"), md=2),
        dbc.Col(metric_card("Unrealized P&L", f"${summary.get('UnrealizedPnL', 0):,.2f}",
                           "success" if summary.get('UnrealizedPnL', 0) >= 0 else "danger", "bi-graph-up"), md=2),
        dbc.Col(metric_card("Realized P&L", f"${summary.get('RealizedPnL', 0):,.2f}",
                           "success" if summary.get('RealizedPnL', 0) >= 0 else "danger", "bi-check-circle"), md=2),
    ]
    return cards


@callback(
    Output("dashboard-positions-table", "children"),
    Input("dashboard-interval", "n_intervals"),
)
def update_positions(n):
    services = get_services()
    if not services:
        return html.P("Connect to IBKR to view positions", className="text-muted")
    data_client = services["data_client"]
    conn_mgr = services["conn_mgr"]
    if not conn_mgr.is_connected:
        return html.P("Connect to IBKR to view positions", className="text-muted")

    positions = data_client.get_positions()
    if not positions:
        return html.P("No open positions", className="text-muted")

    columns = [
        {"headerName": "Symbol", "field": "symbol", "width": 100},
        {"headerName": "Type", "field": "secType", "width": 80},
        {"headerName": "Expiry", "field": "expiry", "width": 110},
        {"headerName": "Strike", "field": "strike", "width": 90, "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Right", "field": "right", "width": 70},
        {"headerName": "Qty", "field": "position", "width": 70},
        {"headerName": "Avg Cost", "field": "avgCost", "width": 100, "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Mkt Value", "field": "marketValue", "width": 110, "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Unrealized P&L", "field": "unrealizedPNL", "width": 130,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"},
         "cellStyle": {"function": "params.value >= 0 ? {'color': '#26a69a'} : {'color': '#ef5350'}"}},
    ]
    return create_data_table(positions, columns, "positions-table", height=300)
