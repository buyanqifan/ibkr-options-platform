"""Settings page: IBKR connection configuration, trading mode, cache management."""

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc
from app.components.connection_status import connection_badge
from app.services import get_services

dash.register_page(__name__, path="/settings", name="Settings", order=5)

layout = html.Div([
    html.H3("Settings", className="mb-4"),

    dbc.Row([
        # Connection settings
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("IBKR Connection"),
                dbc.CardBody([
                    dbc.Label("Host"),
                    dbc.Input(id="set-host", value="ibgateway", className="mb-2"),

                    dbc.Label("Port"),
                    dbc.Input(id="set-port", type="number", value=4002, className="mb-2"),

                    dbc.Label("Client ID"),
                    dbc.Input(id="set-client-id", type="number", value=1, className="mb-3"),

                    dbc.Label("Trading Mode"),
                    dbc.RadioItems(
                        id="set-trading-mode",
                        options=[
                            {"label": "Paper Trading", "value": "paper"},
                            {"label": "Live Trading", "value": "live"},
                        ],
                        value="paper",
                        className="mb-3",
                    ),

                    html.Div(id="set-conn-status", className="mb-3"),

                    dbc.ButtonGroup([
                        dbc.Button("Connect", id="set-connect-btn", color="success", n_clicks=0),
                        dbc.Button("Disconnect", id="set-disconnect-btn", color="danger", outline=True, n_clicks=0),
                    ], className="w-100"),

                    html.Div(id="set-conn-message", className="mt-2"),
                ]),
            ], className="shadow-sm"),
        ], md=4),

        # Data preferences
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Preferences"),
                dbc.CardBody([
                    dbc.Label("Default Exchange"),
                    dbc.Select(
                        id="set-exchange",
                        options=[
                            {"label": "SMART (Best Execution)", "value": "SMART"},
                            {"label": "NYSE", "value": "NYSE"},
                            {"label": "NASDAQ", "value": "NASDAQ"},
                        ],
                        value="SMART",
                        className="mb-3",
                    ),

                    dbc.Label("Currency"),
                    dbc.Select(
                        id="set-currency",
                        options=[
                            {"label": "USD", "value": "USD"},
                            {"label": "EUR", "value": "EUR"},
                            {"label": "GBP", "value": "GBP"},
                        ],
                        value="USD",
                        className="mb-3",
                    ),

                    html.Hr(),

                    html.H6("Cache Management", className="fw-bold mb-3"),
                    html.Div(id="set-cache-info", className="mb-3"),
                    dbc.Button(
                        "Clear All Cache", id="set-clear-cache-btn",
                        color="warning", outline=True, n_clicks=0,
                    ),
                    html.Div(id="set-cache-message", className="mt-2"),
                ]),
            ], className="shadow-sm"),
        ], md=4),

        # System info
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("System Info"),
                dbc.CardBody(id="set-system-info"),
            ], className="shadow-sm"),
        ], md=4),
    ]),

    dcc.Interval(id="set-interval", interval=3000, n_intervals=0),
])


@callback(
    Output("set-conn-status", "children"),
    Output("set-system-info", "children"),
    Input("set-interval", "n_intervals"),
)
def update_status(n):
    services = get_services()

    if not services:
        return (
            connection_badge("disconnected", "Services not initialized"),
            html.P("Initializing...", className="text-muted"),
        )

    conn_mgr = services["conn_mgr"]
    status = conn_mgr.status

    from config.settings import settings
    info = html.Div([
        html.P([html.Strong("IBKR Host: "), settings.IBKR_HOST]),
        html.P([html.Strong("IBKR Port: "), str(settings.IBKR_PORT)]),
        html.P([html.Strong("Mode: "), settings.IBKR_TRADING_MODE]),
        html.P([html.Strong("Account: "), status.account or "N/A"]),
        html.P([html.Strong("Server Version: "), str(status.server_version) if status.server_version else "N/A"]),
        html.P([html.Strong("DB Path: "), settings.DB_PATH]),
    ])

    return connection_badge(status.state.value, status.message), info


@callback(
    Output("set-conn-message", "children"),
    Input("set-connect-btn", "n_clicks"),
    Input("set-disconnect-btn", "n_clicks"),
    State("set-host", "value"),
    State("set-port", "value"),
    State("set-client-id", "value"),
    prevent_initial_call=True,
)
def handle_connection(connect_clicks, disconnect_clicks, host, port, client_id):
    from dash import ctx
    services = get_services()
    if not services:
        return dbc.Alert("Services not initialized", color="warning")

    conn_mgr = services["conn_mgr"]
    triggered = ctx.triggered_id

    if triggered == "set-connect-btn":
        success = conn_mgr.connect(host=host, port=int(port or 4002), client_id=int(client_id or 1))
        if success:
            return dbc.Alert("Connected successfully!", color="success", duration=3000)
        return dbc.Alert(f"Connection failed: {conn_mgr.status.message}", color="danger")
    elif triggered == "set-disconnect-btn":
        conn_mgr.disconnect()
        return dbc.Alert("Disconnected", color="info", duration=3000)

    return no_update


@callback(
    Output("set-cache-message", "children"),
    Input("set-clear-cache-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_cache(n):
    services = get_services()
    if not services:
        return no_update
    services["cache"].clear_all()
    return dbc.Alert("Cache cleared!", color="success", duration=3000)
