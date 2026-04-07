"""Options Chain page: view options chain with Greeks, IV smile chart."""

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from app.components.tables import create_data_table
from app.services import get_services

dash.register_page(__name__, path="/options-chain", name="Options Chain", order=3)

layout = html.Div([
    html.H3("Options Chain", className="mb-3"),

    # Controls
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.Input(id="oc-symbol", placeholder="Symbol (e.g. AAPL)", type="text"),
                dbc.Button("Load Chain", id="oc-load-btn", color="primary", n_clicks=0),
            ]),
        ], md=3),
        dbc.Col([
            dbc.Select(id="oc-expiry-select", placeholder="Select expiration...", options=[]),
        ], md=3),
        dbc.Col([
            dbc.RadioItems(
                id="oc-right-filter",
                options=[
                    {"label": "All", "value": ""},
                    {"label": "Puts", "value": "P"},
                    {"label": "Calls", "value": "C"},
                ],
                value="",
                inline=True,
                className="mt-2",
            ),
        ], md=3),
    ], className="mb-3"),

    # Chain table
    dcc.Loading(
        html.Div(id="oc-chain-container", children=[
            html.P("Enter a symbol and click 'Load Chain'", className="text-muted"),
        ]),
        type="circle",
        show_initially=False,
    ),

    # IV Smile chart
    html.Div(id="oc-smile-container", className="mt-4"),

    # Stores
    dcc.Store(id="oc-params-store", data=[]),
    dcc.Store(id="oc-chain-store", data=[]),
])


@callback(
    Output("oc-params-store", "data"),
    Output("oc-expiry-select", "options"),
    Output("oc-expiry-select", "value"),
    Input("oc-load-btn", "n_clicks"),
    State("oc-symbol", "value"),
    prevent_initial_call=True,
)
def load_option_params(n_clicks, symbol):
    if not symbol:
        return no_update, no_update, no_update
    symbol = symbol.strip().upper()

    services = get_services()
    if not services or not services["conn_mgr"].is_connected:
        return [], [], None

    data_client = services["data_client"]
    try:
        params = data_client.get_option_chain_params(symbol)
    except Exception:
        return [], [], None

    if not params:
        return [], [], None

    expirations = params[0].get("expirations", []) if params else []
    options = [{"label": exp, "value": exp} for exp in expirations[:20]]
    default = expirations[0] if expirations else None
    return params, options, default


@callback(
    Output("oc-chain-store", "data"),
    Output("oc-chain-container", "children"),
    Input("oc-expiry-select", "value"),
    Input("oc-right-filter", "value"),
    State("oc-symbol", "value"),
    prevent_initial_call=True,
)
def load_chain(expiry, right_filter, symbol):
    if not symbol or not expiry:
        return no_update, no_update
    symbol = symbol.strip().upper()

    services = get_services()
    if not services or not services["conn_mgr"].is_connected:
        return [], html.P("Please connect to IBKR", className="text-warning")

    data_client = services["data_client"]
    try:
        chain = data_client.get_option_chain(symbol, expiry, right=right_filter)
    except Exception as e:
        return [], html.P(f"Error: {e}", className="text-danger")

    if not chain:
        return [], html.P("No option data available for this expiry", className="text-muted")

    columns = [
        {"headerName": "Strike", "field": "strike", "width": 90, "pinned": "left",
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Type", "field": "right", "width": 70},
        {"headerName": "Bid", "field": "bid", "width": 80,
         "valueFormatter": {"function": "params.value != null ? d3.format(',.2f')(params.value) : '-'"}},
        {"headerName": "Ask", "field": "ask", "width": 80,
         "valueFormatter": {"function": "params.value != null ? d3.format(',.2f')(params.value) : '-'"}},
        {"headerName": "Last", "field": "last", "width": 80,
         "valueFormatter": {"function": "params.value != null ? d3.format(',.2f')(params.value) : '-'"}},
        {"headerName": "Volume", "field": "volume", "width": 80},
        {"headerName": "OI", "field": "openInterest", "width": 80},
        {"headerName": "IV", "field": "impliedVol", "width": 80,
         "valueFormatter": {"function": "params.value != null ? d3.format('.1%')(params.value) : '-'"}},
        {"headerName": "Delta", "field": "delta", "width": 80,
         "valueFormatter": {"function": "params.value != null ? d3.format('+.3f')(params.value) : '-'"}},
        {"headerName": "Gamma", "field": "gamma", "width": 80,
         "valueFormatter": {"function": "params.value != null ? d3.format('.4f')(params.value) : '-'"}},
        {"headerName": "Theta", "field": "theta", "width": 80,
         "valueFormatter": {"function": "params.value != null ? d3.format('.4f')(params.value) : '-'"}},
        {"headerName": "Vega", "field": "vega", "width": 80,
         "valueFormatter": {"function": "params.value != null ? d3.format('.4f')(params.value) : '-'"}},
    ]

    table = create_data_table(chain, columns, "options-chain-table", height=450)
    return chain, table


@callback(
    Output("oc-smile-container", "children"),
    Input("oc-chain-store", "data"),
    State("oc-symbol", "value"),
)
def update_iv_smile(chain, symbol):
    if not chain:
        return html.Div()

    symbol = (symbol or "").strip().upper()

    puts = [r for r in chain if r.get("right") == "P" and r.get("impliedVol")]
    calls = [r for r in chain if r.get("right") == "C" and r.get("impliedVol")]

    fig = go.Figure()
    if puts:
        puts_sorted = sorted(puts, key=lambda x: x["strike"])
        fig.add_trace(go.Scatter(
            x=[p["strike"] for p in puts_sorted],
            y=[p["impliedVol"] * 100 for p in puts_sorted],
            name="Put IV",
            mode="lines+markers",
            line=dict(color="#ef5350"),
        ))
    if calls:
        calls_sorted = sorted(calls, key=lambda x: x["strike"])
        fig.add_trace(go.Scatter(
            x=[c["strike"] for c in calls_sorted],
            y=[c["impliedVol"] * 100 for c in calls_sorted],
            name="Call IV",
            mode="lines+markers",
            line=dict(color="#26a69a"),
        ))

    fig.update_layout(
        title=f"{symbol} Volatility Smile",
        template="plotly_dark",
        xaxis_title="Strike",
        yaxis_title="Implied Volatility (%)",
        height=350,
        margin=dict(l=50, r=50, t=50, b=30),
    )
    return dcc.Graph(figure=fig)
