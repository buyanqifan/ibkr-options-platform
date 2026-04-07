"""Market Data page: stock search, candlestick chart, technical indicators, real-time quotes."""

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc
from app.components.charts import create_candlestick_chart
from app.components.tables import metric_card
from app.services import get_services

dash.register_page(__name__, path="/market-data", name="Market Data", order=1)

layout = html.Div([
    html.H3("Market Data", className="mb-3"),

    # Search bar
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.Input(id="md-symbol-input", placeholder="Enter symbol (e.g. AAPL)", type="text", value=""),
                dbc.Select(
                    id="md-timeframe",
                    options=[
                        {"label": "1 Day", "value": "1 D|1 min"},
                        {"label": "1 Week", "value": "1 W|5 mins"},
                        {"label": "1 Month", "value": "1 M|15 mins"},
                        {"label": "3 Months", "value": "3 M|1 hour"},
                        {"label": "1 Year", "value": "1 Y|1 day"},
                        {"label": "5 Years", "value": "5 Y|1 week"},
                    ],
                    value="1 Y|1 day",
                    style={"maxWidth": "160px"},
                ),
                dbc.Button("Load", id="md-load-btn", color="primary", n_clicks=0),
            ]),
        ], md=6),
        dbc.Col([
            dbc.Checklist(
                id="md-indicators",
                options=[
                    {"label": " MA20", "value": 20},
                    {"label": " MA50", "value": 50},
                    {"label": " MA200", "value": 200},
                ],
                value=[20, 50],
                inline=True,
                className="mt-2",
            ),
        ], md=6),
    ], className="mb-3"),

    # Real-time quote panel
    dbc.Row(id="md-quote-row", className="mb-3 g-3"),

    # Chart
    dcc.Loading(
        dcc.Graph(id="md-candlestick-chart", figure=create_candlestick_chart([])),
        type="circle",
        show_initially=False,
    ),

    # Hidden store for loaded data
    dcc.Store(id="md-bars-store", data=[]),
])


@callback(
    Output("md-bars-store", "data"),
    Output("md-quote-row", "children"),
    Input("md-load-btn", "n_clicks"),
    State("md-symbol-input", "value"),
    State("md-timeframe", "value"),
    prevent_initial_call=True,
)
def load_market_data(n_clicks, symbol, timeframe):
    if not symbol:
        return no_update, no_update
    symbol = symbol.strip().upper()
    duration, bar_size = timeframe.split("|")

    services = get_services()
    if not services or not services["conn_mgr"].is_connected:
        return [], [dbc.Col(html.P("Please connect to IBKR first", className="text-warning"))]

    data_client = services["data_client"]

    # Fetch bars
    try:
        bars = data_client.get_historical_bars(symbol, duration, bar_size)
    except Exception as e:
        return [], [dbc.Col(html.P(f"Error loading data: {e}", className="text-danger"))]

    # Fetch real-time quote
    try:
        quote = data_client.get_realtime_quote(symbol)
    except Exception:
        quote = {}

    quote_cards = []
    if quote:
        last = quote.get("last") or quote.get("close") or 0
        prev_close = quote.get("close") or 0
        change = last - prev_close if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        color = "success" if change >= 0 else "danger"

        quote_cards = [
            dbc.Col(metric_card("Last", f"${last:,.2f}", color), md=2),
            dbc.Col(metric_card("Change", f"${change:+,.2f} ({change_pct:+.2f}%)", color), md=2),
            dbc.Col(metric_card("Bid", f"${quote.get('bid', 0) or 0:,.2f}", "info"), md=2),
            dbc.Col(metric_card("Ask", f"${quote.get('ask', 0) or 0:,.2f}", "info"), md=2),
            dbc.Col(metric_card("Volume", f"{quote.get('volume', 0):,.0f}", "secondary"), md=2),
            dbc.Col(metric_card("High / Low", f"${quote.get('high', 0) or 0:,.2f} / ${quote.get('low', 0) or 0:,.2f}", "warning"), md=2),
        ]

    return bars, quote_cards


@callback(
    Output("md-candlestick-chart", "figure"),
    Input("md-bars-store", "data"),
    Input("md-indicators", "value"),
    State("md-symbol-input", "value"),
)
def update_chart(bars, indicators, symbol):
    symbol = (symbol or "").strip().upper()
    return create_candlestick_chart(bars, symbol=symbol, ma_periods=indicators)
