"""Top-level layout: navbar + page container with manual URL routing."""

from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
from app.components.navbar import create_navbar


def create_layout():
    return html.Div([
        # URL location for routing
        dcc.Location(id="url", refresh=False),

        # Stores for shared state
        dcc.Store(id="connection-state-store", data={"state": "disconnected", "message": ""}),
        dcc.Store(id="account-store", data={}),
        dcc.Store(id="positions-store", data=[]),

        # Interval for periodic data refresh
        dcc.Interval(id="global-interval", interval=5000, n_intervals=0),

        # Navbar
        create_navbar(),

        # Page content (manually routed)
        dbc.Container(
            id="page-content",
            fluid=True,
            className="px-4",
        ),
    ])


@callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    """Route to the appropriate page based on URL pathname."""
    from app.pages import dashboard, market_data, screener, options_chain, backtester, settings

    routes = {
        "/": dashboard.layout,
        "/market-data": market_data.layout,
        "/screener": screener.layout,
        "/options-chain": options_chain.layout,
        "/backtester": backtester.layout,
        "/settings": settings.layout,
    }

    if pathname in routes:
        return routes[pathname]

    # 404 fallback
    return html.Div([
        html.H3("404 - Page Not Found", className="text-danger"),
        html.P(f"The path '{pathname}' does not exist."),
        dbc.Button("Go to Dashboard", href="/", color="primary"),
    ], className="text-center mt-5")
