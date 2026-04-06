"""Top-level layout: navbar + page container with manual URL routing."""

from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
from app.components.navbar import create_navbar

# Language translations dictionary
TRANSLATIONS = {
    "en": {
        "dashboard": "Dashboard",
        "market_data": "Market Data",
        "screener": "Screener",
        "options_chain": "Options Chain",
        "backtester": "Backtester",
        "settings": "Settings",
        "language": "Language",
    },
    "zh": {
        "dashboard": "仪表盘",
        "market_data": "市场数据",
        "screener": "选股器",
        "options_chain": "期权链",
        "backtester": "回测器",
        "settings": "设置",
        "language": "语言",
    }
}


def create_layout():
    return html.Div([
        # URL location for routing
        dcc.Location(id="url", refresh=False),

        # Stores for shared state
        dcc.Store(id="connection-state-store", data={"state": "disconnected", "message": ""}),
        dcc.Store(id="account-store", data={}),
        dcc.Store(id="positions-store", data=[]),
        
        # Language store - default to English
        dcc.Store(id="language-store", data="en"),

        # Interval for periodic data refresh
        dcc.Interval(id="global-interval", interval=5000, n_intervals=0),

        # Navbar with language selector
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
    from app.pages import (
        dashboard, market_data, screener, options_chain, backtester, 
        settings, backtest_history, binbin_god, binbin_god_live
    )

    routes = {
        "/": dashboard.layout,
        "/market-data": market_data.layout,
        "/screener": screener.layout,
        "/options-chain": options_chain.layout,
        "/backtester": backtester.layout,
        "/backtest-history": backtest_history.layout,
        "/binbin-god": binbin_god.layout,
        "/binbin-god-live": binbin_god_live.layout,
        "/settings": settings.layout,
    }

    if pathname in routes:
        layout = routes[pathname]
        return layout() if callable(layout) else layout

    # 404 fallback
    return html.Div([
        html.H3("404 - Page Not Found", className="text-danger"),
        html.P(f"The path '{pathname}' does not exist."),
        dbc.Button("Go to Dashboard", href="/", color="primary"),
    ], className="text-center mt-5")


@callback(
    Output("language-store", "data"),
    Input("language-selector", "value"),
    prevent_initial_call=True,
)
def update_language(selected_lang):
    """Update language store when user selects a language."""
    return selected_lang if selected_lang else "en"
