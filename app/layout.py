"""Top-level layout with resilient manual routing over Dash page registry."""

import traceback

import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
from flask import has_request_context, request
from urllib.parse import urlparse
from app.components.navbar import create_navbar


def _page_layout(key: str):
    page = dash.page_registry.get(key, {})
    return page.get("layout", html.Div(f"Missing page: {key}", className="text-danger"))


_ROUTE_KEYS = {
    "/": "pages.dashboard",
    "/market-data": "pages.market_data",
    "/screener": "pages.screener",
    "/options-chain": "pages.options_chain",
    "/backtester": "pages.backtester",
    "/backtest-history": "pages.backtest_history",
    "/binbin-god": "pages.binbin_god",
    "/binbin-god-live": "pages.binbin_god_live",
    "/settings": "pages.settings",
}

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
    initial_path = "/"
    if has_request_context():
        request_path = request.path or "/"
        if request_path in _ROUTE_KEYS:
            initial_path = request_path
        else:
            referer = request.headers.get("Referer", "")
            if referer:
                referer_path = urlparse(referer).path or "/"
                if referer_path in _ROUTE_KEYS:
                    initial_path = referer_path

    return html.Div([
        dcc.Location(id="url", refresh=True),
        dcc.Store(id="connection-state-store", data={"state": "disconnected", "message": ""}),
        dcc.Store(id="account-store", data={}),
        dcc.Store(id="positions-store", data=[]),
        dcc.Store(id="language-store", data="en", storage_type="local"),
        dcc.Interval(id="global-interval", interval=5000, n_intervals=0),
        create_navbar(),
        dbc.Container(
            display_page(initial_path),
            id="page-content",
            fluid=True,
            className="px-4",
        ),
        dcc.Markdown(
            """
<style>
._dash-loading,
._dash-loading-callback {
    pointer-events: none !important;
}
</style>
            """,
            dangerously_allow_html=True,
        ),
    ])


def _render_page(pathname):
    if pathname not in _ROUTE_KEYS:
        return html.Div([
            html.H3("404 - Page Not Found", className="text-danger"),
            html.P(f"The path '{pathname}' does not exist."),
            dbc.Button("Go to Dashboard", href="/", color="primary"),
        ], className="text-center mt-5")

    page_key = _ROUTE_KEYS[pathname]
    layout = _page_layout(page_key)
    if getattr(layout, "children", None) == f"Missing page: {page_key}":
        registered = sorted(dash.page_registry.keys())
        return dbc.Alert(
            [
                html.H5("Page registration error", className="alert-heading"),
                html.P(f"Missing page key: {page_key}"),
                html.P(f"Registered pages: {', '.join(registered) if registered else '(none)'}"),
            ],
            color="danger",
            className="mt-4",
        )
    return layout() if callable(layout) else layout


def display_page(pathname):
    """Compatibility helper for routing tests and manual inspection."""
    try:
        return _render_page(pathname)
    except Exception as exc:
        trace = traceback.format_exc()
        return dbc.Alert(
            [
                html.H5("Page render error", className="alert-heading"),
                html.P(f"{exc.__class__.__name__}: {exc}"),
                html.Pre(trace, className="mb-0", style={"whiteSpace": "pre-wrap", "overflowX": "auto"}),
            ],
            color="danger",
            className="mt-4",
        )


@callback(Output("page-content", "children"), Input("url", "pathname"))
def route_page(pathname):
    return display_page(pathname)


@callback(
    Output("language-store", "data"),
    Input("language-selector", "value"),
    prevent_initial_call=True,
)
def update_language(selected_lang):
    """Update language store when user selects a language."""
    return selected_lang if selected_lang else "en"
