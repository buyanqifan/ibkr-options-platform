"""Navigation bar component with multi-language support."""

import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input


def create_navbar_items(lang="en"):
    """Create navbar items based on language."""
    # Navigation labels by language
    labels = {
        "en": {
            "dashboard": "Dashboard",
            "market_data": "Market Data",
            "screener": "Screener",
            "options_chain": "Options Chain",
            "backtester": "Backtester",
            "settings": "Settings",
        },
        "zh": {
            "dashboard": "仪表盘",
            "market_data": "市场数据",
            "screener": "选股器",
            "options_chain": "期权链",
            "backtester": "回测器",
            "settings": "设置",
        }
    }
    
    nav_labels = labels.get(lang, labels["en"])
    
    return [
        dbc.NavItem(dbc.NavLink(nav_labels["dashboard"], href="/", active="exact")),
        dbc.NavItem(dbc.NavLink(nav_labels["market_data"], href="/market-data", active="exact")),
        dbc.NavItem(dbc.NavLink(nav_labels["screener"], href="/screener", active="exact")),
        dbc.NavItem(dbc.NavLink(nav_labels["options_chain"], href="/options-chain", active="exact")),
        dbc.NavItem(dbc.NavLink(nav_labels["backtester"], href="/backtester", active="exact")),
        dbc.NavItem(dbc.NavLink(nav_labels["settings"], href="/settings", active="exact")),
        
        # Language Selector
        dbc.NavItem(
            dbc.InputGroup(
                [
                    dbc.InputGroupText(html.I(className="bi bi-translate", style={"color": "#ffc107"})),
                    dcc.Dropdown(
                        id="language-selector",
                        options=[
                            {"label": "🇺🇸 English", "value": "en"},
                            {"label": "🇨🇳 中文", "value": "zh"},
                        ],
                        value="en",
                        clearable=False,
                        style={"width": "150px"},
                    ),
                ],
                size="sm",
                className="ms-3",
            )
        ),
    ]


def create_navbar():
    navbar_items = create_navbar_items()
    
    return html.Div([
        dbc.Navbar(
            dbc.Container(
                [
                    dbc.NavbarBrand(
                        [html.I(className="bi bi-graph-up-arrow me-2", style={"color": "#0d6efd"}), 
                         html.Span("IBKR Options Platform", id="navbar-brand-text")],
                        href="/",
                        className="fw-bold",
                    ),
                    dbc.NavbarToggler(id="navbar-toggler"),
                    dbc.Collapse(
                        dbc.Nav(
                            navbar_items,
                            className="ms-auto",
                            navbar=True,
                        ),
                        id="navbar-collapse",
                        navbar=True,
                    ),
                    html.Div(id="connection-badge", className="ms-3"),
                ],
                fluid=True,
            ),
            color="dark",
            dark=True,
            className="mb-4",
            sticky="top",
        ),
        # Callback to update navbar text based on language will be registered separately
    ])


@callback(
    Output("navbar-collapse", "children"),
    Input("language-selector", "value"),
    prevent_initial_call=False,
)
def update_navbar_language(lang):
    """Update navbar navigation items based on selected language."""
    if lang is None:
        lang = "en"
    
    navbar_items = create_navbar_items(lang)
    return dbc.Nav(navbar_items, className="ms-auto", navbar=True)
