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
            "backtest": "Backtest",  # Parent menu
            "backtester": "Strategy Backtest",
            "backtest_history": "History",
            "binbin_god": "Binbin God 🤖",
            "binbin_god_live": "Binbin God Live",
            "settings": "Settings",
        },
        "zh": {
            "dashboard": "仪表盘",
            "market_data": "市场数据",
            "screener": "选股器",
            "options_chain": "期权链",
            "backtest": "回测",  # 父菜单
            "backtester": "策略回测",
            "backtest_history": "历史记录",
            "binbin_god": "彬彬神 🤖",
            "binbin_god_live": "彬彬神实盘",
            "settings": "设置",
        }
    }
    
    nav_labels = labels.get(lang, labels["en"])
    
    # Create backtest submenu
    backtest_menu = dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem(
                nav_labels["backtester"],
                href="/backtester",
                external_link=True,
            ),
            dbc.DropdownMenuItem(
                nav_labels["backtest_history"],
                href="/backtest-history",
                external_link=True,
            ),
            dbc.DropdownMenuItem(
                nav_labels["binbin_god"],
                href="/binbin-god",
                external_link=True,
            ),
            dbc.DropdownMenuItem(
                nav_labels["binbin_god_live"],
                href="/binbin-god-live",
                external_link=True,
            ),
        ],
        label=nav_labels["backtest"],
        nav=True,
        in_navbar=True,
        toggle_style={"color": "rgba(255,255,255,0.55)"},
    )
    
    return [
        dbc.NavItem(dbc.NavLink(nav_labels["dashboard"], href="/", active="exact", external_link=True)),
        dbc.NavItem(dbc.NavLink(nav_labels["market_data"], href="/market-data", active="exact", external_link=True)),
        dbc.NavItem(dbc.NavLink(nav_labels["screener"], href="/screener", active="exact", external_link=True)),
        dbc.NavItem(dbc.NavLink(nav_labels["options_chain"], href="/options-chain", active="exact", external_link=True)),
        backtest_menu,  # Dropdown menu for backtest pages
        dbc.NavItem(dbc.NavLink(nav_labels["settings"], href="/settings", active="exact", external_link=True)),

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
