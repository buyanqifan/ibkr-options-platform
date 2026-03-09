"""Navigation bar component."""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_navbar():
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(
                    [html.I(className="bi bi-graph-up-arrow me-2", style={"color": "#0d6efd"}), "IBKR Options Platform"],
                    href="/",
                    className="fw-bold",
                ),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Dashboard", href="/", active="exact")),
                            dbc.NavItem(dbc.NavLink("Market Data", href="/market-data", active="exact")),
                            dbc.NavItem(dbc.NavLink("Screener", href="/screener", active="exact")),
                            dbc.NavItem(dbc.NavLink("Options Chain", href="/options-chain", active="exact")),
                            dbc.NavItem(dbc.NavLink("Backtester", href="/backtester", active="exact")),
                            dbc.NavItem(dbc.NavLink("Settings", href="/settings", active="exact")),
                            
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
                                            value="en",  # Default to English
                                            clearable=False,
                                            style={"width": "150px"},
                                        ),
                                    ],
                                    size="sm",
                                    className="ms-3",
                                )
                            ),
                        ],
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
    )
