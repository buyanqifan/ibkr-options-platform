"""Stock Screener page: configurable filters, results table with scoring."""

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc
from app.components.tables import create_data_table
from app.services import get_services

dash.register_page(__name__, path="/screener", name="Screener", order=2)

layout = html.Div([
    html.H3("Stock Screener", className="mb-3"),

    dbc.Row([
        # Left: Filter panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Screening Criteria"),
                dbc.CardBody([
                    dbc.Label("Symbols (comma-separated, or leave empty for default)"),
                    dbc.Input(
                        id="scr-symbols",
                        placeholder="AAPL, MSFT, GOOGL, AMZN, ...",
                        className="mb-3",
                    ),

                    html.Hr(),
                    html.H6("Financial Filters", className="fw-bold"),

                    dbc.Label("PE Ratio"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="scr-pe-min", type="number", value=0, size="sm"), width=6),
                        dbc.Col(dbc.Input(id="scr-pe-max", type="number", value=50, size="sm"), width=6),
                    ], className="mb-2"),

                    dbc.Label("Market Cap (Billion $)"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="scr-mcap-min", type="number", value=2, size="sm"), width=6),
                        dbc.Col(dbc.Input(id="scr-mcap-max", type="number", placeholder="No limit", size="sm"), width=6),
                    ], className="mb-2"),

                    html.Hr(),
                    html.H6("Options Filters", className="fw-bold"),

                    dbc.Label("IV Rank (%)"),
                    dcc.RangeSlider(
                        id="scr-iv-rank", min=0, max=100, step=5,
                        value=[30, 100],
                        marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
                    ),
                    html.Div(className="mb-2"),

                    dbc.Label("Min Option Volume"),
                    dbc.Input(id="scr-opt-vol", type="number", value=100, size="sm", className="mb-2"),

                    dbc.Label("Min Put Premium Yield (%)"),
                    dbc.Input(
                        id="scr-put-yield", type="number",
                        value=0.5, step=0.1, size="sm", className="mb-3",
                    ),

                    dbc.Button(
                        "Run Screener", id="scr-run-btn",
                        color="primary", className="w-100", n_clicks=0,
                    ),
                ]),
            ], className="shadow-sm"),
        ], md=3),

        # Right: Results
        dbc.Col([
            dcc.Loading(
                html.Div(
                    id="scr-results-container",
                    children=[
                        html.P(
                            "Configure filters and click 'Run Screener' to start",
                            className="text-muted",
                        ),
                    ],
                ),
                type="circle",
            ),
        ], md=9),
    ]),

    dcc.Store(id="scr-results-store", data=[]),
])


@callback(
    Output("scr-results-store", "data"),
    Output("scr-results-container", "children"),
    Input("scr-run-btn", "n_clicks"),
    State("scr-symbols", "value"),
    State("scr-pe-min", "value"),
    State("scr-pe-max", "value"),
    State("scr-mcap-min", "value"),
    State("scr-mcap-max", "value"),
    State("scr-iv-rank", "value"),
    State("scr-opt-vol", "value"),
    State("scr-put-yield", "value"),
    prevent_initial_call=True,
)
def run_screener(
    n_clicks, symbols_str, pe_min, pe_max,
    mcap_min, mcap_max, iv_rank_range, min_opt_vol, min_put_yield,
):
    services = get_services()
    if not services or not services["conn_mgr"].is_connected:
        return [], html.P("Please connect to IBKR first", className="text-warning")

    if symbols_str and symbols_str.strip():
        symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
    else:
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
            "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS",
            "BAC", "XOM", "KO", "PEP", "ABBV", "COST", "MRK", "LLY",
        ]

    screener = services["screener"]
    from core.screener.criteria import ScreeningCriteria
    criteria = ScreeningCriteria(
        pe_min=pe_min or 0,
        pe_max=pe_max or 999,
        market_cap_min=(mcap_min or 0) * 1e9,
        market_cap_max=(mcap_max * 1e9) if mcap_max else None,
        iv_rank_min=iv_rank_range[0] if iv_rank_range else 0,
        iv_rank_max=iv_rank_range[1] if iv_rank_range else 100,
        min_option_volume=min_opt_vol or 0,
        min_put_premium_yield=min_put_yield or 0,
    )

    try:
        results = screener.run(symbols, criteria)
    except Exception as e:
        return [], html.P(f"Screener error: {e}", className="text-danger")

    if not results:
        return [], html.P("No stocks matched the criteria", className="text-muted")

    columns = [
        {"headerName": "Rank", "field": "rank", "width": 70},
        {"headerName": "Symbol", "field": "symbol", "width": 90},
        {"headerName": "Score", "field": "score", "width": 80,
         "valueFormatter": {"function": "d3.format('.1f')(params.value)"}},
        {"headerName": "Price", "field": "price", "width": 90,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "PE", "field": "pe_ratio", "width": 80,
         "valueFormatter": {"function": "params.value ? d3.format('.1f')(params.value) : 'N/A'"}},
        {"headerName": "Mkt Cap (B)", "field": "market_cap_b", "width": 110,
         "valueFormatter": {"function": "params.value ? d3.format(',.1f')(params.value) : 'N/A'"}},
        {"headerName": "IV Rank %", "field": "iv_rank", "width": 100,
         "valueFormatter": {"function": "params.value != null ? d3.format('.1f')(params.value) : 'N/A'"}},
        {"headerName": "ATM IV %", "field": "atm_iv", "width": 100,
         "valueFormatter": {"function": "params.value != null ? d3.format('.1f')(params.value) : 'N/A'"}},
        {"headerName": "Put Yield %", "field": "put_premium_yield", "width": 110,
         "valueFormatter": {"function": "params.value != null ? d3.format('.2f')(params.value) : 'N/A'"}},
    ]

    table = create_data_table(results, columns, "screener-results-table", height=550)
    summary = html.P(
        f"Found {len(results)} stocks matching criteria",
        className="text-success mb-2",
    )
    return results, html.Div([summary, table])
