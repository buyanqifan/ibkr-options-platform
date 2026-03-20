"""Backtest history page: view and manage saved backtest results."""

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc
from app.components.tables import metric_card, create_data_table
from app.components.charts import create_pnl_chart, create_monthly_heatmap
from core.backtesting.storage import get_backtest_storage
import json

dash.register_page(__name__, path="/backtest-history", name="Backtest History", order=5)

# Strategy display names
STRATEGY_NAMES = {
    "sell_put": "Sell Put",
    "covered_call": "Covered Call",
    "iron_condor": "Iron Condor",
    "bull_put_spread": "Bull Put Spread",
    "bear_call_spread": "Bear Call Spread",
    "straddle": "Straddle",
    "strangle": "Strangle",
    "wheel": "Wheel Strategy",
    "binbin_god": "Binbin God",
}

layout = html.Div([
    html.H3("Backtest History", className="mb-3"),
    
    # Filter controls
    dbc.Row([
        dbc.Col([
            dbc.Label("Filter by Strategy"),
            dbc.Select(
                id="bh-strategy-filter",
                options=[{"label": "All Strategies", "value": ""}] + [
                    {"label": v, "value": k} for k, v in STRATEGY_NAMES.items()
                ],
                value="",
                className="mb-3",
            ),
        ], md=3),
        dbc.Col([
            dbc.Label("Filter by Symbol"),
            dbc.Input(
                id="bh-symbol-filter",
                type="text",
                placeholder="e.g., NVDA, AAPL",
                className="mb-3",
            ),
        ], md=3),
        dbc.Col([
            dbc.Label("Actions"),
            html.Div([
                dbc.Button("🔄 Refresh", id="bh-refresh-btn", color="primary", className="me-2"),
                dbc.Button("🗑️ Clear All", id="bh-clear-all-btn", color="danger", outline=True),
            ], className="mb-3"),
        ], md=6),
    ], className="mb-3"),
    
    # Summary stats
    html.Div(id="bh-summary-stats", className="mb-3"),
    
    # Backtest list table
    dbc.Card([
        dbc.CardHeader("Saved Backtests"),
        dbc.CardBody([
            dcc.Loading(
                html.Div(id="bh-backtest-list"),
                type="circle",
            ),
        ]),
    ], className="shadow-sm mb-4"),
    
    # Detail view (hidden by default)
    html.Div(id="bh-detail-container", className="d-none"),
    
    # Stores
    dcc.Store(id="bh-selected-id", data=None),
    
    # Delete confirmation modal
    dbc.Modal([
        dbc.ModalHeader("Confirm Delete"),
        dbc.ModalBody("Are you sure you want to delete this backtest record?"),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="bh-delete-cancel", className="me-2"),
            dbc.Button("Delete", id="bh-delete-confirm", color="danger"),
        ]),
    ], id="bh-delete-modal", is_open=False),
    
    # Clear all confirmation modal
    dbc.Modal([
        dbc.ModalHeader("⚠️ Confirm Clear All"),
        dbc.ModalBody("Are you sure you want to delete ALL backtest records? This cannot be undone."),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="bh-clear-cancel", className="me-2"),
            dbc.Button("Delete All", id="bh-clear-confirm", color="danger"),
        ]),
    ], id="bh-clear-modal", is_open=False),
    
    # Result message container
    html.Div(id="bh-clear-result"),

    # Download component for export
    dcc.Download(id="bh-download"),
])


@callback(
    Output("bh-summary-stats", "children"),
    Input("bh-refresh-btn", "n_clicks"),
)
def update_summary_stats(n_clicks):
    """Update summary statistics."""
    storage = get_backtest_storage()
    stats = storage.get_summary_stats()
    
    if stats["total_backtests"] == 0:
        return dbc.Alert("No saved backtests yet. Run a backtest and save it to see history here.", color="info")
    
    # Create strategy breakdown
    strategy_items = []
    for strategy, count in stats.get("by_strategy", {}).items():
        display_name = STRATEGY_NAMES.get(strategy, strategy)
        strategy_items.append(html.Span([
            f"{display_name}: {count}",
        ], className="badge bg-secondary me-1"))
    
    # Create symbol breakdown
    symbol_items = []
    for symbol, count in stats.get("by_symbol", {}).items():
        symbol_items.append(html.Span([
            f"{symbol}: {count}",
        ], className="badge bg-primary me-1"))
    
    return dbc.Row([
        dbc.Col(metric_card("Total Backtests", str(stats["total_backtests"]), "primary"), md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Small("By Strategy", className="text-muted d-block mb-1"),
                    html.Div(strategy_items) if strategy_items else html.Small("N/A"),
                ]),
            ], className="h-100"),
        ], md=5),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Small("By Symbol", className="text-muted d-block mb-1"),
                    html.Div(symbol_items) if symbol_items else html.Small("N/A"),
                ]),
            ], className="h-100"),
        ], md=4),
    ], className="g-3")


@callback(
    Output("bh-backtest-list", "children"),
    Input("bh-refresh-btn", "n_clicks"),
    Input("bh-strategy-filter", "value"),
    Input("bh-symbol-filter", "value"),
    Input("bh-delete-modal", "is_open"),
    Input("bh-clear-modal", "is_open"),
)
def update_backtest_list(n_clicks, strategy, symbol, delete_closed, clear_closed):
    """Update the list of backtests."""
    storage = get_backtest_storage()
    
    backtests = storage.list_backtests(
        strategy=strategy if strategy else None,
        symbol=symbol if symbol else None,
        limit=100,
    )
    
    if not backtests:
        return html.P("No backtests found matching the criteria.", className="text-muted")
    
    # Create table data
    table_data = []
    for bt in backtests:
        strategy_display = STRATEGY_NAMES.get(bt["strategy_name"], bt["strategy_name"])
        created = bt.get("created_at", "")
        if created:
            # Format datetime for display
            created = created.replace("T", " ")[:19]
        
        ret_color = "success" if bt["total_return_pct"] >= 0 else "danger"
        
        table_data.append({
            "id": bt["id"],
            "strategy": strategy_display,
            "symbol": bt["symbol"],
            "period": f"{bt['start_date']} ~ {bt['end_date']}",
            "capital": f"${bt['initial_capital']:,.0f}",
            "return": f"{bt['total_return_pct']:+.2f}%",
            "sharpe": f"{bt['sharpe_ratio']:.2f}",
            "win_rate": f"{bt['win_rate']:.1f}%",
            "trades": bt["total_trades"],
            "dd": f"{bt['max_drawdown_pct']:.1f}%",
            "created": created,
            "return_raw": bt["total_return_pct"],
        })
    
    columns = [
        {"headerName": "ID", "field": "id", "width": 60},
        {"headerName": "Strategy", "field": "strategy", "width": 120},
        {"headerName": "Symbol", "field": "symbol", "width": 80},
        {"headerName": "Period", "field": "period", "width": 200},
        {"headerName": "Capital", "field": "capital", "width": 100},
        {"headerName": "Return", "field": "return", "width": 90,
         "cellStyle": {"function": "params.data.return_raw >= 0 ? {'color': '#26a69a'} : {'color': '#ef5350'}"}},
        {"headerName": "Sharpe", "field": "sharpe", "width": 80},
        {"headerName": "Win Rate", "field": "win_rate", "width": 90},
        {"headerName": "Trades", "field": "trades", "width": 70},
        {"headerName": "Max DD", "field": "dd", "width": 80},
        {"headerName": "Created", "field": "created", "width": 160},
    ]
    
    table = create_data_table(table_data, columns, "bh-table", height=500)
    
    # Add action buttons above table
    return html.Div([
        html.P(f"Showing {len(table_data)} backtest(s)", className="text-muted mb-2"),
        table,
    ])


@callback(
    Output("bh-detail-container", "children"),
    Output("bh-detail-container", "className"),
    Output("bh-selected-id", "data"),
    Input("bh-table", "cellClicked"),
    prevent_initial_call=True,
)
def show_backtest_detail(cell_clicked):
    """Show detailed view when a row is clicked."""
    if not cell_clicked:
        return no_update, no_update, no_update
    
    # Get row data
    row_id = cell_clicked.get("rowId")
    if not row_id:
        return no_update, no_update, no_update
    
    # Find the backtest ID from the row
    storage = get_backtest_storage()
    backtest = storage.get_backtest(int(row_id))
    
    if not backtest:
        return html.P("Backtest not found", className="text-danger"), "d-block", None
    
    # Build detail view
    strategy_display = STRATEGY_NAMES.get(backtest["strategy_name"], backtest["strategy_name"])
    
    # Metrics row
    metrics = backtest
    total_ret = metrics.get("total_return_pct", 0)
    ret_color = "success" if total_ret >= 0 else "danger"
    
    metrics_row = dbc.Row([
        dbc.Col(metric_card("Total Return", f"{total_ret:+.2f}%", ret_color), md=2),
        dbc.Col(metric_card("Annual Return", f"{metrics.get('annualized_return_pct', 0):+.2f}%", ret_color), md=2),
        dbc.Col(metric_card("Win Rate", f"{metrics.get('win_rate', 0):.1f}%", "info"), md=2),
        dbc.Col(metric_card("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}", "primary"), md=2),
        dbc.Col(metric_card("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%", "danger"), md=2),
        dbc.Col(metric_card("Total Trades", f"{metrics.get('total_trades', 0)}", "secondary"), md=2),
    ], className="mb-3 g-3")
    
    # P&L Chart
    daily_pnl = backtest.get("daily_pnl", [])
    pnl_dates = [p["date"] for p in daily_pnl] if daily_pnl else []
    pnl_values = [p["cumulative_pnl"] for p in daily_pnl] if daily_pnl else []
    initial_capital = backtest.get("initial_capital", 100000)
    
    pnl_chart = dcc.Graph(figure=create_pnl_chart(
        pnl_dates, pnl_values, initial_capital=initial_capital
    ))
    
    # Trades table
    trades = backtest.get("trades", [])
    trade_columns = [
        {"headerName": "Entry", "field": "entry_date", "width": 100},
        {"headerName": "Exit", "field": "exit_date", "width": 100},
        {"headerName": "Type", "field": "trade_type", "width": 100},
        {"headerName": "Strike", "field": "strike", "width": 80,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Expiry", "field": "expiry", "width": 100},
        {"headerName": "Entry $", "field": "entry_price", "width": 80,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "Exit $", "field": "exit_price", "width": 80,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
        {"headerName": "P&L", "field": "pnl", "width": 80,
         "valueFormatter": {"function": "d3.format(',.2f')(params.value)"},
         "cellStyle": {"function": "params.value >= 0 ? {'color': '#26a69a'} : {'color': '#ef5350'}"}},
        {"headerName": "Reason", "field": "exit_reason", "width": 100},
    ]
    trades_table = create_data_table(trades, trade_columns, "bh-detail-trades-table", height=300) if trades else html.P("No trades", className="text-muted")

    detail_content = dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H5(f"Backtest #{backtest['id']}: {strategy_display} on {backtest['symbol']}", className="mb-0"),
                html.Small(f"Period: {backtest['start_date']} to {backtest['end_date']} | Capital: ${backtest['initial_capital']:,.0f}"),
            ], className="d-flex justify-content-between align-items-center"),
        ]),
        dbc.CardBody([
            metrics_row,
            html.H6("Cumulative P&L", className="mt-3 mb-2"),
            pnl_chart,
            html.H6("Trade Log", className="mt-4 mb-2"),
            trades_table,
            html.Div([
                dbc.Button("📤 Export for AI Analysis", id="bh-export-btn", color="info", className="mt-3 me-2"),
                dbc.Button("🗑️ Delete This Record", id="bh-delete-btn", color="danger", outline=True, className="mt-3"),
            ]),
        ]),
    ], className="shadow-sm mt-3")

    return detail_content, "d-block", backtest["id"]


@callback(
    Output("bh-delete-modal", "is_open"),
    Input("bh-delete-btn", "n_clicks"),
    Input("bh-delete-cancel", "n_clicks"),
    Input("bh-delete-confirm", "n_clicks"),
    State("bh-delete-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_delete_modal(n_delete, n_cancel, n_confirm, is_open):
    """Toggle delete confirmation modal."""
    if n_delete or n_cancel or n_confirm:
        return not is_open
    return is_open


@callback(
    Output("bh-clear-modal", "is_open"),
    Input("bh-clear-all-btn", "n_clicks"),
    Input("bh-clear-cancel", "n_clicks"),
    Input("bh-clear-confirm", "n_clicks"),
    State("bh-clear-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_clear_modal(n_clear, n_cancel, n_confirm, is_open):
    """Toggle clear all confirmation modal."""
    if n_clear or n_cancel or n_confirm:
        return not is_open
    return is_open


@callback(
    Output("bh-detail-container", "children", allow_duplicate=True),
    Input("bh-delete-confirm", "n_clicks"),
    State("bh-selected-id", "data"),
    prevent_initial_call=True,
)
def delete_backtest(n_clicks, backtest_id):
    """Delete the selected backtest."""
    if not n_clicks or not backtest_id:
        return no_update
    
    storage = get_backtest_storage()
    storage.delete_backtest(backtest_id)
    
    return dbc.Alert(f"Backtest #{backtest_id} deleted successfully.", color="success")


@callback(
    Output("bh-clear-result", "children", allow_duplicate=True),
    Input("bh-clear-confirm", "n_clicks"),
    prevent_initial_call=True,
)
def clear_all_backtests(n_clicks):
    """Delete all backtests."""
    if not n_clicks:
        return no_update
    
    storage = get_backtest_storage()
    # Get all backtests and delete them
    all_backtests = storage.list_backtests(limit=1000)
    deleted_count = 0
    for bt in all_backtests:
        if storage.delete_backtest(bt["id"]):
            deleted_count += 1
    
    return dbc.Alert(f"Deleted {deleted_count} backtest(s).", color="warning")


@callback(
    Output("bh-download", "data"),
    Input("bh-export-btn", "n_clicks"),
    State("bh-selected-id", "data"),
    prevent_initial_call=True,
)
def export_backtest(n_clicks, backtest_id):
    """Export the selected backtest as JSON for AI analysis."""
    from datetime import datetime

    if not n_clicks or not backtest_id:
        return no_update

    storage = get_backtest_storage()
    export_json = storage.export_backtest_json(backtest_id)

    if not export_json:
        return no_update

    # Get backtest info for filename
    backtest = storage.get_backtest(backtest_id)
    if backtest:
        symbol = backtest.get("symbol", "UNKNOWN")
        strategy = backtest.get("strategy_name", "unknown")
    else:
        symbol = "UNKNOWN"
        strategy = "unknown"

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_{strategy}_{symbol}_{date_str}.json"

    return dict(
        content=export_json,
        filename=filename,
        mime_type="application/json",
    )