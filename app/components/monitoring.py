"""Performance monitoring components for strategy tracking."""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Dict, Any


def create_performance_metrics_card(metrics: Dict[str, Any]) -> dbc.Card:
    """Create a card displaying key performance metrics."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Performance Metrics", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Total Return", className="text-muted mb-1"),
                    html.H4(f"{metrics.get('total_return_pct', 0):+.2f}%", 
                           className=f"text-{'success' if metrics.get('total_return_pct', 0) >= 0 else 'danger'}"),
                ], width=3),
                dbc.Col([
                    html.H6("Win Rate", className="text-muted mb-1"),
                    html.H4(f"{metrics.get('win_rate', 0):.1f}%", 
                           className="text-info"),
                ], width=3),
                dbc.Col([
                    html.H6("Sharpe Ratio", className="text-muted mb-1"),
                    html.H4(f"{metrics.get('sharpe_ratio', 0):.2f}", 
                           className="text-primary"),
                ], width=3),
                dbc.Col([
                    html.H6("Max Drawdown", className="text-muted mb-1"),
                    html.H4(f"{metrics.get('max_drawdown_pct', 0):.2f}%", 
                           className="text-danger"),
                ], width=3),
            ]),
        ]),
    ], className="mb-4 shadow-sm")


def create_strategy_state_card(state: Dict[str, Any]) -> dbc.Card:
    """Create a card displaying current strategy state."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Current Strategy State", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Phase", className="text-muted mb-1"),
                    html.H4(state.get("phase", "N/A"), className="text-primary"),
                ], width=3),
                dbc.Col([
                    html.H6("Shares Held", className="text-muted mb-1"),
                    html.H4(f"{state.get('shares_held', 0):,}", className="text-info"),
                ], width=3),
                dbc.Col([
                    html.H6("Cost Basis", className="text-muted mb-1"),
                    html.H4(f"${state.get('cost_basis', 0):.2f}", className="text-dark"),
                ], width=3),
                dbc.Col([
                    html.H6("Premium Collected", className="text-muted mb-1"),
                    html.H4(f"${state.get('total_premium_collected', 0):,.2f}", 
                           className="text-success"),
                ], width=3),
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H6("Effective Cost Basis", className="text-muted mb-1"),
                    html.H5(f"${state.get('effective_cost_basis', 0):.2f}", 
                           className="text-dark"),
                ], width=4),
                dbc.Col([
                    html.H6("Current Portfolio Value", className="text-muted mb-1"),
                    html.H5(f"${state.get('current_portfolio_value', 0):,.2f}", 
                           className="text-primary"),
                ], width=4),
                dbc.Col([
                    html.H6("Total P&L", className="text-muted mb-1"),
                    html.H5(f"${state.get('total_pnl', 0):+.2f}", 
                           className=f"text-{'success' if state.get('total_pnl', 0) >= 0 else 'danger'}"),
                ], width=4),
            ]),
        ]),
    ], className="mb-4 shadow-sm")


def create_holdings_card(holdings_data: Dict[str, Any]) -> dbc.Card:
    """Create a card displaying current stock and option holdings."""
    if not holdings_data:
        return dbc.Card([
            dbc.CardHeader(html.H5("Current Holdings", className="mb-0")),
            dbc.CardBody([
                html.P("No holdings data available", className="text-muted"),
            ]),
        ], className="mb-4 shadow-sm")
    
    # Extract holdings info
    shares_held = holdings_data.get("shares_held", 0)
    cost_basis = holdings_data.get("cost_basis", 0)
    options_held = holdings_data.get("options_held", [])
    
    # Build holdings rows
    holdings_rows = []
    
    # Stock holding row (if any)
    if shares_held > 0:
        holdings_rows.append(
            html.Tr([
                html.Td([
                    html.I(className="bi bi-building me-2"),
                    "Common Stock"
                ]),
                html.Td(f"{shares_held:,}"),
                html.Td(f"${cost_basis:.2f}"),
                html.Td("-"),
                html.Td(f"${cost_basis:.2f}"),
            ], className="table-success")
        )
    
    # Option holdings rows
    for opt in options_held:
        symbol = opt.get("symbol", "N/A")
        expiry = opt.get("expiry", "N/A")
        strike = opt.get("strike", 0)
        right = opt.get("right", "N/A")
        quantity = opt.get("quantity", 0)
        entry_price = opt.get("entry_price", 0)
        market_value = opt.get("market_value", 0)
        
        # Format option name: e.g., "AAPL 240119 150 Put"
        try:
            expiry_short = expiry[2:] if len(expiry) >= 6 else expiry
            option_type = "Put" if right == "P" else "Call"
            option_name = f"{symbol} {expiry_short} {strike:.0f} {option_type}"
        except:
            option_name = f"{symbol} {expiry} {strike} {right}"
        
        qty_display = f"{abs(quantity)}x {'Long' if quantity > 0 else 'Short'}"
        
        holdings_rows.append(
            html.Tr([
                html.Td(option_name),
                html.Td(qty_display),
                html.Td(f"${entry_price:.2f}"),
                html.Td(f"${market_value:.2f}"),
                html.Td(f"${quantity * entry_price * 100:.2f}", 
                       className="text-success" if quantity * entry_price > 0 else "text-danger"),
            ])
        )
    
    return dbc.Card([
        dbc.CardHeader(html.H5("Current Holdings", className="mb-0")),
        dbc.CardBody([
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Instrument"),
                        html.Th("Quantity"),
                        html.Th("Entry Price"),
                        html.Th("Market Value"),
                        html.Th("Total Value"),
                    ])
                ]),
                html.Tbody(holdings_rows),
            ], className="table table-striped table-sm mb-0"),
            
            # Summary footer
            html.Div([
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.H6("Total Stock Value", className="text-muted"),
                        html.H5(f"${cost_basis:.2f}", className="text-info"),
                    ], width=6),
                    dbc.Col([
                        html.H6("Total Options Value", className="text-muted"),
                        html.H5(f"${sum(opt.get('market_value', 0) for opt in options_held):.2f}", 
                               className="text-warning"),
                    ], width=6),
                ]),
            ]),
        ]),
    ], className="mb-4 shadow-sm")


def create_holdings_card(holdings_data: Dict[str, Any]) -> dbc.Card:
    """Create a card displaying current stock and option holdings."""
    if not holdings_data:
        return dbc.Card([
            dbc.CardHeader(html.H5("Current Holdings", className="mb-0")),
            dbc.CardBody([
                html.P("No holdings data available", className="text-muted"),
            ]),
        ], className="mb-4 shadow-sm")
    
    # Extract holdings info
    shares_held = holdings_data.get("shares_held", 0)
    cost_basis = holdings_data.get("cost_basis", 0)
    options_held = holdings_data.get("options_held", [])
    
    # Build holdings rows
    holdings_rows = []
    
    # Stock holding row (if any)
    if shares_held > 0:
        holdings_rows.append(
            html.Tr([
                html.Td([
                    html.I(className="bi bi-building me-2"),
                    "Common Stock"
                ]),
                html.Td(f"{shares_held:,}"),
                html.Td(f"${cost_basis:.2f}"),
                html.Td("-"),
                html.Td(f"${cost_basis:.2f}"),
            ], className="table-success")
        )
    
    # Option holdings rows
    for opt in options_held:
        symbol = opt.get("symbol", "N/A")
        expiry = opt.get("expiry", "N/A")
        strike = opt.get("strike", 0)
        right = opt.get("right", "N/A")
        quantity = opt.get("quantity", 0)
        entry_price = opt.get("entry_price", 0)
        market_value = opt.get("market_value", 0)
        
        # Format option name: e.g., "AAPL 240119 150 Put"
        try:
            expiry_short = expiry[2:] if len(expiry) >= 6 else expiry
            option_type = "Put" if right == "P" else "Call"
            option_name = f"{symbol} {expiry_short} {strike:.0f} {option_type}"
        except:
            option_name = f"{symbol} {expiry} {strike} {right}"
        
        qty_display = f"{abs(quantity)}x {'Long' if quantity > 0 else 'Short'}"
        
        holdings_rows.append(
            html.Tr([
                html.Td(option_name),
                html.Td(qty_display),
                html.Td(f"${entry_price:.2f}"),
                html.Td(f"${market_value:.2f}"),
                html.Td(f"${quantity * entry_price * 100:.2f}", 
                       className="text-success" if quantity * entry_price > 0 else "text-danger"),
            ])
        )
    
    return dbc.Card([
        dbc.CardHeader(html.H5("Current Holdings", className="mb-0")),
        dbc.CardBody([
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Instrument"),
                        html.Th("Quantity"),
                        html.Th("Entry Price"),
                        html.Th("Market Value"),
                        html.Th("Total Value"),
                    ])
                ]),
                html.Tbody(holdings_rows),
            ], className="table table-striped table-sm mb-0"),
            
            # Summary footer
            html.Div([
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.H6("Total Stock Value", className="text-muted"),
                        html.H5(f"${cost_basis:.2f}", className="text-info"),
                    ], width=6),
                    dbc.Col([
                        html.H6("Total Options Value", className="text-muted"),
                        html.H5(f"${sum(opt.get('market_value', 0) for opt in options_held):.2f}", 
                               className="text-warning"),
                    ], width=6),
                ]),
            ]),
        ]),
    ], className="mb-4 shadow-sm")
    """Create a card displaying detailed performance summary."""
    return dbc.Card([
        dbc.CardHeader(html.H5("Performance Summary", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Total Trades", className="text-muted mb-1"),
                    html.H4(f"{performance_metrics.get('total_trades', 0)}", 
                           className="text-dark"),
                ], width=2),
                dbc.Col([
                    html.H6("Successful Assignments", className="text-muted mb-1"),
                    html.H4(f"{performance_metrics.get('successful_assignments', 0)}", 
                           className="text-success"),
                ], width=2),
                dbc.Col([
                    html.H6("Expired Worthless", className="text-muted mb-1"),
                    html.H4(f"{performance_metrics.get('expired_worthless', 0)}", 
                           className="text-info"),
                ], width=2),
                dbc.Col([
                    html.H6("Profit Target Exits", className="text-muted mb-1"),
                    html.H4(f"{performance_metrics.get('profit_target_exits', 0)}", 
                           className="text-warning"),
                ], width=2),
                dbc.Col([
                    html.H6("Stop Loss Exits", className="text-muted mb-1"),
                    html.H4(f"{performance_metrics.get('stop_loss_exits', 0)}", 
                           className="text-danger"),
                ], width=2),
                dbc.Col([
                    html.H6("Phase Transitions", className="text-muted mb-1"),
                    html.H4(f"{performance_metrics.get('phase_transitions', 0)}", 
                           className="text-primary"),
                ], width=2),
            ]),
            html.Hr(className="my-3"),
            dbc.Row([
                dbc.Col([
                    html.H6("Average P&L per Trade", className="text-muted mb-1"),
                    html.H4(f"${performance_metrics.get('avg_pnl_per_trade', 0):+.2f}", 
                           className=f"text-{'success' if performance_metrics.get('avg_pnl_per_trade', 0) >= 0 else 'danger'}"),
                ], width=4),
                dbc.Col([
                    html.H6("Current Win Rate", className="text-muted mb-1"),
                    html.H4(f"{performance_metrics.get('win_rate', 0):.1f}%", 
                           className="text-info"),
                ], width=4),
                dbc.Col([
                    html.H6("Max Drawdown", className="text-muted mb-1"),
                    html.H4(f"{performance_metrics.get('max_drawdown', 0):.2f}%", 
                           className="text-danger"),
                ], width=4),
            ]),
        ]),
    ], className="mb-4 shadow-sm")


def create_monitoring_dashboard(performance_data: Dict[str, Any]) -> html.Div:
    """Create the complete monitoring dashboard."""
    if not performance_data:
        return html.Div([
            html.H3("Performance Monitoring", className="mb-4"),
            html.P("No performance data available. Run a backtest to see monitoring data.", 
                  className="text-muted"),
        ])
    
    # Extract data components
    current_state = performance_data.get("current_state", {})
    performance_metrics = performance_data.get("performance_metrics", {})
    metrics = performance_data.get("metrics", {})
    
    return html.Div([
        html.H3("Performance Monitoring Dashboard", className="mb-4"),
        
        # Key performance metrics
        create_performance_metrics_card(metrics),
        
        # Current strategy state
        create_strategy_state_card(current_state),
        
        # Additional monitoring sections
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Recent Trade History", className="mb-0")),
                    dbc.CardBody([
                        html.Div(
                            id="monitoring-trade-history",
                            children=html.P("Trade history will appear here after running backtest", 
                                          className="text-muted")
                        ),
                    ]),
                ], className="shadow-sm"),
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Phase Transition Log", className="mb-0")),
                    dbc.CardBody([
                        html.Div(
                            id="monitoring-phase-log",
                            children=html.P("Phase transitions will appear here", 
                                          className="text-muted")
                        ),
                    ]),
                ], className="shadow-sm"),
            ], width=6),
        ], className="mb-4"),
    ])


def create_trade_history_table(trades: list) -> html.Div:
    """Create a table showing recent trades."""
    if not trades:
        return html.P("No trades recorded yet.", className="text-muted")
    
    # Create table rows for recent trades
    rows = []
    for trade in trades[-10:]:  # Show last 10 trades
        pnl_class = "text-success" if trade.get("pnl", 0) >= 0 else "text-danger"
        
        # Extract option contract details
        symbol = trade.get("symbol", "N/A")
        expiry = trade.get("expiry", "N/A")
        strike = trade.get("strike", 0)
        right = trade.get("right", "N/A")  # P (Put) or C (Call)
        quantity = trade.get("quantity", 0)
        trade_type = trade.get("type", "N/A")
        
        # Format option name: e.g., "AAPL 240115 150 Put"
        option_name = f"{symbol} {expiry[2:]} {strike:.0f} {'Put' if right == 'P' else 'Call'}"
        
        # Format quantity with direction
        qty_display = f"{quantity}x"
        
        rows.append(
            html.Tr([
                html.Td(trade.get("date", "N/A")),
                html.Td(option_name, title=f"{trade_type}"),
                html.Td(qty_display),
                html.Td(trade.get("exit_reason", "N/A")),
                html.Td(f"${trade.get('pnl', 0):+.2f}", className=pnl_class),
                html.Td(f"${trade.get('cumulative_pnl', 0):+.2f}"),
            ])
        )
    
    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Date"), 
                html.Th("Option Contract"), 
                html.Th("Qty"),
                html.Th("Exit Reason"), 
                html.Th("P&L"), 
                html.Th("Cumulative P&L")
            ])
        ]),
        html.Tbody(rows),
    ], className="table table-striped table-sm")


def create_phase_transition_log(transitions: list) -> html.Div:
    """Create a log of phase transitions."""
    if not transitions:
        return html.P("No phase transitions recorded yet.", className="text-muted")
    
    # Create log entries for recent transitions
    entries = []
    for transition in transitions[-10:]:  # Show last 10 transitions
        entries.append(
            html.Div([
                html.Small(transition.get("timestamp", ""), className="text-muted"),
                html.P(f"{transition.get('from_phase', 'N/A')} → {transition.get('to_phase', 'N/A')}", 
                      className="mb-1"),
                html.Small(transition.get("reason", ""), className="text-muted"),
                html.Hr(className="my-2"),
            ])
        )
    
    return html.Div(entries)