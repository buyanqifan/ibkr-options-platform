"""Reusable chart components for Plotly Dash."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_candlestick_chart(
    bars: list[dict],
    symbol: str = "",
    ma_periods: list[int] | None = None,
    show_volume: bool = True,
) -> go.Figure:
    """Create a candlestick chart with optional moving averages and volume."""
    if not bars:
        fig = go.Figure()
        fig.update_layout(
            title="No data available",
            template="plotly_dark",
        )
        return fig

    import pandas as pd
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["date"])

    row_heights = [0.7, 0.3] if show_volume else [1.0]
    rows = 2 if show_volume else 1

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    # Moving averages
    if ma_periods is None:
        ma_periods = [20, 50, 200]

    colors = ["#FFA726", "#42A5F5", "#AB47BC", "#66BB6A"]
    for i, period in enumerate(ma_periods):
        if len(df) >= period:
            ma = df["close"].rolling(window=period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=ma,
                    name=f"MA{period}",
                    line=dict(color=colors[i % len(colors)], width=1),
                ),
                row=1,
                col=1,
            )

    # Volume
    if show_volume and "volume" in df.columns:
        colors_vol = [
            "#26a69a" if row["close"] >= row["open"] else "#ef5350"
            for _, row in df.iterrows()
        ]
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="Volume",
                marker_color=colors_vol,
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=f"{symbol} Price Chart" if symbol else "Price Chart",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=50, r=50, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    fig.update_xaxes(type="date")

    return fig


def create_pnl_chart(
    dates: list,
    pnl_values: list[float],
    benchmark_data: dict | None = None,
    initial_capital: float | None = None,
    title: str = "P&L Curve",
) -> go.Figure:
    """Create a P&L curve chart with optional benchmark comparison and percentage display.
    
    benchmark_data: dict with format {'symbol': [{'date': str, 'cumulative_pnl': float, 'percentage_return': float}, ...]}
    """
    fig = go.Figure()

    # Add P&L in dollars (primary trace)
    strategy_trace = go.Scatter(
        x=dates,
        y=pnl_values,
        name="Strategy P&L ($)",
        line=dict(color="#26a69a", width=2),
        fill="tozeroy",
        fillcolor="rgba(38, 166, 154, 0.1)",
    )
    
    # Add percentage trace if initial capital is provided
    if initial_capital and initial_capital > 0:
        percentage_values = [pnl / initial_capital * 100 for pnl in pnl_values]
        percentage_trace = go.Scatter(
            x=dates,
            y=percentage_values,
            name="Strategy Return (%)",
            line=dict(color="#42A5F5", width=2, dash="dot"),
            yaxis="y2",
        )
        fig.add_trace(percentage_trace)
    
    fig.add_trace(strategy_trace)

    # Add multiple benchmarks if provided
    if benchmark_data:
        benchmark_colors = ["#FFA726", "#AB47BC", "#66BB6A", "#FF7043", "#29B6F6", "#FFD54F", "#BA68C8"]
        benchmark_idx = 0
        
        for symbol, benchmark_points in benchmark_data.items():
            if not benchmark_points or len(benchmark_points) == 0:
                logger.warning(f"No data points for benchmark {symbol}")
                continue
            
            # Validate data structure
            try:
                # Extract dates and values
                bench_dates = []
                bench_pnl = []
                bench_percentage = []
                
                for point in benchmark_points:
                    if "date" not in point or "cumulative_pnl" not in point:
                        logger.warning(f"Invalid benchmark data point for {symbol}: missing required fields")
                        continue
                    bench_dates.append(point["date"])
                    bench_pnl.append(point.get("cumulative_pnl", 0))
                    if "percentage_return" in point:
                        bench_percentage.append(point["percentage_return"])
                
                if not bench_dates:
                    logger.warning(f"No valid data points for benchmark {symbol}")
                    continue
                
                color = benchmark_colors[benchmark_idx % len(benchmark_colors)]
                
                # Add benchmark P&L trace
                fig.add_trace(
                    go.Scatter(
                        x=bench_dates,
                        y=bench_pnl,
                        name=f"{symbol} P&L ($)",
                        line=dict(color=color, width=2, dash="dash"),
                        mode="lines",
                    )
                )
                
                # Add benchmark percentage trace if available
                if initial_capital and initial_capital > 0 and bench_percentage:
                    fig.add_trace(
                        go.Scatter(
                            x=bench_dates,
                            y=bench_percentage,
                            name=f"{symbol} Return (%)",
                            line=dict(color=color, width=2, dash="dot"),
                            yaxis="y2",
                            mode="lines",
                        )
                    )
                
                benchmark_idx += 1
                logger.info(f"Added benchmark curve for {symbol}: {len(bench_dates)} data points")
                
            except Exception as e:
                logger.error(f"Error adding benchmark curve for {symbol}: {e}")
                continue

    # Update layout with dual y-axes if percentage is shown
    layout_updates = dict(
        title=title,
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=50, t=50, b=30),
        yaxis_title="P&L ($)",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    # Add second y-axis for percentage if needed
    if initial_capital and initial_capital > 0:
        layout_updates["yaxis2"] = dict(
            title="Return (%)",
            overlaying="y",
            side="right",
            gridcolor="rgba(255,255,255,0.1)",
        )
    
    fig.update_layout(**layout_updates)
    return fig


def create_monthly_heatmap(monthly_returns: dict) -> go.Figure:
    """Create monthly returns heatmap.

    monthly_returns: {(year, month): return_pct, ...}
    """
    import pandas as pd

    if not monthly_returns:
        fig = go.Figure()
        fig.update_layout(title="No data", template="plotly_dark")
        return fig

    df = pd.DataFrame(
        [(y, m, v) for (y, m), v in monthly_returns.items()],
        columns=["year", "month", "return_pct"],
    )
    pivot = df.pivot(index="year", columns="month", values="return_pct")
    pivot = pivot.reindex(columns=range(1, 13))

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=month_labels,
            y=[str(y) for y in pivot.index],
            colorscale="RdYlGn",
            zmid=0,
            text=pivot.values.round(1),
            texttemplate="%{text}%",
            textfont={"size": 10},
            colorbar_title="Return %",
        )
    )
    fig.update_layout(
        title="Monthly Returns",
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=50, t=50, b=30),
    )
    return fig


def create_trade_timeline_chart(
    trades: list[dict],
    daily_pnl: list[dict],
    underlying_prices: list[dict] | None = None,
    selection_history: list[dict] | None = None,
    title: str = "Trade Timeline and Performance",
) -> go.Figure:
    """Create a comprehensive trade timeline chart showing entry/exit points and performance.
    
    Args:
        trades: List of trade dictionaries with entry/exit info
        daily_pnl: List of daily P&L data
        underlying_prices: Optional list of underlying price data
        selection_history: Optional list of stock selection changes for binbin_god
        title: Chart title
    """
    if not trades and not daily_pnl:
        fig = go.Figure()
        fig.update_layout(title="No trade data available", template="plotly_dark")
        return fig
    
    # Create subplots: Price chart (top) and P&L chart (bottom)
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4],
        subplot_titles=('Underlying Price with Trade Points', 'Cumulative P&L')
    )
    
    # Add underlying price data if available
    if underlying_prices:
        price_dates = [p["date"] for p in underlying_prices]
        prices = [p["close"] for p in underlying_prices]
        
        fig.add_trace(
            go.Scatter(
                x=price_dates,
                y=prices,
                name="Price",
                line=dict(color="#29B6F6", width=1.5),
            ),
            row=1, col=1
        )
    
    # Add trade entry points
    entry_dates = []
    entry_prices = []
    entry_annotations = []
    
    exit_dates = []
    exit_prices = []
    exit_annotations = []
    
    for i, trade in enumerate(trades):
        # Generate contract name: e.g., "AAPL 240115 150 Put"
        symbol = trade.get("symbol", "")
        expiry = trade.get("expiry", "")
        strike = trade.get("strike", 0)
        right = trade.get("right", "")
        quantity = trade.get("quantity", 0)
        
        # Format contract name
        try:
            expiry_short = expiry[2:] if len(expiry) >= 6 else expiry
            option_type = "Put" if right == "P" else "Call"
            contract_name = f"{symbol} {expiry_short} {strike:.0f} {option_type}"
        except:
            contract_name = f"{symbol} {expiry} {strike} {right}"
        
        # Entry point
        entry_date = trade.get("entry_date", "")
        entry_price = trade.get("underlying_entry", 0)
        if entry_date and entry_price:
            entry_dates.append(entry_date)
            entry_prices.append(entry_price)
            entry_annotations.append(
                f"Entry {i+1}<br>"
                f"{contract_name}<br>"
                f"Qty: {abs(quantity)}x<br>"
                f"@ ${entry_price:.2f}"
            )
        
        # Exit point
        exit_date = trade.get("exit_date", "")
        exit_price = trade.get("underlying_exit", trade.get("underlying_entry", 0))
        if exit_date and exit_price:
            pnl = trade.get("pnl", 0)
            exit_dates.append(exit_date)
            exit_prices.append(exit_price)
            exit_annotations.append(
                f"Exit {i+1}<br>"
                f"{contract_name}<br>"
                f"Qty: {abs(quantity)}x<br>"
                f"@ ${exit_price:.2f}<br>"
                f"P&L: ${pnl:+.2f}"
            )
    
    # Add entry markers
    if entry_dates:
        fig.add_trace(
            go.Scatter(
                x=entry_dates,
                y=entry_prices,
                mode='markers',
                name="Entry Points",
                marker=dict(
                    color="#4CAF50",
                    size=10,
                    symbol="triangle-up",
                    line=dict(color="white", width=1)
                ),
                text=entry_annotations,
                hovertemplate="<b>%{text}</b><extra></extra>",
            ),
            row=1, col=1
        )
    
    # Add exit markers
    if exit_dates:
        fig.add_trace(
            go.Scatter(
                x=exit_dates,
                y=exit_prices,
                mode='markers',
                name="Exit Points",
                marker=dict(
                    color="#F44336",
                    size=10,
                    symbol="triangle-down",
                    line=dict(color="white", width=1)
                ),
                text=exit_annotations,
                hovertemplate="<b>%{text}</b><extra></extra>",
            ),
            row=1, col=1
        )
    
    # Add stock selection change markers (for binbin_god strategy)
    selection_shapes = []
    selection_annotations = []
    if selection_history:
        for i, sel in enumerate(selection_history):
            sel_date = sel.get("date", "")
            from_stock = sel.get("from", "")
            to_stock = sel.get("to", "")
            score = sel.get("score", 0)
            
            if sel_date:
                # Add vertical line shape
                selection_shapes.append(dict(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=sel_date,
                    x1=sel_date,
                    y0=0,
                    y1=1,
                    line=dict(
                        color="#FFA726",
                        width=2,
                        dash="dash",
                    ),
                ))
                
                # Add annotation for selection change
                selection_annotations.append(dict(
                    x=sel_date,
                    y=1,
                    yref="paper",
                    text=f"{from_stock} → {to_stock}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#FFA726",
                    ax=0,
                    ay=-30,
                    font=dict(color="#FFA726", size=10),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor="#FFA726",
                    borderwidth=1,
                ))
    
    # Add P&L curve
    if daily_pnl:
        pnl_dates = [p["date"] for p in daily_pnl]
        cumulative_pnl = [p["cumulative_pnl"] for p in daily_pnl]
        
        fig.add_trace(
            go.Scatter(
                x=pnl_dates,
                y=cumulative_pnl,
                name="Cumulative P&L",
                line=dict(color="#26a69a", width=2),
                fill="tozeroy",
                fillcolor="rgba(38, 166, 154, 0.1)",
            ),
            row=2, col=1
        )
    
    # Add trade P&L annotations on P&L chart
    for i, trade in enumerate(trades):
        exit_date = trade.get("exit_date", "")
        if exit_date:
            # Find corresponding P&L value
            pnl_value = 0
            for pnl_point in daily_pnl:
                if pnl_point["date"] == exit_date:
                    pnl_value = pnl_point["cumulative_pnl"]
                    break
            
            fig.add_annotation(
                x=exit_date,
                y=pnl_value,
                text=f"Trade {i+1}<br>P&L: ${trade.get('pnl', 0):+.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                ax=0,
                ay=-30,
                row=2,
                col=1
            )
    
    # Update layout
    layout_updates = {
        "title": title,
        "template": "plotly_dark",
        "height": 700,
        "margin": dict(l=50, r=50, t=50, b=30),
        "showlegend": True,
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    }
    
    # Add selection shapes and annotations if present
    if selection_shapes:
        layout_updates["shapes"] = selection_shapes
    if selection_annotations:
        layout_updates["annotations"] = selection_annotations
    
    fig.update_layout(**layout_updates)
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative P&L ($)", row=2, col=1)
    
    return fig
