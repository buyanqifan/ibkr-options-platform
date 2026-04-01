"""State reconstruction helpers for live IBKR positions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _stock_cost_per_share(position: dict[str, Any]) -> float:
    qty = abs(float(position.get("position", 0) or 0))
    if qty <= 0:
        return 0.0
    avg_cost = float(position.get("avgCost", 0) or 0)
    return avg_cost / qty


def reconstruct_live_state(
    *,
    account_summary: dict[str, Any],
    positions: list[dict[str, Any]],
    open_orders: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild strategy-visible phase/risk state from broker snapshots."""
    stock_pool = config.get("stock_pool") or []
    by_symbol: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"stock": [], "puts": [], "calls": []})
    for item in positions:
        symbol = item.get("symbol")
        if not symbol:
            continue
        sec_type = item.get("secType")
        right = item.get("right")
        if sec_type == "STK":
            by_symbol[symbol]["stock"].append(item)
        elif sec_type == "OPT" and right == "P" and float(item.get("position", 0) or 0) < 0:
            by_symbol[symbol]["puts"].append(item)
        elif sec_type == "OPT" and right == "C" and float(item.get("position", 0) or 0) < 0:
            by_symbol[symbol]["calls"].append(item)

    net_liq = float(account_summary.get("NetLiquidation", 0) or 0)
    symbols: dict[str, dict[str, Any]] = {}
    phase_counts: dict[str, int] = defaultdict(int)
    anomaly_count = 0
    for symbol in stock_pool or list(by_symbol.keys()):
        state = by_symbol.get(symbol, {"stock": [], "puts": [], "calls": []})
        blocked_reasons: list[str] = []
        stock_positions = state["stock"]
        stock_position = stock_positions[0] if stock_positions else None
        short_puts = state["puts"]
        short_calls = state["calls"]

        if stock_position and short_calls:
            stock_cost = _stock_cost_per_share(stock_position)
            call_strike = float(short_calls[0].get("strike", 0) or 0)
            phase = "repair_call" if call_strike < stock_cost else "covered_call"
        elif stock_position:
            phase = "assigned_stock"
        elif short_puts:
            phase = "short_put"
        else:
            phase = "cash"

        stock_value = abs(float(stock_position.get("marketValue", 0) or 0)) if stock_position else 0.0
        stock_ratio = (stock_value / net_liq) if net_liq > 0 else 0.0
        if stock_position and stock_ratio >= float(config.get("stock_inventory_block_threshold", 1.0) or 1.0):
            blocked_reasons.append("inventory_cap")
        if phase == "cash" and any(order.get("symbol") == symbol for order in open_orders):
            blocked_reasons.append("pending_entry_order")

        symbol_state = {
            "phase": phase,
            "blocked_reasons": blocked_reasons,
            "stock_position": stock_position,
            "short_puts": short_puts,
            "short_calls": short_calls,
            "open_order_count": sum(1 for order in open_orders if order.get("symbol") == symbol),
            "recent_fill_count": sum(1 for fill in fills if fill.get("symbol") == symbol),
            "stock_value_ratio": round(stock_ratio, 4),
        }
        if phase == "cash" and blocked_reasons:
            anomaly_count += 1
        symbols[symbol] = symbol_state
        phase_counts[phase] += 1

    return {
        "status": "ready",
        "symbols": symbols,
        "phase_counts": dict(phase_counts),
        "open_orders_count": len(open_orders),
        "fills_count": len(fills),
        "anomaly_count": anomaly_count,
    }
