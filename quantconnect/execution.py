"""Trade execution functions for BinbinGod Strategy."""

from typing import Dict, Optional

from AlgorithmImports import OptionRight, OrderStatus, Resolution, SecurityType

from ml_integration import StrategySignal
from signals import calculate_pnl_metrics
from debug_counters import increment_debug_counter
from qc_portfolio import (
    get_option_position_count,
    get_shares_held,
    get_call_position_contracts,
    get_position_for_symbol,
    save_position_metadata,
    remove_position_metadata,
)

MAX_DEFERRED_OPEN_ATTEMPTS = 3


def _log_option_selection_failure(algo, signal: StrategySignal, target_right, target_delta: float):
    stats = getattr(algo, "_last_option_selection_stats", None)
    right_label = "CALL" if target_right == OptionRight.Call else "PUT"
    if signal.action == "SELL_CALL":
        increment_debug_counter(algo, "cc_option_filter_block")
    if not isinstance(stats, dict):
        algo.Log(f"OPTION_FILTER_BLOCK:{signal.symbol}:{right_label}:delta={target_delta:.2f}:stats=unavailable")
        return
    counts = stats.get("stats", {})
    algo.Log(
        f"OPTION_FILTER_BLOCK:{signal.symbol}:{right_label}:delta={target_delta:.2f}:"
        f"dte={stats.get('dte_min')}-{stats.get('dte_max')}:"
        f"tol={stats.get('delta_tolerance', 0):.2f}:"
        f"minstrike={float(stats.get('min_strike') or 0):.2f}:"
        f"minpremium={float(stats.get('min_premium') or 0):.2f}:"
        f"chain={stats.get('total_chain', 0)}:"
        f"right={counts.get('right', 0)}:"
        f"dtepass={counts.get('dte', 0)}:"
        f"strike_block={counts.get('min_strike', 0)}:"
        f"itm_block={counts.get('itm', 0)}:"
        f"delta_missing={counts.get('delta_none', 0)}:"
        f"delta_block={counts.get('tolerance', 0)}:"
        f"premium_block={counts.get('premium', 0)}:"
        f"suitable={stats.get('suitable_count', 0)}"
    )


def calculate_dynamic_max_positions(algo) -> int:
    max_positions_ceiling = max(1, int(getattr(algo, "max_positions_ceiling", 1)))
    symbols = list(getattr(algo, "stock_pool", []) or [])
    if not symbols and getattr(algo, "equities", None):
        symbols = list(getattr(algo, "equities").keys())

    prices = []
    for symbol in symbols:
        equity = getattr(algo, "equities", {}).get(symbol)
        equity_symbol = getattr(equity, "Symbol", symbol)
        security = getattr(algo, "Securities", {}).get(equity_symbol)
        price = float(getattr(security, "Price", 0.0) or 0.0) if security is not None else 0.0
        if price > 0:
            prices.append(price)

    if not prices:
        return max_positions_ceiling

    avg_price = sum(prices) / len(prices)
    portfolio = getattr(algo, "Portfolio", None)
    portfolio_value = float(
        getattr(portfolio, "TotalPortfolioValue", getattr(algo, "initial_capital", 0.0)) or 0.0
    )
    margin_budget = max(0.0, portfolio_value) * float(getattr(algo, "target_margin_utilization", 1.0))
    margin_per_contract = avg_price * 100 * 0.20
    if margin_per_contract <= 0:
        return max_positions_ceiling

    dynamic_max = int(margin_budget / margin_per_contract)
    return max(1, min(dynamic_max, max_positions_ceiling))


def _format_pending_expiry(expiry) -> str:
    if hasattr(expiry, "strftime"):
        return expiry.strftime("%Y%m%d")
    return str(expiry).replace("-", "")


def _format_pending_right(target_right) -> str:
    return "C" if target_right in (OptionRight.Call, "Call", "C") else "P"


def _build_pending_open_key(signal, selected, target_right, quantity):
    return (
        f"{signal.symbol}_{_format_pending_expiry(selected['expiry'])}_"
        f"{float(selected['strike']):.0f}_{_format_pending_right(target_right)}_{quantity}"
    )


def _enqueue_pending_open_order(algo, queue_key, payload):
    if not hasattr(algo, "pending_open_orders") or algo.pending_open_orders is None:
        algo.pending_open_orders = {}
    algo.pending_open_orders[queue_key] = payload


def safe_execute_option_order(algo, option_symbol, quantity, theoretical_price, deferred_context=None):
    if not algo.Securities.ContainsKey(option_symbol):
        algo.Log(f"Subscribing to option contract: {option_symbol}")
        algo.AddOptionContract(option_symbol, Resolution.Minute)

    security = algo.Securities[option_symbol]
    if security.HasData and security.Price > 0:
        return algo.MarketOrder(option_symbol, quantity)

    algo.Log(f"ORDER_DEFERRED: {option_symbol} waiting for first bar")
    if deferred_context:
        _enqueue_pending_open_order(
            algo,
            deferred_context["queue_key"],
            {
                **deferred_context,
                "option_symbol": option_symbol,
                "quantity": quantity,
                "theoretical_price": theoretical_price,
                "attempt_count": deferred_context.get("attempt_count", 0),
            },
        )
    return None


def make_signal(symbol, action, delta=0, dte_min=30, dte_max=45, num_contracts=1, confidence=0.5, reasoning=""):
    return StrategySignal(
        symbol=symbol,
        action=action,
        delta=delta,
        dte_min=dte_min,
        dte_max=dte_max,
        num_contracts=num_contracts,
        confidence=confidence,
        reasoning=reasoning,
        expected_premium=0.0,
        expected_return=0.0,
        expected_risk=0.0,
        assignment_probability=0.0,
    )


def _enqueue_open_order_metadata(algo, ticket, signal: StrategySignal, selected: Dict, target_right):
    if not ticket or ticket.OrderId is None:
        return
    if not hasattr(algo, "pending_order_metadata"):
        algo.pending_order_metadata = {}
    strategy_phase = "SP" if target_right == OptionRight.Put else "CC"
    algo.pending_order_metadata[ticket.OrderId] = {
        "symbol": signal.symbol,
        "right": "P" if target_right == OptionRight.Put else "C",
        "strike": float(selected["strike"]),
        "expiry": selected["expiry"].strftime("%Y%m%d"),
        "delta_at_entry": selected.get("delta", 0),
        "iv_at_entry": selected.get("iv", 0.25),
        "strategy_phase": strategy_phase,
        "entry_date": algo.Time.strftime("%Y-%m-%d"),
        "ml_signal": signal,
    }


def _build_position_key(symbol: str, expiry, strike: float, right: str) -> str:
    return f"{symbol}_{expiry.strftime('%Y%m%d')}_{strike:.0f}_{right}"


def retry_pending_open_orders(algo, _find_option_func=None):
    pending = getattr(algo, "pending_open_orders", None)
    if not pending:
        return []

    completed = []
    items = sorted(pending.items(), key=lambda kv: 0 if kv[1]["signal"].action == "SELL_CALL" else 1)
    for queue_key, item in items:
        attempts = int(item.get("attempt_count", 0))
        if attempts >= MAX_DEFERRED_OPEN_ATTEMPTS:
            algo.Log(f"ORDER_DEFERRED_EXPIRED: {queue_key}")
            completed.append(queue_key)
            continue

        security = algo.Securities[item["option_symbol"]] if algo.Securities.ContainsKey(item["option_symbol"]) else None
        if security is None or not security.HasData or security.Price <= 0:
            item["attempt_count"] = attempts + 1
            continue

        algo.Log(f"RETRY_DEFERRED_OPEN: {queue_key}")
        ticket = algo.MarketOrder(item["option_symbol"], item["quantity"])
        _enqueue_open_order_metadata(algo, ticket, item["signal"], item["selected"], item["target_right"])
        completed.append(queue_key)

    for queue_key in completed:
        pending.pop(queue_key, None)
    return completed


def handle_order_event(algo, order_event):
    if order_event.Status != OrderStatus.Filled:
        return
    symbol = order_event.Symbol
    if not hasattr(symbol, "SecurityType") or symbol.SecurityType != SecurityType.Option:
        return
    if order_event.FillQuantity >= 0:
        return
    pending = getattr(algo, "pending_order_metadata", {}).pop(order_event.OrderId, None)
    if not pending:
        return
    pos_id = f"{pending['symbol']}_{pending['expiry']}_{pending['strike']:.0f}_{pending['right']}"
    save_position_metadata(
        algo,
        pos_id,
        {
            "delta_at_entry": pending["delta_at_entry"],
            "iv_at_entry": pending["iv_at_entry"],
            "strategy_phase": pending["strategy_phase"],
            "entry_date": pending["entry_date"],
            "ml_signal": pending["ml_signal"],
        },
    )


def execute_signal(algo, signal: StrategySignal, find_option_func):
    if not signal or signal.action == "HOLD":
        return
    if signal.action == "ROLL":
        execute_roll(algo, signal, find_option_func)
        return
    if signal.action == "CLOSE":
        execute_close(algo, signal)
        return

    equity = algo.equities.get(signal.symbol)
    if not equity:
        return
    underlying_price = algo.Securities[equity.Symbol].Price
    target_right = OptionRight.Put if signal.action == "SELL_PUT" else OptionRight.Call
    target_delta = -signal.delta if target_right == OptionRight.Put else signal.delta
    min_strike = getattr(signal, "min_strike", 0.0)
    selected = find_option_func(
        algo,
        symbol=signal.symbol,
        equity_symbol=equity.Symbol,
        target_right=target_right,
        target_delta=target_delta,
        dte_min=signal.dte_min,
        dte_max=signal.dte_max,
        delta_tolerance=0.08,
        min_strike=min_strike if min_strike > 0 else None,
        selection_tiers=getattr(signal, "selection_tiers", None),
    )
    if not selected:
        algo.Log(f"No suitable options for {signal.symbol} delta ~{target_delta:.2f}")
        _log_option_selection_failure(algo, signal, target_right, target_delta)
        increment_debug_counter(algo, "no_suitable_options")
        return
    if signal.action == "SELL_CALL" and selected.get("selection_tier") and selected["selection_tier"] != "primary":
        algo.Log(f"CC_SELECTION_TIER:{signal.symbol}:{selected['selection_tier']}")

    current_positions = get_option_position_count(algo)
    if target_right == OptionRight.Put:
        quantity = calculate_put_quantity(algo, selected, current_positions, underlying_price, signal.symbol)
    else:
        shares_held = get_shares_held(algo, signal.symbol)
        existing_call_contracts = get_call_position_contracts(algo, signal.symbol)
        shares_covered = existing_call_contracts * 100
        shares_available = shares_held - shares_covered
        quantity = min(max(0, shares_available // 100), algo.max_positions)
        if quantity <= 0:
            increment_debug_counter(algo, "cc_share_block")
            algo.Log(f"No available shares for {signal.symbol} call: held={shares_held}, covered={shares_covered}")
            return

    if quantity <= 0:
        return

    quantity = -quantity
    option_symbol = selected["option_symbol"]
    queue_key = _build_pending_open_key(signal, selected, target_right, quantity)
    deferred_context = {
        "queue_key": queue_key,
        "signal": signal,
        "selected": selected,
        "target_right": target_right,
        "attempt_count": getattr(algo, "pending_open_orders", {}).get(queue_key, {}).get("attempt_count", 0),
    }
    ticket = safe_execute_option_order(algo, option_symbol, quantity, selected["premium"], deferred_context=deferred_context)
    _enqueue_open_order_metadata(algo, ticket, signal, selected, target_right)


def calculate_put_quantity(algo, selected: Dict, current_positions: int, underlying_price: float, symbol: str) -> int:
    strike = float(selected["strike"])
    premium = float(selected.get("premium", 0) or 0)
    otm_amount = max(0.0, underlying_price - strike)
    margin_method_1 = 0.20 * underlying_price * 100 - otm_amount * 100
    margin_method_2 = 0.10 * strike * 100
    estimated_margin_per_contract = max(margin_method_1, margin_method_2) + premium * 100
    estimated_margin_per_contract = max(estimated_margin_per_contract, strike * 100 * 0.20)

    portfolio_value = max(algo.Portfolio.TotalPortfolioValue, 0.0)
    remaining_margin_budget = max(0.0, portfolio_value * algo.target_margin_utilization - algo.Portfolio.TotalMarginUsed)
    current_inventory_value = 0.0
    for holding in algo.Portfolio.Values:
        if not holding.Invested:
            continue
        hs = holding.Symbol
        if not (hasattr(hs, "SecurityType") and hs.SecurityType == SecurityType.Equity):
            continue
        current_inventory_value += abs(float(getattr(holding, "HoldingsValue", 0.0) or 0.0))
    remaining_inventory_budget = max(0.0, portfolio_value * getattr(algo, "assigned_stock_inventory_cap_pct", 0.30) - current_inventory_value)

    symbol_assignment_exposure = get_shares_held(algo, symbol) * max(underlying_price, 0)
    for holding in algo.Portfolio.Values:
        if not holding.Invested:
            continue
        hs = holding.Symbol
        if not (hasattr(hs, "SecurityType") and hs.SecurityType == SecurityType.Option):
            continue
        if hs.ID.OptionRight != OptionRight.Put:
            continue
        if hasattr(hs, "Underlying") and hs.Underlying and hs.Underlying.Value == symbol:
            symbol_assignment_exposure += abs(holding.Quantity) * float(hs.ID.StrikePrice) * 100

    candidate_assignment = strike * 100
    remaining_symbol_cap = max(0.0, portfolio_value * algo.symbol_assignment_base_cap - symbol_assignment_exposure)
    remaining_trade_cap = max(0.0, portfolio_value * algo.max_assignment_risk_per_trade)
    remaining_slot_count = max(0, algo.max_positions - current_positions)

    portfolio_qty = int(remaining_margin_budget / estimated_margin_per_contract) if estimated_margin_per_contract > 0 else 0
    inventory_qty = int(remaining_inventory_budget / candidate_assignment) if candidate_assignment > 0 else 0
    symbol_qty = int(remaining_symbol_cap / candidate_assignment) if candidate_assignment > 0 else 0
    trade_qty = int(remaining_trade_cap / candidate_assignment) if candidate_assignment > 0 else 0
    quantity = min(portfolio_qty, inventory_qty, symbol_qty, trade_qty, remaining_slot_count)
    if quantity > 0:
        return quantity

    reasons = {
        "portfolio_margin": portfolio_qty,
        "inventory_cap": inventory_qty,
        "symbol_assignment": symbol_qty,
        "trade_assignment": trade_qty,
        "position_slots": remaining_slot_count,
    }
    block_reason = min(reasons.items(), key=lambda item: item[1])[0]
    increment_debug_counter(algo, "put_block")
    if block_reason == "inventory_cap":
        increment_debug_counter(algo, "assigned_stock_inventory_block")
    algo.Log(
        f"PUT_BLOCK:{symbol}:reason={block_reason}:"
        f"portfolio={portfolio_qty}:inventory={inventory_qty}:symbol={symbol_qty}:trade={trade_qty}:slots={remaining_slot_count}"
    )
    return 0


def execute_roll(algo, signal: StrategySignal, find_option_func, existing_position: Optional[Dict] = None):
    existing = existing_position or get_position_for_symbol(algo, signal.symbol)
    if not existing:
        return
    pos_info = existing
    pos_id = _build_position_key(signal.symbol, pos_info["expiry"], pos_info["strike"], pos_info["right"])
    close_ticket = safe_execute_option_order(algo, pos_info["option_symbol"], -pos_info["quantity"], pos_info["entry_price"])
    if not close_ticket or close_ticket.Status != OrderStatus.Filled:
        if not hasattr(algo, "pending_roll_orders"):
            algo.pending_roll_orders = {}
        algo.pending_roll_orders[pos_id] = {
            "symbol": signal.symbol,
            "existing_position": pos_info,
            "signal": signal,
            "queued_at": getattr(algo, "Time", None),
            "close_order_id": getattr(close_ticket, "OrderId", None) if close_ticket else None,
        }
        return

    pnl, _ = calculate_pnl_metrics(pos_info["entry_price"], close_ticket.AverageFillPrice, pos_info["quantity"])
    record_trade(algo, signal.symbol, pos_info["right"], pnl, "ROLL")
    remove_position_metadata(algo, pos_id)

    equity = algo.equities.get(signal.symbol)
    if not equity:
        return
    target_right = OptionRight.Put if pos_info["right"] == "P" else OptionRight.Call
    target_delta = -signal.delta if target_right == OptionRight.Put else signal.delta
    new_selected = find_option_func(
        algo,
        symbol=signal.symbol,
        equity_symbol=equity.Symbol,
        target_right=target_right,
        target_delta=target_delta,
        dte_min=signal.dte_min,
        dte_max=signal.dte_max,
    )
    if new_selected:
        new_qty = pos_info["quantity"]
        new_ticket = safe_execute_option_order(algo, new_selected["option_symbol"], new_qty, new_selected["premium"])
        _enqueue_open_order_metadata(algo, new_ticket, signal, new_selected, target_right)


def execute_close(algo, signal: StrategySignal, existing_position: Optional[Dict] = None):
    pos_info = existing_position or get_position_for_symbol(algo, signal.symbol)
    if not pos_info:
        return
    pos_id = _build_position_key(signal.symbol, pos_info["expiry"], pos_info["strike"], pos_info["right"])
    close_ticket = safe_execute_option_order(algo, pos_info["option_symbol"], -pos_info["quantity"], pos_info["entry_price"])
    if not close_ticket or close_ticket.Status != OrderStatus.Filled:
        if not hasattr(algo, "pending_close_orders"):
            algo.pending_close_orders = {}
        algo.pending_close_orders[pos_id] = {
            "symbol": signal.symbol,
            "existing_position": pos_info,
            "reason": signal.reasoning or "SIGNAL_CLOSE",
            "queued_at": getattr(algo, "Time", None),
            "close_order_id": getattr(close_ticket, "OrderId", None) if close_ticket else None,
        }
        return

    pnl, _ = calculate_pnl_metrics(pos_info["entry_price"], close_ticket.AverageFillPrice, pos_info["quantity"])
    record_trade(algo, signal.symbol, pos_info["right"], pnl, signal.reasoning or "SIGNAL_CLOSE")
    remove_position_metadata(algo, pos_id)


def record_trade(algo, symbol: str, right: str, pnl: float, reason: str):
    algo.total_trades += 1
    algo.total_pnl += pnl
    if pnl > 0:
        algo.winning_trades += 1
    algo.trade_history.append(
        {"date": algo.Time.strftime("%Y-%m-%d"), "symbol": symbol, "type": right, "pnl": pnl, "exit_reason": reason}
    )
