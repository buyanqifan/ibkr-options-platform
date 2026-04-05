"""Position management functions for BinbinGod Strategy."""
from datetime import datetime

from option_utils import should_roll_position, calculate_dte
from signals import calculate_pnl_metrics
from execution import make_signal, execute_roll
from debug_counters import increment_debug_counter
from qc_portfolio import get_option_positions, get_shares_held, get_call_position_contracts


def _clear_assigned_stock_state(algo, symbol: str):
    if hasattr(algo, "assigned_stock_state"):
        algo.assigned_stock_state.pop(symbol, None)


def _normalize_assignment_date(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def _manage_assigned_stock_fail_safe(algo):
    if not getattr(algo, "assigned_stock_fail_safe_enabled", False):
        return

    for symbol, state in list(getattr(algo, "assigned_stock_state", {}).items()):
        shares_held = get_shares_held(algo, symbol)
        if shares_held <= 0:
            _clear_assigned_stock_state(algo, symbol)
            continue

        state["has_covered_call"] = get_call_position_contracts(algo, symbol) > 0
        if state["has_covered_call"] or state.get("force_exit_triggered"):
            continue

        assignment_cost_basis = float(state.get("assignment_cost_basis", 0) or 0)
        assignment_date = _normalize_assignment_date(state.get("assignment_date"))
        equity = algo.equities.get(symbol)
        underlying_price = algo.Securities[equity.Symbol].Price if equity and algo.Securities.ContainsKey(equity.Symbol) else 0
        if assignment_cost_basis <= 0 or underlying_price <= 0 or assignment_date is None:
            continue

        days_held = max(0, (algo.Time.date() - assignment_date.date()).days)
        state["days_held"] = days_held
        if days_held < getattr(algo, "assigned_stock_min_days_held", 5):
            continue

        drawdown_pct = max(0.0, (assignment_cost_basis - underlying_price) / assignment_cost_basis)
        if drawdown_pct < getattr(algo, "assigned_stock_drawdown_pct", 0.12):
            continue

        shares_to_sell = int(shares_held * getattr(algo, "assigned_stock_force_exit_pct", 1.0))
        shares_to_sell = min(shares_held, max(0, shares_to_sell))
        if shares_to_sell <= 0:
            continue

        algo.MarketOrder(equity.Symbol, -shares_to_sell)
        state["force_exit_triggered"] = True
        increment_debug_counter(algo, "assigned_stock_exit")
        algo.Log(
            f"ASSIGNED_STOCK_EXIT:{symbol}:shares={shares_to_sell}:"
            f"drawdown={drawdown_pct:.1%}:days={days_held}"
        )


def check_position_management(algo, execute_signal_func, find_option_func):
    _manage_assigned_stock_fail_safe(algo)
    positions = get_option_positions(algo)
    for pos_info in positions.values():
        option_symbol = pos_info["option_symbol"]
        security = algo.Securities.get(option_symbol)
        if not security:
            continue

        current_price, entry_price = security.Price, pos_info["entry_price"]
        if current_price <= 0 or entry_price <= 0:
            continue

        _, pnl_pct = calculate_pnl_metrics(entry_price, current_price, pos_info["quantity"])
        premium_captured_pct = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0
        dte = calculate_dte(pos_info["expiry"], algo.Time)
        if premium_captured_pct < 0 or dte <= algo.min_dte_for_roll:
            algo.Log(f"POS: {pos_info['symbol']} {pos_info['right']} cap={premium_captured_pct:.0f}% DTE={dte}")

        action, reasoning = should_roll_position(
            premium_captured_pct=premium_captured_pct,
            dte=dte,
            roll_threshold_pct=algo.roll_threshold_pct,
            min_dte_for_roll=algo.min_dte_for_roll,
        )

        if action != "ROLL":
            continue

        delta_at_entry = abs(pos_info.get("delta_at_entry", 0)) or 0.30
        signal = make_signal(
            pos_info["symbol"],
            "ROLL",
            delta=delta_at_entry,
            dte_min=algo.roll_target_dte_min,
            dte_max=algo.roll_target_dte_max,
            num_contracts=abs(pos_info["quantity"]),
            confidence=0.8,
            reasoning=reasoning,
        )
        algo.Log(f"SP_ROLL:{pos_info['symbol']}:{reasoning}")
        execute_roll(algo, signal, find_option_func, existing_position=pos_info)
