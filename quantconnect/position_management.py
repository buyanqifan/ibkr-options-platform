"""Position management functions for BinbinGod Strategy."""
from datetime import datetime
from AlgorithmImports import OptionRight
from ml_integration import StrategySignal
from option_utils import should_roll_position, calculate_dte, should_defensively_roll_short_put
from signals import build_position_data, calculate_pnl_metrics
from execution import make_signal, execute_roll, execute_close
from helpers import set_symbol_cooldown
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

        covered_call_contracts = get_call_position_contracts(algo, symbol)
        if covered_call_contracts > 0:
            state["repair_failures"] = 0
            continue

        assignment_cost_basis = float(state.get("assignment_cost_basis", 0) or 0)
        assignment_date = _normalize_assignment_date(state.get("assignment_date"))
        equity = algo.equities.get(symbol)
        underlying_price = algo.Securities[equity.Symbol].Price if equity and algo.Securities.ContainsKey(equity.Symbol) else 0
        if assignment_cost_basis <= 0 or underlying_price <= 0 or assignment_date is None:
            continue

        days_held = max(0, (algo.Time.date() - assignment_date.date()).days)
        if days_held < getattr(algo, "assigned_stock_min_days_held", 5):
            continue

        drawdown_pct = max(0.0, (assignment_cost_basis - underlying_price) / assignment_cost_basis)
        if drawdown_pct < getattr(algo, "assigned_stock_drawdown_pct", 0.12):
            continue

        last_attempt = state.get("last_repair_attempt")
        if isinstance(last_attempt, datetime) and last_attempt.date() == algo.Time.date():
            continue

        state["repair_failures"] = int(state.get("repair_failures", 0)) + 1
        state["last_repair_attempt"] = algo.Time
        increment_debug_counter(algo, "assigned_repair_fail")
        algo.Log(
            f"ASSIGNED_REPAIR_FAIL:{symbol}:failures={state['repair_failures']}:"
            f"drawdown={drawdown_pct:.1%}:days={days_held}"
        )

        if state["repair_failures"] < getattr(algo, "assigned_stock_repair_attempt_limit", 3):
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
    for pos_id, pos_info in positions.items():
        option_symbol = pos_info['option_symbol']
        security = algo.Securities.get(option_symbol)
        if not security:
            continue
        current_price, entry_price = security.Price, pos_info['entry_price']
        if current_price <= 0 or entry_price <= 0:
            continue
        pnl, pnl_pct = calculate_pnl_metrics(entry_price, current_price, pos_info['quantity'])
        premium_captured_pct = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0
        dte = calculate_dte(pos_info['expiry'], algo.Time)
        equity = algo.equities.get(pos_info['symbol'])
        underlying_price = algo.Securities[equity.Symbol].Price if equity and algo.Securities.ContainsKey(equity.Symbol) else 0
        # Only log positions with issues (losing or near expiry)
        if premium_captured_pct < 0 or dte <= 7:
            algo.Log(f"POS: {pos_info['symbol']} {pos_info['right']} cap={premium_captured_pct:.0f}% DTE={dte}")

        if pos_info.get("right") == "P" and algo.defensive_put_roll_enabled:
            should_def_roll, def_reason = should_defensively_roll_short_put(
                underlying_price=underlying_price,
                strike=pos_info['strike'],
                dte=dte,
                pnl_pct=pnl_pct,
                min_dte=algo.defensive_put_roll_min_dte,
                max_dte=algo.defensive_put_roll_max_dte,
                itm_buffer_pct=algo.defensive_put_roll_itm_buffer_pct,
                max_loss_pct=algo.defensive_put_roll_loss_pct,
            )
            if should_def_roll:
                target_delta = abs(pos_info['delta_at_entry']) if pos_info.get('delta_at_entry', 0) > 0 else 0.30
                target_delta = min(target_delta, algo.defensive_put_roll_delta)
                signal = make_signal(
                    pos_info['symbol'], "ROLL",
                    delta=target_delta,
                    dte_min=algo.defensive_put_roll_dte_min,
                    dte_max=max(algo.defensive_put_roll_dte_min, algo.defensive_put_roll_dte_max),
                    num_contracts=abs(pos_info['quantity']),
                    confidence=0.95,
                    reasoning=def_reason,
                )
                set_symbol_cooldown(
                    algo,
                    pos_info['symbol'],
                    algo.large_loss_cooldown_days,
                    "put_defensive_roll",
                )
                execute_roll(algo, signal, find_option_func, existing_position=pos_info)
                continue
        
        if algo.ml_enabled and hasattr(algo.ml_integration, 'roll_optimizer'):
            position_data = build_position_data(pos_info, current_price, pnl_pct, dte)
            market_data = {
                'price': underlying_price,
                'iv': 0.25,
                'vix': algo._current_vix,
                'option_price': current_price
            }
            try:
                should_roll, roll_rec = algo.ml_integration.roll_optimizer.should_roll(
                    position=position_data, market_data=market_data,
                    current_date=algo.Time.strftime('%Y-%m-%d'), min_confidence=algo.ml_min_confidence)
                
                # SP policy: short-put positions should not use CLOSE_EARLY/ROLL_OUT.
                if pos_info.get("right") == "P" and roll_rec.action == "CLOSE_EARLY":
                    should_roll = False
                elif pos_info.get("right") == "P" and roll_rec.action == "ROLL_OUT" and not algo.defensive_put_roll_enabled:
                    should_roll = False

                if should_roll and roll_rec.action in ["ROLL_FORWARD", "ROLL_OUT", "CLOSE_EARLY"]:
                    algo.Log(f"ML triggered action: {roll_rec.action} - {roll_rec.reasoning}")
                    if pos_info.get("right") == "P" and roll_rec.action == "ROLL_OUT":
                        set_symbol_cooldown(algo, pos_info['symbol'], algo.large_loss_cooldown_days, "ml_put_roll")
                    handle_roll_action(algo, roll_rec, pos_info, pos_id, find_option_func)
                    continue
            except Exception as e:
                pass
        
        strategy_phase = "SP" if pos_info.get("right") == "P" else pos_info.get("strategy_phase", "CC")
        action, reasoning = should_roll_position(
            premium_captured_pct, dte, pnl_pct,
            algo.profit_target_pct, algo._profit_target_disabled, algo.stop_loss_pct, algo._stop_loss_disabled,
            strategy_phase=strategy_phase)
        
        if action == "ROLL":
            if pos_info.get("right") == "P" and pnl_pct <= -algo.large_loss_cooldown_pct:
                set_symbol_cooldown(algo, pos_info['symbol'], algo.large_loss_cooldown_days, "put_loss_roll")
            signal = make_signal(pos_info['symbol'], "ROLL", delta=abs(pos_info['delta_at_entry']),
                dte_min=30, dte_max=45, num_contracts=abs(pos_info['quantity']), confidence=0.8, reasoning=reasoning)
            execute_roll(algo, signal, find_option_func, existing_position=pos_info)
        elif action.startswith("CLOSE"):
            algo.Log(f"CLOSE: {reasoning}")
            if pos_info.get("right") == "P" and pnl_pct <= -algo.large_loss_cooldown_pct:
                set_symbol_cooldown(algo, pos_info['symbol'], algo.large_loss_cooldown_days, "put_loss_close")
            signal = make_signal(pos_info['symbol'], "CLOSE", dte_min=0, dte_max=0, num_contracts=0, confidence=0.9, reasoning=reasoning)
            execute_close(algo, signal, existing_position=pos_info)


def handle_roll_action(algo, roll_rec, pos_info, position_id, find_option_func):
    if roll_rec.action == "ROLL_FORWARD":
        signal = make_signal(pos_info['symbol'], "ROLL",
            delta=roll_rec.optimal_delta or abs(pos_info['delta_at_entry']),
            dte_min=roll_rec.optimal_dte or 30, dte_max=roll_rec.optimal_dte or 45,
            num_contracts=abs(pos_info['quantity']), confidence=roll_rec.confidence, reasoning=roll_rec.reasoning)
        execute_roll(algo, signal, find_option_func, existing_position=pos_info)
    elif roll_rec.action == "ROLL_OUT":
        signal = make_signal(pos_info['symbol'], "ROLL",
            delta=roll_rec.optimal_delta or abs(pos_info['delta_at_entry']),
            dte_min=roll_rec.optimal_dte or 30, dte_max=roll_rec.optimal_dte or 45,
            num_contracts=abs(pos_info['quantity']), confidence=roll_rec.confidence, reasoning=roll_rec.reasoning)
        execute_roll(algo, signal, find_option_func, existing_position=pos_info)
    elif roll_rec.action == "CLOSE_EARLY":
        signal = make_signal(pos_info['symbol'], "CLOSE", dte_min=0, dte_max=0, num_contracts=0, confidence=roll_rec.confidence, reasoning=roll_rec.reasoning)
        execute_close(algo, signal, existing_position=pos_info)
