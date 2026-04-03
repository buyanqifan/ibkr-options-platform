"""Strategy mixin for BinbinGod - Main entry point coordinating all modules."""

from debug_counters import DEFAULT_DEBUG_COUNTERS, increment_debug_counter
from signals import select_best_signal_with_memory
from strategy_init import init_dates, init_parameters, init_ml, init_securities, init_state, schedule_events
from signal_generation import generate_ml_signals
from execution import execute_signal, calculate_dynamic_max_positions
from position_management import check_position_management
from expiry import check_expired_options, update_ml_models
from option_selector import find_option_by_greeks
from qc_portfolio import get_option_position_count, get_symbols_with_holdings


def _select_sp_candidates_for_execution(algo, sp_signals, available_slots: int):
    """Select up to N distinct SP signals, preserving memory bias for the first pick."""
    if not sp_signals or available_slots <= 0:
        return []

    selected = []
    remaining = list(sp_signals)
    max_new_puts = max(1, min(getattr(algo, "max_new_puts_per_day", 1), available_slots))

    best_sp, algo._last_selected_stock, algo._selection_count, algo._last_stock_scores = select_best_signal_with_memory(
        remaining,
        algo._last_selected_stock,
        algo._selection_count,
        algo._min_hold_cycles,
        algo._last_stock_scores,
    )
    if not best_sp or best_sp.confidence < algo.ml_min_confidence:
        return []

    selected.append(best_sp)
    remaining = [signal for signal in remaining if signal.symbol != best_sp.symbol]

    while remaining and len(selected) < max_new_puts:
        candidate = max(remaining, key=lambda x: x.confidence)
        if candidate.confidence < algo.ml_min_confidence:
            break
        selected.append(candidate)
        remaining = [signal for signal in remaining if signal.symbol != candidate.symbol]

    return selected


def rebalance(algo):
    if algo.IsWarmingUp:
        return

    # Dynamically update max_positions based on current stock prices
    algo.max_positions = calculate_dynamic_max_positions(algo)

    check_position_management(algo, execute_signal, find_option_by_greeks)
    signals = generate_ml_signals(algo)
    if not signals:
        return

    # Separate SP and CC signals - both can execute in same cycle
    sp_signals = [s for s in signals if s.action == "SELL_PUT"]
    cc_signals = [s for s in signals if s.action == "SELL_CALL"]

    open_count = get_option_position_count(algo)

    # Execute all eligible CC signals first (if we have stock, we should sell calls)
    if cc_signals:
        for cc_signal in sorted(cc_signals, key=lambda x: x.confidence, reverse=True):
            algo.Log(f"CC_SIGNAL: {cc_signal.symbol} delta={cc_signal.delta:.2f}")
            if cc_signal.confidence >= algo.ml_min_confidence:
                execute_signal(algo, cc_signal, find_option_by_greeks)

    open_count = get_option_position_count(algo)

    # Execute top SP signals, preserving memory for the first pick
    if not sp_signals or open_count >= algo.max_positions:
        return

    available_slots = max(0, algo.max_positions - open_count)
    for sp_signal in _select_sp_candidates_for_execution(algo, sp_signals, available_slots):
        algo.Log(f"SP_SIGNAL: {sp_signal.symbol} delta={sp_signal.delta:.2f}")
        execute_signal(algo, sp_signal, find_option_by_greeks)
        open_count = get_option_position_count(algo)
        if open_count >= algo.max_positions:
            break


def on_end_of_algorithm(algo):
    """Called at end of algorithm - log final results using QC Portfolio data."""
    wr = (algo.winning_trades / algo.total_trades * 100) if algo.total_trades > 0 else 0
    total_profit = algo.Portfolio.TotalProfit
    total_value = algo.Portfolio.TotalPortfolioValue
    initial_capital = algo.initial_capital
    total_return = (total_value - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0
    
    held_symbols = get_symbols_with_holdings(algo, algo.stock_pool)
    total_shares = sum(algo.Portfolio[algo.equities[s].Symbol].Quantity for s in held_symbols if algo.equities.get(s) and algo.Portfolio.ContainsKey(algo.equities[s].Symbol))
    
    algo.Log("=" * 60)
    algo.Log("BINBINGOD STRATEGY RESULTS")
    algo.Log("=" * 60)
    algo.Log(f"Total Trades: {algo.total_trades}")
    algo.Log(f"Winning Trades: {algo.winning_trades}")
    algo.Log(f"Win Rate: {wr:.1f}%")
    algo.Log(f"Initial Capital: ${initial_capital:,.2f}")
    algo.Log(f"Final Portfolio: ${total_value:,.2f}")
    algo.Log(f"Total Profit: ${total_profit:,.2f}")
    algo.Log(f"Total Return: {total_return:.1f}%")
    algo.Log(f"Shares Held: {total_shares}")
    if held_symbols:
        holdings_info = {s: algo.Portfolio[algo.equities[s].Symbol].Quantity for s in held_symbols if algo.equities.get(s) and algo.Portfolio.ContainsKey(algo.equities[s].Symbol)}
        algo.Log(f"Holdings: {holdings_info}")
    algo.Log("")
    algo.Log(algo.ml_integration.get_status_report())
    algo.Log("=" * 60)
