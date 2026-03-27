"""Strategy mixin for BinbinGod - Main entry point coordinating all modules."""
from signals import select_best_signal_with_memory
from strategy_init import init_dates, init_parameters, init_ml, init_securities, init_state, schedule_events
from signal_generation import generate_ml_signals
from execution import execute_signal, calculate_dynamic_max_positions
from position_management import check_position_management
from expiry import check_expired_options, update_ml_models
from option_selector import find_option_by_greeks
from qc_portfolio import get_option_position_count, get_symbols_with_holdings


def rebalance(algo):
    if algo.IsWarmingUp:
        return
    
    # Dynamically update max_positions based on current stock prices
    algo.max_positions = calculate_dynamic_max_positions(algo)
    
    check_position_management(algo, execute_signal, find_option_by_greeks)
    open_count = get_option_position_count(algo)
    if open_count >= algo.max_positions:
        return
    signals = generate_ml_signals(algo)
    if not signals:
        return
    
    # Separate SP and CC signals - both can execute in same cycle
    sp_signals = [s for s in signals if s.action == "SELL_PUT"]
    cc_signals = [s for s in signals if s.action == "SELL_CALL"]
    
    # Execute best CC signal first (if we have stock, we should sell calls)
    if cc_signals and open_count < algo.max_positions:
        best_cc = max(cc_signals, key=lambda x: x.confidence)
        algo.Log(f"CC_SIGNAL: {best_cc.symbol} delta={best_cc.delta:.2f}")
        if best_cc.confidence >= algo.ml_min_confidence:
            execute_signal(algo, best_cc, find_option_by_greeks)
            open_count = get_option_position_count(algo)
    
    # Execute best SP signal (with memory to avoid frequent switching)
    if sp_signals and open_count < algo.max_positions:
        best_sp, algo._last_selected_stock, algo._selection_count, algo._last_stock_scores = \
            select_best_signal_with_memory(sp_signals, algo._last_selected_stock, algo._selection_count, algo._min_hold_cycles, algo._last_stock_scores)
        if best_sp:
            algo.Log(f"SP_SIGNAL: {best_sp.symbol} delta={best_sp.delta:.2f}")
            if best_sp.confidence >= algo.ml_min_confidence:
                execute_signal(algo, best_sp, find_option_by_greeks)


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
