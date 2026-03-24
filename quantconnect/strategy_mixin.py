"""Strategy mixin for BinbinGod - Main entry point coordinating all modules."""
from signals import select_best_signal_with_memory
from strategy_init import init_dates, init_parameters, init_ml, init_securities, init_state, schedule_events
from signal_generation import generate_ml_signals
from execution import execute_signal
from position_management import check_position_management
from expiry import check_expired_options, update_ml_models
from option_selector import find_option_by_greeks


def rebalance(algo):
    if algo.IsWarmingUp:
        return
    check_position_management(algo, execute_signal, find_option_by_greeks)
    open_count = len(algo.open_option_positions)
    algo.Log(f"Rebalance: open_positions={open_count}, max_positions={algo.max_positions}")
    if open_count >= algo.max_positions:
        algo.Log("Rebalance: max positions reached, skipping")
        return
    signals = generate_ml_signals(algo)
    algo.Log(f"Rebalance: generated {len(signals)} signals")
    if not signals:
        algo.Log("Rebalance: no signals generated")
        return
    best_signal, algo._last_selected_stock, algo._selection_count, algo._last_stock_scores = \
        select_best_signal_with_memory(signals, algo._last_selected_stock, algo._selection_count, algo._min_hold_cycles, algo._last_stock_scores)
    if best_signal:
        algo.Log(f"Rebalance: best_signal={best_signal.symbol} confidence={best_signal.confidence:.2f}")
        if best_signal.confidence >= algo.ml_min_confidence:
            execute_signal(algo, best_signal, find_option_by_greeks)
        else:
            algo.Log(f"Rebalance: confidence too low (min={algo.ml_min_confidence})")
    else:
        algo.Log("Rebalance: no best_signal selected")


def on_end_of_algorithm(algo):
    """Called at end of algorithm - log final results using QC Portfolio data."""
    wr = (algo.winning_trades / algo.total_trades * 100) if algo.total_trades > 0 else 0
    total_profit = algo.Portfolio.TotalProfit
    total_value = algo.Portfolio.TotalPortfolioValue
    initial_capital = algo.initial_capital
    total_return = (total_value - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0
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
    algo.Log(f"Final Phase: {algo.phase}")
    algo.Log(f"Shares Held: {algo.stock_holding.shares}")
    if algo.stock_holding.holdings:
        algo.Log(f"Holdings: {algo.stock_holding.holdings}")
    algo.Log("")
    algo.Log(algo.ml_integration.get_status_report())
    algo.Log("=" * 60)