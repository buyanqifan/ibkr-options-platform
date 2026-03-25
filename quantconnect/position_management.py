"""Position management functions for BinbinGod Strategy."""
from datetime import datetime
from AlgorithmImports import OptionRight
from ml_integration import StrategySignal
from option_utils import should_roll_position, calculate_dte
from signals import build_position_data, calculate_pnl_metrics
from execution import make_signal, execute_roll, execute_close
from qc_portfolio import get_option_positions


def check_position_management(algo, execute_signal_func, find_option_func):
    # Get all option positions from QC Portfolio
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
        algo.Log(f"Position check: {pos_info['symbol']} {pos_info['right']} captured={premium_captured_pct:.1f}% DTE={dte} PnL={pnl_pct:.1f}%")
        if algo.ml_enabled and hasattr(algo.ml_integration, 'roll_optimizer'):
            position_data = build_position_data(pos_info, current_price, pnl_pct, dte)
            # pos_info['symbol'] is the underlying stock symbol (e.g., 'NVDA'), not option symbol
            equity = algo.equities.get(pos_info['symbol'])
            underlying_price = algo.Securities[equity.Symbol].Price if equity and algo.Securities.ContainsKey(equity.Symbol) else 0
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
                algo.Log(f"ML Roll: action={roll_rec.action} confidence={roll_rec.confidence:.2f} improvement={roll_rec.expected_pnl_improvement:.2f}")
                if should_roll and roll_rec.action in ["ROLL_FORWARD", "ROLL_OUT", "CLOSE_EARLY"]:
                    algo.Log(f"ML triggered action: {roll_rec.action} - {roll_rec.reasoning}")
                    handle_roll_action(algo, roll_rec, pos_info, pos_id, find_option_func)
                    continue
            except Exception as e:
                algo.Log(f"ML Roll error: {e}")
        action, reasoning = should_roll_position(premium_captured_pct, dte, pnl_pct,
            algo.profit_target_pct, algo._profit_target_disabled, algo.stop_loss_pct, algo._stop_loss_disabled)
        algo.Log(f"Rule-based: action={action} reasoning={reasoning}")
        if action == "ROLL":
            signal = make_signal(pos_info['symbol'], "ROLL", delta=abs(pos_info['delta_at_entry']),
                dte_min=30, dte_max=45, num_contracts=abs(pos_info['quantity']), confidence=0.8, reasoning=reasoning)
            execute_roll(algo, signal, find_option_func)
        elif action.startswith("CLOSE"):
            algo.Log(f"Closing position: {reasoning}")
            signal = make_signal(pos_info['symbol'], "CLOSE", dte_min=0, dte_max=0, num_contracts=0, confidence=0.9, reasoning=reasoning)
            execute_close(algo, signal)


def handle_roll_action(algo, roll_rec, pos_info, position_id, find_option_func):
    if roll_rec.action == "ROLL_FORWARD":
        signal = make_signal(pos_info['symbol'], "ROLL",
            delta=roll_rec.optimal_delta or abs(pos_info['delta_at_entry']),
            dte_min=roll_rec.optimal_dte or 30, dte_max=roll_rec.optimal_dte or 45,
            num_contracts=abs(pos_info['quantity']), confidence=roll_rec.confidence, reasoning=roll_rec.reasoning)
        execute_roll(algo, signal, find_option_func)
    elif roll_rec.action == "ROLL_OUT":
        signal = make_signal(pos_info['symbol'], "ROLL",
            delta=roll_rec.optimal_delta or abs(pos_info['delta_at_entry']),
            dte_min=roll_rec.optimal_dte or 30, dte_max=roll_rec.optimal_dte or 45,
            num_contracts=abs(pos_info['quantity']), confidence=roll_rec.confidence, reasoning=roll_rec.reasoning)
        execute_roll(algo, signal, find_option_func)
    elif roll_rec.action == "CLOSE_EARLY":
        signal = make_signal(pos_info['symbol'], "CLOSE", dte_min=0, dte_max=0, num_contracts=0, confidence=roll_rec.confidence, reasoning=roll_rec.reasoning)
        execute_close(algo, signal)