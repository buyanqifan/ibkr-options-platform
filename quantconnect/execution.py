"""Trade execution functions for BinbinGod Strategy."""
from typing import Dict, Optional
from AlgorithmImports import OptionRight, OrderStatus, Resolution
from ml_integration import StrategySignal
from option_utils import calculate_dte
from signals import calculate_pnl_metrics
from option_pricing import BlackScholes
from qc_portfolio import (
    get_option_position_count, get_shares_held, get_call_position_contracts,
    get_position_for_symbol, save_position_metadata, remove_position_metadata,
    get_total_stock_holdings_value, get_stock_holding_count
)

RISK_FREE_RATE = 0.05


def bs_put_price(S, K, T, sigma):
    return BlackScholes.put_price(S, K, T, RISK_FREE_RATE, sigma)


def bs_call_price(S, K, T, sigma):
    return BlackScholes.call_price(S, K, T, RISK_FREE_RATE, sigma)


def safe_execute_option_order(algo, option_symbol, quantity, theoretical_price):
    """Safely execute option order with data readiness check.
    
    QC dynamically subscribed options may not have price data immediately.
    This function checks if data is ready and uses LimitOrder as fallback.
    
    Args:
        algo: QCAlgorithm instance
        option_symbol: Option Symbol object
        quantity: Number of contracts (negative for sell)
        theoretical_price: BS theoretical price for limit order fallback
    
    Returns:
        OrderTicket or None
    """
    # Subscribe if not already in Securities
    if not algo.Securities.ContainsKey(option_symbol):
        algo.Log(f"Subscribing to option contract: {option_symbol}")
        algo.AddOptionContract(option_symbol, Resolution.Daily)
    
    security = algo.Securities[option_symbol]
    
    # Check if price data is available
    if security.HasData and security.Price > 0:
        # Data ready - use MarketOrder
        
        return algo.MarketOrder(option_symbol, quantity)
    else:
        # Data not ready - use LimitOrder with theoretical price
        # For sell orders (negative quantity), set limit slightly below theoretical
        limit_price = theoretical_price * 0.98 if quantity < 0 else theoretical_price * 1.02
        
        return algo.LimitOrder(option_symbol, quantity, limit_price)


def make_signal(symbol, action, delta=0, dte_min=30, dte_max=45, num_contracts=1, confidence=0.5, reasoning=""):
    return StrategySignal(symbol=symbol, action=action, delta=delta, dte_min=dte_min, dte_max=dte_max,
        num_contracts=num_contracts, confidence=confidence, reasoning=reasoning,
        expected_premium=0.0, expected_return=0.0, expected_risk=0.0, assignment_probability=0.0)


def execute_signal(algo, signal: StrategySignal, find_option_func):
    if not signal or signal.action == "HOLD": return
    if signal.action == "ROLL": execute_roll(algo, signal, find_option_func); return
    if signal.action == "CLOSE": execute_close(algo, signal); return
    equity = algo.equities.get(signal.symbol)
    if not equity: return
    underlying_price = algo.Securities[equity.Symbol].Price
    target_right = OptionRight.Put if signal.action == "SELL_PUT" else OptionRight.Call
    target_delta = -signal.delta if target_right == OptionRight.Put else signal.delta
    min_strike = getattr(signal, 'min_strike', 0.0)
    selected = find_option_func(algo, symbol=signal.symbol, equity_symbol=equity.Symbol,
        target_right=target_right, target_delta=target_delta, dte_min=signal.dte_min, dte_max=signal.dte_max,
        delta_tolerance=0.05, min_strike=min_strike if min_strike > 0 else None)
    if not selected:
        algo.Log(f"No suitable options for {signal.symbol} delta ~{target_delta:.2f}")
        return
    current_positions = get_option_position_count(algo)
    if target_right == OptionRight.Put:
        quantity = calculate_put_quantity(algo, selected, current_positions, underlying_price, signal.symbol)
    else:
        shares_held = get_shares_held(algo, signal.symbol)
        existing_call_contracts = get_call_position_contracts(algo, signal.symbol)
        shares_covered = existing_call_contracts * 100  # Each contract covers 100 shares
        shares_available = shares_held - shares_covered
        quantity = min(max(0, shares_available // 100), algo.max_positions)
        if quantity <= 0:
            algo.Log(f"No available shares for {signal.symbol} call: held={shares_held}, covered={shares_covered}")
            return
    if quantity <= 0: return
    quantity = -quantity
    option_symbol = selected['option_symbol']
    # Use safe execution to handle data readiness
    ticket = safe_execute_option_order(algo, option_symbol, quantity, selected['premium'])
    if ticket.Status == OrderStatus.Filled:
        fill_price = ticket.AverageFillPrice or selected['premium']
        right_str = 'P' if target_right == OptionRight.Put else 'C'
        position_id = f"{signal.symbol}_{algo.Time.strftime('%Y%m%d')}_{selected['strike']:.0f}_{right_str}"
        # Save position metadata (entry Greeks, etc.) - QC doesn't track this
        strategy_phase = "SP" if "PUT" in signal.action else "CC"
        save_position_metadata(algo, position_id, {
            'delta_at_entry': selected['delta'],
            'iv_at_entry': selected['iv'],
            'strategy_phase': strategy_phase,
            'entry_date': algo.Time.strftime('%Y-%m-%d'),
            'ml_signal': signal,
        })


def calculate_put_quantity(algo, selected: Dict, current_positions: int, underlying_price: float, symbol: str) -> int:
    """Calculate number of Put contracts to sell with improved margin management.
    
    Improvements:
    1. Use more accurate margin estimation (QC standard formula)
    2. Reduce positions when holding stock (stock ties up capital)
    3. Apply global margin budget constraint
    """
    strike = selected['strike']
    premium = selected.get('premium', 0)
    
    # === More accurate margin estimation ===
    # QC uses: max(20% * underlying - OTM amount, 10% * strike) + premium
    # For naked short put, OTM amount = max(0, underlying - strike)
    otm_amount = max(0, underlying_price - strike)
    margin_method_1 = 0.20 * underlying_price * 100 - otm_amount * 100
    margin_method_2 = 0.10 * strike * 100
    estimated_margin_per_contract = max(margin_method_1, margin_method_2) + premium * 100
    
    # Ensure minimum margin estimate
    fallback_margin = strike * 100 * algo.margin_rate_per_contract
    estimated_margin_per_contract = max(estimated_margin_per_contract, fallback_margin)
    
    # === Get available margin ===
    available_margin = algo.Portfolio.MarginRemaining
    usable_margin = available_margin * (1 - algo.margin_buffer_pct)
    
    # === Reduce positions when holding stock ===
    # Stock holdings tie up capital, reduce max positions accordingly
    stock_value = get_total_stock_holdings_value(algo, algo.stock_pool)
    stock_holding_count = get_stock_holding_count(algo, algo.stock_pool)
    
    # Each stock holding reduces available Put slots
    # If we have 2 stocks, reduce max_positions by 2-3 for new Puts
    adjusted_max_positions = algo.max_positions - stock_holding_count
    
    # Also reduce based on stock value relative to initial capital
    # If stock value > 30% of capital, further reduce positions
    stock_value_ratio = stock_value / algo.initial_capital if algo.initial_capital > 0 else 0
    if stock_value_ratio > 0.30:
        # Reduce positions proportionally (max 50% reduction)
        reduction_factor = min(0.5, stock_value_ratio)
        adjusted_max_positions = max(1, int(adjusted_max_positions * (1 - reduction_factor)))
    
    # === Calculate quantity limits ===
    max_by_margin = max(1, int(usable_margin / estimated_margin_per_contract)) if estimated_margin_per_contract > 0 else 1
    max_by_limit = max(0, adjusted_max_positions - current_positions)
    
    # === Global margin budget: don't use more than 60% of initial capital for new positions ===
    total_margin_used = algo.Portfolio.TotalMarginUsed
    margin_budget = algo.initial_capital * 0.60
    remaining_budget = max(0, margin_budget - total_margin_used)
    max_by_budget = max(0, int(remaining_budget / estimated_margin_per_contract)) if estimated_margin_per_contract > 0 else 0
    
    quantity = min(max_by_margin, max_by_limit, max_by_budget)
    
    # Log if position is limited due to stock holdings or margin
    if stock_holding_count > 0 and quantity < (algo.max_positions - current_positions):
        algo.Log(f"PUT_LIMIT: stocks={stock_holding_count}, adj_max={adjusted_max_positions}, qty={quantity}")
    
    return max(0, quantity)


def execute_roll(algo, signal: StrategySignal, find_option_func):
    existing = get_position_for_symbol(algo, signal.symbol)
    if not existing: return
    pos_info = existing
    pos_id = f"{signal.symbol}_{pos_info['expiry'].strftime('%Y%m%d')}_{pos_info['strike']:.0f}_{pos_info['right']}"
    # Use safe execution for closing the existing position
    close_ticket = safe_execute_option_order(
        algo, pos_info['option_symbol'], -pos_info['quantity'], pos_info['entry_price'])
    if close_ticket.Status != OrderStatus.Filled: return
    pnl, _ = calculate_pnl_metrics(pos_info['entry_price'], close_ticket.AverageFillPrice, pos_info['quantity'])
    record_trade(algo, signal.symbol, pos_info['right'], pnl, "ROLL")
    remove_position_metadata(algo, pos_id)
    equity = algo.equities.get(signal.symbol)
    if not equity: return
    target_right = OptionRight.Put if pos_info['right'] == 'P' else OptionRight.Call
    target_delta = -signal.delta if target_right == OptionRight.Put else signal.delta
    new_selected = find_option_func(algo, symbol=signal.symbol, equity_symbol=equity.Symbol,
        target_right=target_right, target_delta=target_delta, dte_min=signal.dte_min, dte_max=signal.dte_max)
    if new_selected:
        new_qty = pos_info['quantity']
        new_option_symbol = new_selected['option_symbol']
        # Use safe execution to handle data readiness
        new_ticket = safe_execute_option_order(algo, new_option_symbol, new_qty, new_selected['premium'])
        if new_ticket.Status == OrderStatus.Filled:
            right_str = 'P' if target_right == OptionRight.Put else 'C'
            new_pos_id = f"{signal.symbol}_{algo.Time.strftime('%Y%m%d')}_{new_selected['strike']:.0f}_{right_str}"
            # Save position metadata for new position
            strategy_phase = "SP" if "PUT" in signal.action else "CC"
            save_position_metadata(algo, new_pos_id, {
                'delta_at_entry': new_selected['delta'],
                'iv_at_entry': new_selected['iv'],
                'strategy_phase': strategy_phase,
                'entry_date': algo.Time.strftime('%Y-%m-%d'),
                'ml_signal': signal,
            })


def execute_close(algo, signal: StrategySignal):
    pos_info = get_position_for_symbol(algo, signal.symbol)
    if not pos_info: return
    pos_id = f"{signal.symbol}_{pos_info['expiry'].strftime('%Y%m%d')}_{pos_info['strike']:.0f}_{pos_info['right']}"
    # For closing, use entry price as reference for limit order fallback
    # (positions should have data, but use safe execution for consistency)
    close_ticket = safe_execute_option_order(
        algo, pos_info['option_symbol'], -pos_info['quantity'], pos_info['entry_price'])
    if close_ticket.Status == OrderStatus.Filled:
        pnl, _ = calculate_pnl_metrics(pos_info['entry_price'], close_ticket.AverageFillPrice, pos_info['quantity'])
        record_trade(algo, signal.symbol, pos_info['right'], pnl, signal.reasoning or "SIGNAL_CLOSE")
        remove_position_metadata(algo, pos_id)


def record_trade(algo, symbol: str, right: str, pnl: float, reason: str):
    algo.total_trades += 1
    algo.total_pnl += pnl
    if pnl > 0: algo.winning_trades += 1
    algo.trade_history.append({"date": algo.Time.strftime("%Y-%m-%d"), "symbol": symbol, "type": right, "pnl": pnl, "exit_reason": reason})