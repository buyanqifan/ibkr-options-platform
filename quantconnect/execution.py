"""Trade execution functions for BinbinGod Strategy."""
from typing import Dict, Optional
from AlgorithmImports import OptionRight, OrderStatus, Resolution
from ml_integration import StrategySignal
from option_utils import calculate_dte
from signals import calculate_pnl_metrics
from option_pricing import BlackScholes

RISK_FREE_RATE = 0.05


def bs_put_price(S, K, T, sigma):
    return BlackScholes.put_price(S, K, T, RISK_FREE_RATE, sigma)


def bs_call_price(S, K, T, sigma):
    return BlackScholes.call_price(S, K, T, RISK_FREE_RATE, sigma)


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
    current_positions = len(algo.open_option_positions)
    if target_right == OptionRight.Put:
        estimated_margin_per_contract = selected['strike'] * 100 * algo.margin_rate_per_contract
        available_margin = algo.Portfolio.MarginRemaining
        usable_margin = available_margin * (1 - algo.margin_buffer_pct)
        max_by_margin = max(1, int(usable_margin / estimated_margin_per_contract)) if estimated_margin_per_contract > 0 else 1
        max_by_limit = algo.max_positions - current_positions
        quantity = min(max_by_margin, max_by_limit)
        algo.Log(f"Position sizing: available_margin=${available_margin:.0f}, usable=${usable_margin:.0f}, margin_per_contract=${estimated_margin_per_contract:.0f}, max_by_margin={max_by_margin}, quantity={quantity}")
    else:
        shares_held = algo.stock_holding.get_shares(signal.symbol)
        existing_call_contracts = sum(abs(p.get('quantity', 0)) for p in algo.open_option_positions.values()
            if p.get('symbol') == signal.symbol and p.get('right') == 'C')
        shares_covered = existing_call_contracts * 100
        shares_available = shares_held - shares_covered
        quantity = min(max(0, shares_available // 100), algo.max_positions)
        if quantity <= 0:
            algo.Log(f"No available shares for {signal.symbol} call: held={shares_held}, covered={shares_covered}")
            return
    if quantity <= 0: return
    quantity = -quantity
    algo.Log(f"Selling {abs(quantity)} {signal.symbol} {target_right} @ ${selected['premium']:.2f}")
    option_symbol = selected['option_symbol']
    if not algo.Securities.ContainsKey(option_symbol):
        algo.Log(f"Subscribing to option contract: {option_symbol}")
        algo.AddOptionContract(option_symbol, Resolution.Daily)
    ticket = algo.MarketOrder(option_symbol, quantity)
    if ticket.Status == OrderStatus.Filled:
        fill_price = ticket.AverageFillPrice or selected['premium']
        right_str = 'P' if target_right == OptionRight.Put else 'C'
        position_id = f"{signal.symbol}_{algo.Time.strftime('%Y%m%d')}_{selected['strike']:.0f}_{right_str}"
        algo.open_option_positions[position_id] = {
            'symbol': signal.symbol, 'option_symbol': selected['option_symbol'], 'right': right_str,
            'strike': selected['strike'], 'expiry': selected['expiry'], 'entry_date': algo.Time.strftime('%Y-%m-%d'),
            'entry_price': fill_price, 'quantity': quantity, 'delta_at_entry': selected['delta'],
            'iv_at_entry': selected['iv'], 'strategy_phase': algo.phase, 'ml_signal': signal}
        algo.Log(f"Executed: {signal.action} {signal.num_contracts} {signal.symbol} @ ${fill_price:.2f}")


def execute_roll(algo, signal: StrategySignal, find_option_func):
    existing = None
    for pos_id, pos_info in algo.open_option_positions.items():
        if pos_info.get('symbol') == signal.symbol: existing = (pos_id, pos_info); break
    if not existing: return
    pos_id, pos_info = existing
    close_ticket = algo.MarketOrder(pos_info['option_symbol'], -pos_info['quantity'])
    if close_ticket.Status != OrderStatus.Filled: return
    pnl, _ = calculate_pnl_metrics(pos_info['entry_price'], close_ticket.AverageFillPrice, pos_info['quantity'])
    record_trade(algo, signal.symbol, pos_info['right'], pnl, "ROLL")
    del algo.open_option_positions[pos_id]
    equity = algo.equities.get(signal.symbol)
    if not equity: return
    target_right = OptionRight.Put if pos_info['right'] == 'P' else OptionRight.Call
    target_delta = -signal.delta if target_right == OptionRight.Put else signal.delta
    new_selected = find_option_func(algo, symbol=signal.symbol, equity_symbol=equity.Symbol,
        target_right=target_right, target_delta=target_delta, dte_min=signal.dte_min, dte_max=signal.dte_max)
    if new_selected:
        new_qty = pos_info['quantity']
        new_option_symbol = new_selected['option_symbol']
        if not algo.Securities.ContainsKey(new_option_symbol):
            algo.Log(f"Subscribing to option contract: {new_option_symbol}")
            algo.AddOptionContract(new_option_symbol, Resolution.Daily)
        new_ticket = algo.MarketOrder(new_option_symbol, new_qty)
        if new_ticket.Status == OrderStatus.Filled:
            right_str = 'P' if target_right == OptionRight.Put else 'C'
            algo.open_option_positions[f"{signal.symbol}_{algo.Time.strftime('%Y%m%d')}_{new_selected['strike']:.0f}_{right_str}"] = {
                'symbol': signal.symbol, 'option_symbol': new_selected['option_symbol'], 'right': right_str,
                'strike': new_selected['strike'], 'expiry': new_selected['expiry'],
                'entry_date': algo.Time.strftime('%Y-%m-%d'), 'entry_price': new_ticket.AverageFillPrice,
                'quantity': new_qty, 'delta_at_entry': new_selected['delta'],
                'iv_at_entry': new_selected['iv'], 'strategy_phase': algo.phase, 'ml_signal': signal}


def execute_close(algo, signal: StrategySignal):
    for pos_id, pos_info in list(algo.open_option_positions.items()):
        if pos_info.get('symbol') != signal.symbol: continue
        close_ticket = algo.MarketOrder(pos_info['option_symbol'], -pos_info['quantity'])
        if close_ticket.Status == OrderStatus.Filled:
            pnl, _ = calculate_pnl_metrics(pos_info['entry_price'], close_ticket.AverageFillPrice, pos_info['quantity'])
            record_trade(algo, signal.symbol, pos_info['right'], pnl, signal.reasoning or "SIGNAL_CLOSE")
            del algo.open_option_positions[pos_id]


def record_trade(algo, symbol: str, right: str, pnl: float, reason: str):
    algo.total_trades += 1
    algo.total_pnl += pnl
    if pnl > 0: algo.winning_trades += 1
    algo.trade_history.append({"date": algo.Time.strftime("%Y-%m-%d"), "symbol": symbol, "type": right, "pnl": pnl, "exit_reason": reason})