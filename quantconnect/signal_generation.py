"""Signal generation functions for BinbinGod Strategy.

No phase concept - signals are generated based on actual holdings:
- Has stock -> can sell Call
- No stock -> can sell Put
- Both can happen simultaneously
"""
from typing import Dict, List, Optional
from ml_integration import StrategySignal
from signals import get_cc_optimization_params
from scoring import score_single_stock
from helpers import is_symbol_on_cooldown
from qc_portfolio import (
    get_symbols_with_holdings, get_cost_basis,
    get_position_for_symbol, get_option_position_count, get_call_position_contracts,
    get_shares_held
)


def get_portfolio_state(algo) -> Dict:
    positions = [{'symbol': h.Symbol.Value if hasattr(h.Symbol, 'Value') else str(h.Symbol), 'quantity': h.Quantity, 'market_value': h.HoldingsValue}
                 for h in algo.Portfolio.Values if h.Invested]
    cost_basis_dict = {}
    for symbol in algo.stock_pool:
        cb = get_cost_basis(algo, symbol)
        if cb > 0:
            cost_basis_dict[symbol] = cb
    return {'total_capital': algo.Portfolio.TotalPortfolioValue, 'available_margin': algo.Portfolio.Cash,
            'margin_used': algo.Portfolio.TotalMarginUsed, 'drawdown': calculate_drawdown(algo),
            'positions': positions, 'cost_basis': cost_basis_dict}


def calculate_drawdown(algo) -> float:
    current = algo.Portfolio.TotalPortfolioValue
    if current < algo.initial_capital: return (algo.initial_capital - current) / algo.initial_capital * 100
    return 0.0


def get_current_position(algo, symbol: str) -> Optional[Dict]:
    return get_position_for_symbol(algo, symbol)


def generate_ml_signals(algo) -> List[StrategySignal]:
    """Generate signals based on holdings, not phase.
    
    - Symbols with stock -> SELL_CALL signals
    - All symbols in pool -> SELL_PUT signals (allowed anytime)
    - Both can be generated simultaneously
    """
    signals, portfolio_state = [], get_portfolio_state(algo)
    
    # Get current state
    held_symbols = get_symbols_with_holdings(algo, algo.stock_pool)  # Symbols with stock
    
    # Log when we have stock holdings (important for CC debugging)
    if held_symbols:
        shares_info = {s: get_shares_held(algo, s) for s in held_symbols}
        algo.Log(f"HOLDINGS: {shares_info}")
    
    # Generate SELL_CALL for symbols we hold stock
    for symbol in held_symbols:
        sig = generate_signal_for_symbol(algo, symbol, "CC", portfolio_state)
        if sig: signals.append(sig)
    
    # Generate SELL_PUT for ALL symbols in pool (allowed anytime)
    # This allows: SP while holding stock, SP while having other Puts
    for symbol in algo.stock_pool:
        if is_symbol_on_cooldown(algo, symbol):
            continue
        sig = generate_signal_for_symbol(algo, symbol, "SP", portfolio_state)
        if sig: signals.append(sig)
    
    return signals


def generate_signal_for_symbol(algo, symbol: str, strategy_phase: str, portfolio_state: Dict) -> Optional[StrategySignal]:
    equity = algo.equities.get(symbol)
    if not equity: return None
    underlying_price = algo.Securities[equity.Symbol].Price
    if underlying_price <= 0: return None
    bars = algo.price_history.get(symbol, [])
    if len(bars) < 20: return None
    cost_basis = get_cost_basis(algo, symbol)
    preferred_right = "C" if strategy_phase == "CC" else "P"
    current_position = get_position_for_symbol(algo, symbol, preferred_right=preferred_right)
    traditional_delta, cc_min_strike = 0.30, None
    repair_mode = False
    if strategy_phase == "CC" and algo.cc_optimization_enabled and cost_basis > 0:
        adj_delta, cc_min_strike, log_msg = get_cc_optimization_params(cost_basis, underlying_price,
            algo.cc_optimization_enabled, algo.cc_min_delta_cost, algo.cc_cost_basis_threshold, algo.cc_min_strike_premium)
        # Apply CC-adjusted delta based on optimization result, not log message presence.
        traditional_delta = adj_delta
        if log_msg:
            algo.Log(log_msg)
        drawdown_vs_cost = (cost_basis - underlying_price) / cost_basis if cost_basis > 0 else 0.0
        if drawdown_vs_cost >= algo.repair_call_threshold_pct:
            repair_mode = True
            traditional_delta = max(traditional_delta, algo.repair_call_delta)
            repair_min_strike = max(underlying_price * 1.01, cost_basis * (1 - algo.repair_call_max_discount_pct))
            if cc_min_strike is None:
                cc_min_strike = repair_min_strike
            else:
                cc_min_strike = max(underlying_price * 1.01, min(cc_min_strike, repair_min_strike))
            algo.Log(
                f"CC_REPAIR:{symbol}:drawdown={drawdown_vs_cost:.1%}:"
                f"delta>={traditional_delta:.2f}:minstrike={cc_min_strike:.2f}"
            )
    signal = algo.ml_integration.generate_signal(symbol=symbol, current_price=underlying_price, cost_basis=cost_basis,
        bars=bars, strategy_phase=strategy_phase, portfolio_state=portfolio_state, current_position=current_position)
    if signal and algo.ml_enabled:
        right = "P" if strategy_phase == "SP" else "C"
        if right == "P":
            adaptive_delta, _ = algo.adaptive_strategy.select_put_delta(traditional_delta, signal.delta, signal.delta_confidence, signal.reasoning)
        else:
            adaptive_delta, _ = algo.adaptive_strategy.select_call_delta(traditional_delta, signal.delta, signal.delta_confidence, signal.reasoning)
        signal.delta = adaptive_delta
    if signal:
        if repair_mode and strategy_phase == "CC":
            signal.delta = max(signal.delta, algo.repair_call_delta)
            signal.dte_min = max(algo.repair_call_dte_min, min(signal.dte_min, algo.repair_call_dte_max))
            signal.dte_max = min(max(signal.dte_max, signal.dte_min), algo.repair_call_dte_max)
        score = score_single_stock(symbol, bars, underlying_price, algo.weights)
        signal.ml_score_adjustment = (score.total_score - 50) / 100
        if cc_min_strike is not None: signal.min_strike = cc_min_strike
    return signal
