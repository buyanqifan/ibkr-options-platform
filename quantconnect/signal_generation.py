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
from qc_portfolio import (
    get_symbols_with_holdings, get_cost_basis, get_put_position_symbols,
    get_position_for_symbol, get_option_position_count, get_call_position_contracts,
    get_shares_held
)


def get_portfolio_state(algo) -> Dict:
    positions = [{'symbol': str(h.Symbol), 'quantity': h.Quantity, 'market_value': h.HoldingsValue}
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
    - Symbols without stock and without Put -> SELL_PUT signals
    - Both can be generated simultaneously
    """
    signals, portfolio_state = [], get_portfolio_state(algo)
    
    # Get current state
    held_symbols = get_symbols_with_holdings(algo, algo.stock_pool)  # Symbols with stock
    put_symbols = get_put_position_symbols(algo)  # Symbols with open Put
    
    # Generate SELL_CALL for symbols we hold stock
    for symbol in held_symbols:
        sig = generate_signal_for_symbol(algo, symbol, "CC", portfolio_state)
        if sig: signals.append(sig)
    
    # Generate SELL_PUT for symbols we don't hold stock and don't have Put
    for symbol in algo.stock_pool:
        if symbol not in held_symbols and symbol not in put_symbols:
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
    current_position = get_current_position(algo, symbol)
    traditional_delta, cc_min_strike = 0.30, None
    if strategy_phase == "CC" and algo.cc_optimization_enabled and cost_basis > 0:
        adj_delta, cc_min_strike, log_msg = get_cc_optimization_params(
            cost_basis, underlying_price,
            algo.cc_optimization_enabled, algo.cc_min_delta_cost, 
            algo.cc_cost_basis_threshold, algo.cc_min_strike_premium,
            getattr(algo, 'cc_profit_protection_enabled', True),
            getattr(algo, 'cc_profit_threshold', 0.20),
            getattr(algo, 'cc_profit_delta', 0.20)
        )
        if log_msg: algo.Log(log_msg); traditional_delta = adj_delta
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
        score = score_single_stock(symbol, bars, underlying_price, algo.weights)
        signal.ml_score_adjustment = (score.total_score - 50) / 100
        if cc_min_strike is not None: signal.min_strike = cc_min_strike
    return signal