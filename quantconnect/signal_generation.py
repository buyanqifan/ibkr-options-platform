"""Signal generation functions for BinbinGod Strategy."""
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
    return {'total_capital': algo.Portfolio.TotalPortfolioValue, 'available_margin': algo.Portfolio.Cash,
            'margin_used': algo.Portfolio.TotalMarginUsed, 'drawdown': calculate_drawdown(algo),
            'positions': positions, 'cost_basis': algo.stock_holding.cost_basis}


def calculate_drawdown(algo) -> float:
    current = algo.Portfolio.TotalPortfolioValue
    if current < algo.initial_capital: return (algo.initial_capital - current) / algo.initial_capital * 100
    return 0.0


def get_current_position(algo, symbol: str) -> Optional[Dict]:
    return get_position_for_symbol(algo, symbol)


def generate_ml_signals(algo) -> List[StrategySignal]:
    signals, portfolio_state = [], get_portfolio_state(algo)
    algo.Log(f"generate_ml_signals: phase={algo.phase}, stock_pool={algo.stock_pool}")
    if algo.phase == "SP":
        symbols_with_put = get_put_position_symbols(algo)
        algo.Log(f"generate_ml_signals: symbols_with_put={symbols_with_put}")
        for symbol in algo.stock_pool:
            if symbol not in symbols_with_put:
                sig = generate_signal_for_symbol(algo, symbol, "SP", portfolio_state)
                algo.Log(f"generate_ml_signals: symbol={symbol}, sig={'None' if sig is None else 'generated'}")
                if sig: signals.append(sig)
    elif algo.phase == "CC":
        for symbol in get_symbols_with_holdings(algo, algo.stock_pool):
            sig = generate_signal_for_symbol(algo, symbol, "CC", portfolio_state)
            if sig: signals.append(sig)
        if algo.allow_sp_in_cc_phase:
            signals.extend(generate_cc_sp_signals(algo, portfolio_state))
    return signals


def generate_signal_for_symbol(algo, symbol: str, strategy_phase: str, portfolio_state: Dict) -> Optional[StrategySignal]:
    equity = algo.equities.get(symbol)
    if not equity:
        algo.Log(f"generate_signal_for_symbol: {symbol} - no equity")
        return None
    underlying_price = algo.Securities[equity.Symbol].Price
    if underlying_price <= 0:
        algo.Log(f"generate_signal_for_symbol: {symbol} - price <= 0")
        return None
    bars = algo.price_history.get(symbol, [])
    if len(bars) < 20:
        algo.Log(f"generate_signal_for_symbol: {symbol} - bars={len(bars)} < 20")
        return None
    cost_basis = get_cost_basis(algo, symbol)
    current_position = get_current_position(algo, symbol)
    traditional_delta, cc_min_strike = 0.30, None
    if strategy_phase == "CC" and algo.cc_optimization_enabled and cost_basis > 0:
        adj_delta, cc_min_strike, log_msg = get_cc_optimization_params(cost_basis, underlying_price,
            algo.cc_optimization_enabled, algo.cc_min_delta_cost, algo.cc_cost_basis_threshold, algo.cc_min_strike_premium)
        if log_msg: algo.Log(log_msg); traditional_delta = adj_delta
    signal = algo.ml_integration.generate_signal(symbol=symbol, current_price=underlying_price, cost_basis=cost_basis,
        bars=bars, strategy_phase=strategy_phase, portfolio_state=portfolio_state, current_position=current_position)
    if signal and algo.ml_enabled:
        right = "P" if strategy_phase in ("SP", "CC+SP") else "C"
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


def generate_cc_sp_signals(algo, portfolio_state: Dict) -> List[StrategySignal]:
    signals = []
    margin_util = algo.Portfolio.TotalMarginUsed / algo.Portfolio.TotalPortfolioValue
    if margin_util > algo.sp_in_cc_margin_threshold: return signals
    
    # Count put positions using QC Portfolio
    put_symbols = get_put_position_symbols(algo)
    if len(put_symbols) >= algo.sp_in_cc_max_positions: return signals
    
    open_count = get_option_position_count(algo)
    if open_count >= algo.max_positions: return signals
    
    held = get_symbols_with_holdings(algo, algo.stock_pool)
    available = [s for s in algo.stock_pool if s not in held] or algo.stock_pool
    available = [s for s in available if s not in put_symbols]
    best_signal = None
    for symbol in available:
        signal = generate_signal_for_symbol(algo, symbol, "CC+SP", portfolio_state)
        if signal and signal.confidence > 0.6:
            if best_signal is None or signal.confidence > best_signal.confidence: best_signal = signal
    if best_signal: signals.append(best_signal)
    return signals