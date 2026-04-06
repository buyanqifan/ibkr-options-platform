"""Signal generation functions for BinbinGod Strategy.

No phase concept - signals are generated based on actual holdings:
- Has stock -> can sell Call
- No stock -> can sell Put
- Both can happen simultaneously
"""
import math
from typing import Dict, List, Optional, Tuple
from ml_integration import StrategySignal
from scoring import score_single_stock
from debug_counters import increment_debug_counter
from qc_portfolio import (
    get_symbols_with_holdings, get_cost_basis,
    get_position_for_symbol, get_option_position_count, get_call_position_contracts,
    get_shares_held, get_put_position_symbols
)


def get_portfolio_state(algo) -> Dict:
    positions = [{'symbol': h.Symbol.Value if hasattr(h.Symbol, 'Value') else str(h.Symbol), 'quantity': h.Quantity, 'market_value': h.HoldingsValue}
                 for h in algo.Portfolio.Values if h.Invested]
    cost_basis_dict = {}
    for symbol in algo.stock_pool:
        cb = get_cost_basis(algo, symbol)
        if cb > 0:
            cost_basis_dict[symbol] = cb
    return {'total_capital': algo.Portfolio.TotalPortfolioValue, 'available_margin': algo.Portfolio.MarginRemaining,
            'margin_used': algo.Portfolio.TotalMarginUsed, 'drawdown': calculate_drawdown(algo),
            'positions': positions, 'cost_basis': cost_basis_dict}


def calculate_drawdown(algo) -> float:
    current = algo.Portfolio.TotalPortfolioValue
    peak = getattr(algo, "_portfolio_peak_value", max(getattr(algo, "initial_capital", current), current))
    peak = max(peak, current)
    algo._portfolio_peak_value = peak
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - current) / peak * 100)


def _annualized_realized_volatility(closes: List[float]) -> float:
    if len(closes) < 2:
        return 0.0
    returns = []
    for previous, current in zip(closes[:-1], closes[1:]):
        if previous <= 0 or current <= 0:
            continue
        returns.append(math.log(current / previous))
    if len(returns) < 2:
        return 0.0
    mean_return = sum(returns) / len(returns)
    variance = sum((ret - mean_return) ** 2 for ret in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(252)


def _build_cc_selection_tiers(target_dte_min: int, target_dte_max: int, min_strike: Optional[float]) -> List[Dict]:
    return [
        {
            "name": "primary",
            "delta_tolerance": 0.08,
            "dte_min": target_dte_min,
            "dte_max": target_dte_max,
            "min_strike": min_strike,
        },
        {
            "name": "delta_relaxed",
            "delta_tolerance": 0.16,
            "dte_min": target_dte_min,
            "dte_max": target_dte_max,
            "min_strike": min_strike,
        },
    ]


def _build_sp_selection_tiers(algo, target_dte_min: int, target_dte_max: int) -> List[Dict]:
    return [
        {
            "name": "primary",
            "delta_tolerance": algo.sp_primary_delta_tolerance,
            "dte_min": target_dte_min,
            "dte_max": target_dte_max,
        },
        {
            "name": "delta_relaxed",
            "delta_tolerance": algo.sp_relaxed_delta_tolerance,
            "dte_min": target_dte_min,
            "dte_max": target_dte_max,
        },
    ]


def _extract_closes(bars: List[Dict]) -> List[float]:
    closes = []
    for bar in bars:
        close = bar.get("close") if isinstance(bar, dict) else None
        if close is None:
            continue
        close = float(close)
        if close > 0:
            closes.append(close)
    return closes


def _is_symbol_in_assignment_cooldown(algo, symbol: str) -> Tuple[bool, Optional[object]]:
    state = getattr(algo, "assigned_stock_state", {}).get(symbol)
    if not isinstance(state, dict):
        return False, None
    block_until = state.get("sp_reentry_block_until")
    if block_until is None:
        return False, None
    return algo.Time < block_until, block_until


def _should_block_sp_for_weakness(algo, symbol: str, underlying_price: float, bars: List[Dict]) -> Tuple[bool, Dict]:
    diagnostics = {
        "inventory_risk": False,
        "trend_breakdown": False,
        "vol_spike": False,
        "price": underlying_price,
        "ma20": 0.0,
        "rv10": 0.0,
        "rv20": 0.0,
    }
    if not getattr(algo, "sp_weak_filter_enabled", False):
        return False, diagnostics

    has_shares = get_shares_held(algo, symbol) > 0
    has_short_put = symbol in get_put_position_symbols(algo)
    in_assignment_state = symbol in getattr(algo, "assigned_stock_state", {})
    diagnostics["inventory_risk"] = has_shares or has_short_put or in_assignment_state
    if not diagnostics["inventory_risk"]:
        return False, diagnostics

    closes = _extract_closes(bars)
    if len(closes) < 20:
        return False, diagnostics

    ma20 = sum(closes[-20:]) / 20.0
    rv10 = _annualized_realized_volatility(closes[-10:])
    rv20 = _annualized_realized_volatility(closes[-20:])
    diagnostics["ma20"] = ma20
    diagnostics["rv10"] = rv10
    diagnostics["rv20"] = rv20

    if ma20 > 0:
        diagnostics["trend_breakdown"] = underlying_price < ma20 * (1 - getattr(algo, "sp_weak_filter_ma20_break_pct", 0.03))
    if rv20 > 0:
        diagnostics["vol_spike"] = rv10 >= rv20 * getattr(algo, "sp_weak_filter_vol_spike_ratio", 1.25)

    return diagnostics["inventory_risk"] and diagnostics["trend_breakdown"] and diagnostics["vol_spike"], diagnostics


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
        increment_debug_counter(algo, "holdings_seen")
        shares_info = {s: get_shares_held(algo, s) for s in held_symbols}
        algo.Log(f"HOLDINGS: {shares_info}")
    
    # Generate SELL_CALL for symbols we hold stock
    for symbol in held_symbols:
        sig = generate_signal_for_symbol(algo, symbol, "CC", portfolio_state)
        if sig: signals.append(sig)
    
    # Generate SELL_PUT for ALL symbols in pool (allowed anytime)
    # This allows: SP while holding stock, SP while having other Puts
    for symbol in algo.stock_pool:
        shares_held = get_shares_held(algo, symbol)
        if shares_held > 0:
            increment_debug_counter(algo, "sp_held_block")
            algo.Log(f"SP_HELD_BLOCK:{symbol}:shares={shares_held}")
            continue
        in_cooldown, block_until = _is_symbol_in_assignment_cooldown(algo, symbol)
        if in_cooldown:
            increment_debug_counter(algo, "sp_assignment_cooldown_block")
            algo.Log(
                f"SP_ASSIGNMENT_COOLDOWN:{symbol}:until={block_until.strftime('%Y-%m-%d')}:"
                f"shares={get_shares_held(algo, symbol)}"
            )
            continue
        sig = generate_signal_for_symbol(algo, symbol, "SP", portfolio_state)
        if sig: signals.append(sig)
    
    return signals


def generate_signal_for_symbol(algo, symbol: str, strategy_phase: str, portfolio_state: Dict) -> Optional[StrategySignal]:
    equity = algo.equities.get(symbol)
    if not equity: return None
    underlying_price = algo.Securities[equity.Symbol].Price
    if underlying_price <= 0:
        if strategy_phase == "CC":
            increment_debug_counter(algo, "cc_signal_missing")
            algo.Log(f"CC_SKIP:{symbol}:invalid_underlying_price={underlying_price}")
        return None
    bars = algo.price_history.get(symbol, [])
    if len(bars) < 20:
        if strategy_phase == "CC":
            increment_debug_counter(algo, "cc_signal_missing")
            algo.Log(f"CC_SKIP:{symbol}:insufficient_history={len(bars)}")
        return None
    if strategy_phase == "SP":
        weak_filter_block, diagnostics = _should_block_sp_for_weakness(algo, symbol, underlying_price, bars)
        if weak_filter_block:
            increment_debug_counter(algo, "sp_weak_filter_block")
            algo.Log(
                f"SP_WEAK_FILTER:{symbol}:price={diagnostics['price']:.2f}:ma20={diagnostics['ma20']:.2f}:"
                f"rv10={diagnostics['rv10']:.4f}:rv20={diagnostics['rv20']:.4f}:inventory_risk=1"
            )
            return None
    cost_basis = get_cost_basis(algo, symbol)
    preferred_right = "C" if strategy_phase == "CC" else "P"
    current_position = get_position_for_symbol(algo, symbol, preferred_right=preferred_right)
    traditional_delta = 0.30 if strategy_phase == "SP" else algo.cc_target_delta
    cc_min_strike = None
    if strategy_phase == "CC":
        cc_min_strike = underlying_price * 1.01
        if cost_basis > 0 and getattr(algo, "cc_below_cost_enabled", True) and underlying_price < cost_basis:
            cc_min_strike = max(cc_min_strike, cost_basis * (1 - algo.cc_max_discount_to_cost))
            algo.Log(
                f"CC_COST_FLOOR:{symbol}:cost={cost_basis:.2f}:price={underlying_price:.2f}:"
                f"min_strike={cc_min_strike:.2f}"
            )
    signal = algo.ml_integration.generate_signal(symbol=symbol, current_price=underlying_price, cost_basis=cost_basis,
        bars=bars, strategy_phase=strategy_phase, portfolio_state=portfolio_state, current_position=current_position)
    if not signal and strategy_phase == "CC":
        increment_debug_counter(algo, "cc_signal_missing")
        algo.Log(
            f"CC_SIGNAL_MISSING:{symbol}:shares={get_shares_held(algo, symbol)}:"
            f"cost={cost_basis:.2f}:price={underlying_price:.2f}"
        )
    if signal and algo.ml_enabled:
        right = "P" if strategy_phase == "SP" else "C"
        if right == "P":
            adaptive_delta, _ = algo.adaptive_strategy.select_put_delta(traditional_delta, signal.delta, signal.delta_confidence, signal.reasoning)
        else:
            adaptive_delta, _ = algo.adaptive_strategy.select_call_delta(traditional_delta, signal.delta, signal.delta_confidence, signal.reasoning)
        signal.delta = adaptive_delta
    if signal:
        if strategy_phase == "CC":
            signal.delta = max(signal.delta, algo.cc_target_delta)
            signal.dte_min = algo.cc_target_dte_min
            signal.dte_max = max(algo.cc_target_dte_min, algo.cc_target_dte_max)
            algo.Log(
                f"CC_SIGNAL_READY:{symbol}:delta={signal.delta:.2f}:dte={signal.dte_min}-{signal.dte_max}:"
                f"min_strike={cc_min_strike if cc_min_strike is not None else 0:.2f}:"
                f"confidence={signal.confidence:.2f}:shares={get_shares_held(algo, symbol)}"
            )
        score = score_single_stock(symbol, bars, underlying_price, algo.weights)
        signal.ml_score_adjustment = (score.total_score - 50) / 100
        if cc_min_strike is not None:
            signal.min_strike = cc_min_strike
        if strategy_phase == "CC":
            signal.selection_tiers = _build_cc_selection_tiers(
                target_dte_min=signal.dte_min,
                target_dte_max=signal.dte_max,
                min_strike=cc_min_strike,
            )
        elif strategy_phase == "SP":
            signal.selection_tiers = _build_sp_selection_tiers(
                algo,
                target_dte_min=signal.dte_min,
                target_dte_max=signal.dte_max,
            )
    return signal
