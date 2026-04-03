"""Signal generation functions for BinbinGod Strategy.

No phase concept - signals are generated based on actual holdings:
- Has stock -> can sell Call
- No stock -> can sell Put
- Both can happen simultaneously
"""
import math
from typing import Dict, List, Optional
from ml_integration import StrategySignal
from signals import build_cc_selection_tiers, get_cc_optimization_params
from scoring import score_single_stock
from helpers import is_symbol_on_cooldown
from debug_counters import increment_debug_counter
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


def _should_block_extreme_sell_put(algo, symbol: str, bars: List[Dict], underlying_price: float) -> bool:
    lookback = max(20, int(getattr(algo, "volatility_lookback", 20) or 20))
    if len(bars) < lookback:
        return False
    closes = [float(bar["close"]) for bar in bars[-lookback:] if bar.get("close", 0) > 0]
    if len(closes) < lookback:
        return False
    annualized_vol = _annualized_realized_volatility(closes)
    ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
    return_20d = underlying_price / closes[-20] - 1 if len(closes) >= 20 and closes[-20] > 0 else 0.0

    vol_threshold = float(getattr(algo, "put_hard_filter_vol_threshold", 0.75) or 0.75)
    downtrend_threshold = float(getattr(algo, "put_hard_filter_20d_drop_threshold", 0.08) or 0.08)
    ma_threshold = float(getattr(algo, "put_hard_filter_ma_threshold", 0.97) or 0.97)

    is_extreme_high_vol = annualized_vol >= vol_threshold
    is_downtrend = return_20d <= -downtrend_threshold and underlying_price <= ma20 * ma_threshold
    if is_extreme_high_vol and is_downtrend:
        increment_debug_counter(algo, "sp_quality_block")
        algo.Log(
            f"SP_QUALITY_BLOCK:{symbol}:vol={annualized_vol:.2f}:"
            f"ret20={return_20d:.1%}:ma20={ma20:.2f}:price={underlying_price:.2f}"
        )
        return True
    return False


def _get_assignment_repair_state(algo, symbol: str) -> Optional[Dict]:
    if not getattr(algo, "assigned_stock_fail_safe_enabled", False):
        return None
    state = getattr(algo, "assigned_stock_state", {}).get(symbol)
    if not state or state.get("force_exit_triggered"):
        return None
    if get_shares_held(algo, symbol) <= 0:
        return None
    return state


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
        if is_symbol_on_cooldown(algo, symbol):
            continue
        shares_held = get_shares_held(algo, symbol)
        if getattr(algo, "stock_inventory_cap_enabled", True):
            equity = algo.equities.get(symbol)
            underlying_price = algo.Securities[equity.Symbol].Price if equity and algo.Securities.ContainsKey(equity.Symbol) else 0
            stock_notional = shares_held * max(underlying_price, 0)
            portfolio_value = max(algo.Portfolio.TotalPortfolioValue, 0.0)
            inventory_cap = portfolio_value * getattr(algo, "stock_inventory_base_cap", 0.17)
            inventory_block_threshold = getattr(algo, "stock_inventory_block_threshold", 0.85)
            if shares_held > 0 and inventory_cap > 0 and stock_notional >= inventory_cap * inventory_block_threshold:
                increment_debug_counter(algo, "sp_stock_block")
                algo.Log(f"SP_STOCK_BLOCK:{symbol}:stock={stock_notional:.0f}:cap={inventory_cap:.0f}")
        if shares_held > 0:
            increment_debug_counter(algo, "sp_held_block")
            algo.Log(f"SP_HELD_BLOCK:{symbol}:shares={shares_held}")
            continue
        bars = algo.price_history.get(symbol, [])
        equity = algo.equities.get(symbol)
        underlying_price = algo.Securities[equity.Symbol].Price if equity and algo.Securities.ContainsKey(equity.Symbol) else 0
        if _should_block_extreme_sell_put(algo, symbol, bars, underlying_price):
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
    assignment_repair_state = _get_assignment_repair_state(algo, symbol) if strategy_phase == "CC" else None
    if strategy_phase == "CC" and algo.cc_optimization_enabled and cost_basis > 0:
        adj_delta, cc_min_strike, log_msg = get_cc_optimization_params(cost_basis, underlying_price,
            algo.cc_optimization_enabled, algo.cc_min_delta_cost, algo.cc_cost_basis_threshold, algo.cc_min_strike_premium)
        # Apply CC-adjusted delta based on optimization result, not log message presence.
        traditional_delta = adj_delta
        if log_msg:
            algo.Log(log_msg)
        drawdown_vs_cost = (cost_basis - underlying_price) / cost_basis if cost_basis > 0 else 0.0
        if drawdown_vs_cost >= algo.repair_call_threshold_pct or assignment_repair_state:
            repair_mode = True
            repair_delta = algo.repair_call_delta
            repair_max_discount_pct = algo.repair_call_max_discount_pct
            if assignment_repair_state:
                repair_delta = min(0.60, repair_delta + getattr(algo, "assigned_stock_repair_delta_boost", 0.10))
                repair_max_discount_pct = max(
                    repair_max_discount_pct,
                    getattr(algo, "assigned_stock_repair_max_discount_pct", repair_max_discount_pct),
                )
                increment_debug_counter(algo, "assigned_repair_attempt")
                algo.Log(
                    f"ASSIGNED_REPAIR_ATTEMPT:{symbol}:failures={assignment_repair_state.get('repair_failures', 0)}:"
                    f"cost_basis={cost_basis:.2f}:price={underlying_price:.2f}"
                )
            traditional_delta = max(traditional_delta, repair_delta)
            repair_min_strike = max(underlying_price * 1.01, cost_basis * (1 - repair_max_discount_pct))
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
            repair_delta = algo.repair_call_delta
            if assignment_repair_state:
                repair_delta = min(0.60, repair_delta + getattr(algo, "assigned_stock_repair_delta_boost", 0.10))
            signal.delta = max(signal.delta, repair_delta)
            if assignment_repair_state:
                signal.dte_min = getattr(algo, "assigned_stock_repair_dte_min", signal.dte_min)
                signal.dte_max = max(signal.dte_min, getattr(algo, "assigned_stock_repair_dte_max", signal.dte_max))
            else:
                signal.dte_min = max(algo.repair_call_dte_min, min(signal.dte_min, algo.repair_call_dte_max))
                signal.dte_max = min(max(signal.dte_max, signal.dte_min), algo.repair_call_dte_max)
        score = score_single_stock(symbol, bars, underlying_price, algo.weights)
        signal.ml_score_adjustment = (score.total_score - 50) / 100
        if cc_min_strike is not None: signal.min_strike = cc_min_strike
        if strategy_phase == "CC":
            signal.selection_tiers = build_cc_selection_tiers(
                underlying_price=underlying_price,
                cost_basis=cost_basis,
                primary_dte_min=signal.dte_min,
                primary_dte_max=signal.dte_max,
                primary_delta_tolerance=0.08,
                primary_min_strike=cc_min_strike,
                fallback_delta_tolerance_1=float(getattr(algo, "cc_fallback_delta_tolerance_1", 0.12) or 0.12),
                fallback_delta_tolerance_2=float(getattr(algo, "cc_fallback_delta_tolerance_2", 0.15) or 0.15),
                fallback_dte_min=int(getattr(algo, "cc_fallback_dte_min", 14) or 14),
                fallback_dte_max=int(getattr(algo, "cc_fallback_dte_max", 30) or 30),
                fallback_min_cost_basis_ratio=float(getattr(algo, "cc_fallback_min_cost_basis_ratio", 0.85) or 0.85),
            )
    return signal
