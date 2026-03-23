"""
Signal generation utilities for BinbinGod Strategy.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SignalCandidate:
    """Candidate signal for evaluation."""
    symbol: str
    action: str
    confidence: float
    ml_score_adjustment: float = 0.0
    delta: float = 0.0
    dte_min: int = 30
    dte_max: int = 45
    reasoning: str = ""


def select_best_signal_with_memory(
    signals: List,
    last_selected_stock: Optional[str],
    selection_count: int,
    min_hold_cycles: int,
    last_stock_scores: Dict[str, float]
) -> Tuple[object, str, int, Dict[str, float]]:
    """Select best signal with stock selection memory mechanism.
    
    Implements memory to avoid frequent stock switching:
    - Keep same stock for at least min_hold_cycles
    - Only switch if new stock score is 10%+ better
    
    Args:
        signals: List of StrategySignal objects
        last_selected_stock: Previously selected stock symbol
        selection_count: Current hold cycle count
        min_hold_cycles: Minimum cycles before allowing switch
        last_stock_scores: Historical scores by symbol
        
    Returns:
        Tuple of (best_signal, new_last_selected_stock, new_selection_count, updated_scores)
    """
    if not signals:
        return None, last_selected_stock, selection_count, last_stock_scores
    
    # Adjust confidence with stock score and track scores
    updated_scores = last_stock_scores.copy()
    for signal in signals:
        signal.confidence += signal.ml_score_adjustment * 0.5
        updated_scores[signal.symbol] = signal.confidence
    
    # Sort by confidence
    signals_sorted = sorted(signals, key=lambda x: x.confidence, reverse=True)
    best_signal = signals_sorted[0]
    best_symbol = best_signal.symbol
    
    new_count = selection_count
    new_last = last_selected_stock
    log_message = ""
    
    # Memory mechanism
    if last_selected_stock is not None:
        new_count += 1
        
        if last_selected_stock != best_symbol:
            prev_score = last_stock_scores.get(last_selected_stock, 0)
            new_score = best_signal.confidence
            
            if prev_score > 0:
                score_improvement = (new_score - prev_score) / prev_score * 100
            else:
                score_improvement = 100
            
            # Only switch if held enough AND improvement is significant
            if new_count < min_hold_cycles and score_improvement < 10:
                log_message = f"KEEP:{last_selected_stock}:held_{new_count}_improvement_{score_improvement:.1f}%"
                # Return previous stock's signal if available
                for signal in signals_sorted:
                    if signal.symbol == last_selected_stock:
                        return signal, last_selected_stock, new_count, updated_scores
                # Previous not available, use best
                new_count = 0
                new_last = best_symbol
                log_message = f"SWITCH:{best_symbol}:prev_unavailable"
            else:
                new_count = 0
                new_last = best_symbol
                log_message = f"SWITCH:{last_selected_stock}_to_{best_symbol}:improvement_{score_improvement:.1f}%"
    else:
        # First selection
        new_last = best_symbol
        new_count = 0
        log_message = f"INITIAL:{best_symbol}"
    
    return best_signal, new_last, new_count, updated_scores


def calculate_position_risk(
    premium: float,
    quantity: int,
    portfolio_value: float,
    max_risk_per_trade: float,
    max_leverage: float,
    current_margin_used: float
) -> Tuple[int, str]:
    """Calculate position size based on risk limits.
    
    Args:
        premium: Option premium
        quantity: Proposed quantity (negative for short)
        portfolio_value: Current portfolio value
        max_risk_per_trade: Max risk as fraction of portfolio
        max_leverage: Max leverage allowed
        current_margin_used: Current margin in use
        
    Returns:
        Tuple of (adjusted_quantity, risk_message)
    """
    risk_message = ""
    adjusted_quantity = quantity
    
    # Max risk per trade check
    trade_risk = abs(premium * quantity * 100)
    risk_pct = trade_risk / portfolio_value if portfolio_value > 0 else 0
    
    if risk_pct > max_risk_per_trade:
        max_contracts = int((portfolio_value * max_risk_per_trade) / (premium * 100))
        if max_contracts < 1:
            return 0, f"RISK_LIMIT_TOO_LOW:min_{premium * 100 / portfolio_value / max_risk_per_trade:.1f}_contracts"
        adjusted_quantity = -max_contracts
        risk_message = f"RISK_ADJUSTED:{max_contracts}_contracts"
    
    # Max leverage check
    new_margin_estimate = current_margin_used + abs(premium * adjusted_quantity * 100)
    leverage = new_margin_estimate / portfolio_value if portfolio_value > 0 else 0
    
    if leverage > max_leverage:
        return 0, f"LEVERAGE_EXCEEDED:{leverage:.2f}_vs_{max_leverage}"
    
    return adjusted_quantity, risk_message


def get_cc_optimization_params(
    cost_basis: float,
    underlying_price: float,
    cc_optimization_enabled: bool,
    cc_min_delta_cost: float,
    cc_cost_basis_threshold: float,
    cc_min_strike_premium: float
) -> Tuple[float, Optional[float], str]:
    """Calculate CC optimization parameters when stock is below cost.
    
    Args:
        cost_basis: Current cost basis for the stock
        underlying_price: Current stock price
        cc_optimization_enabled: Whether CC optimization is enabled
        cc_min_delta_cost: Minimum delta when price below cost
        cc_cost_basis_threshold: Threshold % below cost to trigger
        cc_min_strike_premium: Min premium % for strike selection
        
    Returns:
        Tuple of (adjusted_delta, min_strike, log_message)
    """
    if not cc_optimization_enabled or cost_basis <= 0:
        return 0.30, None, ""
    
    price_cost_ratio = underlying_price / cost_basis
    
    if price_cost_ratio >= (1 - cc_cost_basis_threshold):
        # Price is not significantly below cost
        return 0.30, None, ""
    
    # Stock price is below cost basis - use protective delta
    adjusted_delta = cc_min_delta_cost
    min_strike = cost_basis * (1 - cc_min_strike_premium)  # Want strike near cost basis
    
    log_message = (f"CC_OPT:{underlying_price:.2f}_below_cost_{cost_basis:.2f}_"
                   f"delta_{adjusted_delta:.2f}_minstrike_{min_strike:.2f}")
    
    return adjusted_delta, min_strike, log_message


def build_position_data(
    pos_info: Dict,
    current_price: float,
    pnl_pct: float,
    dte: int
) -> Dict:
    """Build position data dict for ML roll optimizer."""
    return {
        **pos_info,
        'current_price': current_price,
        'pnl_pct': pnl_pct,
        'dte': dte
    }


def calculate_pnl_metrics(
    entry_price: float,
    current_price: float,
    quantity: int
) -> Tuple[float, float]:
    """Calculate P&L metrics for a short option position.
    
    Args:
        entry_price: Entry price of the position
        current_price: Current market price
        quantity: Number of contracts (negative for short)
        
    Returns:
        Tuple of (pnl_dollars, pnl_percentage)
    """
    if entry_price <= 0:
        return 0.0, 0.0
    
    # Short position: profit when price drops
    pnl = (entry_price - current_price) * abs(quantity) * 100
    pnl_pct = pnl / (entry_price * abs(quantity) * 100) * 100
    
    return pnl, pnl_pct