"""
Option utility functions for BinbinGod Strategy.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np


def calculate_historical_vol(bars: List[Dict], window: int = 20) -> float:
    """Calculate historical volatility from price bars.
    
    Args:
        bars: List of price bars with 'close' key
        window: Lookback window in days
        
    Returns:
        Annualized historical volatility or 0.25 as fallback
    """
    if len(bars) < window + 1:
        return 0.25
    
    closes = np.array([b['close'] for b in bars[-(window+1):]])
    returns = np.diff(closes) / closes[:-1]
    
    if len(returns) < 2:
        return 0.25
    
    daily_vol = np.std(returns)
    annual_vol = daily_vol * np.sqrt(252)
    
    return annual_vol if annual_vol > 0 else 0.25


def estimate_premium_approx(underlying_price: float, delta: float, dte: int, 
                            iv: float, right: str) -> float:
    """Estimate option premium using simplified approximation.
    
    Note: In QC, we should use actual option chain prices.
    This provides a rough estimate for ML optimization only.
    """
    T = dte / 365.0
    
    if right == "P":
        strike = underlying_price * (1 + delta * iv * np.sqrt(T) * 0.5)
        moneyness = strike / underlying_price
        premium = underlying_price * delta * iv * np.sqrt(T) * (1 - moneyness * 0.5)
    else:
        strike = underlying_price * (1 - delta * iv * np.sqrt(T) * 0.5)
        moneyness = strike / underlying_price
        premium = underlying_price * delta * iv * np.sqrt(T) * (moneyness * 0.5 - 0.5)
    
    return max(premium, 0.01)


def filter_option_by_itm_protection(
    strike: float,
    underlying_price: float,
    target_right: Any,
    itm_buffer_pct: float = 0.01
) -> bool:
    """Check if option passes ITM protection filter.
    
    Returns True if option is OTM and passes protection.
    
    Args:
        strike: Option strike price
        underlying_price: Current underlying price
        target_right: OptionRight.Put or OptionRight.Call
        itm_buffer_pct: Minimum OTM buffer (default 1%)
        
    Returns:
        True if option passes ITM protection
    """
    # Check string representation for comparison
    right_str = str(target_right).upper()
    
    if 'CALL' in right_str:
        # For calls, ITM means strike < underlying price
        if strike < underlying_price:
            return False  # Skip ITM calls
        if strike < underlying_price * (1 + itm_buffer_pct):
            return False  # Skip near-ITM calls
    else:
        # For puts, ITM means strike > underlying price
        if strike > underlying_price:
            return False  # Skip ITM puts
        if strike > underlying_price * (1 - itm_buffer_pct):
            return False  # Skip near-ITM puts
    
    return True


def estimate_delta_from_moneyness(
    strike: float,
    underlying_price: float,
    target_right: Any
) -> Optional[float]:
    """Estimate delta from moneyness when QC Greeks not available.
    
    Args:
        strike: Option strike price
        underlying_price: Current underlying price
        target_right: OptionRight.Put or OptionRight.Call
        
    Returns:
        Estimated delta or None if ITM
    """
    right_str = str(target_right).upper()
    moneyness = strike / underlying_price
    
    if 'PUT' in right_str:
        # For Put: OTM when strike < underlying (moneyness < 1.0)
        # Return delta for all OTM puts (moneyness < 0.99 to match ITM protection)
        if moneyness < 0.99:
            # Delta roughly equals OTM probability
            # moneyness 0.90 -> delta ~0.20, moneyness 0.95 -> delta ~0.30
            return -max(0.10, min(0.45, (1 - moneyness) * 3))
    else:
        # For Call: OTM when strike > underlying (moneyness > 1.0)
        if moneyness > 1.01:
            return max(0.10, min(0.45, (moneyness - 1) * 3))
    
    return None


def build_option_result(
    option_symbol: Any,
    strike: float,
    expiry: datetime,
    dte: int,
    delta: float,
    iv: float,
    premium: float,
    delta_diff: float,
    bid: float = 0,
    ask: float = 0
) -> Dict:
    """Build standardized option result dictionary.
    
    Args:
        option_symbol: QC option symbol
        strike: Strike price
        expiry: Expiration date
        dte: Days to expiry
        delta: Option delta
        iv: Implied volatility
        premium: Option premium (mid price)
        delta_diff: Difference from target delta
        bid: Bid price
        ask: Ask price
        
    Returns:
        Standardized option result dict
    """
    return {
        'option_symbol': option_symbol,
        'strike': strike,
        'expiry': expiry,
        'dte': dte,
        'delta': delta,
        'iv': iv,
        'premium': premium,
        'delta_diff': delta_diff,
        'bid': bid,
        'ask': ask
    }


def get_premium_from_security(security: Any) -> float:
    """Get premium from QC security using bid/ask or last price.
    
    Args:
        security: QC security object
        
    Returns:
        Premium (mid price) or 0 if not available
    """
    bid = getattr(security, 'BidPrice', 0) or 0
    ask = getattr(security, 'AskPrice', 0) or 0
    
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    
    return security.Price or 0


def should_roll_position(
    premium_captured_pct: float,
    dte: int,
    pnl_pct: float,
    profit_target_pct: float,
    profit_target_disabled: bool,
    stop_loss_pct: float,
    stop_loss_disabled: bool,
    roll_threshold_pct: float = 80,
    min_dte_for_roll: int = 7,
    strategy_phase: str = "SP",
) -> Tuple[str, str]:
    """Determine if position should be rolled or closed.
    
    This matches the original binbin_god.py logic:
    1. Roll Forward when premium_captured >= 80% and DTE > 7
    2. Profit target when pnl_pct >= profit_target_pct (if enabled)
    3. Stop loss when pnl_pct <= -stop_loss_pct (if enabled)
    
    Args:
        premium_captured_pct: Premium captured percentage (same as pnl_pct for short options)
        dte: Days to expiry
        pnl_pct: P&L percentage (for short options, equals premium_captured_pct)
        profit_target_pct: Profit target percentage (e.g., 50 means capture 50% of premium)
        profit_target_disabled: Whether profit target is disabled
        stop_loss_pct: Stop loss percentage
        stop_loss_disabled: Whether stop loss is disabled
        roll_threshold_pct: Threshold for rolling (default 80%)
        min_dte_for_roll: Minimum DTE to consider roll
        
    Returns:
        Tuple of (action, reasoning)
        action: "ROLL", "CLOSE_PROFIT", "CLOSE_LOSS", or "HOLD"
    """
    # PRIORITY 1: Roll Forward - capture 80%+ premium with time remaining
    if premium_captured_pct >= roll_threshold_pct and dte > min_dte_for_roll:
        return "ROLL", f"Roll: {premium_captured_pct:.0f}% premium captured, {dte} DTE"
    
    # PRIORITY 2: Expiry check - let it expire when DTE <= 0
    if dte <= 0:
        return "EXPIRY", f"Option expired, DTE={dte}"
    
    # PRIORITY 3: Profit target - DISABLED for Wheel strategy
    # Wheel strategy should ROLL to capture more premium (handled by PRIORITY 1)
    # Closing at profit_target would break the Wheel cycle
    # if not profit_target_disabled:
    #     if pnl_pct >= profit_target_pct:
    #         return "CLOSE_PROFIT", f"Profit target: {pnl_pct:.0f}% captured (target: {profit_target_pct:.0f}%)"
    
    # PRIORITY 4: Stop loss (disabled by default for Wheel)
    # Strategy design: SP phase should not stop-loss; accept assignment in wheel cycle.
    if strategy_phase != "SP" and not stop_loss_disabled and pnl_pct <= -stop_loss_pct:
        return "CLOSE_LOSS", f"Stop loss: {pnl_pct:.0f}% loss"
    
    return "HOLD", "Position within normal parameters"


def should_defensively_roll_short_put(
    underlying_price: float,
    strike: float,
    dte: int,
    pnl_pct: float,
    min_dte: int,
    max_dte: int,
    itm_buffer_pct: float,
    max_loss_pct: float,
) -> Tuple[bool, str]:
    """Determine whether a short put should be rolled defensively."""
    if underlying_price <= 0 or strike <= 0 or dte < min_dte:
        return False, ""
    if max_dte > 0 and dte > max_dte:
        return False, ""

    itm_pct = max(0.0, (strike - underlying_price) / strike)
    if itm_pct >= itm_buffer_pct:
        return True, f"Defensive roll: underlying {itm_pct:.1%} below strike with {dte} DTE"

    if pnl_pct <= -max_loss_pct:
        return True, f"Defensive roll: option loss {pnl_pct:.0f}% exceeds {max_loss_pct:.0f}%"

    return False, ""


def calculate_dte(expiry: Any, current_time: datetime) -> int:
    """Calculate days to expiry.
    
    Args:
        expiry: Expiration date (datetime or date object)
        current_time: Current datetime
        
    Returns:
        Days to expiry
    """
    if hasattr(expiry, '__sub__'):
        return (expiry - current_time).days
    return 30  # Default fallback
