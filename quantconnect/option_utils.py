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
        if moneyness < 0.98:  # OTM put
            return -max(0.15, min(0.40, (1 - moneyness) * 2))
    else:
        if moneyness > 1.02:  # OTM call
            return max(0.15, min(0.40, (moneyness - 1) * 2))
    
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
    min_dte_for_roll: int = 7
) -> Tuple[str, str]:
    """Determine if position should be rolled or closed.
    
    Args:
        premium_captured_pct: Premium captured percentage
        dte: Days to expiry
        pnl_pct: P&L percentage
        profit_target_pct: Profit target percentage
        profit_target_disabled: Whether profit target is disabled
        stop_loss_pct: Stop loss percentage
        stop_loss_disabled: Whether stop loss is disabled
        roll_threshold_pct: Threshold for rolling (default 80%)
        min_dte_for_roll: Minimum DTE to consider roll
        
    Returns:
        Tuple of (action, reasoning)
        action: "ROLL", "CLOSE_PROFIT", "CLOSE_LOSS", or "HOLD"
    """
    # Check roll forward: 80%+ premium captured with 7+ DTE
    if premium_captured_pct >= roll_threshold_pct and dte > min_dte_for_roll:
        return "ROLL", f"Roll: {premium_captured_pct:.0f}% premium captured, {dte} DTE"
    
    # Check profit target
    if not profit_target_disabled and pnl_pct >= profit_target_pct:
        return "CLOSE_PROFIT", f"Profit target: {pnl_pct:.0f}% gain"
    
    # Check stop loss (disabled by default for Wheel)
    if not stop_loss_disabled and pnl_pct <= -stop_loss_pct:
        return "CLOSE_LOSS", f"Stop loss: {pnl_pct:.0f}% loss"
    
    return "HOLD", "Position within normal parameters"


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