"""
Stock scoring functions for BinbinGod Strategy.
"""

import numpy as np
from typing import Dict, List
from helpers import StockScore


# Default scoring weights
DEFAULT_WEIGHTS = {
    "iv_rank": 0.35,
    "technical": 0.25,
    "momentum": 0.20,
    "pe_score": 0.20,
}


def score_single_stock(symbol: str, bars: List[Dict], current_price: float, 
                       weights: Dict = None) -> StockScore:
    """Score a single stock for selection.
    
    Args:
        symbol: Stock symbol
        bars: List of price bars with 'close' and 'volume' keys
        current_price: Current stock price
        weights: Scoring weights dict (optional)
        
    Returns:
        StockScore with all scoring components
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    if len(bars) < 20:
        return StockScore(symbol=symbol, total_score=50.0)
    
    closes = np.array([b['close'] for b in bars])
    
    # Momentum (20-day price change)
    prev_20_price = closes[-20]
    raw_momentum = ((current_price - prev_20_price) / prev_20_price) * 100
    momentum = max(0, min(100, (raw_momentum + 20) / 70 * 100))
    
    # IV Rank - aligned with original binbin_god.py (no annualization)
    lookback = min(30, len(closes))
    if lookback > 2:
        recent = closes[-lookback:]
        returns = np.diff(recent) / recent[:-1]
        vol = np.std(returns)
        iv_rank = min(100, max(0, vol * 200))  # No annualization
    else:
        iv_rank = 50.0
    
    # Technical Score
    rsi_score = calculate_rsi_score(closes)
    ma_score = calculate_ma_score(closes)
    technical_score = rsi_score * 0.6 + ma_score * 0.4
    
    # Liquidity Score
    volumes = [b.get('volume', 1) for b in bars[-10:]]
    if volumes:
        recent_vol = np.mean(volumes[-5:])
        prior_vol = np.mean(volumes[-10:-5])
        liquidity_score = min(100, (recent_vol / prior_vol) * 50) if prior_vol > 0 else 70.0
    else:
        liquidity_score = 70.0
    
    # PE Score (proxied from momentum)
    pe_ratio = max(5, min(60, 35 + raw_momentum * 0.3))
    pe_score = max(0, min(100, 100 - pe_ratio))
    
    # Calculate weighted total score
    total_score = (
        iv_rank * weights["iv_rank"] +
        technical_score * weights["technical"] +
        momentum * weights["momentum"] +
        pe_score * weights["pe_score"]
    )
    
    # Apply liquidity penalty for very low liquidity (aligned with original binbin_god.py)
    if liquidity_score < 30:
        total_score *= 0.8  # 20% penalty for low liquidity
    
    return StockScore(
        symbol=symbol,
        pe_ratio=pe_ratio,
        iv_rank=iv_rank,
        momentum=momentum,
        technical_score=technical_score,
        liquidity_score=liquidity_score,
        total_score=total_score
    )


def calculate_rsi_score(closes: np.ndarray) -> float:
    """Calculate RSI score from closing prices.
    
    Args:
        closes: Array of closing prices
        
    Returns:
        RSI score (0-100)
    """
    if len(closes) < 14:
        return 50.0
    
    deltas = np.diff(closes[-14:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    # Score: 50-70 RSI is ideal (trending but not overbought)
    rsi_score = max(0, min(100, 100 - abs(rsi - 60) * 2))
    return rsi_score


def calculate_ma_score(closes: np.ndarray) -> float:
    """Calculate MA position score.
    
    Args:
        closes: Array of closing prices
        
    Returns:
        MA position score (0-100)
    """
    if len(closes) < 20:
        return 50.0
    
    current_price = closes[-1]
    sma_20 = np.mean(closes[-20:])
    ma_position = (current_price - sma_20) / sma_20 * 100
    
    ma_score = max(0, min(100, 50 + ma_position * 5))
    return ma_score


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
    
    # Annualize the volatility
    daily_vol = np.std(returns)
    annual_vol = daily_vol * np.sqrt(252)
    
    return annual_vol if annual_vol > 0 else 0.25


def calculate_iv_rank(bars: List[Dict]) -> float:
    """Calculate IV rank based on 30-day historical volatility.
    
    Aligned with original binbin_god.py calculation (no annualization).
    
    Args:
        bars: List of price bars with 'close' key
        
    Returns:
        IV rank (0-100)
    """
    if len(bars) < 30:
        return 50.0
    
    # Calculate 30-day volatility
    prices_30 = [bar['close'] for bar in bars[-30:]]
    returns = [(prices_30[i] - prices_30[i-1]) / prices_30[i-1] 
              for i in range(1, len(prices_30))]
    
    if not returns or len(returns) < 2:
        return 50.0
    
    # Daily volatility (NOT annualized, aligned with original)
    vol = np.std(returns)
    
    # Same formula as original binbin_god.py
    iv_rank = min(100, max(0, vol * 200))
    
    return iv_rank