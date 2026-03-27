"""QuantConnect Portfolio wrapper - Use QC native interfaces for position tracking.

This module provides helper functions to access QC Portfolio data, reducing
the need for manual position tracking dictionaries.

QC Portfolio provides:
- Portfolio.Values: All holdings (equity + options)
- Portfolio[symbol].Quantity: Current position size
- Portfolio[symbol].AveragePrice: Cost basis per share/contract
- Portfolio[symbol].HoldingsValue: Market value
- Portfolio[symbol].UnrealizedProfit: P&L

We still need to track:
- Entry Greeks (delta_at_entry, iv_at_entry) for ML learning
- Strategy metadata (strategy_phase, ml_signal) for signal generation
"""
from typing import Dict, List, Optional, Any
from AlgorithmImports import OptionRight, SecurityType


def _get_underlying_ticker(option_symbol) -> str:
    """Extract underlying ticker from QC option symbol.
    
    QC's symbol.ID.Underlying returns a Symbol object, not a string.
    Use symbol.Underlying.Value for the ticker string.
    """
    if hasattr(option_symbol, 'Underlying') and option_symbol.Underlying:
        return option_symbol.Underlying.Value
    # Fallback: parse from string representation
    underlying_str = str(option_symbol.ID.Underlying)
    return underlying_str.split()[0] if ' ' in underlying_str else underlying_str


def get_option_positions(algo) -> Dict[str, Dict]:
    """Get all open option positions from QC Portfolio.
    
    Returns dict keyed by position_id with position info from QC.
    """
    positions = {}
    for holding in algo.Portfolio.Values:
        if not holding.Invested:
            continue
        symbol = holding.Symbol
        # Check if it's an option by SecurityType first (avoids OptionRight access error)
        if hasattr(symbol, 'SecurityType') and symbol.SecurityType == SecurityType.Option:
            # It's an option position
            right_str = 'P' if symbol.ID.OptionRight == OptionRight.Put else 'C'
            underlying_ticker = _get_underlying_ticker(symbol)
            pos_id = f"{underlying_ticker}_{symbol.ID.Date.strftime('%Y%m%d')}_{symbol.ID.StrikePrice:.0f}_{right_str}"
            
            # Get metadata from our tracking (if exists)
            metadata = getattr(algo, 'position_metadata', {}).get(pos_id, {})
            
            positions[pos_id] = {
                'symbol': underlying_ticker,
                'option_symbol': symbol,
                'right': right_str,
                'strike': float(symbol.ID.StrikePrice),
                'expiry': symbol.ID.Date,
                'quantity': holding.Quantity,  # Negative for short
                'entry_price': holding.AveragePrice,
                'current_price': algo.Securities[symbol].Price if algo.Securities.ContainsKey(symbol) else 0,
                'market_value': holding.HoldingsValue,
                'unrealized_pnl': holding.UnrealizedProfit,
                # Metadata from tracking (QC doesn't have this)
                'delta_at_entry': metadata.get('delta_at_entry', 0),
                'iv_at_entry': metadata.get('iv_at_entry', 0.25),
                'strategy_phase': metadata.get('strategy_phase', 'SP'),
                'entry_date': metadata.get('entry_date', ''),
                'ml_signal': metadata.get('ml_signal', None),
            }
    return positions


def get_equity_positions(algo, stock_pool: List[str] = None) -> Dict[str, Dict]:
    """Get equity positions from QC Portfolio.
    
    Returns dict keyed by symbol with position info.
    """
    positions = {}
    for holding in algo.Portfolio.Values:
        if not holding.Invested:
            continue
        symbol = holding.Symbol
        # Check if it's equity (not an option) by SecurityType
        is_option = hasattr(symbol, 'SecurityType') and symbol.SecurityType == SecurityType.Option
        if not is_option:
            # Use symbol.Value for ticker string (e.g., 'AMZN')
            sym_str = symbol.Value if hasattr(symbol, 'Value') else str(symbol).split()[0]
            if stock_pool and sym_str not in stock_pool:
                continue
            positions[sym_str] = {
                'symbol': sym_str,
                'quantity': holding.Quantity,
                'average_price': holding.AveragePrice,
                'market_value': holding.HoldingsValue,
                'unrealized_pnl': holding.UnrealizedProfit,
            }
    return positions


def get_shares_held(algo, symbol: str) -> int:
    """Get shares held for a symbol from QC Portfolio."""
    equity = algo.equities.get(symbol)
    if not equity:
        return 0
    if algo.Portfolio.ContainsKey(equity.Symbol):
        return algo.Portfolio[equity.Symbol].Quantity
    return 0


def get_symbols_with_holdings(algo, stock_pool: List[str] = None) -> List[str]:
    """Get list of symbols with equity holdings from QC Portfolio."""
    symbols = []
    for holding in algo.Portfolio.Values:
        if not holding.Invested or holding.Quantity <= 0:
            continue
        symbol = holding.Symbol
        # Check if it's equity (not an option) by SecurityType
        is_option = hasattr(symbol, 'SecurityType') and symbol.SecurityType == SecurityType.Option
        if not is_option:
            # Use symbol.Value for ticker string (e.g., 'AMZN')
            sym_str = symbol.Value if hasattr(symbol, 'Value') else str(symbol).split()[0]
            if stock_pool is None or sym_str in stock_pool:
                symbols.append(sym_str)
    return symbols


def get_cost_basis(algo, symbol: str) -> float:
    """Get cost basis for a symbol from QC Portfolio.
    
    Note: QC's AveragePrice is the actual cost basis for current holdings.
    For wheel strategy, this reflects the strike price when shares were assigned.
    """
    equity = algo.equities.get(symbol)
    if not equity:
        return 0.0
    if algo.Portfolio.ContainsKey(equity.Symbol):
        return algo.Portfolio[equity.Symbol].AveragePrice
    return 0.0


def get_option_position_count(algo) -> int:
    """Get count of open option positions from QC Portfolio."""
    count = 0
    for holding in algo.Portfolio.Values:
        if not holding.Invested:
            continue
        symbol = holding.Symbol
        if hasattr(symbol, 'SecurityType') and symbol.SecurityType == SecurityType.Option:
            count += 1
    return count


def get_put_position_symbols(algo) -> set:
    """Get set of symbols with open put positions from QC Portfolio."""
    symbols = set()
    for holding in algo.Portfolio.Values:
        if not holding.Invested:
            continue
        symbol = holding.Symbol
        if hasattr(symbol, 'SecurityType') and symbol.SecurityType == SecurityType.Option:
            if symbol.ID.OptionRight == OptionRight.Put:
                symbols.add(_get_underlying_ticker(symbol))
    return symbols


def get_call_position_symbols(algo) -> set:
    """Get set of symbols with open call positions from QC Portfolio."""
    symbols = set()
    for holding in algo.Portfolio.Values:
        if not holding.Invested:
            continue
        symbol = holding.Symbol
        if hasattr(symbol, 'SecurityType') and symbol.SecurityType == SecurityType.Option:
            if symbol.ID.OptionRight == OptionRight.Call:
                symbols.add(_get_underlying_ticker(symbol))
    return symbols


def get_call_position_contracts(algo, symbol: str) -> int:
    """Get number of call contracts for a symbol from QC Portfolio."""
    contracts = 0
    for holding in algo.Portfolio.Values:
        if not holding.Invested:
            continue
        pos_symbol = holding.Symbol
        if hasattr(pos_symbol, 'SecurityType') and pos_symbol.SecurityType == SecurityType.Option:
            if (pos_symbol.ID.OptionRight == OptionRight.Call and 
                _get_underlying_ticker(pos_symbol) == symbol):
                contracts += abs(holding.Quantity)
    return contracts


def get_position_for_symbol(algo, symbol: str, preferred_right: str = None) -> Optional[Dict]:
    """Get option position info for a specific symbol from QC Portfolio.

    If multiple option positions exist for the same underlying, prioritize:
    1) preferred_right (if provided, "P" or "C")
    2) nearest expiry
    """
    candidates = []
    for holding in algo.Portfolio.Values:
        if not holding.Invested:
            continue
        pos_symbol = holding.Symbol
        if hasattr(pos_symbol, 'SecurityType') and pos_symbol.SecurityType == SecurityType.Option:
            if _get_underlying_ticker(pos_symbol) == symbol:
                right_str = 'P' if pos_symbol.ID.OptionRight == OptionRight.Put else 'C'
                pos_id = f"{symbol}_{pos_symbol.ID.Date.strftime('%Y%m%d')}_{pos_symbol.ID.StrikePrice:.0f}_{right_str}"
                metadata = getattr(algo, 'position_metadata', {}).get(pos_id, {})
                candidates.append({
                    'symbol': symbol,
                    'option_symbol': pos_symbol,
                    'right': right_str,
                    'strike': float(pos_symbol.ID.StrikePrice),
                    'expiry': pos_symbol.ID.Date,
                    'quantity': holding.Quantity,
                    'entry_price': holding.AveragePrice,
                    'delta_at_entry': metadata.get('delta_at_entry', 0),
                    'iv_at_entry': metadata.get('iv_at_entry', 0.25),
                    'strategy_phase': metadata.get('strategy_phase', 'SP'),
                })
    if not candidates:
        return None
    if preferred_right in ("P", "C"):
        filtered = [c for c in candidates if c["right"] == preferred_right]
        if filtered:
            candidates = filtered
    candidates.sort(key=lambda p: p["expiry"])
    return candidates[0]


# ============ Metadata tracking (for data QC doesn't provide) ============

def init_position_tracking(algo):
    """Initialize minimal position metadata tracking.
    
    We only track what QC Portfolio doesn't provide:
    - delta_at_entry, iv_at_entry (for ML learning)
    - strategy_phase at entry
    - ml_signal (for reference)
    """
    algo.position_metadata = {}  # pos_id -> {delta_at_entry, iv_at_entry, ...}


def save_position_metadata(algo, pos_id: str, metadata: Dict):
    """Save position metadata when opening a trade."""
    if not hasattr(algo, 'position_metadata'):
        init_position_tracking(algo)
    algo.position_metadata[pos_id] = metadata


def remove_position_metadata(algo, pos_id: str):
    """Remove position metadata when closing a trade."""
    if hasattr(algo, 'position_metadata') and pos_id in algo.position_metadata:
        del algo.position_metadata[pos_id]


def get_position_metadata(algo, pos_id: str) -> Dict:
    """Get position metadata."""
    return getattr(algo, 'position_metadata', {}).get(pos_id, {})


def get_total_stock_holdings_value(algo, stock_pool: List[str] = None) -> float:
    """Get total market value of all stock holdings.
    
    This is used to determine how much capital is tied up in stocks,
    which affects how many new Put positions we should open.
    """
    total_value = 0.0
    for holding in algo.Portfolio.Values:
        if not holding.Invested or holding.Quantity <= 0:
            continue
        symbol = holding.Symbol
        # Check if it's equity (not an option)
        is_option = hasattr(symbol, 'SecurityType') and symbol.SecurityType == SecurityType.Option
        if not is_option:
            sym_str = symbol.Value if hasattr(symbol, 'Value') else str(symbol).split()[0]
            if stock_pool is None or sym_str in stock_pool:
                total_value += abs(holding.HoldingsValue)
    return total_value


def get_stock_holding_count(algo, stock_pool: List[str] = None) -> int:
    """Get count of symbols with stock holdings.
    
    Returns the number of different stocks held (not the total shares).
    """
    return len(get_symbols_with_holdings(algo, stock_pool))
