"""
Helper classes for BinbinGod Strategy.
"""

from datetime import timedelta
import math
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class StockScore:
    """Score for a single stock."""
    symbol: str
    pe_ratio: float = 0.0
    iv_rank: float = 50.0
    momentum: float = 50.0
    technical_score: float = 50.0
    liquidity_score: float = 70.0
    ml_score_adjustment: float = 0.0
    total_score: float = 0.0


@dataclass
class StockHolding:
    """Tracks stock position from put assignment."""
    shares: int = 0
    cost_basis: float = 0.0
    total_premium_collected: float = 0.0
    symbol: str = ""
    holdings: Dict = field(default_factory=dict)
    
    def add_shares(self, symbol: str, shares: int, cost_basis: float):
        """Add shares of a stock to holdings."""
        if symbol not in self.holdings:
            self.holdings[symbol] = {"shares": 0, "cost_basis": 0.0, "premium": 0.0}
        
        existing = self.holdings[symbol]
        total_shares = existing["shares"] + shares
        if total_shares > 0:
            total_cost = existing["shares"] * existing["cost_basis"] + shares * cost_basis
            existing["cost_basis"] = total_cost / total_shares
        existing["shares"] = total_shares
        self._update_legacy_fields()
    
    def remove_shares(self, symbol: str, shares: int) -> int:
        """Remove shares of a stock."""
        if symbol not in self.holdings:
            return 0
        existing = self.holdings[symbol]
        removed = min(shares, existing["shares"])
        existing["shares"] -= removed
        if existing["shares"] <= 0:
            del self.holdings[symbol]
        self._update_legacy_fields()
        return removed
    
    def get_shares(self, symbol: str) -> int:
        """Get shares for a specific symbol."""
        return self.holdings.get(symbol, {}).get("shares", 0)
    
    def get_symbols(self) -> List[str]:
        """Get list of symbols with holdings."""
        return list(self.holdings.keys())
    
    def add_premium(self, symbol: str, premium: float):
        """Add premium collected for a symbol."""
        if symbol not in self.holdings:
            self.holdings[symbol] = {"shares": 0, "cost_basis": 0.0, "premium": 0.0}
        self.holdings[symbol]["premium"] = self.holdings[symbol].get("premium", 0.0) + premium
        self.total_premium_collected += premium
    
    def _update_legacy_fields(self):
        """Update legacy fields for backward compatibility."""
        self.shares = sum(h["shares"] for h in self.holdings.values())
        if self.holdings:
            primary = max(self.holdings.items(), key=lambda x: x[1]["shares"])
            self.symbol = primary[0]
            self.cost_basis = primary[1]["cost_basis"]


def set_symbol_cooldown(algo, symbol: str, days: int, reason: str = ""):
    """Block new short-put entries for a symbol until a future date."""
    if days <= 0:
        return
    if not hasattr(algo, "symbol_cooldowns"):
        algo.symbol_cooldowns = {}
    current_end = algo.symbol_cooldowns.get(symbol)
    new_end = algo.Time + timedelta(days=days)
    if current_end is None or new_end > current_end:
        algo.symbol_cooldowns[symbol] = new_end
        msg = f"COOLDOWN_SET:{symbol}:{days}d"
        if reason:
            msg += f":{reason}"
        algo.Log(msg)


def get_symbol_cooldown_days_remaining(algo, symbol: str) -> int:
    """Return remaining cooldown days for a symbol."""
    expiry = getattr(algo, "symbol_cooldowns", {}).get(symbol)
    if not expiry:
        return 0
    remaining_seconds = (expiry - algo.Time).total_seconds()
    if remaining_seconds <= 0:
        return 0
    return max(0, int(math.ceil(remaining_seconds / 86400)))


def is_symbol_on_cooldown(algo, symbol: str) -> bool:
    """Check whether a symbol is currently blocked from new short puts."""
    return get_symbol_cooldown_days_remaining(algo, symbol) > 0
