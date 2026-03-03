"""Abstract base class for all options strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Signal:
    """Signal to open a new position."""
    symbol: str
    trade_type: str      # SELL_PUT, COVERED_CALL, etc.
    right: str           # P or C
    strike: float
    expiry: str          # YYYYMMDD
    quantity: int         # negative for sell
    iv: float
    delta: float
    premium: float       # expected premium per share


class BaseStrategy(ABC):
    """Abstract base for options strategies."""

    def __init__(self, params: dict):
        self.params = params
        self.dte_min = params.get("dte_min", 21)
        self.dte_max = params.get("dte_max", 45)
        self.delta_target = params.get("delta_target", 0.30)
        self.profit_target_pct = params.get("profit_target_pct", 50)
        self.stop_loss_pct = params.get("stop_loss_pct", 200)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate_signals(
        self,
        current_date: str,
        underlying_price: float,
        iv: float,
        open_positions: list,
    ) -> list[Signal]:
        """Generate trading signals for the current date."""
        ...

    def select_strike(
        self,
        underlying_price: float,
        iv: float,
        T: float,
        right: str,
    ) -> float:
        """Select strike price based on target delta."""
        from core.backtesting.pricing import OptionsPricer
        # Binary search for the strike that gives target delta
        if right == "P":
            target_delta = -abs(self.delta_target)
            # For put options: when S > K (OTM), delta approaches 0
            # When S < K (ITM), delta approaches -1
            # So to get delta near -0.3 (slightly OTM), we want K slightly less than S
            low = underlying_price * 0.9  # e.g. 135 for S=150
            high = underlying_price * 1.05  # e.g. 157.5 for S=150
        else:
            target_delta = abs(self.delta_target)
            # For call options: when S > K (ITM), delta approaches 1
            # When S < K (OTM), delta approaches 0
            # So to get delta near 0.3 (slightly ITM), we want K slightly less than S
            low = underlying_price * 0.9
            high = underlying_price * 1.05

        for _ in range(50):
            mid = (low + high) / 2
            d = OptionsPricer.delta(underlying_price, mid, T, iv, right)
            if right == "P":
                # For puts: increasing strike makes delta more negative
                # So if d < target_delta (too negative), decrease strike (high = mid)
                # If d > target_delta (too positive), increase strike (low = mid)
                if d < target_delta:
                    high = mid
                else:
                    low = mid
            else:
                # For calls: increasing strike makes delta more negative
                # So if d > target_delta (too positive), decrease strike (high = mid) 
                # If d < target_delta (too negative), increase strike (low = mid)
                if d > target_delta:
                    high = mid
                else:
                    low = mid
            if abs(d - target_delta) < 0.005:
                break

        # Round to nearest 0.5 or 1.0
        if underlying_price > 100:
            return round(mid)
        return round(mid * 2) / 2

    def select_expiry_dte(self) -> float:
        """Return target DTE in the middle of the range."""
        return (self.dte_min + self.dte_max) / 2
