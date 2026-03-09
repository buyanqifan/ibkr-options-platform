"""Covered Call strategy."""

from core.backtesting.strategies.base import BaseStrategy, Signal
from core.backtesting.pricing import OptionsPricer
from datetime import datetime, timedelta


class CoveredCallStrategy(BaseStrategy):
    """Covered Call strategy with full configuration support."""

    def __init__(self, params: dict):
        super().__init__(params)
        # Check if profit target/stop loss are disabled (special value 999999 means disabled)
        self._profit_target_disabled = params.get("profit_target_pct", 50) >= 999999
        self._stop_loss_disabled = params.get("stop_loss_pct", 200) >= 999999

    @property
    def name(self) -> str:
        return "covered_call"

    def generate_signals(
        self,
        current_date: str,
        underlying_price: float,
        iv: float,
        open_positions: list,
        position_mgr=None,
    ) -> list[Signal]:
        max_pos = self.params.get("max_positions", 1)
        if len(open_positions) >= max_pos:
            return []

        T = self.select_expiry_dte() / 365.0
        strike = self.select_strike(underlying_price, iv, T, "C")
        premium = OptionsPricer.call_price(underlying_price, strike, T, iv)
        delta = OptionsPricer.delta(underlying_price, strike, T, iv, "C")

        dte_days = int(self.select_expiry_dte())
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=dte_days)
        expiry_str = expiry_date.strftime("%Y%m%d")

        # Calculate position size using position manager if available
        if position_mgr:
            # For covered call, need to own 100 shares per contract
            max_contracts = position_mgr.calculate_position_size(
                margin_per_contract=underlying_price * 100,
                max_positions=min(max_pos, 10),
            )
            if max_contracts <= 0:
                return []  # No signal if can't afford
        else:
            # Fallback to legacy calculation
            available_capital = self.initial_capital * self.position_percentage
            leveraged_capital = available_capital * self.max_leverage
            
            max_contracts_by_capital = int(leveraged_capital / (underlying_price * 100))
            max_contracts = min(max_pos, max_contracts_by_capital, 10)
            if max_contracts <= 0:
                return []  # No signal if insufficient capital
        
        quantity = -max_contracts  # Sell calls (negative quantity)
        
        return [Signal(
            symbol=self.params["symbol"],
            trade_type="COVERED_CALL",
            right="C",
            strike=strike,
            expiry=expiry_str,
            quantity=quantity,  # Dynamic quantity based on capital
            iv=iv,
            delta=delta,
            premium=premium,
        )]
