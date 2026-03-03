"""Sell Put (Cash Secured Put) strategy."""

from core.backtesting.strategies.base import BaseStrategy, Signal
from core.backtesting.pricing import OptionsPricer
from datetime import datetime, timedelta


class SellPutStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "sell_put"

    def generate_signals(
        self,
        current_date: str,
        underlying_price: float,
        iv: float,
        open_positions: list,
    ) -> list[Signal]:
        max_pos = self.params.get("max_positions", 1)
        if len(open_positions) >= max_pos:
            return []

        T = self.select_expiry_dte() / 365.0
        strike = self.select_strike(underlying_price, iv, T, "P")
        premium = OptionsPricer.put_price(underlying_price, strike, T, iv)
        delta = OptionsPricer.delta(underlying_price, strike, T, iv, "P")

        # Calculate position size based on available capital, position percentage and leverage
        # For cash secured put, need to reserve cash equal to strike * 100 per contract
        available_capital = self.initial_capital * self.position_percentage
        leveraged_capital = available_capital * self.max_leverage
        
        max_contracts_by_capital = int(leveraged_capital / (strike * 100))
        
        # For risk-based sizing, use premium income as the measure
        # We can risk a percentage of account value per trade based on potential premium income
        expected_premium_income = premium * 100  # per contract premium income
        max_contracts_by_risk = int((self.initial_capital * self.max_risk_per_trade) / expected_premium_income) if expected_premium_income > 0 else max_pos
        
        # Take minimum of capital-based sizing, risk-based sizing and max position constraint
        max_contracts = min(max_pos, max_contracts_by_capital, max_contracts_by_risk)
        quantity = max(1, max_contracts) * -1  # Sell contracts (negative quantity)

        dte_days = int(self.select_expiry_dte())
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=dte_days)
        expiry_str = expiry_date.strftime("%Y%m%d")

        return [Signal(
            symbol=self.params["symbol"],
            trade_type="SELL_PUT",
            right="P",
            strike=strike,
            expiry=expiry_str,
            quantity=quantity,
            iv=iv,
            delta=delta,
            premium=premium,
        )]
