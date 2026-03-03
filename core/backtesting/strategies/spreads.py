"""Bull Put Spread and Bear Call Spread strategies."""

from core.backtesting.strategies.base import BaseStrategy, Signal
from core.backtesting.pricing import OptionsPricer
from datetime import datetime, timedelta


class BullPutSpreadStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "bull_put_spread"

    def __init__(self, params: dict):
        super().__init__(params)
        self.spread_width = params.get("spread_width", 5.0)

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
        dte_days = int(self.select_expiry_dte())
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=dte_days)
        expiry_str = expiry_date.strftime("%Y%m%d")

        short_strike = self.select_strike(underlying_price, iv, T, "P")
        long_strike = short_strike - self.spread_width

        short_premium = OptionsPricer.put_price(underlying_price, short_strike, T, iv)
        long_premium = OptionsPricer.put_price(underlying_price, long_strike, T, iv)
        short_delta = OptionsPricer.delta(underlying_price, short_strike, T, iv, "P")

        # Calculate position size based on available capital, position percentage and leverage
        available_capital = self.initial_capital * self.position_percentage
        leveraged_capital = available_capital * self.max_leverage
        
        # For bull put spread, margin requirement is the spread width minus net credit
        net_credit = short_premium - long_premium
        estimated_margin_per_spread = (self.spread_width * 100) - (net_credit * 100)  # Per spread
        estimated_margin_per_spread = max(estimated_margin_per_spread, 100)  # Minimum margin of $100
        
        max_spreads_by_capital = int(leveraged_capital / estimated_margin_per_spread)
        
        # Limit position size to a reasonable number
        max_spreads = min(max_pos, max_spreads_by_capital, 10)  # Cap at 10 spreads
        quantity = max(1, max_spreads)
        
        symbol = self.params["symbol"]
        return [
            Signal(symbol=symbol, trade_type="BULL_PUT_SHORT", right="P",
                   strike=short_strike, expiry=expiry_str, quantity=-quantity,
                   iv=iv, delta=short_delta, premium=short_premium),
            Signal(symbol=symbol, trade_type="BULL_PUT_LONG", right="P",
                   strike=long_strike, expiry=expiry_str, quantity=quantity,
                   iv=iv, delta=0, premium=long_premium),
        ]


class BearCallSpreadStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "bear_call_spread"

    def __init__(self, params: dict):
        super().__init__(params)
        self.spread_width = params.get("spread_width", 5.0)

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
        dte_days = int(self.select_expiry_dte())
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=dte_days)
        expiry_str = expiry_date.strftime("%Y%m%d")

        short_strike = self.select_strike(underlying_price, iv, T, "C")
        long_strike = short_strike + self.spread_width

        short_premium = OptionsPricer.call_price(underlying_price, short_strike, T, iv)
        long_premium = OptionsPricer.call_price(underlying_price, long_strike, T, iv)
        short_delta = OptionsPricer.delta(underlying_price, short_strike, T, iv, "C")

        # Calculate position size based on available capital, position percentage and leverage
        available_capital = self.initial_capital * self.position_percentage
        leveraged_capital = available_capital * self.max_leverage
        
        # For bear call spread, margin requirement is the spread width minus net credit
        net_credit = short_premium - long_premium
        estimated_margin_per_spread = (self.spread_width * 100) - (net_credit * 100)  # Per spread
        estimated_margin_per_spread = max(estimated_margin_per_spread, 100)  # Minimum margin of $100
        
        max_spreads_by_capital = int(leveraged_capital / estimated_margin_per_spread)
        
        # Limit position size to a reasonable number
        max_spreads = min(max_pos, max_spreads_by_capital, 10)  # Cap at 10 spreads
        quantity = max(1, max_spreads)
        
        symbol = self.params["symbol"]
        return [
            Signal(symbol=symbol, trade_type="BEAR_CALL_SHORT", right="C",
                   strike=short_strike, expiry=expiry_str, quantity=-quantity,
                   iv=iv, delta=short_delta, premium=short_premium),
            Signal(symbol=symbol, trade_type="BEAR_CALL_LONG", right="C",
                   strike=long_strike, expiry=expiry_str, quantity=quantity,
                   iv=iv, delta=0, premium=long_premium),
        ]
