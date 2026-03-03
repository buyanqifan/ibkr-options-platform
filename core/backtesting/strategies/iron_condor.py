"""Iron Condor strategy: sell OTM put + sell OTM call + buy wings."""

from core.backtesting.strategies.base import BaseStrategy, Signal
from core.backtesting.pricing import OptionsPricer
from datetime import datetime, timedelta


class IronCondorStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "iron_condor"

    def __init__(self, params: dict):
        super().__init__(params)
        self.put_delta = params.get("put_delta_target", 0.16)
        self.call_delta = params.get("call_delta_target", 0.16)
        self.wing_width = params.get("wing_width", 5.0)

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

        # Short put (OTM)
        saved = self.delta_target
        self.delta_target = self.put_delta
        short_put_strike = self.select_strike(underlying_price, iv, T, "P")
        self.delta_target = saved
        short_put_premium = OptionsPricer.put_price(underlying_price, short_put_strike, T, iv)
        short_put_delta = OptionsPricer.delta(underlying_price, short_put_strike, T, iv, "P")

        # Short call (OTM)
        saved = self.delta_target
        self.delta_target = self.call_delta
        short_call_strike = self.select_strike(underlying_price, iv, T, "C")
        self.delta_target = saved
        short_call_premium = OptionsPricer.call_price(underlying_price, short_call_strike, T, iv)
        short_call_delta = OptionsPricer.delta(underlying_price, short_call_strike, T, iv, "C")

        # Long wings
        long_put_strike = short_put_strike - self.wing_width
        long_call_strike = short_call_strike + self.wing_width
        long_put_premium = OptionsPricer.put_price(underlying_price, long_put_strike, T, iv)
        long_call_premium = OptionsPricer.call_price(underlying_price, long_call_strike, T, iv)

        # Calculate position size based on available capital, position percentage and leverage
        available_capital = self.initial_capital * self.position_percentage
        leveraged_capital = available_capital * self.max_leverage
        
        # For iron condor, estimate margin requirement (worst case scenario)
        # Margin for iron condor is typically the difference between strikes minus net credit
        max_spread_width = self.wing_width  # Assuming wing_width is the spread width
        estimated_margin_per_spread = (max_spread_width * 100) - (net_premium * 100)  # Per spread
        estimated_margin_per_spread = max(estimated_margin_per_spread, 100)  # Minimum margin of $100
        
        max_spreads_by_capital = int(leveraged_capital / estimated_margin_per_spread)
        
        # Limit position size to a reasonable number
        max_spreads = min(max_pos, max_spreads_by_capital, 5)  # Cap at 5 spreads
        quantity = max(1, max_spreads)
        
        net_premium = short_put_premium + short_call_premium - long_put_premium - long_call_premium
        symbol = self.params["symbol"]

        signals = [
            Signal(symbol=symbol, trade_type="IRON_CONDOR_SP", right="P",
                   strike=short_put_strike, expiry=expiry_str, quantity=-quantity,
                   iv=iv, delta=short_put_delta, premium=short_put_premium),
            Signal(symbol=symbol, trade_type="IRON_CONDOR_LP", right="P",
                   strike=long_put_strike, expiry=expiry_str, quantity=quantity,
                   iv=iv, delta=0, premium=long_put_premium),
            Signal(symbol=symbol, trade_type="IRON_CONDOR_SC", right="C",
                   strike=short_call_strike, expiry=expiry_str, quantity=-quantity,
                   iv=iv, delta=short_call_delta, premium=short_call_premium),
            Signal(symbol=symbol, trade_type="IRON_CONDOR_LC", right="C",
                   strike=long_call_strike, expiry=expiry_str, quantity=quantity,
                   iv=iv, delta=0, premium=long_call_premium),
        ]
        return signals
