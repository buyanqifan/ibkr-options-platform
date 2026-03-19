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
        position_mgr=None,
    ) -> list[Signal]:
        max_pos = self.params.get("max_positions", 1)
        if len(open_positions) >= max_pos:
            return []

        T = self.select_expiry_dte() / 365.0
        dte_days = int(self.select_expiry_dte())
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=dte_days)
        expiry_str = expiry_date.strftime("%Y%m%d")

        # Get optimized delta for short put (ML or traditional)
        optimized_delta = self.get_optimized_delta(underlying_price, iv, "P")
        original_delta = self.delta_target
        self.delta_target = optimized_delta
        short_strike = self.select_strike(underlying_price, iv, T, "P")
        self.delta_target = original_delta  # Restore original
        
        long_strike = short_strike - self.spread_width

        short_premium = OptionsPricer.put_price(underlying_price, short_strike, T, iv)
        long_premium = OptionsPricer.put_price(underlying_price, long_strike, T, iv)
        short_delta = OptionsPricer.delta(underlying_price, short_strike, T, iv, "P")

        # Calculate position size using position manager if available
        if position_mgr:
            net_credit = short_premium - long_premium
            estimated_margin_per_spread = (self.spread_width * 100) - (net_credit * 100)
            estimated_margin_per_spread = max(estimated_margin_per_spread, 100)
            
            num_spreads = position_mgr.calculate_position_size(
                margin_per_contract=estimated_margin_per_spread,
                max_positions=min(max_pos, 10),
            )
            if num_spreads <= 0:
                return []  # No signal if insufficient capital
        else:
            # Fallback to legacy calculation
            available_capital = self.initial_capital * self.position_percentage
            leveraged_capital = available_capital * self.max_leverage
            
            net_credit = short_premium - long_premium
            estimated_margin_per_spread = (self.spread_width * 100) - (net_credit * 100)
            estimated_margin_per_spread = max(estimated_margin_per_spread, 100)
            
            max_spreads_by_capital = int(leveraged_capital / estimated_margin_per_spread)
            num_spreads = min(max_pos, max_spreads_by_capital, 10)
            if num_spreads <= 0:
                return []  # No signal if insufficient capital
        
        quantity = num_spreads
        
        symbol = self.params["symbol"]
        
        # Calculate margin requirement for this spread
        net_credit = short_premium - long_premium
        margin_per_spread = (self.spread_width * 100) - (net_credit * 100)
        margin_per_spread = max(margin_per_spread, 100)  # Minimum $100
        
        return [
            Signal(symbol=symbol, trade_type="BULL_PUT_SHORT", right="P",
                   strike=short_strike, expiry=expiry_str, quantity=-quantity,
                   iv=iv, delta=short_delta, premium=short_premium,
                   margin_requirement=margin_per_spread),  # Margin for the spread
            Signal(symbol=symbol, trade_type="BULL_PUT_LONG", right="P",
                   strike=long_strike, expiry=expiry_str, quantity=quantity,
                   iv=iv, delta=0, premium=long_premium,
                   margin_requirement=0),  # Long leg is hedged, no margin needed
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
        position_mgr=None,
    ) -> list[Signal]:
        max_pos = self.params.get("max_positions", 1)
        if len(open_positions) >= max_pos:
            return []

        T = self.select_expiry_dte() / 365.0
        dte_days = int(self.select_expiry_dte())
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=dte_days)
        expiry_str = expiry_date.strftime("%Y%m%d")

        # Get optimized delta for short call (ML or traditional)
        optimized_delta = self.get_optimized_delta(underlying_price, iv, "C")
        original_delta = self.delta_target
        self.delta_target = optimized_delta
        short_strike = self.select_strike(underlying_price, iv, T, "C")
        self.delta_target = original_delta  # Restore original
        
        long_strike = short_strike + self.spread_width

        short_premium = OptionsPricer.call_price(underlying_price, short_strike, T, iv)
        long_premium = OptionsPricer.call_price(underlying_price, long_strike, T, iv)
        short_delta = OptionsPricer.delta(underlying_price, short_strike, T, iv, "C")

        # Calculate position size using position manager if available
        if position_mgr:
            net_credit = short_premium - long_premium
            estimated_margin_per_spread = (self.spread_width * 100) - (net_credit * 100)
            estimated_margin_per_spread = max(estimated_margin_per_spread, 100)
            
            num_spreads = position_mgr.calculate_position_size(
                margin_per_contract=estimated_margin_per_spread,
                max_positions=min(max_pos, 10),
            )
        else:
            # Fallback to legacy calculation
            available_capital = self.initial_capital * self.position_percentage
            leveraged_capital = available_capital * self.max_leverage
            
            net_credit = short_premium - long_premium
            estimated_margin_per_spread = (self.spread_width * 100) - (net_credit * 100)
            estimated_margin_per_spread = max(estimated_margin_per_spread, 100)
            
            max_spreads_by_capital = int(leveraged_capital / estimated_margin_per_spread)
            num_spreads = min(max_pos, max_spreads_by_capital, 10)
            num_spreads = max(1, num_spreads)
        
        quantity = num_spreads
        
        symbol = self.params["symbol"]
        return [
            Signal(symbol=symbol, trade_type="BEAR_CALL_SHORT", right="C",
                   strike=short_strike, expiry=expiry_str, quantity=-quantity,
                   iv=iv, delta=short_delta, premium=short_premium,
                   margin_requirement=margin_per_spread),  # Margin for the spread
            Signal(symbol=symbol, trade_type="BEAR_CALL_LONG", right="C",
                   strike=long_strike, expiry=expiry_str, quantity=quantity,
                   iv=iv, delta=0, premium=long_premium,
                   margin_requirement=0),  # Long leg is hedged, no margin needed
        ]
