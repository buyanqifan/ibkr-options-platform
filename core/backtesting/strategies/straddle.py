"""Short Straddle strategy."""

from core.backtesting.strategies.base import BaseStrategy, Signal
from core.backtesting.pricing import OptionsPricer
from datetime import datetime, timedelta


class StraddleStrategy(BaseStrategy):
    """Short straddle: sell ATM put + sell ATM call."""

    def __init__(self, params: dict):
        super().__init__(params)
        # Check if profit target/stop loss are disabled (special value 999999 means disabled)
        self._profit_target_disabled = params.get("profit_target_pct", 50) >= 999999
        self._stop_loss_disabled = params.get("stop_loss_pct", 200) >= 999999

    @property
    def name(self) -> str:
        return "straddle"

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

        # ATM strike
        if underlying_price > 100:
            strike = round(underlying_price)
        else:
            strike = round(underlying_price * 2) / 2

        put_premium = OptionsPricer.put_price(underlying_price, strike, T, iv)
        call_premium = OptionsPricer.call_price(underlying_price, strike, T, iv)
        put_delta = OptionsPricer.delta(underlying_price, strike, T, iv, "P")
        call_delta = OptionsPricer.delta(underlying_price, strike, T, iv, "C")

        # Calculate quantity using position manager if available
        if position_mgr:
            # For short straddle, use the higher premium as margin reference
            # Short options require margin for potential assignment
            higher_premium = max(put_premium, call_premium)
            # Margin estimate: higher premium × 100 × leverage factor (typically 10-20x)
            margin_per_contract = higher_premium * 100 * 15  # Conservative 15x multiplier
            margin_per_contract = max(margin_per_contract, strike * 100 * 0.2)  # Min 20% of underlying
            
            num_contracts = position_mgr.calculate_position_size(
                margin_per_contract=margin_per_contract,
                max_positions=max_pos,
            )
        else:
            # Legacy: always 1 contract
            num_contracts = 1
        
        symbol = self.params["symbol"]
        
        # Calculate and assign margin requirement
        higher_premium = max(put_premium, call_premium)
        margin_per_contract = higher_premium * 100 * 15
        margin_per_contract = max(margin_per_contract, strike * 100 * 0.2)
        
        return [
            Signal(symbol=symbol, trade_type="STRADDLE_PUT", right="P",
                   strike=strike, expiry=expiry_str, quantity=-num_contracts,
                   iv=iv, delta=put_delta, premium=put_premium,
                   margin_requirement=margin_per_contract),
            Signal(symbol=symbol, trade_type="STRADDLE_CALL", right="C",
                   strike=strike, expiry=expiry_str, quantity=-num_contracts,
                   iv=iv, delta=call_delta, premium=call_premium,
                   margin_requirement=margin_per_contract),
        ]


class StrangleStrategy(BaseStrategy):
    """Short strangle: sell OTM put + sell OTM call."""

    @property
    def name(self) -> str:
        return "strangle"

    def __init__(self, params: dict):
        super().__init__(params)
        self.put_delta = params.get("put_delta_target", 0.20)
        self.call_delta = params.get("call_delta_target", 0.20)
        # Check if profit target/stop loss are disabled (special value 999999 means disabled)
        self._profit_target_disabled = params.get("profit_target_pct", 50) >= 999999
        self._stop_loss_disabled = params.get("stop_loss_pct", 200) >= 999999

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

        # OTM put
        saved = self.delta_target
        self.delta_target = self.put_delta
        put_strike = self.select_strike(underlying_price, iv, T, "P")
        self.delta_target = saved

        # OTM call
        saved = self.delta_target
        self.delta_target = self.call_delta
        call_strike = self.select_strike(underlying_price, iv, T, "C")
        self.delta_target = saved

        put_premium = OptionsPricer.put_price(underlying_price, put_strike, T, iv)
        call_premium = OptionsPricer.call_price(underlying_price, call_strike, T, iv)
        put_delta = OptionsPricer.delta(underlying_price, put_strike, T, iv, "P")
        call_delta = OptionsPricer.delta(underlying_price, call_strike, T, iv, "C")

        # Calculate quantity using position manager if available
        if position_mgr:
            # For short strangle, similar to straddle but with OTM strikes
            # Risk is slightly lower due to OTM strikes, but still significant
            higher_premium = max(put_premium, call_premium)
            # Margin estimate: higher premium × 100 × leverage factor (slightly lower than straddle)
            margin_per_contract = higher_premium * 100 * 12  # Conservative 12x multiplier
            margin_per_contract = max(margin_per_contract, min(put_strike, call_strike) * 100 * 0.15)  # Min 15% of lower strike
            
            num_contracts = position_mgr.calculate_position_size(
                margin_per_contract=margin_per_contract,
                max_positions=max_pos,
            )
        else:
            # Legacy: always 1 contract
            num_contracts = 1
        
        symbol = self.params["symbol"]
        
        # Calculate and assign margin requirement
        higher_premium = max(put_premium, call_premium)
        margin_per_contract = higher_premium * 100 * 12
        margin_per_contract = max(margin_per_contract, min(put_strike, call_strike) * 100 * 0.15)
        
        return [
            Signal(symbol=symbol, trade_type="STRANGLE_PUT", right="P",
                   strike=put_strike, expiry=expiry_str, quantity=-num_contracts,
                   iv=iv, delta=put_delta, premium=put_premium,
                   margin_requirement=margin_per_contract),
            Signal(symbol=symbol, trade_type="STRANGLE_CALL", right="C",
                   strike=call_strike, expiry=expiry_str, quantity=-num_contracts,
                   iv=iv, delta=call_delta, premium=call_premium,
                   margin_requirement=margin_per_contract),
        ]
