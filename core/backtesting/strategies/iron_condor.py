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
        self._profit_target_disabled = params.get("profit_target_pct", 50) >= 999999
        self._stop_loss_disabled = params.get("stop_loss_pct", 200) >= 999999
        # Check if profit target/stop loss are disabled (special value 999999 means disabled)
        self._profit_target_disabled = params.get("profit_target_pct", 50) >= 999999
        self._stop_loss_disabled = params.get("stop_loss_pct", 200) >= 999999

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

        # Get optimized DTE (ML or traditional)
        dte_days = self.select_expiry_dte(underlying_price=underlying_price, iv=iv, right="P")
        T = dte_days / 365.0
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=int(dte_days))
        expiry_str = expiry_date.strftime("%Y%m%d")

        # Short put (OTM) - with ML optimization if enabled
        if self.ml_delta_optimization:
            optimized_put_delta = self.get_optimized_delta(underlying_price, iv, "P")
            original_delta = self.delta_target
            self.delta_target = optimized_put_delta
            short_put_strike = self.select_strike(underlying_price, iv, T, "P")
            self.delta_target = original_delta
        else:
            saved = self.delta_target
            self.delta_target = self.put_delta
            short_put_strike = self.select_strike(underlying_price, iv, T, "P")
            self.delta_target = saved
        short_put_premium = OptionsPricer.put_price(underlying_price, short_put_strike, T, iv)
        short_put_delta = OptionsPricer.delta(underlying_price, short_put_strike, T, iv, "P")

        # Short call (OTM) - with ML optimization if enabled
        if self.ml_delta_optimization:
            optimized_call_delta = self.get_optimized_delta(underlying_price, iv, "C")
            original_delta = self.delta_target
            self.delta_target = optimized_call_delta
            short_call_strike = self.select_strike(underlying_price, iv, T, "C")
            self.delta_target = original_delta
        else:
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

        net_premium = short_put_premium + short_call_premium - long_put_premium - long_call_premium

        # Calculate position size using position manager if available
        if position_mgr:
            # Iron Condor margin: spread width - net credit
            max_spread_width = self.wing_width
            estimated_margin_per_spread = (max_spread_width * 100) - (net_premium * 100)
            estimated_margin_per_spread = max(estimated_margin_per_spread, 100)
            
            num_spreads = position_mgr.calculate_position_size(
                margin_per_contract=estimated_margin_per_spread,
                max_positions=min(max_pos, 5),  # Cap at 5 spreads
            )
        else:
            # Fallback to legacy calculation - use initial_capital with leverage
            available_capital = self.initial_capital
            leveraged_capital = available_capital * self.max_leverage
            
            max_spread_width = self.wing_width
            estimated_margin_per_spread = (max_spread_width * 100) - (net_premium * 100)
            estimated_margin_per_spread = max(estimated_margin_per_spread, 100)
            
            max_spreads_by_capital = int(leveraged_capital / estimated_margin_per_spread)
            num_spreads = min(max_pos, max_spreads_by_capital, 5)
            num_spreads = max(1, num_spreads)
        
        quantity = num_spreads
        symbol = self.params["symbol"]

        # Calculate margin requirements for each side of the iron condor
        # Put side margin
        put_net_credit = short_put_premium - long_put_premium
        put_margin = (self.wing_width * 100) - (put_net_credit * 100)
        put_margin = max(put_margin, 100)  # Minimum $100
        
        # Call side margin
        call_net_credit = short_call_premium - long_call_premium
        call_margin = (self.wing_width * 100) - (call_net_credit * 100)
        call_margin = max(call_margin, 100)  # Minimum $100
        
        # Iron Condor uses the wider of the two margins (they don't stack)
        # Only one side will be tested at expiration
        total_margin_per_contract = max(put_margin, call_margin)

        signals = [
            Signal(symbol=symbol, trade_type="IRON_CONDOR_SP", right="P",
                   strike=short_put_strike, expiry=expiry_str, quantity=-quantity,
                   iv=iv, delta=short_put_delta, premium=short_put_premium,
                   margin_requirement=total_margin_per_contract),  # Share total margin
            Signal(symbol=symbol, trade_type="IRON_CONDOR_LP", right="P",
                   strike=long_put_strike, expiry=expiry_str, quantity=quantity,
                   iv=iv, delta=0, premium=long_put_premium,
                   margin_requirement=0),  # Long leg hedged
            Signal(symbol=symbol, trade_type="IRON_CONDOR_SC", right="C",
                   strike=short_call_strike, expiry=expiry_str, quantity=-quantity,
                   iv=iv, delta=short_call_delta, premium=short_call_premium,
                   margin_requirement=0),  # Already accounted for in SP leg
            Signal(symbol=symbol, trade_type="IRON_CONDOR_LC", right="C",
                   strike=long_call_strike, expiry=expiry_str, quantity=quantity,
                   iv=iv, delta=0, premium=long_call_premium,
                   margin_requirement=0),  # Long leg hedged
        ]
        return signals
