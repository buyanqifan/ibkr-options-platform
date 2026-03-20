"""Trade simulator: manages open positions, checks exit conditions, tracks P&L."""

from dataclasses import dataclass, field
from datetime import date
from core.backtesting.pricing import OptionsPricer


@dataclass
class OptionPosition:
    """Represents an open option position."""
    symbol: str
    entry_date: str
    expiry: str
    strike: float
    right: str           # "P" or "C"
    trade_type: str      # e.g. "SELL_PUT"
    quantity: int         # negative for short
    entry_price: float   # premium per share
    underlying_entry: float
    iv_at_entry: float
    delta_at_entry: float
    current_price: float = 0.0
    current_pnl: float = 0.0
    capital_at_entry: float = 0.0  # Total portfolio capital when opening position


@dataclass
class TradeRecord:
    """Completed trade record."""
    symbol: str
    trade_type: str
    entry_date: str
    exit_date: str
    expiry: str
    strike: float
    right: str  # P (Put) or C (Call)
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    underlying_entry: float
    underlying_exit: float
    iv_at_entry: float
    delta_at_entry: float
    capital_at_entry: float = 0.0      # Total capital when opening position
    capital_at_exit: float = 0.0       # Total capital when closing position

    def to_dict(self) -> dict:
        # Generate formatted option contract name
        # Note: This is a theoretical option based on BS model pricing
        # Real options have discrete strikes (e.g., $2.5/$5 intervals), not continuous values
        try:
            expiry_short = self.expiry[2:] if len(self.expiry) >= 6 else self.expiry
            option_type = "Put" if self.right == "P" else "Call"
            
            # Format strike with appropriate precision
            # Use integer if whole number, otherwise keep 1-2 decimals
            if self.strike == int(self.strike):
                strike_display = int(self.strike)
            elif self.strike * 10 == int(self.strike * 10):
                strike_display = round(self.strike, 1)  # e.g., 745.5
            else:
                strike_display = round(self.strike, 2)  # e.g., 745.37
            
            # Add "~" prefix to indicate this is a theoretical/synthetic option
            # Real-world options only trade at discrete strike intervals
            contract_name = f"{self.symbol} {expiry_short} ~{strike_display} {option_type} (Theoretical)"
        except:
            contract_name = f"{self.symbol} {self.expiry} {self.strike} {self.right}"
        
        return {
            "symbol": self.symbol,
            "trade_type": self.trade_type,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "expiry": self.expiry,
            "strike": self.strike,
            "right": self.right,
            "quantity": self.quantity,
            "contract_name": contract_name,  # Formatted option contract name (theoretical)
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 1),
            "exit_reason": self.exit_reason,
            "underlying_entry": round(self.underlying_entry, 2),
            "underlying_exit": round(self.underlying_exit, 2),
            "iv_at_entry": round(self.iv_at_entry, 4),
            "delta_at_entry": round(self.delta_at_entry, 3),
            "capital_at_entry": round(self.capital_at_entry, 2),
            "capital_at_exit": round(self.capital_at_exit, 2),
        }


class TradeSimulator:
    """Manages option positions and checks exit conditions."""

    def __init__(self):
        self.open_positions: list[OptionPosition] = []
        self.closed_trades: list[TradeRecord] = []

    def open_position(self, position: OptionPosition):
        self.open_positions.append(position)

    def check_exits_for_position(
        self,
        pos: OptionPosition,
        current_date: str,
        underlying_price: float,
        iv: float,
        profit_target_pct: float,
        stop_loss_pct: float,
        min_dte: int = 0,
    ) -> TradeRecord | None:
        """Check a single position for exit conditions.

        This method is used for multi-stock strategies where each position
        may have a different underlying price.

        Returns:
            TradeRecord if position was closed, None otherwise
        """
        from datetime import datetime

        exit_reason = None
        exit_price = 0.0

        # Calculate DTE
        try:
            expiry_date = datetime.strptime(pos.expiry, "%Y%m%d").date()
            curr_date = datetime.strptime(current_date, "%Y-%m-%d").date()
            dte = (expiry_date - curr_date).days
        except (ValueError, TypeError):
            dte = 999

        # Calculate current option price using Black-Scholes
        T = max(dte / 365.0, 0.001)
        if pos.right == "P":
            current_opt_price = OptionsPricer.put_price(
                underlying_price, pos.strike, T, iv
            )
        else:
            current_opt_price = OptionsPricer.call_price(
                underlying_price, pos.strike, T, iv
            )

        # For short positions, profit = entry_premium - current_premium
        if pos.quantity < 0:
            pnl_per_share = pos.entry_price - current_opt_price
            pnl_pct = (pnl_per_share / pos.entry_price * 100) if pos.entry_price > 0 else 0
        else:
            pnl_per_share = current_opt_price - pos.entry_price
            pnl_pct = (pnl_per_share / pos.entry_price * 100) if pos.entry_price > 0 else 0

        # Check profit target (dynamic adjustment based on DTE)
        adjusted_profit_target = profit_target_pct
        if dte <= 7:  # Last week: reduce target to capture remaining premium
            adjusted_profit_target = profit_target_pct * 0.7

        if pnl_pct >= adjusted_profit_target:
            exit_reason = "PROFIT_TARGET"
            exit_price = current_opt_price

        # Check stop loss (dynamic adjustment based on DTE and holding period)
        adjusted_stop_loss = stop_loss_pct
        if dte <= 7:  # Last week: widen stop to avoid whipsaw
            adjusted_stop_loss = stop_loss_pct * 1.3

        if pnl_pct <= -adjusted_stop_loss:
            exit_reason = "STOP_LOSS"
            exit_price = current_opt_price

        # Check expiration
        elif dte <= min_dte:
            # At expiration, option value is its intrinsic value
            if pos.right == "P":  # Put option
                intrinsic_value = max(0, pos.strike - underlying_price)
            else:  # Call option
                intrinsic_value = max(0, underlying_price - pos.strike)

            exit_price = intrinsic_value
            exit_reason = "EXPIRY"

            # Check assignment for short puts (stock price below strike)
            if pos.right == "P" and pos.quantity < 0 and underlying_price < pos.strike:
                exit_reason = "ASSIGNMENT"

            # Check assignment for short calls (stock price above strike)
            if pos.right == "C" and pos.quantity < 0 and underlying_price > pos.strike:
                exit_reason = "ASSIGNMENT"

        if exit_reason:
            total_pnl = pnl_per_share * abs(pos.quantity) * 100  # options multiplier
            total_pnl_pct = pnl_pct

            trade = TradeRecord(
                symbol=pos.symbol,
                trade_type=pos.trade_type,
                entry_date=pos.entry_date,
                exit_date=current_date,
                expiry=pos.expiry,
                strike=pos.strike,
                right=pos.right,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                quantity=pos.quantity,
                pnl=total_pnl,
                pnl_pct=total_pnl_pct,
                exit_reason=exit_reason,
                underlying_entry=pos.underlying_entry,
                underlying_exit=underlying_price,
                iv_at_entry=pos.iv_at_entry,
                delta_at_entry=pos.delta_at_entry,
                capital_at_entry=getattr(pos, 'capital_at_entry', 0.0),
                capital_at_exit=0.0,
            )
            self.closed_trades.append(trade)
            return trade
        else:
            pos.current_price = current_opt_price
            pos.current_pnl = pnl_per_share * abs(pos.quantity) * 100
            return None

    def check_exits(
        self,
        current_date: str,
        underlying_price: float,
        iv: float,
        profit_target_pct: float,
        stop_loss_pct: float,
        min_dte: int = 0,
    ) -> list[TradeRecord]:
        """Check all open positions for exit conditions. Returns closed trades."""
        closed = []
        remaining = []

        for pos in self.open_positions:
            exit_reason = None
            exit_price = 0.0

            # Calculate DTE
            from datetime import datetime
            try:
                expiry_date = datetime.strptime(pos.expiry, "%Y%m%d").date()
                curr_date = datetime.strptime(current_date, "%Y-%m-%d").date()
                dte = (expiry_date - curr_date).days
            except (ValueError, TypeError):
                dte = 999

            # Calculate current option price using Black-Scholes
            T = max(dte / 365.0, 0.001)
            if pos.right == "P":
                current_opt_price = OptionsPricer.put_price(
                    underlying_price, pos.strike, T, iv
                )
            else:
                current_opt_price = OptionsPricer.call_price(
                    underlying_price, pos.strike, T, iv
                )

            # For short positions, profit = entry_premium - current_premium
            if pos.quantity < 0:
                pnl_per_share = pos.entry_price - current_opt_price
                pnl_pct = (pnl_per_share / pos.entry_price * 100) if pos.entry_price > 0 else 0
            else:
                pnl_per_share = current_opt_price - pos.entry_price
                pnl_pct = (pnl_per_share / pos.entry_price * 100) if pos.entry_price > 0 else 0

            # Check profit target (dynamic adjustment based on DTE)
            adjusted_profit_target = profit_target_pct
            if dte <= 7:  # Last week: reduce target to capture remaining premium
                adjusted_profit_target = profit_target_pct * 0.7
            
            if pnl_pct >= adjusted_profit_target:
                exit_reason = "PROFIT_TARGET"
                exit_price = current_opt_price

            # Check stop loss (dynamic adjustment based on DTE and holding period)
            adjusted_stop_loss = stop_loss_pct
            if dte <= 7:  # Last week: widen stop to avoid whipsaw
                adjusted_stop_loss = stop_loss_pct * 1.3
            
            if pnl_pct <= -adjusted_stop_loss:
                exit_reason = "STOP_LOSS"
                exit_price = current_opt_price

            # Check expiration
            elif dte <= min_dte:
                # At expiration, option value is its intrinsic value
                if pos.right == "P":  # Put option
                    intrinsic_value = max(0, pos.strike - underlying_price)
                else:  # Call option
                    intrinsic_value = max(0, underlying_price - pos.strike)
                
                exit_price = intrinsic_value
                exit_reason = "EXPIRY"
                
                # Check assignment for short puts (stock price below strike)
                if pos.right == "P" and pos.quantity < 0 and underlying_price < pos.strike:
                    exit_reason = "ASSIGNMENT"
                
                # Check assignment for short calls (stock price above strike)
                if pos.right == "C" and pos.quantity < 0 and underlying_price > pos.strike:
                    exit_reason = "ASSIGNMENT"

            if exit_reason:
                total_pnl = pnl_per_share * abs(pos.quantity) * 100  # options multiplier
                total_pnl_pct = pnl_pct

                trade = TradeRecord(
                    symbol=pos.symbol,
                    trade_type=pos.trade_type,
                    entry_date=pos.entry_date,
                    exit_date=current_date,
                    expiry=pos.expiry,
                    strike=pos.strike,
                    right=pos.right,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    quantity=pos.quantity,
                    pnl=total_pnl,
                    pnl_pct=total_pnl_pct,
                    exit_reason=exit_reason,
                    underlying_entry=pos.underlying_entry,
                    underlying_exit=underlying_price,
                    iv_at_entry=pos.iv_at_entry,
                    delta_at_entry=pos.delta_at_entry,
                    capital_at_entry=getattr(pos, 'capital_at_entry', 0.0),  # Pass from position
                    capital_at_exit=0.0,  # Will be set by engine after margin release
                )
                closed.append(trade)
                self.closed_trades.append(trade)
            else:
                pos.current_price = current_opt_price
                pos.current_pnl = pnl_per_share * abs(pos.quantity) * 100
                remaining.append(pos)

        self.open_positions = remaining
        return closed

    def get_total_open_pnl(self) -> float:
        return sum(p.current_pnl for p in self.open_positions)
