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
    position_id: str = ""  # Unique identifier for margin tracking (must match allocation)
    current_price: float = 0.0
    current_pnl: float = 0.0
    capital_at_entry: float = 0.0  # Total portfolio capital when opening position
    strategy_phase: str = "SP"  # SP, CC, or CC+SP (binbingod策略优化)
    # Entry costs tracking for P&L breakdown
    entry_commission: float = 0.0  # Commission paid at entry
    entry_slippage: float = 0.0    # Slippage at entry


@dataclass
class PnLBreakdown:
    """Detailed P&L breakdown for transparency and verification."""
    option_pnl: float = 0.0        # P&L from option premium
    stock_pnl: float = 0.0         # P&L from stock position (CC assignment, etc.)
    entry_commission: float = 0.0  # Commission at entry
    exit_commission: float = 0.0   # Commission at exit
    entry_slippage: float = 0.0    # Slippage at entry
    exit_slippage: float = 0.0     # Slippage at exit
    margin_interest: float = 0.0   # Margin interest accrued
    
    @property
    def total_commission(self) -> float:
        return self.entry_commission + self.exit_commission
    
    @property
    def total_slippage(self) -> float:
        return self.entry_slippage + self.exit_slippage
    
    @property
    def total_costs(self) -> float:
        return self.total_commission + self.total_slippage + self.margin_interest
    
    @property
    def gross_pnl(self) -> float:
        return self.option_pnl + self.stock_pnl
    
    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.total_costs
    
    def to_dict(self) -> dict:
        return {
            "option_pnl": round(self.option_pnl, 2),
            "stock_pnl": round(self.stock_pnl, 2),
            "gross_pnl": round(self.gross_pnl, 2),
            "entry_commission": round(self.entry_commission, 2),
            "exit_commission": round(self.exit_commission, 2),
            "total_commission": round(self.total_commission, 2),
            "entry_slippage": round(self.entry_slippage, 2),
            "exit_slippage": round(self.exit_slippage, 2),
            "total_slippage": round(self.total_slippage, 2),
            "margin_interest": round(self.margin_interest, 2),
            "total_costs": round(self.total_costs, 2),
            "net_pnl": round(self.net_pnl, 2),
        }


@dataclass
class TradeRecord:
    """Completed trade record with detailed P&L breakdown."""
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
    position_id: str = ""               # Unique identifier for margin tracking
    capital_at_entry: float = 0.0      # Total capital when opening position
    capital_at_exit: float = 0.0       # Total capital when closing position
    strategy_phase: str = "SP"  # SP, CC, or CC+SP (binbingod策略优化)
    pnl_breakdown: PnLBreakdown = None  # Detailed P&L breakdown for verification

    def to_dict(self) -> dict:
        # Generate formatted contract name based on type
        try:
            if self.right == "S":
                # Stock trade
                if self.trade_type == "STOCK_BUY":
                    contract_name = f"{self.symbol} STOCK BUY @ ${self.strike:.2f}"
                elif self.trade_type == "STOCK_SELL":
                    contract_name = f"{self.symbol} STOCK SELL @ ${self.strike:.2f}"
                else:
                    contract_name = f"{self.symbol} STOCK"
            else:
                # Option trade
                expiry_short = self.expiry[2:] if len(self.expiry) >= 6 else self.expiry
                option_type = "Put" if self.right == "P" else "Call"
                    
                # Format strike with appropriate precision
                if self.strike == int(self.strike):
                    strike_display = int(self.strike)
                elif self.strike * 10 == int(self.strike * 10):
                    strike_display = round(self.strike, 1)
                else:
                    strike_display = round(self.strike, 2)
                    
                contract_name = f"{self.symbol} {expiry_short} ~{strike_display} {option_type} (Theoretical)"
        except (ValueError, TypeError, AttributeError):
            contract_name = f"{self.symbol} {self.expiry} {self.strike} {self.right}"
        
        result = {
            "symbol": self.symbol,
            "trade_type": self.trade_type,
            "strategy_phase": self.strategy_phase,  # SP, CC, or CC+SP
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
        
        # Add P&L breakdown if available
        if self.pnl_breakdown:
            result["pnl_breakdown"] = self.pnl_breakdown.to_dict()
        
        return result


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
                # CRITICAL FIX: When assigned, we don't buy back the option
                # The option pnl should be the premium received, not (entry - intrinsic)
                # Recalculate pnl_per_share for assignment case
                pnl_per_share = pos.entry_price  # Premium received = profit
                pnl_pct = 100.0  # 100% profit on premium received
                exit_price = 0.0  # No cost to close when assigned

            # Check assignment for short calls (stock price above strike)
            if pos.right == "C" and pos.quantity < 0 and underlying_price > pos.strike:
                exit_reason = "ASSIGNMENT"
                # CRITICAL FIX: When assigned, we don't buy back the option
                # The option pnl should be the premium received, not (entry - intrinsic)
                # Recalculate pnl_per_share for assignment case
                pnl_per_share = pos.entry_price  # Premium received = profit
                pnl_pct = 100.0  # 100% profit on premium received
                exit_price = 0.0  # No cost to close when assigned

        if exit_reason:
            total_pnl = pnl_per_share * abs(pos.quantity) * 100  # options multiplier
            total_pnl_pct = pnl_pct

            # Create P&L breakdown for transparency
            pnl_breakdown = PnLBreakdown(
                option_pnl=total_pnl,
                stock_pnl=0.0,  # Will be updated by engine for assignments
                entry_commission=getattr(pos, 'entry_commission', 0.0),
                exit_commission=0.0,  # Will be updated by engine
                entry_slippage=getattr(pos, 'entry_slippage', 0.0),
                exit_slippage=0.0,  # Will be updated by engine
                margin_interest=0.0,
            )

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
                position_id=getattr(pos, 'position_id', ''),
                capital_at_entry=getattr(pos, 'capital_at_entry', 0.0),
                capital_at_exit=0.0,
                strategy_phase=getattr(pos, 'strategy_phase', 'SP'),  # 传递strategy_phase
                pnl_breakdown=pnl_breakdown,
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
                # CRITICAL FIX: When assigned, we don't buy back the option
                # The option pnl should be the premium received, not (entry - intrinsic)
                if pos.right == "P" and pos.quantity < 0 and underlying_price < pos.strike:
                    exit_reason = "ASSIGNMENT"
                    pnl_per_share = pos.entry_price  # Premium received = profit
                    pnl_pct = 100.0  # 100% profit on premium received
                    exit_price = 0.0  # No cost to close when assigned
                
                # Check assignment for short calls (stock price above strike)
                if pos.right == "C" and pos.quantity < 0 and underlying_price > pos.strike:
                    exit_reason = "ASSIGNMENT"
                    pnl_per_share = pos.entry_price  # Premium received = profit
                    pnl_pct = 100.0  # 100% profit on premium received
                    exit_price = 0.0  # No cost to close when assigned

            if exit_reason:
                total_pnl = pnl_per_share * abs(pos.quantity) * 100  # options multiplier
                total_pnl_pct = pnl_pct

                # Create P&L breakdown for transparency
                pnl_breakdown = PnLBreakdown(
                    option_pnl=total_pnl,
                    stock_pnl=0.0,  # Will be updated by engine for assignments
                    entry_commission=getattr(pos, 'entry_commission', 0.0),
                    exit_commission=0.0,  # Will be updated by engine
                    entry_slippage=getattr(pos, 'entry_slippage', 0.0),
                    exit_slippage=0.0,  # Will be updated by engine
                    margin_interest=0.0,
                )

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
                    position_id=getattr(pos, 'position_id', ''),
                    capital_at_entry=getattr(pos, 'capital_at_entry', 0.0),  # Pass from position
                    capital_at_exit=0.0,  # Will be set by engine after margin release
                    strategy_phase=getattr(pos, 'strategy_phase', 'SP'),  # 传递strategy_phase
                    pnl_breakdown=pnl_breakdown,
                )
                closed.append(trade)
                self.closed_trades.append(trade)
            else:
                pos.current_price = current_opt_price
                pos.current_pnl = pnl_per_share * abs(pos.quantity) * 100
                remaining.append(pos)

        self.open_positions = remaining
        return closed

    def check_position_management(
        self,
        current_date: str,
        price_lookup: dict[str, float],
        iv: float,
        profit_target_pct: float,
        stop_loss_pct: float,
    ) -> tuple[list[TradeRecord], list[OptionPosition]]:
        """Check only proactive close/roll rules, leaving expiry for a later phase."""
        closed: list[TradeRecord] = []
        remaining: list[OptionPosition] = []
        for pos in self.open_positions:
            symbol_price = price_lookup.get(pos.symbol, pos.underlying_entry)
            trade = self.check_exits_for_position(
                pos=pos,
                current_date=current_date,
                underlying_price=symbol_price,
                iv=iv,
                profit_target_pct=profit_target_pct,
                stop_loss_pct=stop_loss_pct,
                min_dte=-999,
            )
            if trade and trade.exit_reason not in ("EXPIRY", "ASSIGNMENT"):
                closed.append(trade)
            else:
                if trade and trade in self.closed_trades:
                    self.closed_trades.remove(trade)
                remaining.append(pos)
        return closed, remaining

    def check_expiries(
        self,
        current_date: str,
        price_lookup: dict[str, float],
        iv: float,
    ) -> tuple[list[TradeRecord], list[OptionPosition]]:
        """Check only expiry/assignment events for the current date."""
        from datetime import datetime

        closed: list[TradeRecord] = []
        remaining: list[OptionPosition] = []
        for pos in self.open_positions:
            symbol_price = price_lookup.get(pos.symbol, pos.underlying_entry)
            try:
                expiry_date = datetime.strptime(pos.expiry, "%Y%m%d").date()
                curr_date = datetime.strptime(current_date, "%Y-%m-%d").date()
                dte = (expiry_date - curr_date).days
            except (ValueError, TypeError):
                dte = 999

            if dte > 0:
                remaining.append(pos)
                continue

            trade = self.check_exits_for_position(
                pos=pos,
                current_date=current_date,
                underlying_price=symbol_price,
                iv=iv,
                profit_target_pct=999999,
                stop_loss_pct=999999,
                min_dte=0,
            )
            if trade:
                closed.append(trade)
            else:
                remaining.append(pos)
        return closed, remaining

    def get_total_open_pnl(self) -> float:
        return sum(p.current_pnl for p in self.open_positions)
