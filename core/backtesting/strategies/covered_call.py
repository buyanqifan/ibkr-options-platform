"""Covered Call strategy with proper stock tracking."""

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from core.backtesting.strategies.base import BaseStrategy, Signal
from core.backtesting.pricing import OptionsPricer


@dataclass
class StockHolding:
    """Tracks stock position for covered call writing."""
    shares: int = 0
    cost_basis: float = 0.0
    total_premium_collected: float = 0.0


class CoveredCallStrategy(BaseStrategy):
    """Covered Call strategy - must own stock before selling calls.
    
    Logic Flow:
    1. Own shares (assumed to be purchased at start)
    2. Sell OTM calls against owned shares
    3. Track stock P&L when calls are assigned
    """

    def __init__(self, params: dict):
       super().__init__(params)
       self.stock_holding = StockHolding()
       self._profit_target_disabled = params.get("profit_target_pct", 50) >= 999999
       self._stop_loss_disabled = params.get("stop_loss_pct", 200) >= 999999
       self.logger= logging.getLogger(f"covered_call_{params.get('symbol', 'unknown')}")

    @property
    def name(self) -> str:
        return "covered_call"

    def initialize_stock_position(self, initial_price: float):
        """Initialize stock position at the start of backtest.
        
        This simulates buying the underlying stock before selling calls.
        Called once by the engine at the beginning.
        """
        # Buy stock with most of the capital (leave some for flexibility)
        shares_to_buy = int((self.initial_capital * 0.95) / initial_price)
        shares_to_buy = (shares_to_buy // 100) * 100  # Round to nearest 100
        
        if shares_to_buy > 0:
           self.stock_holding.shares = shares_to_buy
           self.stock_holding.cost_basis = initial_price
           self.logger.info(
                f"Initialized with {shares_to_buy} shares @ ${initial_price:.2f} "
                f"(cost basis: ${shares_to_buy * initial_price:.2f})"
            )

    def generate_signals(
        self,
        current_date: str,
        underlying_price: float,
        iv: float,
        open_positions: list,
        position_mgr=None,
    ) -> list[Signal]:
        max_pos = self.params.get("max_positions", 1)
        
        # Check if we already have an open call position
        cc_positions = [p for p in open_positions if p.trade_type == "COVERED_CALL"]
        if len(cc_positions) >= max_pos:
            return []
        
        # Must own shares to sell covered calls!
        if self.stock_holding.shares <= 0:
            self.logger.warning("No shares owned - cannot sell covered calls")
            return []
        
        # Can only sell calls for shares we own (1 contract per 100 shares)
        max_contracts = min(self.stock_holding.shares // 100, max_pos, 10)
        if max_contracts <= 0:
            return []
        
        T = self.select_expiry_dte() / 365.0
        strike = self.select_strike(underlying_price, iv, T, "C")
        premium = OptionsPricer.call_price(underlying_price, strike, T, iv)
        delta = OptionsPricer.delta(underlying_price, strike, T, iv, "C")
        
        dte_days = int(self.select_expiry_dte())
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=dte_days)
        expiry_str = expiry_date.strftime("%Y%m%d")
        
        quantity = -max_contracts  # Sell calls
        
        return [Signal(
            symbol=self.params["symbol"],
            trade_type="COVERED_CALL",
            right="C",
            strike=strike,
            expiry=expiry_str,
            quantity=quantity,
            iv=iv,
            delta=delta,
            premium=premium,
            underlying_price=underlying_price,
           margin_requirement=0.0,  # No additional margin - shares are collateral
        )]

    def on_trade_closed(self, trade: dict):
        """Called when option position is closed."""
        exit_reason = trade.get("exit_reason", "")
        option_pnl = trade.get("pnl", 0)
        
        if exit_reason == "EXPIRY":
            # Call expired worthless - keep premium and shares
            premium_kept = abs(trade.get("entry_price", 0)) * abs(trade.get("quantity", 0)) * 100
            self.stock_holding.total_premium_collected += premium_kept
            self.logger.info(f"Call expired worthless, keeping ${premium_kept:.2f} premium")
            
        elif exit_reason == "ASSIGNMENT":
            # Call assigned - sell shares at strike price
            strike = trade.get("strike", 0)
            quantity = abs(trade.get("quantity", 0))
            shares_sold = quantity * 100
            
            if shares_sold <= self.stock_holding.shares:
                # Calculate stock P&L
                stock_cost = self.stock_holding.cost_basis * shares_sold
                stock_proceeds = strike * shares_sold
                stock_pnl = stock_proceeds - stock_cost
                
                # Add option premium received
                option_premium = trade.get("entry_price", 0) * shares_sold
                
                # Total P&L from this assignment
                total_pnl = stock_pnl + option_premium
                
                # Update holdings
                self.stock_holding.shares -= shares_sold
                if self.stock_holding.shares == 0:
                    self.stock_holding.cost_basis = 0.0
                
                self.logger.info(
                    f"Call assigned: Option P&L=${option_pnl:+.2f}, Stock P&L=${stock_pnl:+.2f}, "
                    f"Total=${total_pnl:+.2f} (sold {shares_sold} shares @ ${strike:.2f})"
                )
            else:
                self.logger.error(f"Assignment error: trying to sell {shares_sold} but only have {self.stock_holding.shares}")
