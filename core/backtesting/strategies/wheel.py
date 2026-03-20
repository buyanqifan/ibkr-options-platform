"""Wheel Strategy: cycles between Sell Put and Covered Call.

Phase 1 (SP): Sell OTM puts, collect premium
  - If expires worthless: keep premium, continue selling puts
  - If assigned: buy shares at strike, switch to Phase 2

Phase 2 (CC): Sell OTM calls against owned shares
  - If expires worthless: keep premium + shares, continue selling calls
  - If assigned: sell shares at strike, return to Phase 1
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any
from core.backtesting.strategies.base import BaseStrategy, Signal
from core.backtesting.pricing import OptionsPricer
import logging


@dataclass
class StockHolding:
    """Tracks stock position from put assignment."""
    shares: int = 0
    cost_basis: float = 0.0  # average cost per share
    total_premium_collected: float = 0.0  # cumulative premium from both phases

@dataclass
class PerformanceMetrics:
    """Tracks detailed performance metrics for monitoring."""
    total_trades: int = 0
    successful_assignments: int = 0
    expired_worthless: int = 0
    profit_target_exits: int = 0
    stop_loss_exits: int = 0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    phase_transitions: int = 0
    
    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "successful_assignments": self.successful_assignments,
            "expired_worthless": self.expired_worthless,
            "profit_target_exits": self.profit_target_exits,
            "stop_loss_exits": self.stop_loss_exits,
            "total_pnl": round(self.total_pnl, 2),
            "avg_pnl_per_trade": round(self.avg_pnl_per_trade, 2),
            "win_rate": round(self.win_rate, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "phase_transitions": self.phase_transitions,
        }


class WheelStrategy(BaseStrategy):
    """Wheel strategy implementation with state machine and performance monitoring."""

    def __init__(self, params: dict):
        super().__init__(params)
        self.phase = "SP"  # Start with Sell Put phase
        self.stock_holding = StockHolding()
        self.put_delta = params.get("put_delta", 0.30)
        self.call_delta = params.get("call_delta", 0.30)
        # Track pending assignments
        self._pending_assignment = None
        
        # Performance monitoring
        self.logger = logging.getLogger(f"wheel_strategy_{params.get('symbol', 'unknown')}")
        self.performance_metrics = PerformanceMetrics()
        self.trade_history: List[Dict[str, Any]] = []
        self.phase_history: List[Dict[str, Any]] = []
        self.daily_stats: List[Dict[str, Any]] = []
        self._current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._peak_value = self.initial_capital
        self._trough_value = self.initial_capital
        
        # Check if profit target/stop loss are disabled (special value 999999 means disabled)
        self._profit_target_disabled = params.get("profit_target_pct", 50) >= 999999
        self._stop_loss_disabled = params.get("stop_loss_pct", 200) >= 999999

    @property
    def name(self) -> str:
        return "wheel"

    def on_trade_closed(self, trade: dict):
        """Called by engine when a trade is closed. Updates internal state and tracks performance.
        
        Returns:
            float: Additional P&L from stock position (e.g., when call is assigned).
                   This should be added to cumulative_pnl by the engine.
        """
        # Record trade in history
        self._record_trade(trade)
        
        # Track additional stock P&L that needs to be added to cumulative_pnl
        additional_stock_pnl = 0.0
        
        if trade.get("exit_reason") == "ASSIGNMENT":
            self.performance_metrics.successful_assignments += 1
            if trade.get("trade_type") == "WHEEL_PUT":
                # Put assigned: we bought shares at strike price
                # Note: The option P&L has already been calculated by the simulator
                # We only need to track the stock position for the CC phase
                strike = trade["strike"]
                quantity = abs(trade["quantity"])
                shares_acquired = quantity * 100
                
                # Update stock holding - track shares acquired from assignment
                # Cost basis is tracked separately for stock (not including option premium)
                total_stock_cost = self.stock_holding.shares * self.stock_holding.cost_basis
                total_stock_cost += shares_acquired * strike
                self.stock_holding.shares += shares_acquired
                if self.stock_holding.shares > 0:
                    self.stock_holding.cost_basis = total_stock_cost / self.stock_holding.shares
                
                # Track premium collected from the put sale (already included in option P&L)
                # We track it separately for informational purposes only
                premium_from_put = trade["entry_price"] * shares_acquired
                self.stock_holding.total_premium_collected += premium_from_put
                
                # Log transition
                self._log_transition("SP", "CC", f"Put assigned at ${strike}, acquired {shares_acquired} shares")
                # Switch to Covered Call phase
                self.phase = "CC"
                self.performance_metrics.phase_transitions += 1
                
            elif trade.get("trade_type") == "WHEEL_CALL":
                # Call assigned: we sold shares at strike price
                # The option P&L has already been calculated by the simulator
                # We need to realize the stock P&L here
                strike = trade["strike"]
                quantity = abs(trade["quantity"])
                shares_sold = quantity * 100
                
                # CRITICAL: Defensive check - if no shares held, this is an error
                # This should not happen after the fix to generate_signals, but we guard against it
                if self.stock_holding.shares <= 0 or self.stock_holding.cost_basis <= 0:
                    self.logger.warning(
                        f"Call assigned but no shares held! This indicates a bug. "
                        f"shares={self.stock_holding.shares}, cost_basis={self.stock_holding.cost_basis:.2f}"
                    )
                    # Return 0 to avoid incorrect PnL
                    return 0.0
                
                # Calculate stock P&L from this assignment
                # Limit shares_sold to actual shares held to avoid negative positions
                actual_shares_sold = min(shares_sold, self.stock_holding.shares)
                if actual_shares_sold != shares_sold:
                    self.logger.warning(
                        f"Call assignment for {shares_sold} shares but only {self.stock_holding.shares} held. "
                        f"Adjusting to {actual_shares_sold} shares."
                    )
                    shares_sold = actual_shares_sold
                
                stock_cost_basis = self.stock_holding.cost_basis * shares_sold
                stock_proceeds = strike * shares_sold
                stock_pnl = stock_proceeds - stock_cost_basis  # Realized stock P&L
                
                # IMPORTANT: Record stock P&L to be added to cumulative_pnl
                additional_stock_pnl = stock_pnl
                
                # Reduce stock holding
                self.stock_holding.shares = max(0, self.stock_holding.shares - shares_sold)
                
                # Add premium from call sale (already included in option P&L)
                premium_from_call = trade["entry_price"] * shares_sold
                self.stock_holding.total_premium_collected += premium_from_call
                
                # Log the complete P&L breakdown for transparency
                option_pnl = trade.get("pnl", 0)
                total_trade_pnl = option_pnl + stock_pnl
                self.logger.info(
                    f"Call assigned: Option P&L=${option_pnl:+.2f}, Stock P&L=${stock_pnl:+.2f}, "
                    f"Total=${total_trade_pnl:+.2f} (bought at ${self.stock_holding.cost_basis:.2f}, "
                    f"sold at ${strike:.2f}, {shares_sold} shares)"
                )
                
                # If no more shares, switch back to Sell Put phase
                if self.stock_holding.shares == 0:
                   self._log_transition("CC", "SP", f"Call assigned at ${strike}, sold {shares_sold} shares")
                   self.phase = "SP"
                   self.stock_holding.cost_basis = 0.0
                   self.performance_metrics.phase_transitions += 1
        
        # For Wheel Put (SP phase), ONLY track assignments and expiry
        # Profit target and stop loss should NOT apply - we want to hold until assignment or expiry
        elif trade.get("trade_type") == "WHEEL_PUT":
            if trade.get("exit_reason") == "EXPIRY":
                self.performance_metrics.expired_worthless += 1
                self.logger.info(f"Sell Put expired worthless, keeping premium - ready for next SP cycle")
            # Note: PROFIT_TARGET and STOP_LOSS exits are logged but don't trigger phase transitions
            # This is correct behavior - premature exits reduce probability of assignment
            else:
                exit_reason = trade.get("exit_reason", "UNKNOWN")
                if exit_reason == "PROFIT_TARGET":
                    self.performance_metrics.profit_target_exits += 1
                    self.logger.warning(
                        f"SP phase exited early with profit target - this reduces assignment probability. "
                        f"Consider holding until expiry."
                    )
                elif exit_reason == "STOP_LOSS":
                    self.performance_metrics.stop_loss_exits += 1
                    self.logger.warning(
                        f"SP phase hit stop loss - Wheel strategy aims for assignment, not premium trading. "
                        f"Stop loss may be counterproductive for SP phase."
                    )
        
        # For Wheel Call (CC phase), process normally
        elif trade.get("trade_type") == "WHEEL_CALL":
            # Track exit reasons
            exit_reason = trade.get("exit_reason", "UNKNOWN")
            if exit_reason == "PROFIT_TARGET":
                self.performance_metrics.profit_target_exits += 1
            elif exit_reason == "STOP_LOSS":
                self.performance_metrics.stop_loss_exits += 1
            elif exit_reason == "EXPIRED_WORTHLESS":
                self.performance_metrics.expired_worthless += 1
            
            # Log trade completion
            self.logger.info(f"Trade closed: {trade.get('trade_type')} {exit_reason} PnL: ${trade.get('pnl', 0):+.2f}")
        
        # Return additional stock P&L to be added to cumulative_pnl by the engine
        return additional_stock_pnl

    def generate_signals(
        self,
        current_date: str,
        underlying_price: float,
        iv: float,
        open_positions: list,
        position_mgr=None,
    ) -> list[Signal]:
        """Generate signals based on current phase."""
        max_pos = self.params.get("max_positions", 1)
        
        # Check if we already have an open position
        wheel_positions = [
            p for p in open_positions 
            if p.trade_type in ("WHEEL_PUT", "WHEEL_CALL")
        ]
        if len(wheel_positions) >= max_pos:
            return []
        
        # CRITICAL: For CC phase, check if we already have enough Call contracts
        # Each Call contract locks 100 shares. We cannot sell more Calls than shares owned.
        if self.phase == "CC":
            # Count already sold Call contracts (negative quantity means sold)
            existing_call_contracts = sum(
                abs(p.quantity) for p in wheel_positions 
                if p.trade_type == "WHEEL_CALL"
            )
            # Each contract covers 100 shares
            shares_already_covered = existing_call_contracts * 100
            shares_available = self.stock_holding.shares - shares_already_covered
            
            # If no shares available for more Calls, don't generate signal
            if shares_available <= 0:
                return []

        T = self.select_expiry_dte() / 365.0
        dte_days = int(self.select_expiry_dte())
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=dte_days)
        expiry_str = expiry_date.strftime("%Y%m%d")

        if self.phase == "SP":
            return self._generate_sell_put_signal(
                underlying_price, iv, T, expiry_str, position_mgr
            )
        else:  # phase == "CC"
            return self._generate_covered_call_signal(
                underlying_price, iv, T, expiry_str, position_mgr
            )

    def _generate_sell_put_signal(
        self,
        underlying_price: float,
        iv: float,
        T: float,
        expiry_str: str,
        position_mgr=None,
    ) -> list[Signal]:
        """Generate Sell Put signal for Phase 1."""
        # Use put-specific delta
        original_delta = self.delta_target
        self.delta_target = self.put_delta
        strike = self.select_strike(underlying_price, iv, T, "P")
        self.delta_target = original_delta

        premium = OptionsPricer.put_price(underlying_price, strike, T, iv)
        delta = OptionsPricer.delta(underlying_price, strike, T, iv, "P")

        # Position sizing using position manager
        # Cash-secured put: reserve strike * 100 per contract
        if position_mgr:
            max_contracts = position_mgr.calculate_position_size(
                margin_per_contract=strike * 100,
                max_positions=self.params.get("max_positions", 10),
            )
        else:
            # Fallback when no position manager (e.g., in unit tests)
            max_contracts = self.params.get("max_positions", 1)
        if max_contracts <= 0:
            return []  # No signal if insufficient capital
        
        quantity = -max_contracts  # Sell
        
        # Explicitly set margin requirement for cash-secured put
        margin_per_contract = strike * 100

        return [Signal(
            symbol=self.params["symbol"],
            trade_type="WHEEL_PUT",
            right="P",
            strike=strike,
            expiry=expiry_str,
            quantity=quantity,
            iv=iv,
            delta=delta,
            premium=premium,
            underlying_price=underlying_price,  # Pass stock price at entry
            margin_requirement=margin_per_contract,  # Cash-secured put requires strike × 100
        )]

    def _generate_covered_call_signal(
        self,
        underlying_price: float,
        iv: float,
        T: float,
        expiry_str: str,
        position_mgr=None,
    ) -> list[Signal]:
        """Generate Covered Call signal for Phase 2."""
        # Can only sell calls for shares we own
        if self.stock_holding.shares <= 0:
            # Shouldn't happen, but fallback to SP phase
            self.phase = "SP"
            return self._generate_sell_put_signal(underlying_price, iv, T, expiry_str, position_mgr)

        # Use call-specific delta
        original_delta = self.delta_target
        self.delta_target = self.call_delta
        strike = self.select_strike(underlying_price, iv, T, "C")
        self.delta_target = original_delta

        premium = OptionsPricer.call_price(underlying_price, strike, T, iv)
        delta = OptionsPricer.delta(underlying_price, strike, T, iv, "C")

        # Covered call: 1 contract per 100 shares owned
        # Calculate contracts based on shares held (same as CoveredCallStrategy)
        # generate_signals() already checks for existing call positions to prevent over-selling
        max_by_shares = self.stock_holding.shares // 100
        max_contracts = min(max_by_shares, self.params.get("max_positions", 10))
        
        # Return empty if no shares to cover
        if max_contracts <= 0:
            return []
        
        quantity = -max_contracts  # Sell
        
        # No additional margin required - shares are already owned
        # The margin was allocated in the Sell Put phase when we bought the shares
        margin_requirement = 0.0

        return [Signal(
            symbol=self.params["symbol"],
            trade_type="WHEEL_CALL",
            right="C",
            strike=strike,
            expiry=expiry_str,
            quantity=quantity,
            iv=iv,
            delta=delta,
            premium=premium,
            underlying_price=underlying_price,  # Pass stock price at entry
            margin_requirement=margin_requirement,  # No additional margin needed for covered call
        )]

    def _record_trade(self, trade: dict):
        """Record trade details for performance tracking."""
        self.performance_metrics.total_trades += 1
        pnl = trade.get("pnl", 0)
        self.performance_metrics.total_pnl += pnl
        self.performance_metrics.avg_pnl_per_trade = (
            self.performance_metrics.total_pnl / self.performance_metrics.total_trades
            if self.performance_metrics.total_trades > 0 else 0
        )
        
        # Update win rate
        winning_trades = sum(1 for t in self.trade_history if t.get("pnl", 0) > 0)
        self.performance_metrics.win_rate = (
            (winning_trades + (1 if pnl > 0 else 0)) / self.performance_metrics.total_trades * 100
        )
        
        # Track drawdown
        current_value = self.initial_capital + self.performance_metrics.total_pnl
        if current_value > self._peak_value:
            self._peak_value = current_value
        drawdown = (self._peak_value - current_value) / self._peak_value * 100
        self.performance_metrics.max_drawdown = max(self.performance_metrics.max_drawdown, drawdown)
        
        # Add to trade history
        trade_record = {
            "trade_id": len(self.trade_history) + 1,
            "date": trade.get("exit_date", datetime.now().strftime("%Y-%m-%d")),
            "type": trade.get("trade_type", "UNKNOWN"),
            "symbol": trade.get("symbol", "N/A"),
            "expiry": trade.get("expiry", "N/A"),
            "strike": trade.get("strike", 0),
            "right": trade.get("right", "N/A"),  # P (Put) or C (Call)
            "quantity": trade.get("quantity", 0),
            "exit_reason": trade.get("exit_reason", "UNKNOWN"),
            "pnl": round(pnl, 2),
            "phase": self.phase,
            "cumulative_pnl": round(self.performance_metrics.total_pnl, 2),
        }
        self.trade_history.append(trade_record)
    
    def _log_transition(self, from_phase: str, to_phase: str, reason: str):
        """Log phase transitions for monitoring."""
        transition = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "from_phase": from_phase,
            "to_phase": to_phase,
            "reason": reason,
            "shares_held": self.stock_holding.shares,
            "premium_collected": round(self.stock_holding.total_premium_collected, 2),
        }
        self.phase_history.append(transition)
        self.logger.info(f"Phase transition: {from_phase} → {to_phase} ({reason})")
    
    def update_daily_stats(self, date: str, portfolio_value: float, open_pnl: float):
        """Update daily statistics for performance monitoring."""
        daily_record = {
            "date": date,
            "portfolio_value": round(portfolio_value, 2),
            "open_pnl": round(open_pnl, 2),
            "closed_pnl": round(self.performance_metrics.total_pnl, 2),
            "total_pnl": round(self.performance_metrics.total_pnl + open_pnl, 2),
            "phase": self.phase,
            "shares_held": self.stock_holding.shares,
            "premium_collected": round(self.stock_holding.total_premium_collected, 2),
        }
        self.daily_stats.append(daily_record)
    
    def get_performance_report(self) -> dict:
        """Generate comprehensive performance report."""
        return {
            "strategy": "wheel",
            "run_id": self._current_run_id,
            "current_state": self.get_state_summary(),
            "performance_metrics": self.performance_metrics.to_dict(),
            "trade_history": self.trade_history[-10:],  # Last 10 trades
            "phase_history": self.phase_history[-5:],   # Last 5 transitions
            "daily_stats": self.daily_stats[-30:],      # Last 30 days
        }
    
    def get_state_summary(self) -> dict:
        """Return current strategy state for debugging/display."""
        effective_cost_basis = 0
        if self.stock_holding.shares > 0:
            effective_cost_basis = self.stock_holding.cost_basis - (
                self.stock_holding.total_premium_collected / self.stock_holding.shares
            )
        
        return {
            "phase": self.phase,
            "shares_held": self.stock_holding.shares,
            "cost_basis": round(self.stock_holding.cost_basis, 2),
            "total_premium_collected": round(self.stock_holding.total_premium_collected, 2),
            "effective_cost_basis": round(effective_cost_basis, 2),
            "current_portfolio_value": round(
                self.initial_capital + self.performance_metrics.total_pnl, 2
            ),
            "total_pnl": round(self.performance_metrics.total_pnl, 2),
            "win_rate": round(self.performance_metrics.win_rate, 2),
        }
