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

    @property
    def name(self) -> str:
        return "wheel"

    def on_trade_closed(self, trade: dict):
        """Called by engine when a trade is closed. Updates internal state and tracks performance."""
        # Record trade in history
        self._record_trade(trade)
        
        if trade.get("exit_reason") == "ASSIGNMENT":
            self.performance_metrics.successful_assignments += 1
            if trade.get("trade_type") == "WHEEL_PUT":
                # Put assigned: we bought shares at strike price
                strike = trade["strike"]
                quantity = abs(trade["quantity"])
                shares_acquired = quantity * 100
                
                # Update stock holding
                total_cost = self.stock_holding.shares * self.stock_holding.cost_basis
                total_cost += shares_acquired * strike
                self.stock_holding.shares += shares_acquired
                if self.stock_holding.shares > 0:
                    self.stock_holding.cost_basis = total_cost / self.stock_holding.shares
                
                # Adjust cost basis by premium collected
                premium_collected = trade["entry_price"] * shares_acquired
                self.stock_holding.total_premium_collected += premium_collected
                
                # Log transition
                self._log_transition("SP", "CC", f"Put assigned at ${strike}, acquired {shares_acquired} shares")
                # Switch to Covered Call phase
                self.phase = "CC"
                self.performance_metrics.phase_transitions += 1
                
            elif trade.get("trade_type") == "WHEEL_CALL":
                # Call assigned: we sold shares at strike price
                strike = trade["strike"]
                quantity = abs(trade["quantity"])
                shares_sold = quantity * 100
                
                # Reduce stock holding
                self.stock_holding.shares = max(0, self.stock_holding.shares - shares_sold)
                
                # Add premium to total collected
                premium_collected = trade["entry_price"] * shares_sold
                self.stock_holding.total_premium_collected += premium_collected
                
                # If no more shares, switch back to Sell Put phase
                if self.stock_holding.shares == 0:
                    self._log_transition("CC", "SP", f"Call assigned at ${strike}, sold {shares_sold} shares")
                    self.phase = "SP"
                    self.stock_holding.cost_basis = 0.0
                    self.performance_metrics.phase_transitions += 1
        else:
            # For non-assignment exits (profit target, stop loss, expiry worthless)
            if trade.get("trade_type") in ("WHEEL_PUT", "WHEEL_CALL"):
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

        # Position sizing using position manager if available
        if position_mgr:
            # Cash-secured put: reserve strike * 100 per contract
            max_contracts = position_mgr.calculate_position_size(
                margin_per_contract=strike * 100,
                max_positions=self.params.get("max_positions", 1),
            )
        else:
            # Fallback to legacy calculation
            available_capital = self.initial_capital * self.position_percentage
            leveraged_capital = available_capital * self.max_leverage
            max_contracts = int(leveraged_capital / (strike * 100))
            max_contracts = max(1, min(max_contracts, self.params.get("max_positions", 1)))
        
        quantity = -max_contracts  # Sell

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
        max_contracts_by_shares = self.stock_holding.shares // 100
        
        # Also check capital constraints if position_mgr available
        if position_mgr:
            # For covered call, need to verify we have enough capital for shares
            max_contracts_by_capital = position_mgr.calculate_position_size(
                margin_per_contract=underlying_price * 100,
                max_positions=self.params.get("max_positions", 1),
            )
            # Take minimum of shares-based and capital-based sizing
            max_contracts = min(max_contracts_by_shares, max_contracts_by_capital)
            max_contracts = max(1, min(max_contracts, self.params.get("max_positions", 1)))
        else:
            max_contracts = max_contracts_by_shares
            max_contracts = max(1, min(max_contracts, self.params.get("max_positions", 1)))
        
        quantity = -max_contracts  # Sell

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
            "exit_reason": trade.get("exit_reason", "UNKNOWN"),
            "pnl": round(pnl, 2),
            "strike": trade.get("strike", 0),
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
