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
    symbol: str = ""  # Symbol of the stock being held

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
        self.symbol = params.get("symbol", "")  # Stock symbol for this strategy
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

        # ML Position Optimization
        self.ml_position_optimization = params.get("ml_position_optimization", False)
        self.ml_position_optimizer = None
        self.position_recommendations = []  # Track ML recommendations for analysis

        if self.ml_position_optimization:
            try:
                from core.ml.position_optimizer import MLPositionOptimizer, WheelPositionIntegration
                self.ml_position_optimizer = WheelPositionIntegration(
                    optimizer=MLPositionOptimizer(model_path=params.get("ml_position_model_path")),
                    enabled=True,
                    fallback_to_rules=True,
                )
                self.logger.info("ML Position Optimizer initialized for Wheel strategy")
            except ImportError as e:
                self.logger.warning(f"ML Position optimization not available: {e}")
                self.ml_position_optimization = False
            except Exception as e:
                self.logger.warning(f"ML Position optimization initialization failed: {e}")
                self.ml_position_optimization = False

    @property
    def name(self) -> str:
        return "wheel"

    def _calculate_ml_position_size(
        self,
        underlying_price: float,
        iv: float,
        strike: float,
        premium: float,
        dte: int,
        delta: float,
        position_mgr,
        strategy_phase: str,
    ) -> int:
        """Calculate position size using ML optimizer.

        Args:
            underlying_price: Current stock price
            iv: Implied volatility
            strike: Option strike
            premium: Option premium
            dte: Days to expiry
            delta: Option delta
            position_mgr: Position manager
            strategy_phase: "SP" or "CC"

        Returns:
            Recommended number of contracts
        """
        # Base position from position manager
        if position_mgr:
            base_position = position_mgr.calculate_position_size(
                margin_per_contract=strike * 100 if strategy_phase == "SP" else 0,
                max_positions=self.params.get("max_positions", 10),
            )
        else:
            base_position = self.params.get("max_positions", 1)

        if base_position <= 0:
            return 0

        # If ML optimization disabled, return base position
        if not self.ml_position_optimization or self.ml_position_optimizer is None:
            return base_position

        try:
            # Build market data for ML
            market_data = {
                'price': underlying_price,
                'iv': iv,
                'iv_rank': getattr(self, '_current_iv_rank', 50),
                'iv_percentile': getattr(self, '_current_iv_percentile', 50),
                'historical_volatility': iv * 100,
                'vix': getattr(self, '_current_vix', 20),
                'momentum': getattr(self, '_current_momentum', {}),
            }

            # Build portfolio state
            portfolio_state = {
                'total_capital': self.initial_capital,
                'available_margin': position_mgr.available_margin if position_mgr else self.initial_capital,
                'margin_used': position_mgr.total_margin_used if position_mgr else 0,
                'drawdown': self.performance_metrics.max_drawdown,
                'positions': [],  # Would need to track open positions
                'cost_basis': self.stock_holding.cost_basis,
            }

            # Build option info
            option_info = {
                'underlying_price': underlying_price,
                'strike': strike,
                'premium': premium,
                'dte': dte,
                'delta': delta,
            }

            # Get ML recommendation
            max_position = self.params.get("max_positions", 10)
            if strategy_phase == "CC":
                # For CC, limit by shares held
                max_position = min(max_position, self.stock_holding.shares // 100)

            num_contracts, recommendation = self.ml_position_optimizer.get_position_size(
                symbol=self.params.get("symbol", "UNKNOWN"),
                market_data=market_data,
                portfolio_state=portfolio_state,
                strategy_phase=strategy_phase,
                option_info=option_info,
                base_position=base_position,
                max_position=max_position,
            )

            # Log recommendation for analysis
            if recommendation:
                self.position_recommendations.append({
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'phase': strategy_phase,
                    'contracts': num_contracts,
                    'multiplier': recommendation.position_multiplier,
                    'confidence': recommendation.confidence,
                    'reasoning': recommendation.reasoning,
                })

                self.logger.info(
                    f"ML Position: {num_contracts} contracts ({recommendation.position_multiplier:.2f}x, "
                    f"confidence: {recommendation.confidence:.0%}) - {recommendation.reasoning}"
                )

            return num_contracts

        except Exception as e:
            self.logger.warning(f"ML position optimization failed, using base: {e}")
            return base_position

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
                # Ensure shares acquired is a multiple of 100 to maintain proper lot sizes
                shares_acquired = quantity * 100  # This should already be multiple of 100, but ensure it
                total_stock_cost = self.stock_holding.shares * self.stock_holding.cost_basis
                total_stock_cost += shares_acquired * strike
                self.stock_holding.shares += shares_acquired
                self.stock_holding.symbol = trade.get("symbol", self.symbol)  # Track which stock we're holding
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
                # Ensure shares_sold is a multiple of 100 to maintain proper lot sizes
                # But also ensure we don't sell more shares than we hold
                actual_shares_sold = min(shares_sold, self.stock_holding.shares)
                            
                # Make sure we're selling in lots of 100 shares (or all remaining shares if less than 100)
                if actual_shares_sold >= 100:
                    # Round down to nearest 100 to maintain lot size
                    actual_shares_sold = (actual_shares_sold // 100) * 100
                # If less than 100 shares held, we'll sell all of them
                            
                if actual_shares_sold != shares_sold:
                    self.logger.info(
                        f"Adjusted call assignment from {shares_sold} to {actual_shares_sold} shares "
                        f"to maintain proper lot size (100-share multiples)."
                    )
                    shares_sold = actual_shares_sold
                            
                stock_cost_basis = self.stock_holding.cost_basis * shares_sold
                stock_proceeds = strike * shares_sold
                stock_pnl = stock_proceeds - stock_cost_basis  # Realized stock P&L
                            
                # Log the complete P&L breakdown for transparency
                # Note: stock_pnl is for logging only; engine handles cumulative_pnl
                option_pnl = trade.get("pnl", 0)
                total_trade_pnl = option_pnl + stock_pnl
                self.logger.info(
                    f"Call assigned: Option P&L=${option_pnl:+.2f}, Stock P&L=${stock_pnl:+.2f}, "
                    f"Total=${total_trade_pnl:+.2f} (bought at ${self.stock_holding.cost_basis:.2f}, "
                    f"sold at ${strike:.2f}, {shares_sold} shares)"
                )
                
                # IMPORTANT: Return 0 because engine already adds stock_proceeds to cumulative_pnl
                # The cash flow is handled in engine.py, not here
                additional_stock_pnl = 0
                            
                # Reduce stock holding
                self.stock_holding.shares = max(0, self.stock_holding.shares - shares_sold)
                
                # Add premium from call sale (already included in option P&L)
                premium_from_call = trade["entry_price"] * shares_sold
                self.stock_holding.total_premium_collected += premium_from_call
                
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
        
        # CRITICAL FIX: In CC phase, we should be more flexible about max_pos
        # because we can have multiple covered calls as long as we have shares to back them
        if self.phase == "CC":
            # For CC phase, check how many calls we can potentially sell based on shares held
            max_calls_by_shares = self.stock_holding.shares // 100
            max_calls_allowed = min(max_pos, max_calls_by_shares)
            
            # Count already sold Call contracts (negative quantity means sold)
            existing_call_contracts = sum(
                abs(p.quantity) for p in wheel_positions 
                if p.trade_type == "WHEEL_CALL"
            )
            
            # If we can still sell more calls based on shares we have, generate signal
            if existing_call_contracts < max_calls_allowed:
                # We have capacity to sell more covered calls
                pass  # Continue to generate signal
            else:
                # No more calls can be sold (either hit max_pos or ran out of shares)
                return []
        else:  # SP phase
            # Original logic for SP phase: respect max_pos strictly
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
            
            # Ensure we have shares in lots of 100 for proper covered calls
            # If fractional shares exist, log them and proceed with available whole lots
            if self.stock_holding.shares % 100 != 0:
                fractional_shares = self.stock_holding.shares % 100
                whole_share_lots = self.stock_holding.shares - fractional_shares
                self.logger.info(
                    f"Detected fractional shares ({fractional_shares}). "
                    f"Will use {whole_share_lots} shares in {whole_share_lots // 100} lots for covered calls."
                )

        # Get optimized DTE (ML or traditional)
        dte_days = self.select_expiry_dte(underlying_price=underlying_price, iv=iv, right="P" if self.phase == "SP" else "C", cost_basis=self.stock_holding.cost_basis)
        T = dte_days / 365.0
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=int(dte_days))
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
        # Use put-specific delta with ML optimization if enabled
        original_delta = self.delta_target
        optimized_delta = self.get_optimized_delta(underlying_price, iv, "P")
        self.delta_target = optimized_delta
        strike = self.select_strike(underlying_price, iv, T, "P")
        self.delta_target = original_delta

        premium = OptionsPricer.put_price(underlying_price, strike, T, iv)
        delta = OptionsPricer.delta(underlying_price, strike, T, iv, "P")

        # Calculate DTE from T
        dte = int(T * 365)

        # Position sizing: use ML optimizer if enabled, otherwise traditional
        if self.ml_position_optimization and self.ml_position_optimizer:
            max_contracts = self._calculate_ml_position_size(
                underlying_price=underlying_price,
                iv=iv,
                strike=strike,
                premium=premium,
                dte=dte,
                delta=delta,
                position_mgr=position_mgr,
                strategy_phase="SP",
            )
        else:
            # Traditional position sizing using position manager
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

        # Use call-specific delta with ML optimization if enabled
        original_delta = self.delta_target
        optimized_delta = self.get_optimized_delta(underlying_price, iv, "C", self.stock_holding.cost_basis)
        self.delta_target = optimized_delta
        
        try:
            strike = self.select_strike(underlying_price, iv, T, "C")
        except Exception as e:
            # If we can't find a suitable strike, try a more flexible approach
            self.logger.warning(f"Could not select strike with target delta {self.call_delta}: {e}")
            # Try to find a reasonable strike even if it doesn't match the target delta perfectly
            strike = self._find_alternative_call_strike(underlying_price)
        
        self.delta_target = original_delta

        # Calculate premium and delta for the selected strike
        from core.backtesting.pricing import OptionsPricer
        premium = OptionsPricer.call_price(underlying_price, strike, T, iv)
        delta = OptionsPricer.delta(underlying_price, strike, T, iv, "C")

        # Calculate DTE from T
        dte = int(T * 365)

        # Position sizing: use ML optimizer if enabled, otherwise traditional
        if self.ml_position_optimization and self.ml_position_optimizer:
            max_contracts = self._calculate_ml_position_size(
                underlying_price=underlying_price,
                iv=iv,
                strike=strike,
                premium=premium,
                dte=dte,
                delta=delta,
                position_mgr=position_mgr,
                strategy_phase="CC",
            )
            # For CC, ensure we don't exceed shares held
            max_by_shares = self.stock_holding.shares // 100
            max_contracts = min(max_contracts, max_by_shares)
        else:
            # Traditional: Covered call: 1 contract per 100 shares owned
            # Calculate contracts based on shares held (same as CoveredCallStrategy)
            # generate_signals() already checks for existing call positions to prevent over-selling
            max_by_shares = self.stock_holding.shares // 100
            max_contracts = min(max_by_shares, self.params.get("max_positions", 10))

        # If we have shares but not enough to sell another call (less than 100 shares),
        # we should handle the fractional shares appropriately
        if max_contracts <= 0:
            # If we have shares but less than 100, we can't sell covered calls
            # If we have fractional shares (<100), we should liquidate them to avoid getting stuck
            if 0 < self.stock_holding.shares < 100:
                self.logger.info(
                    f"Liquidating fractional shares ({self.stock_holding.shares}) to avoid getting stuck in CC phase."
                )
                # Liquidate fractional shares by switching to SP phase
                self.phase = "SP"
                return self._generate_sell_put_signal(underlying_price, iv, T, expiry_str, position_mgr)
            else:
                return []

        # Double-check that we're using a valid number of contracts based on available shares
        available_share_lots = self.stock_holding.shares // 100
        max_contracts = min(max_contracts, available_share_lots)
        
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
    
    def _find_alternative_call_strike(self, underlying_price: float) -> float:
        """Find an alternative call strike when target delta cannot be achieved."""
        # If we can't find a strike with the target delta, try a more practical approach
        # For covered calls, we often want OTM strikes, but let's be more flexible
        # Try strikes at various percentages of underlying price
        
        # Try various percentage levels above the current price for OTM calls
        for factor in [1.05, 1.10, 1.15, 1.20, 1.25]:
            trial_strike = underlying_price * factor
            # Round to nearest dollar or half-dollar
            if underlying_price > 100:
                trial_strike = round(trial_strike)
            else:
                trial_strike = round(trial_strike * 2) / 2
            
            # If the strike is reasonable, return it
            if trial_strike > underlying_price * 0.9:  # Reasonable range
                return trial_strike
        
        # If all else fails, use a strike slightly above current price
        fallback_strike = underlying_price * 1.05
        if underlying_price > 100:
            fallback_strike = round(fallback_strike)
        else:
            fallback_strike = round(fallback_strike * 2) / 2
        
        return fallback_strike

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
        self.logger.info(f"Phase transition: {from_phase} -> {to_phase} ({reason})")
    
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
        report = {
            "strategy": "wheel",
            "run_id": self._current_run_id,
            "current_state": self.get_state_summary(),
            "performance_metrics": self.performance_metrics.to_dict(),
            "trade_history": self.trade_history[-10:],  # Last 10 trades
            "phase_history": self.phase_history[-5:],   # Last 5 transitions
            "daily_stats": self.daily_stats[-30:],      # Last 30 days
        }

        # Add ML position optimization info if enabled
        if self.ml_position_optimization:
            report["ml_position_optimization"] = {
                "enabled": True,
                "recommendations": self.position_recommendations[-20:],  # Last 20 recommendations
                "total_recommendations": len(self.position_recommendations),
            }
        else:
            report["ml_position_optimization"] = {
                "enabled": False,
            }

        return report
    
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
