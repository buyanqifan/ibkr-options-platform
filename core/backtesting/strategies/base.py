"""Abstract base class for all options strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import logging


@dataclass
class Signal:
    """Signal to open a new position."""
    symbol: str
    trade_type: str      # SELL_PUT, COVERED_CALL, etc.
    right: str           # P or C
    strike: float
    expiry: str          # YYYYMMDD
    quantity: int         # negative for sell
    iv: float
    delta: float
    premium: float       # expected premium per share
    underlying_price: float = 0.0  # Stock price at entry (for tracking in backtest)
    margin_requirement: float = None  # Optional: strategy can provide specific margin requirement


class BaseStrategy(ABC):
    """Abstract base for options strategies."""

    def __init__(self, params: dict):
        self.params = params
        self.dte_min = params.get("dte_min", 21)
        self.dte_max = params.get("dte_max", 45)
        self.delta_target = params.get("delta_target", 0.30)
        self.profit_target_pct = params.get("profit_target_pct", 50)
        self.stop_loss_pct = params.get("stop_loss_pct", 200)
        self.initial_capital = params.get("initial_capital", 100000)
        self.max_risk_per_trade = params.get("max_risk_per_trade", 0.02)  # 2% risk per trade
        # Position management: max_leverage only (position_percentage removed)
        self.max_leverage = params.get("max_leverage", 1.0)  # No leverage by default
        
        # ML Delta optimization parameters
        self.ml_delta_optimization = params.get("ml_delta_optimization", False)
        self.ml_adoption_rate = params.get("ml_adoption_rate", 0.5)
        self.ml_integration = None
        self.logger = logging.getLogger(f"strategy_{self.__class__.__name__}")
        
        # Initialize ML integration if enabled
        if self.ml_delta_optimization:
            try:
                from core.ml.delta_strategy_integration import BinGodDeltaIntegration, AdaptiveDeltaStrategy
                self.ml_integration = BinGodDeltaIntegration(
                    ml_optimization_enabled=True,
                    fallback_delta=self.delta_target,
                    config=params.get("ml_config")
                )
                self.adaptive_strategy = AdaptiveDeltaStrategy(
                    ml_integration=self.ml_integration,
                    adoption_rate=self.ml_adoption_rate
                )
                self.logger.info("ML Delta optimizer initialized for strategy")
            except ImportError as e:
                self.logger.warning(f"ML Delta optimization not available: {e}")
                self.ml_delta_optimization = False
            except Exception as e:
                self.logger.warning(f"ML Delta optimization initialization failed: {e}")
                self.ml_delta_optimization = False
    
    def pretrain_ml_model(self, historical_bars: list, iv_estimate: float = 0.25) -> dict:
        """
        Pretrain ML model with historical data before backtesting.
        
        Should be called by the backtest engine before running the strategy.
        
        Args:
            historical_bars: List of historical price bars
            iv_estimate: Estimated implied volatility
            
        Returns:
            Dict with pretraining statistics
        """
        if not self.ml_delta_optimization or not self.ml_integration:
            return {"status": "skipped", "reason": "ml_not_enabled"}
        
        if not historical_bars or len(historical_bars) < 60:
            return {"status": "skipped", "reason": "insufficient_data"}
        
        try:
            # Access the optimizer from the integration
            optimizer = self.ml_integration.optimizer
            
            if optimizer is None:
                return {"status": "skipped", "reason": "no_optimizer"}
            
            symbol = self.params.get("symbol", "UNKNOWN")
            
            # Pretrain for both puts and calls
            stats_put = optimizer.pretrain_with_history(
                symbol=symbol,
                historical_bars=historical_bars,
                iv_estimate=iv_estimate,
                right="P",
                training_ratio=0.5  # Use 50% of history for pretraining
            )
            
            stats_call = optimizer.pretrain_with_history(
                symbol=symbol,
                historical_bars=historical_bars,
                iv_estimate=iv_estimate,
                right="C",
                training_ratio=0.5
            )
            
            self.logger.info(f"ML model pretrained: Put={stats_put.get('total_simulations', 0)} sims, "
                           f"Call={stats_call.get('total_simulations', 0)} sims")
            
            return {
                "status": "success",
                "put_simulations": stats_put.get("total_simulations", 0),
                "call_simulations": stats_call.get("total_simulations", 0),
                "regimes_tested": stats_put.get("regimes_tested", []),
                "best_delta_by_regime": stats_put.get("best_delta_by_regime", {})
            }
            
        except Exception as e:
            self.logger.error(f"ML pretraining failed: {e}")
            return {"status": "error", "message": str(e)}

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate_signals(
        self,
        current_date: str,
        underlying_price: float,
        iv: float,
        open_positions: list,
        position_mgr: Any = None,  # Optional PositionManager for capital-aware sizing
    ) -> list[Signal]:
        """Generate trading signals for the current date.
        
        Args:
            current_date: Current trading date
            underlying_price: Current underlying price
            iv: Implied volatility
            open_positions: List of currently open positions
            position_mgr: Optional PositionManager instance for margin tracking
            
        Returns:
            List of Signal objects
        """
        ...

    def select_strike(
        self,
        underlying_price: float,
        iv: float,
        T: float,
        right: str,
    ) -> float:
        """Select strike price based on target delta."""
        from core.backtesting.pricing import OptionsPricer
        # Binary search for the strike that gives target delta
        if right == "P":
            target_delta = -abs(self.delta_target)
            # For put options: when S > K (OTM), delta approaches 0
            # When S < K (ITM), delta approaches -1
            # So to get delta near -0.3 (slightly OTM), we want K slightly less than S
            low = underlying_price * 0.9  # e.g. 135 for S=150
            high = underlying_price * 1.05  # e.g. 157.5 for S=150
        else:
            target_delta = abs(self.delta_target)
            # For call options: when S > K (ITM), delta approaches 1
            # When S < K (OTM), delta approaches 0
            # So to get delta near 0.3 (slightly OTM for covered calls), we want K > S
            low = underlying_price * 0.8  # Wider range for better convergence
            high = underlying_price * 1.2

        for _ in range(50):
            mid = (low + high) / 2
            d = OptionsPricer.delta(underlying_price, mid, T, iv, right)
            if right == "P":
                # For puts: increasing strike makes delta more negative
                # So if d < target_delta (too negative), decrease strike (high = mid)
                # If d > target_delta (too positive), increase strike (low = mid)
                if d < target_delta:
                    high = mid
                else:
                    low = mid
            else:
                # For calls: increasing strike makes delta decrease (more negative)
                # So if d > target_delta (too high), increase strike (low = mid)
                # If d < target_delta (too low), decrease strike (high = mid)
                if d > target_delta:
                    low = mid
                else:
                    high = mid
            if abs(d - target_delta) < 0.005:
                break

        # Round to nearest 0.5 or 1.0
        if underlying_price > 100:
            return round(mid)
        return round(mid * 2) / 2

    def get_optimized_delta(self, underlying_price: float, iv: float, right: str, 
                            cost_basis: float = 0.0) -> float:
        """Get delta value, using ML optimization if enabled.
        
        Args:
            underlying_price: Current underlying price
            iv: Implied volatility
            right: Option type ('P' for put, 'C' for call)
            cost_basis: Cost basis for position (relevant for covered calls)
            
        Returns:
            float: Optimized delta value
        """
        if not self.ml_delta_optimization or not self.ml_integration:
            return self.delta_target
        
        try:
            # Use ML integration to get optimized delta
            T = self.select_expiry_dte() / 365.0
            
            if right == "P":
                ml_result = self.ml_integration.optimize_put_delta(
                    symbol=self.params.get("symbol", ""),
                    current_price=underlying_price,
                    cost_basis=cost_basis,
                    bars=[],
                    options_data=[],
                    iv=iv,
                    time_to_expiry=T
                )
            else:
                ml_result = self.ml_integration.optimize_call_delta(
                    symbol=self.params.get("symbol", ""),
                    current_price=underlying_price,
                    cost_basis=cost_basis,
                    bars=[],
                    options_data=[],
                    iv=iv,
                    time_to_expiry=T
                )
            
            # Use adaptive strategy to combine traditional and ML deltas
            if hasattr(self, 'adaptive_strategy') and ml_result:
                final_delta = self.adaptive_strategy.select_put_delta(
                    traditional_delta=self.delta_target,
                    ml_result=ml_result
                ) if right == "P" else self.adaptive_strategy.select_call_delta(
                    traditional_delta=self.delta_target,
                    ml_result=ml_result
                )
                self.logger.info(
                    f"ML optimized {right} delta: traditional={self.delta_target:.3f}, "
                    f"ml={ml_result.optimal_delta:.3f}, final={final_delta:.3f}"
                )
                return final_delta
            elif ml_result and ml_result.confidence > 0.7:
                self.logger.info(f"ML delta (high confidence): {ml_result.optimal_delta:.3f}")
                return ml_result.optimal_delta
                
        except Exception as e:
            self.logger.warning(f"ML delta optimization failed, using traditional: {e}")
        
        return self.delta_target

    def select_expiry_dte(self) -> float:
        """Return target DTE in the middle of the range."""
        return (self.dte_min + self.dte_max) / 2

    def get_performance_report(self) -> dict:
        """Default performance report for strategies that don't override it.
        
        This provides basic structure for UI compatibility, though specific 
        strategies may override this with more detailed information.
        """
        return {
            "strategy": self.name,
            "current_state": {
                "phase": "N/A",  # Default for strategies without phases
                "shares_held": 0,  # Default: no shares held
                "cost_basis": 0.0,  # Default: no cost basis
                "total_premium_collected": 0.0,  # Default: no premium collected
            },
            "performance_metrics": {
                "total_trades": 0,
                "successful_assignments": 0,
                "expired_worthless": 0,
                "profit_target_exits": 0,
                "stop_loss_exits": 0,
                "total_pnl": 0.0,
                "avg_pnl_per_trade": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "phase_transitions": 0,
            },
            # Top-level fields for UI compatibility
            "phase": "N/A",
            "shares_held": 0,
            "cost_basis": 0.0,
            "total_premium_collected": 0.0,
            "open_positions": [],  # Default: no open positions
            "selection_history": [],
            "trade_history": [],
            "phase_history": [],
        }
