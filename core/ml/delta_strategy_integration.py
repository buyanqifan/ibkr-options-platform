"""
Integration of ML Delta Optimizer with Binbin God Strategy.

This module provides the bridge between the ML Delta optimizer and the 
options strategy implementation, replacing static delta values with 
intelligent, dynamic optimization.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.ml.delta_optimizer import DeltaOptimizerML, DeltaOptimizationConfig, MarketContext, OptimizationResult
from core.backtesting.pricing import OptionsPricer

logger = logging.getLogger("delta_integration")


class BinGodDeltaIntegration:
    """
    Integration class for ML Delta optimization in Binbin God strategy.
    
    This class wraps the ML optimizer and provides methods that are compatible
    with the existing Binbin God strategy interface.
    """
    
    def __init__(self, 
                 ml_optimization_enabled: bool = True,
                 fallback_delta: float = 0.30,
                 config: Optional[DeltaOptimizationConfig] = None):
        
        self.ml_optimization_enabled = ml_optimization_enabled
        self.fallback_delta = fallback_delta
        
        # Initialize ML optimizer if enabled
        if ml_optimization_enabled:
            self.ml_config = config or DeltaOptimizationConfig()
            self.optimizer = DeltaOptimizerML(self.ml_config)
            logger.info("ML Delta optimizer initialized")
        else:
            self.optimizer = None
            logger.info("ML Delta optimization disabled, using static delta")
    
    def optimize_put_delta(self,
                          symbol: str,
                          current_price: float,
                          cost_basis: float,
                          bars: List[Dict],
                          options_data: List[Dict],
                          fundamentals: Dict = None,
                          iv: float = 0.25,
                          time_to_expiry: float = 30/365.0) -> OptimizationResult:
        """
        Optimize put delta selection using ML models.
        """
        
        if not self.ml_optimization_enabled or not self.optimizer:
            # Return fallback result
            return OptimizationResult(
                optimal_delta=self.fallback_delta,
                expected_premium=self._estimate_premium(self.fallback_delta, current_price, "P", iv, time_to_expiry),
                expected_probability_assignment=self.fallback_delta * 0.8,  # Estimate
                risk_score=self.fallback_delta,
                confidence=1.0,
                reasoning="Static delta mode (ML disabled)"
            )
        
        try:
            # Extract market context
            market_context = self.optimizer.extract_market_context(
                symbol, current_price, cost_basis, bars, options_data, fundamentals
            )
            
            # Optimize put delta
            result = self.optimizer.optimize_delta(
                market_context=market_context,
                right="P",
                iv=iv,
                time_to_expiry=time_to_expiry
            )
            
            logger.info(f"ML optimized put delta: {result.optimal_delta:.3f} "
                       f"(confidence: {result.confidence:.2f}) - {result.reasoning}")
            
            return result
            
        except Exception as e:
            logger.error(f"ML put optimization failed: {e}, falling back to static delta")
            
            return OptimizationResult(
                optimal_delta=self.fallback_delta,
                expected_premium=self._estimate_premium(self.fallback_delta, current_price, "P", iv, time_to_expiry),
                expected_probability_assignment=self.fallback_delta * 0.8,
                risk_score=self.fallback_delta,
                confidence=0.5,
                reasoning=f"ML optimization failed, fallback to delta {self.fallback_delta:.2f}"
            )
    
    def optimize_call_delta(self,
                           symbol: str,
                           current_price: float,
                           cost_basis: float,
                           bars: List[Dict],
                           options_data: List[Dict],
                           fundamentals: Dict = None,
                           iv: float = 0.25,
                           time_to_expiry: float = 30/365.0) -> OptimizationResult:
        """
        Optimize call delta selection using ML models with special handling for CC optimization.
        """
        
        if not self.ml_optimization_enabled or not self.optimizer:
            # Return fallback result
            return OptimizationResult(
                optimal_delta=self.fallback_delta,
                expected_premium=self._estimate_premium(self.fallback_delta, current_price, "C", iv, time_to_expiry),
                expected_probability_assignment=self.fallback_delta * 0.6,  # Lower for calls
                risk_score=self.fallback_delta * 0.7,
                confidence=1.0,
                reasoning="Static delta mode (ML disabled)"
            )
        
        try:
            # Extract market context
            market_context = self.optimizer.extract_market_context(
                symbol, current_price, cost_basis, bars, options_data, fundamentals
            )
            
            # Apply CC optimization if cost basis > current price
            if cost_basis > 0 and current_price < cost_basis:
                price_ratio = current_price / cost_basis
                
                # Enhanced context for loss position
                logger.info(f"CC optimization: Loss detected ({price_ratio:.2f}), adjusting market context")
                market_context.market_regime = "protective"
                market_context.cost_basis = cost_basis  # Emphasize cost basis
            
            # Optimize call delta
            result = self.optimizer.optimize_delta(
                market_context=market_context,
                right="C",
                iv=iv,
                time_to_expiry=time_to_expiry
            )
            
            logger.info(f"ML optimized call delta: {result.optimal_delta:.3f} "
                       f"(confidence: {result.confidence:.2f}) - {result.reasoning}")
            
            return result
            
        except Exception as e:
            logger.error(f"ML call optimization failed: {e}, falling back to static delta")
            
            return OptimizationResult(
                optimal_delta=self.fallback_delta,
                expected_premium=self._estimate_premium(self.fallback_delta, current_price, "C", iv, time_to_expiry),
                expected_probability_assignment=self.fallback_delta * 0.6,
                risk_score=self.fallback_delta * 0.7,
                confidence=0.5,
                reasoning=f"ML optimization failed, fallback to delta {self.fallback_delta:.2f}"
            )
    
    def _estimate_premium(self, delta: float, price: float, right: str, iv: float, time_to_expiry: float) -> float:
        """Estimate premium for given delta."""
        if right == "P":
            strike = price * (1 - delta)
            premium = OptionsPricer.put_price(price, strike, time_to_expiry, iv)
        else:
            strike = price * (1 + delta)
            premium = OptionsPricer.call_price(price, strike, time_to_expiry, iv)
        
        return premium
    
    def update_performance(self, 
                          delta: float,
                          symbol: str,
                          current_price: float,
                          cost_basis: float,
                          bars: List[Dict],
                          options_data: List[Dict],
                          actual_pnl: float,
                          actual_assignment: bool):
        """Update ML optimizer with actual performance data."""
        
        if self.ml_optimization_enabled and self.optimizer:
            try:
                market_context = self.optimizer.extract_market_context(
                    symbol, current_price, cost_basis, bars, options_data
                )
                
                self.optimizer.update_performance(
                    delta=delta,
                    symbol=symbol,
                    context=market_context,
                    actual_pnl=actual_pnl,
                    actual_assignment=actual_assignment
                )
                
                logger.info(f"Updated ML performance: delta={delta}, pnl={actual_pnl:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to update ML performance: {e}")
    
    def should_retrain(self) -> bool:
        """Check if ML model should be retrained."""
        if self.ml_optimization_enabled and self.optimizer:
            return self.optimizer.should_retrain()
        return False
    
    def save_model(self):
        """Save ML model to disk."""
        if self.ml_optimization_enabled and self.optimizer:
            self.optimizer.save_model()
    
    def get_insights(self) -> Dict[str, Any]:
        """Get ML optimization insights."""
        if self.ml_optimization_enabled and self.optimizer:
            return self.optimizer.get_optimization_insights()
        else:
            return {"status": "ML optimization disabled"}


class AdaptiveDeltaStrategy:
    """
    Adaptive strategy that combines traditional delta selection with ML optimization.
    
    This provides a smooth transition path and allows comparison between
    static and ML-optimized delta selection.
    """
    
    def __init__(self, 
                 ml_integration: BinGodDeltaIntegration,
                 adoption_rate: float = 0.5):
        """
        Initialize adaptive strategy.
        
        Args:
            ml_integration: ML integration instance
            adoption_rate: How much to trust ML vs traditional (0.0 = traditional only, 1.0 = ML only)
        """
        self.ml_integration = ml_integration
        self.adoption_rate = adoption_rate
        self.performance_comparison = {
            'traditional': [],
            'ml_optimized': []
        }
    
    def select_put_delta(self,
                        traditional_delta: float,
                        ml_result: OptimizationResult) -> float:
        """
        Select put delta using adaptive combination of traditional and ML methods.
        """
        if ml_result.confidence < 0.7:  # Low confidence ML results
            logger.warning(f"Low ML confidence ({ml_result.confidence:.2f}), using traditional delta")
            return traditional_delta
        
        # Weighted combination
        adaptive_delta = (
            traditional_delta * (1 - self.adoption_rate) +
            ml_result.optimal_delta * self.adoption_rate
        )
        
        logger.info(f"Adaptive put delta: traditional={traditional_delta:.3f}, "
                   f"ml={ml_result.optimal_delta:.3f}, combined={adaptive_delta:.3f}")
        
        return adaptive_delta
    
    def select_call_delta(self,
                         traditional_delta: float,
                         ml_result: OptimizationResult) -> float:
        """
        Select call delta using adaptive combination of traditional and ML methods.
        """
        if ml_result.confidence < 0.7:  # Low confidence ML results
            logger.warning(f"Low ML confidence ({ml_result.confidence:.2f}), using traditional delta")
            return traditional_delta
        
        # Weighted combination with additional factors for CC optimization
        if "protective" in ml_result.reasoning:
            # In protective mode, trust ML more
            enhanced_adoption_rate = min(1.0, self.adoption_rate * 1.5)
        else:
            enhanced_adoption_rate = self.adoption_rate
        
        adaptive_delta = (
            traditional_delta * (1 - enhanced_adoption_rate) +
            ml_result.optimal_delta * enhanced_adoption_rate
        )
        
        logger.info(f"Adaptive call delta: traditional={traditional_delta:.3f}, "
                   f"ml={ml_result.optimal_delta:.3f}, combined={adaptive_delta:.3f}")
        
        return adaptive_delta
    
    def record_performance(self, 
                         method: str,  # "traditional" or "ml_optimized"
                         delta: float,
                         pnl: float):
        """Record performance for method comparison."""
        self.performance_comparison[method].append({
            'delta': delta,
            'pnl': pnl,
            'timestamp': datetime.now()
        })
        
        # Keep only recent performance
        if len(self.performance_comparison[method]) > 100:
            self.performance_comparison[method] = self.performance_comparison[method][-100:]
    
    def get_method_comparison(self) -> Dict[str, Any]:
        """Compare performance between traditional and ML methods."""
        comparison = {}
        
        for method in ['traditional', 'ml_optimized']:
            if self.performance_comparison[method]:
                recent = self.performance_comparison[method][-50:]  # Last 50 trades
                avg_pnl = np.mean([p['pnl'] for p in recent])
                win_rate = sum(1 for p in recent if p['pnl'] > 0) / len(recent)
                
                comparison[method] = {
                    'average_pnl': avg_pnl,
                    'win_rate': win_rate,
                    'trades': len(recent),
                    'last_trade': recent[-1]['timestamp'] if recent else None
                }
        
        return comparison