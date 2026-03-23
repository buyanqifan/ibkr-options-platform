"""
ML Integration Module for BinbinGod Strategy
=============================================

Integrates all ML modules:
- Delta Optimizer
- DTE Optimizer  
- Roll Optimizer
- Position Optimizer
- Volatility Model

Provides unified interface for the main strategy.
"""

from AlgorithmImports import *
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# Import all ML modules
from ml_delta_optimizer import (
    DeltaOptimizerML, 
    DeltaOptimizationConfig, 
    MarketContext, 
    OptimizationResult
)
from ml_dte_optimizer import (
    DTEOptimizerML,
    DTEOptimizationConfig,
    DTEMarketContext,
    DTEOptimizationResult
)
from ml_roll_optimizer import MLRollOptimizer, RollRecommendation
from ml_position_optimizer import MLPositionOptimizer, PositionRecommendation, WheelPositionIntegration
from ml_volatility_model import VolatilityModel, VolatilityPrediction, VolatilityIntegration


@dataclass
class MLOptimizationConfig:
    """Configuration for all ML optimizations."""
    
    # Delta optimization
    ml_delta_enabled: bool = True
    delta_fallback: float = 0.30
    
    # DTE optimization
    ml_dte_enabled: bool = True
    dte_min: int = 30  # Aligned with original binbin_god.py default
    dte_max: int = 45  # Aligned with original binbin_god.py default
    
    # Roll optimization
    ml_roll_enabled: bool = True
    roll_min_confidence: float = 0.6
    
    # Position optimization
    ml_position_enabled: bool = True
    
    # Volatility model
    vol_lookback: int = 60
    
    # Learning parameters
    exploration_rate: float = 0.1
    learning_rate: float = 0.01
    
    # Model paths
    model_dir: str = "models/"


@dataclass
class StrategySignal:
    """Complete strategy signal with ML optimization."""
    
    # Basic signal
    action: str  # "SELL_PUT", "SELL_CALL", "ROLL", "CLOSE", "HOLD"
    symbol: str
    
    # ML-optimized parameters
    delta: float
    dte_min: int
    dte_max: int
    num_contracts: int
    
    # Expected metrics
    expected_premium: float
    expected_return: float
    expected_risk: float
    assignment_probability: float
    
    # Confidence and reasoning
    confidence: float
    reasoning: str
    
    # Risk metrics
    var_95: float = 0.0
    max_loss: float = 0.0
    
    # ML model insights
    delta_confidence: float = 0.0
    dte_confidence: float = 0.0
    position_confidence: float = 0.0
    
    # Stock scoring adjustment (from stock selection algorithm)
    ml_score_adjustment: float = 0.0  # Range: -0.5 to 0.5
    
    # CC optimization constraints (set when stock price below cost basis)
    min_strike: float = 0.0  # Minimum strike price for CC optimization


class BinbinGodMLIntegration:
    """
    Unified ML integration for BinbinGod strategy.
    
    Coordinates all ML models to provide optimized trading signals.
    """
    
    def __init__(self, config: MLOptimizationConfig = None):
        self.config = config or MLOptimizationConfig()
        
        # Initialize all ML models
        self.delta_optimizer = DeltaOptimizerML(
            DeltaOptimizationConfig(
                exploration_rate=self.config.exploration_rate,
                learning_rate=self.config.learning_rate
            )
        )
        
        self.dte_optimizer = DTEOptimizerML(
            DTEOptimizationConfig(
                dte_min=self.config.dte_min,
                dte_max=self.config.dte_max,
                exploration_rate=self.config.exploration_rate,
                learning_rate=self.config.learning_rate
            )
        )
        
        self.roll_optimizer = MLRollOptimizer()
        self.position_optimizer = MLPositionOptimizer()
        self.volatility_model = VolatilityModel(lookback=self.config.vol_lookback)
        
        # Integration helpers
        self.vol_integration = VolatilityIntegration(self.volatility_model)
        self.position_integration = WheelPositionIntegration(
            self.position_optimizer,
            enabled=self.config.ml_position_enabled
        )
        
        # Performance tracking
        self.trade_history = []
        self.model_performance = {
            'delta': {'correct': 0, 'total': 0},
            'dte': {'correct': 0, 'total': 0},
            'position': {'correct': 0, 'total': 0},
            'roll': {'correct': 0, 'total': 0},
        }
    
    def generate_signal(
        self,
        symbol: str,
        current_price: float,
        cost_basis: float,
        bars: List[Dict],
        strategy_phase: str,
        portfolio_state: Dict[str, Any],
        options_data: List[Dict] = None,
        fundamentals: Dict = None,
        current_position: Dict = None
    ) -> StrategySignal:
        """
        Generate complete trading signal with ML optimization.
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            cost_basis: Cost basis if holding shares
            bars: Historical price bars
            strategy_phase: "SP", "CC", or "CC+SP"
            portfolio_state: Portfolio state dict
            options_data: Available options data
            fundamentals: Fundamental data
            current_position: Current position if any
            
        Returns:
            StrategySignal with all ML-optimized parameters
        """
        
        # 1. Volatility prediction
        vol_prediction = self.volatility_model.predict(bars)
        iv = vol_prediction.iv_estimate
        iv_rank = vol_prediction.iv_rank
        
        # 2. Market context
        delta_context = self.delta_optimizer.extract_market_context(
            symbol, current_price, cost_basis, bars, options_data, fundamentals
        )
        
        dte_context = self.dte_optimizer.extract_market_context(
            symbol, current_price, cost_basis, bars, options_data, fundamentals, strategy_phase
        )
        
        # 3. Check for roll decision if we have a position
        if current_position and self.config.ml_roll_enabled:
            should_roll, roll_rec = self.roll_optimizer.should_roll(
                current_position,
                {'price': current_price, 'iv': iv, 'iv_rank': iv_rank},
                datetime.now().strftime('%Y-%m-%d'),
                self.config.roll_min_confidence
            )
            
            if should_roll:
                return self._create_roll_signal(symbol, roll_rec, current_price, iv, strategy_phase)
        
        # 4. Determine action based on phase
        if strategy_phase == "SP" or strategy_phase == "CC+SP":
            action = "SELL_PUT"
            right = "P"
        else:
            action = "SELL_CALL"
            right = "C"
        
        # 5. Optimize delta
        if self.config.ml_delta_enabled:
            delta_result = self.delta_optimizer.optimize_delta(
                delta_context, right, iv, 30/365.0
            )
            optimal_delta = delta_result.optimal_delta
            delta_confidence = delta_result.confidence
            delta_reasoning = delta_result.reasoning
        else:
            optimal_delta = self.config.delta_fallback
            delta_confidence = 1.0
            delta_reasoning = "Static delta"
        
        # 6. Optimize DTE
        if self.config.ml_dte_enabled:
            dte_result = self.dte_optimizer.optimize_dte_range(
                dte_context, right, iv
            )
            optimal_dte_min = dte_result.optimal_dte_min
            optimal_dte_max = dte_result.optimal_dte_max
            dte_confidence = dte_result.confidence
            dte_reasoning = dte_result.reasoning
        else:
            optimal_dte_min = self.config.dte_min
            optimal_dte_max = self.config.dte_max
            dte_confidence = 1.0
            dte_reasoning = "Static DTE"
        
        # 7. Optimize position size
        avg_dte = (optimal_dte_min + optimal_dte_max) // 2
        option_info = {
            'underlying_price': current_price,
            'strike': current_price * (1 - optimal_delta if right == "P" else 1 + optimal_delta),
            'delta': optimal_delta,
            'premium': self._estimate_premium(current_price, optimal_delta, avg_dte, iv, right),
            'dte': avg_dte
        }
        
        market_data = {
            'price': current_price,
            'iv': iv,
            'iv_rank': iv_rank,
            'vix': 20,  # Would get from market data
        }
        
        num_contracts, position_rec = self.position_integration.get_position_size(
            symbol, market_data, portfolio_state, strategy_phase, option_info
        )
        
        position_confidence = position_rec.confidence if position_rec else 0.5
        
        # 8. Calculate expected metrics
        expected_premium = option_info['premium'] * num_contracts * 100
        assignment_prob = optimal_delta * 0.8
        
        if position_rec:
            expected_return = position_rec.expected_return
            expected_risk = position_rec.expected_risk
            var_95 = position_rec.var_95
            max_loss = position_rec.max_loss_scenario
        else:
            expected_return = expected_premium * (1 - assignment_prob * 0.5)
            expected_risk = option_info['strike'] * optimal_delta * num_contracts * 100
            var_95 = 0
            max_loss = expected_risk
        
        # 9. Overall confidence
        overall_confidence = (
            delta_confidence * 0.35 +
            dte_confidence * 0.30 +
            position_confidence * 0.35
        )
        
        # 10. Combined reasoning
        reasoning_parts = [
            f"Phase: {strategy_phase}",
            f"IV Rank: {iv_rank:.0f}%",
            f"Vol Regime: {vol_prediction.vol_regime}",
            f"Delta: {optimal_delta:.2f} ({delta_reasoning})",
            f"DTE: {optimal_dte_min}-{optimal_dte_max} ({dte_reasoning})",
            f"Position: {num_contracts} contracts"
        ]
        
        return StrategySignal(
            action=action,
            symbol=symbol,
            delta=optimal_delta,
            dte_min=optimal_dte_min,
            dte_max=optimal_dte_max,
            num_contracts=num_contracts,
            expected_premium=expected_premium,
            expected_return=expected_return,
            expected_risk=expected_risk,
            assignment_probability=assignment_prob,
            confidence=overall_confidence,
            reasoning=" | ".join(reasoning_parts),
            var_95=var_95,
            max_loss=max_loss,
            delta_confidence=delta_confidence,
            dte_confidence=dte_confidence,
            position_confidence=position_confidence
        )
    
    def _create_roll_signal(
        self,
        symbol: str,
        roll_rec: RollRecommendation,
        current_price: float,
        iv: float,
        strategy_phase: str
    ) -> StrategySignal:
        """Create signal for roll action."""
        
        action_map = {
            "ROLL_FORWARD": "ROLL",
            "ROLL_OUT": "ROLL",
            "CLOSE_EARLY": "CLOSE",
            "LET_EXPIRE": "HOLD"
        }
        
        return StrategySignal(
            action=action_map.get(roll_rec.action, "HOLD"),
            symbol=symbol,
            delta=roll_rec.optimal_delta or 0.30,
            dte_min=roll_rec.optimal_dte or 30,
            dte_max=roll_rec.optimal_dte or 45,
            num_contracts=1,
            expected_premium=0,
            expected_return=roll_rec.expected_pnl_improvement,
            expected_risk=0,
            assignment_probability=0,
            confidence=roll_rec.confidence,
            reasoning=f"Roll: {roll_rec.action} - {roll_rec.reasoning}"
        )
    
    def _estimate_premium(
        self,
        price: float,
        delta: float,
        dte: int,
        iv: float,
        right: str
    ) -> float:
        """Estimate option premium using simplified approximation.
        
        Note: In QC, we should use actual option chain prices instead of 
        theoretical pricing. This method provides a rough estimate for 
        ML pretraining only.
        """
        T = dte / 365.0
        
        # Simplified premium estimation without Black-Scholes
        # For OTM options, premium ≈ delta * underlying_price * iv * sqrt(T)
        if right == "P":
            strike = price * (1 + delta * iv * np.sqrt(T) * 0.5)
            # OTM put premium approximation
            moneyness = strike / price
            premium = price * delta * iv * np.sqrt(T) * (1 - moneyness * 0.5)
        else:
            strike = price * (1 - delta * iv * np.sqrt(T) * 0.5)
            # OTM call premium approximation
            moneyness = strike / price
            premium = price * delta * iv * np.sqrt(T) * (moneyness * 0.5 - 0.5)
        
        return max(premium, 0.01)  # Minimum premium
    
    def update_performance(
        self,
        trade_result: Dict[str, Any]
    ):
        """
        Update all ML models with trade result.
        
        Args:
            trade_result: Dict containing:
                - symbol: str
                - delta: float
                - dte: int
                - num_contracts: int
                - pnl: float
                - assigned: bool
                - bars: List[Dict]
                - strategy_phase: str
        """
        
        symbol = trade_result['symbol']
        pnl = trade_result['pnl']
        assigned = trade_result['assigned']
        bars = trade_result.get('bars', [])
        strategy_phase = trade_result['strategy_phase']
        
        # Create contexts
        current_price = bars[-1]['close'] if bars else 100
        cost_basis = trade_result.get('cost_basis', current_price)
        
        delta_context = self.delta_optimizer.extract_market_context(
            symbol, current_price, cost_basis, bars
        )
        
        dte_context = self.dte_optimizer.extract_market_context(
            symbol, current_price, cost_basis, bars, None, None, strategy_phase
        )
        
        # Update delta optimizer
        self.delta_optimizer.update_performance(
            delta=trade_result['delta'],
            symbol=symbol,
            context=delta_context,
            actual_pnl=pnl,
            actual_assignment=assigned
        )
        
        # Update DTE optimizer
        self.dte_optimizer.update_performance(
            dte=trade_result['dte'],
            symbol=symbol,
            context=dte_context,
            actual_pnl=pnl,
            actual_assignment=assigned
        )
        
        # Record trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            **trade_result
        })
        
        # Update model performance tracking
        if pnl > 0:
            self.model_performance['delta']['correct'] += 1
            self.model_performance['dte']['correct'] += 1
        self.model_performance['delta']['total'] += 1
        self.model_performance['dte']['total'] += 1
    
    def pretrain_models(
        self,
        symbol: str,
        historical_bars: List[Dict],
        iv_estimate: float = 0.25
    ) -> Dict[str, Any]:
        """
        Pretrain all models with historical data.
        
        Args:
            symbol: Stock symbol
            historical_bars: Historical price bars
            iv_estimate: IV estimate for simulation
            
        Returns:
            Training statistics
        """
        
        stats = {}
        
        # Pretrain DTE optimizer
        dte_stats = self.dte_optimizer.pretrain_with_history(
            symbol, historical_bars, iv_estimate, "P", "SP"
        )
        stats['dte_pretrain'] = dte_stats
        
        return stats
    
    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights from all ML models."""
        
        return {
            'delta': self.delta_optimizer.get_optimization_insights(),
            'dte': self.dte_optimizer.get_optimization_insights(),
            'performance': self.model_performance,
            'total_trades': len(self.trade_history)
        }
    
    def should_retrain(self) -> bool:
        """Check if models need retraining."""
        
        return (
            self.delta_optimizer.should_retrain() or
            self.dte_optimizer.should_retrain()
        )
    
    def get_status_report(self) -> str:
        """Generate human-readable status report."""
        
        insights = self.get_model_insights()
        
        lines = [
            "=" * 50,
            "BINBINGOD ML MODELS STATUS",
            "=" * 50,
            f"Total Trades: {insights['total_trades']}",
            "",
            "--- Delta Optimizer ---",
            f"Q-Table Size: {insights['delta'].get('q_table_size', 0)}",
            f"Best Deltas: {insights['delta'].get('best_performing_deltas', [])}",
            "",
            "--- DTE Optimizer ---",
            f"Average PnL: {insights['dte'].get('average_pnl', 0):.3f}",
            f"Win Rate: {insights['dte'].get('win_rate', 0):.1%}",
            f"Best DTEs: {insights['dte'].get('best_performing_dtes', [])}",
            "",
            "--- Model Performance ---",
            f"Delta Accuracy: {self.model_performance['delta']['correct']}/{self.model_performance['delta']['total']}",
            f"DTE Accuracy: {self.model_performance['dte']['correct']}/{self.model_performance['dte']['total']}",
            "=" * 50
        ]
        
        return "\n".join(lines)


class AdaptiveDeltaStrategy:
    """
    Adaptive strategy that combines traditional delta selection with ML optimization.
    
    This provides a smooth transition path and allows fallback to traditional
    delta when ML confidence is low.
    
    Key features:
    - Low confidence threshold: Falls back to traditional delta
    - Weighted combination: Blends traditional and ML deltas
    - Performance tracking: Compares methods over time
    """
    
    # Minimum confidence threshold for using ML results
    MIN_CONFIDENCE_THRESHOLD = 0.4
    
    def __init__(self, 
                 ml_integration: 'BinbinGodMLIntegration',
                 adoption_rate: float = 0.5,
                 min_confidence: float = 0.4):
        """
        Initialize adaptive strategy.
        
        Args:
            ml_integration: ML integration instance
            adoption_rate: How much to trust ML vs traditional (0.0 = traditional only, 1.0 = ML only)
            min_confidence: Minimum ML confidence to use ML result (below this, use traditional)
        """
        self.ml_integration = ml_integration
        self.adoption_rate = adoption_rate
        self.min_confidence = min_confidence
        self.performance_comparison = {
            'traditional': [],
            'ml_optimized': []
        }
    
    def select_delta(self,
                    traditional_delta: float,
                    ml_delta: float,
                    ml_confidence: float,
                    right: str = "P",
                    reasoning: str = "") -> Tuple[float, str]:
        """
        Select delta using adaptive combination of traditional and ML methods.
        
        Args:
            traditional_delta: Traditional/static delta value
            ml_delta: ML-optimized delta value
            ml_confidence: Confidence of ML prediction
            right: "P" for put, "C" for call
            reasoning: ML reasoning string
            
        Returns:
            Tuple of (selected_delta, explanation)
        """
        # Low confidence: fall back to traditional
        if ml_confidence < self.min_confidence:
            return traditional_delta, f"Low ML confidence ({ml_confidence:.2f}), using traditional delta {traditional_delta:.3f}"
        
        # Protective mode for calls (loss position): trust ML more
        if right == "C" and "protective" in reasoning.lower():
            enhanced_adoption_rate = min(1.0, self.adoption_rate * 1.5)
        else:
            enhanced_adoption_rate = self.adoption_rate
        
        # Weighted combination
        adaptive_delta = (
            traditional_delta * (1 - enhanced_adoption_rate) +
            ml_delta * enhanced_adoption_rate
        )
        
        explanation = (f"Adaptive delta: traditional={traditional_delta:.3f}, "
                      f"ml={ml_delta:.3f}, combined={adaptive_delta:.3f} "
                      f"(adoption={enhanced_adoption_rate:.0%})")
        
        return adaptive_delta, explanation
    
    def select_put_delta(self,
                        traditional_delta: float,
                        ml_delta: float,
                        ml_confidence: float,
                        reasoning: str = "") -> Tuple[float, str]:
        """Select put delta with adaptive strategy."""
        return self.select_delta(traditional_delta, ml_delta, ml_confidence, "P", reasoning)
    
    def select_call_delta(self,
                         traditional_delta: float,
                         ml_delta: float,
                         ml_confidence: float,
                         reasoning: str = "") -> Tuple[float, str]:
        """Select call delta with adaptive strategy."""
        return self.select_delta(traditional_delta, ml_delta, ml_confidence, "C", reasoning)
    
    def record_performance(self, 
                         method: str,  # "traditional" or "ml_optimized"
                         delta: float,
                         pnl: float):
        """Record performance for method comparison."""
        self.performance_comparison[method].append({
            'delta': delta,
            'pnl': pnl
        })
        
        # Keep only recent performance
        if len(self.performance_comparison[method]) > 100:
            self.performance_comparison[method] = self.performance_comparison[method][-100:]
    
    def get_method_comparison(self) -> Dict[str, Any]:
        """Compare performance between traditional and ML methods."""
        comparison = {}
        
        for method in ['traditional', 'ml_optimized']:
            if self.performance_comparison[method]:
                recent = self.performance_comparison[method][-50:]
                avg_pnl = np.mean([p['pnl'] for p in recent])
                win_rate = sum(1 for p in recent if p['pnl'] > 0) / len(recent)
                
                comparison[method] = {
                    'average_pnl': avg_pnl,
                    'win_rate': win_rate,
                    'trades': len(recent)
                }
        
        return comparison