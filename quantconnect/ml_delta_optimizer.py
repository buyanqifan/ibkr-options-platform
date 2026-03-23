"""
ML Delta Optimizer for QuantConnect
====================================

Machine learning model that optimizes delta selection based on:
- Market regime (bull/bear/neutral/high_vol)
- Volatility features (IV rank, historical vol)
- Technical indicators (RSI, momentum, MA position)
- Fundamental factors (PE ratio, earnings)

Uses Q-learning for reinforcement learning optimization.
"""

from AlgorithmImports import *
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np


@dataclass
class DeltaOptimizationConfig:
    """Configuration for ML Delta optimization."""
    model_path: str = "models/delta_optimizer.pkl"
    retrain_interval_days: int = 30
    min_training_samples: int = 100
    exploration_rate: float = 0.1
    learning_rate: float = 0.01
    
    # Delta bounds
    put_delta_min: float = 0.20
    put_delta_max: float = 0.40
    call_delta_min: float = 0.20
    call_delta_max: float = 0.40


@dataclass
class MarketContext:
    """Market environment features for delta optimization."""
    symbol: str
    current_price: float
    cost_basis: float
    volatility_20d: float
    volatility_30d: float
    momentum_5d: float
    momentum_20d: float
    pe_ratio: float
    iv_rank: float
    market_regime: str  # "bull", "bear", "neutral", "high_vol"
    rsi_14: float
    price_vs_ma20: float
    price_vs_ma50: float
    volume_ratio: float
    days_to_earnings: int


@dataclass
class OptimizationResult:
    """Result of delta optimization."""
    optimal_delta: float
    expected_premium: float
    expected_probability_assignment: float
    risk_score: float
    confidence: float
    reasoning: str


class DeltaOptimizerML:
    """
    Machine Learning Delta Optimizer for Options Strategies.
    
    Uses Q-learning to optimize delta selection based on market conditions.
    """
    
    def __init__(self, config: DeltaOptimizationConfig = None):
        self.config = config or DeltaOptimizationConfig()
        self.q_table = {}
        self.performance_history = []
        self.last_retrain = None
        self.model = {
            'delta_performance': {},
            'market_regime_patterns': {},
            'symbol_specific': {}
        }
        
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def extract_market_context(
        self,
        symbol: str,
        current_price: float,
        cost_basis: float,
        bars: List[Dict],
        options_data: List[Dict] = None,
        fundamentals: Dict = None
    ) -> MarketContext:
        """Extract market context features from data."""
        
        if len(bars) < 20:
            return self._create_default_context(symbol, current_price, cost_basis)
        
        # Calculate volatility
        closes = np.array([b['close'] for b in bars[-30:]])
        returns = np.diff(closes) / closes[:-1]
        volatility_20d = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.25
        volatility_30d = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.25
        
        # Calculate momentum
        momentum_5d = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        momentum_20d = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0
        
        # Calculate RSI
        rsi_14 = self._calculate_rsi(closes[-15:]) if len(closes) >= 15 else 50
        
        # MA position
        ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
        ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_price
        price_vs_ma20 = (current_price - ma20) / ma20
        price_vs_ma50 = (current_price - ma50) / ma50
        
        # Volume ratio
        volumes = [b.get('volume', 1) for b in bars[-10:]]
        avg_volume = np.mean(volumes) if volumes else 1
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
        
        # Market regime
        market_regime = self._determine_market_regime(
            volatility_30d, momentum_5d, momentum_20d, rsi_14
        )
        
        # Get fundamentals
        pe_ratio = fundamentals.get('pe_ratio', 25.0) if fundamentals else 25.0
        iv_rank = 50.0
        
        if options_data:
            ivs = [opt.get('iv', 0.25) for opt in options_data if 'iv' in opt]
            if ivs:
                avg_iv = np.mean(ivs)
                # Approximate IV rank from IV level
                iv_rank = min(100, max(0, (avg_iv - 0.15) / 0.35 * 100))
        
        return MarketContext(
            symbol=symbol,
            current_price=current_price,
            cost_basis=cost_basis,
            volatility_20d=volatility_20d,
            volatility_30d=volatility_30d,
            momentum_5d=momentum_5d,
            momentum_20d=momentum_20d,
            pe_ratio=pe_ratio,
            iv_rank=iv_rank,
            market_regime=market_regime,
            rsi_14=rsi_14,
            price_vs_ma20=price_vs_ma20,
            price_vs_ma50=price_vs_ma50,
            volume_ratio=volume_ratio,
            days_to_earnings=30  # Approximate
        )
    
    def _create_default_context(self, symbol: str, current_price: float, cost_basis: float) -> MarketContext:
        """Create default context when insufficient data."""
        return MarketContext(
            symbol=symbol,
            current_price=current_price,
            cost_basis=cost_basis,
            volatility_20d=0.25,
            volatility_30d=0.25,
            momentum_5d=0.0,
            momentum_20d=0.0,
            pe_ratio=25.0,
            iv_rank=50.0,
            market_regime="neutral",
            rsi_14=50.0,
            price_vs_ma20=0.0,
            price_vs_ma50=0.0,
            volume_ratio=1.0,
            days_to_earnings=30
        )
    
    def _calculate_rsi(self, closes: np.ndarray) -> float:
        """Calculate RSI from closing prices."""
        if len(closes) < 14:
            return 50.0
        
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _determine_market_regime(
        self,
        volatility: float,
        momentum_5d: float,
        momentum_20d: float,
        rsi: float
    ) -> str:
        """Determine current market regime."""
        
        if volatility > 0.35:
            return "high_vol"
        elif momentum_5d > 0.03 and momentum_20d > 0.05 and rsi > 55:
            return "bull"
        elif momentum_5d < -0.03 and momentum_20d < -0.05 and rsi < 45:
            return "bear"
        else:
            return "neutral"
    
    def optimize_delta(
        self,
        market_context: MarketContext,
        right: str,  # "P" or "C"
        iv: float,
        time_to_expiry: float
    ) -> OptimizationResult:
        """
        Optimize delta selection using ML model.
        
        Args:
            market_context: Market environment features
            right: Option type ("P" for put, "C" for call)
            iv: Implied volatility
            time_to_expiry: Time to expiry in years
            
        Returns:
            OptimizationResult with optimal delta
        """
        
        # Get state key for Q-table lookup
        state_key = self._get_state_key(market_context, right)
        
        # Check Q-table for learned values
        if state_key in self.q_table:
            learned_delta = self._get_best_delta_from_qtable(state_key)
            confidence = 0.85
            reasoning = f"ML optimized delta based on {market_context.market_regime} regime"
        else:
            learned_delta = None
            confidence = 0.6
            reasoning = f"Rule-based delta for {market_context.market_regime} regime"
        
        # Calculate optimal delta based on market conditions
        if learned_delta is None:
            optimal_delta = self._calculate_rule_based_delta(market_context, right)
        else:
            # Blend learned delta with rule-based for stability
            rule_delta = self._calculate_rule_based_delta(market_context, right)
            optimal_delta = learned_delta * 0.7 + rule_delta * 0.3
        
        # Clamp to bounds
        if right == "P":
            optimal_delta = max(self.config.put_delta_min, 
                              min(self.config.put_delta_max, optimal_delta))
        else:
            optimal_delta = max(self.config.call_delta_min,
                              min(self.config.call_delta_max, optimal_delta))
        
        # Calculate expected metrics
        expected_premium = self._estimate_premium(
            optimal_delta, market_context.current_price, right, iv, time_to_expiry
        )
        
        expected_prob_assignment = optimal_delta * 0.8  # Approximation
        
        risk_score = self._calculate_risk_score(
            optimal_delta, market_context, right
        )
        
        return OptimizationResult(
            optimal_delta=optimal_delta,
            expected_premium=expected_premium,
            expected_probability_assignment=expected_prob_assignment,
            risk_score=risk_score,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _get_state_key(self, context: MarketContext, right: str) -> str:
        """Generate state key for Q-table."""
        return f"{context.symbol}_{context.market_regime}_{right}_{int(context.iv_rank/20)*20}"
    
    def _get_best_delta_from_qtable(self, state_key: str) -> Optional[float]:
        """Get best delta from Q-table."""
        if state_key not in self.q_table:
            return None
        
        actions = self.q_table[state_key]
        if not actions:
            return None
        
        best_delta_key = max(actions.items(), key=lambda x: x[1])[0]
        # Extract numeric value from key like "delta_0.30"
        try:
            return float(best_delta_key.replace("delta_", ""))
        except (ValueError, AttributeError):
            return None
    
    def _calculate_rule_based_delta(self, context: MarketContext, right: str) -> float:
        """Calculate delta using rule-based approach."""
        
        base_delta = 0.30
        
        # Adjust for market regime
        regime_adjustments = {
            "bull": {"P": -0.03, "C": 0.03},  # More conservative puts, aggressive calls
            "bear": {"P": 0.03, "C": -0.03},  # More aggressive puts, conservative calls
            "high_vol": {"P": -0.05, "C": -0.05},  # More conservative in high vol
            "neutral": {"P": 0.0, "C": 0.0}
        }
        
        regime_adj = regime_adjustments.get(context.market_regime, {"P": 0, "C": 0})
        delta = base_delta + regime_adj.get(right, 0)
        
        # Adjust for IV rank
        if context.iv_rank > 70:
            delta -= 0.03  # More conservative when IV is high
        elif context.iv_rank < 30:
            delta += 0.02  # Slightly more aggressive when IV is low
        
        # Adjust for momentum
        if right == "P":
            if context.momentum_20d > 0.10:
                delta += 0.02  # Can be more aggressive in uptrend
            elif context.momentum_20d < -0.10:
                delta -= 0.03  # More conservative in downtrend
        else:
            if context.momentum_20d > 0.10:
                delta -= 0.02  # More conservative CC in strong uptrend
            elif context.momentum_20d < -0.10:
                delta += 0.02  # Can be more aggressive CC in downtrend
        
        # Adjust for RSI
        if right == "P":
            if context.rsi_14 > 70:
                delta += 0.02  # Overbought - can be more aggressive
            elif context.rsi_14 < 30:
                delta -= 0.03  # Oversold - more conservative
        else:
            if context.rsi_14 > 70:
                delta -= 0.03  # Overbought - more conservative CC
            elif context.rsi_14 < 30:
                delta += 0.02  # Oversold - can be more aggressive CC
        
        return delta
    
    def _estimate_premium(
        self,
        delta: float,
        price: float,
        right: str,
        iv: float,
        T: float
    ) -> float:
        """Estimate option premium using simplified approximation.
        
        Note: In QC, we should use actual option chain prices.
        This provides a rough estimate for ML optimization only.
        """
        
        # Approximate strike from delta
        if right == "P":
            strike = price * (1 + delta * iv * np.sqrt(T))
            moneyness = strike / price
            # OTM put premium approximation
            premium = price * delta * iv * np.sqrt(T) * (1 - moneyness * 0.5)
        else:
            strike = price * (1 - delta * iv * np.sqrt(T))
            moneyness = strike / price
            # OTM call premium approximation
            premium = price * delta * iv * np.sqrt(T) * (moneyness * 0.5 - 0.5)
        
        return max(premium, 0.01)  # Minimum premium
    
    def _calculate_risk_score(
        self,
        delta: float,
        context: MarketContext,
        right: str
    ) -> float:
        """Calculate risk score for delta selection."""
        
        # Base risk from delta
        risk = delta
        
        # Adjust for market regime
        if context.market_regime == "high_vol":
            risk *= 1.3
        elif context.market_regime == "bear" and right == "P":
            risk *= 1.2
        elif context.market_regime == "bull" and right == "C":
            risk *= 1.1
        
        # Adjust for cost basis if we own shares
        if context.cost_basis > 0:
            price_ratio = context.current_price / context.cost_basis
            if price_ratio < 0.95 and right == "C":
                # Underwater - higher risk
                risk *= 1.3
            elif price_ratio > 1.10 and right == "P":
                # Significant gain - can be more aggressive
                risk *= 0.9
        
        return min(1.0, risk)
    
    def update_performance(
        self,
        delta: float,
        symbol: str,
        context: MarketContext,
        actual_pnl: float,
        actual_assignment: bool
    ):
        """Update Q-table with actual performance data."""
        
        state_key = self._get_state_key(context, "P")  # Simplified
        delta_key = f"delta_{delta:.2f}"
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Calculate reward
        reward = actual_pnl
        if actual_assignment and actual_pnl > 0:
            reward *= 1.2  # Bonus for successful assignment
        
        # Q-learning update
        if delta_key in self.q_table[state_key]:
            old_value = self.q_table[state_key][delta_key]
            self.q_table[state_key][delta_key] = old_value + self.config.learning_rate * (
                reward - old_value
            )
        else:
            self.q_table[state_key][delta_key] = reward
        
        # Update performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'delta': delta,
            'symbol': symbol,
            'pnl': actual_pnl,
            'assignment': actual_assignment,
            'regime': context.market_regime
        })
        
        # Limit history size
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if self.last_retrain is None:
            return True
        
        if len(self.performance_history) >= self.config.min_training_samples:
            return True
        
        return False
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights about optimization performance."""
        
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        recent = self.performance_history[-100:]
        
        avg_pnl = np.mean([p['pnl'] for p in recent])
        win_rate = sum(1 for p in recent if p['pnl'] > 0) / len(recent)
        
        # Best performing deltas
        delta_performance = {}
        for p in recent:
            delta = p['delta']
            if delta not in delta_performance:
                delta_performance[delta] = []
            delta_performance[delta].append(p['pnl'])
        
        best_deltas = sorted(
            [(d, np.mean(perfs)) for d, perfs in delta_performance.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "average_pnl": avg_pnl,
            "win_rate": win_rate,
            "total_trades": len(self.performance_history),
            "best_performing_deltas": best_deltas,
            "q_table_size": len(self.q_table)
        }