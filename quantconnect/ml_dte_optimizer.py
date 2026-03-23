"""
ML DTE (Days to Expiration) Optimizer for QuantConnect
=======================================================

Machine learning model that optimizes DTE selection using Q-learning.

Supports three strategy phases:
- "SP": Sell Put phase (standard)
- "CC": Covered Call phase (standard)
- "CC+SP": Simultaneous mode (binbingod策略优化)
"""

from AlgorithmImports import *
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np


@dataclass
class DTEOptimizationConfig:
    """Configuration for ML DTE optimization."""
    model_path: str = "models/dte_optimizer.pkl"
    retrain_interval_days: int = 30
    min_training_samples: int = 100
    exploration_rate: float = 0.1
    learning_rate: float = 0.01
    
    # DTE search space
    dte_min: int = 21
    dte_max: int = 60
    dte_step: int = 5


@dataclass
class DTEMarketContext:
    """Market environment features for DTE optimization."""
    symbol: str
    current_price: float
    cost_basis: float
    volatility_20d: float
    volatility_30d: float
    momentum_5d: float
    momentum_20d: float
    pe_ratio: float
    iv_rank: float
    market_regime: str
    days_to_earnings: int
    option_liquidity: float
    strategy_phase: str  # "SP", "CC", or "CC+SP"


@dataclass
class DTEOptimizationResult:
    """Result of DTE optimization."""
    optimal_dte_min: int
    optimal_dte_max: int
    expected_premium: float
    expected_probability_assignment: float
    risk_score: float
    confidence: float
    reasoning: str


class DTEOptimizerML:
    """
    Machine Learning DTE Optimizer for Options Strategies.
    
    Uses Q-learning to optimize DTE selection based on market conditions.
    """
    
    def __init__(self, config: DTEOptimizationConfig = None):
        self.config = config or DTEOptimizationConfig()
        self.q_table = {}
        self.performance_history = []
        self.last_retrain = None
        self.model = {
            'dte_performance': {},
            'market_regime_patterns': {},
            'strategy_phase_preferences': {}
        }
        
        np.random.seed(42)
    
    def extract_market_context(
        self,
        symbol: str,
        current_price: float,
        cost_basis: float,
        bars: List[Dict],
        options_data: List[Dict] = None,
        fundamentals: Dict = None,
        strategy_phase: str = "SP"
    ) -> DTEMarketContext:
        """Extract market context features for DTE optimization."""
        
        if len(bars) < 20:
            return self._create_default_context(symbol, current_price, cost_basis, strategy_phase)
        
        closes = np.array([b['close'] for b in bars[-30:]])
        returns = np.diff(closes) / closes[:-1]
        
        volatility_20d = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.25
        volatility_30d = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.25
        
        momentum_5d = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        momentum_20d = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0
        
        market_regime = self._determine_market_regime(volatility_30d, momentum_5d, momentum_20d)
        
        pe_ratio = fundamentals.get('pe_ratio', 25.0) if fundamentals else 25.0
        iv_rank = 50.0
        
        if options_data:
            ivs = [opt.get('iv', 0.25) for opt in options_data if 'iv' in opt]
            if ivs:
                avg_iv = np.mean(ivs)
                iv_rank = min(100, max(0, (avg_iv - 0.15) / 0.35 * 100))
        
        return DTEMarketContext(
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
            days_to_earnings=30,
            option_liquidity=0.5,
            strategy_phase=strategy_phase
        )
    
    def _create_default_context(
        self,
        symbol: str,
        current_price: float,
        cost_basis: float,
        strategy_phase: str
    ) -> DTEMarketContext:
        """Create default context when insufficient data."""
        
        # Determine regime based on cost basis
        if cost_basis > 0:
            price_ratio = current_price / cost_basis
            if price_ratio < 0.95:
                market_regime = "bear"
            elif price_ratio > 1.05:
                market_regime = "bull"
            else:
                market_regime = "neutral"
        else:
            market_regime = "neutral"
        
        return DTEMarketContext(
            symbol=symbol,
            current_price=current_price,
            cost_basis=cost_basis,
            volatility_20d=0.25,
            volatility_30d=0.25,
            momentum_5d=0.0,
            momentum_20d=0.0,
            pe_ratio=25.0,
            iv_rank=50.0,
            market_regime=market_regime,
            days_to_earnings=30,
            option_liquidity=0.5,
            strategy_phase=strategy_phase
        )
    
    def _determine_market_regime(
        self,
        volatility: float,
        momentum_5d: float,
        momentum_20d: float
    ) -> str:
        """Determine current market regime."""
        
        if volatility > 0.30:
            return "high_vol"
        elif momentum_5d > 0.05 and momentum_20d > 0.10:
            return "bull"
        elif momentum_5d < -0.05 and momentum_20d < -0.10:
            return "bear"
        else:
            return "neutral"
    
    def optimize_dte_range(
        self,
        market_context: DTEMarketContext,
        right: str,
        iv: float,
        current_positions: List[Dict] = None
    ) -> DTEOptimizationResult:
        """
        Optimize DTE range selection using ML models and Q-learning.
        
        Returns the optimal DTE range with confidence and reasoning.
        """
        
        # Generate candidate DTE values
        candidate_dtes = list(range(
            self.config.dte_min,
            self.config.dte_max + 1,
            self.config.dte_step
        ))
        
        # Score each candidate DTE
        dte_scores = []
        
        for dte in candidate_dtes:
            score = self._score_dte_candidate(
                dte, market_context, right, iv, current_positions
            )
            dte_scores.append((dte, score))
        
        # Select optimal DTE with exploration
        optimal_dte, confidence, reasoning = self._select_optimal_dte(
            dte_scores, market_context
        )
        
        # Calculate expected metrics
        expected_premium = self._estimate_expected_premium(
            optimal_dte, market_context, right, iv
        )
        
        expected_prob_assignment = self._estimate_assignment_probability(
            optimal_dte, market_context, right, iv
        )
        
        risk_score = self._calculate_risk_score(
            optimal_dte, market_context, right
        )
        
        # Calculate DTE range around optimal
        dte_range_buffer = 7
        optimal_dte_min = max(self.config.dte_min, optimal_dte - dte_range_buffer)
        optimal_dte_max = min(self.config.dte_max, optimal_dte + dte_range_buffer)
        
        return DTEOptimizationResult(
            optimal_dte_min=optimal_dte_min,
            optimal_dte_max=optimal_dte_max,
            expected_premium=expected_premium,
            expected_probability_assignment=expected_prob_assignment,
            risk_score=risk_score,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _score_dte_candidate(
        self,
        dte: int,
        context: DTEMarketContext,
        right: str,
        iv: float,
        positions: List[Dict] = None
    ) -> float:
        """Score a DTE candidate using multiple factors."""
        
        score = 0.0
        
        # 1. Historical performance (30%)
        hist_score = self._get_historical_score(
            dte, context.symbol, context.market_regime, context.strategy_phase
        )
        score += hist_score * 0.30
        
        # 2. Market regime suitability (25%)
        regime_score = self._get_regime_suitability(
            dte, context.market_regime, right, context.strategy_phase
        )
        score += regime_score * 0.25
        
        # 3. Risk-adjusted return (25%)
        risk_score = self._calculate_risk_adjusted_return(dte, context, right, iv)
        score += risk_score * 0.25
        
        # 4. Strategy phase optimization (20%)
        phase_score = self._optimize_for_strategy_phase(
            dte, context.strategy_phase, context
        )
        score += phase_score * 0.20
        
        return score
    
    def _get_historical_score(
        self,
        dte: int,
        symbol: str,
        regime: str,
        strategy_phase: str
    ) -> float:
        """Get historical performance score for a DTE value."""
        
        key = f"{symbol}_{dte}_{regime}_{strategy_phase}"
        
        if key in self.model['dte_performance']:
            performance = self.model['dte_performance'][key]
            return min(1.0, max(0.0, np.mean(performance) + 0.5))
        else:
            return 0.5
    
    def _get_regime_suitability(
        self,
        dte: int,
        regime: str,
        right: str,
        strategy_phase: str
    ) -> float:
        """Get suitability score for current market regime."""
        
        # Optimal DTE by regime and strategy phase
        regime_patterns = {
            'bull': {'SP': 25, 'CC': 40},
            'bear': {'SP': 40, 'CC': 25},
            'neutral': {'SP': 30, 'CC': 35},
            'high_vol': {'SP': 21, 'CC': 25}
        }
        
        optimal_dte = regime_patterns.get(regime, {}).get(strategy_phase, 30)
        
        # Score based on distance from optimal
        distance = abs(dte - optimal_dte)
        return max(0.0, 1.0 - distance / 20.0)
    
    def _calculate_risk_adjusted_return(
        self,
        dte: int,
        context: DTEMarketContext,
        right: str,
        iv: float
    ) -> float:
        """Calculate risk-adjusted return score for DTE candidate."""
        
        T = dte / 365.0
        delta = 0.30
        
        # Estimate premium using simplified approximation
        if right == "P":
            strike = context.current_price * (1 - delta)
            moneyness = strike / context.current_price
            premium = context.current_price * delta * iv * np.sqrt(T) * (1 - moneyness * 0.5)
        else:
            strike = context.current_price * (1 + delta)
            moneyness = strike / context.current_price
            premium = context.current_price * delta * iv * np.sqrt(T) * (moneyness * 0.5 - 0.5)
        
        premium = max(premium, 0.01)
        
        # Time decay factor
        time_decay_factor = min(1.0, T * 365 / 45)
        
        # Risk from assignment
        assignment_risk = 0.0
        if right == "P" and context.cost_basis > 0:
            assignment_risk = max(0, context.cost_basis - strike) / context.cost_basis
        elif right == "C" and context.cost_basis > 0:
            assignment_risk = max(0, strike - context.cost_basis) / context.cost_basis
        
        # Risk-adjusted score
        risk_score = premium * (1.0 + time_decay_factor) * (1.0 - assignment_risk)
        return min(1.0, risk_score / 0.1)
    
    def _optimize_for_strategy_phase(
        self,
        dte: int,
        strategy_phase: str,
        context: DTEMarketContext
    ) -> float:
        """Optimize DTE based on strategy phase."""
        
        if strategy_phase == "SP" or strategy_phase == "CC+SP":
            # Sell Put: prefer moderate DTE
            if 25 <= dte <= 40:
                return 1.0
            elif 21 <= dte <= 45:
                return 0.8
            else:
                return 0.3
        elif strategy_phase == "CC":
            # Covered Call: prefer moderate to longer DTE
            if 30 <= dte <= 45:
                return 1.0
            elif 25 <= dte <= 50:
                return 0.8
            else:
                return 0.3
        else:
            return 0.5
    
    def _select_optimal_dte(
        self,
        dte_scores: List[Tuple[int, float]],
        context: DTEMarketContext
    ) -> Tuple[int, float, str]:
        """Select optimal DTE with exploration mechanism."""
        
        dte_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Exploration: occasionally try random DTE
        if np.random.random() < self.config.exploration_rate:
            selected_dte = np.random.choice([d[0] for d in dte_scores])
            confidence = 0.5
            reasoning = f"Exploring DTE {selected_dte} for learning in {context.strategy_phase} phase"
        else:
            selected_dte = dte_scores[0][0]
            confidence = max(0.7, min(1.0, dte_scores[0][1] * 2))
            reasoning = f"Optimal DTE {selected_dte} for {context.market_regime} regime, {context.strategy_phase} phase"
        
        return selected_dte, confidence, reasoning
    
    def _estimate_expected_premium(
        self,
        dte: int,
        context: DTEMarketContext,
        right: str,
        iv: float
    ) -> float:
        """Estimate expected premium for selected DTE using simplified approximation."""
        
        T = dte / 365.0
        delta = 0.30
        
        if right == "P":
            strike = context.current_price * (1 - delta)
            moneyness = strike / context.current_price
            premium = context.current_price * delta * iv * np.sqrt(T) * (1 - moneyness * 0.5)
        else:
            strike = context.current_price * (1 + delta)
            moneyness = strike / context.current_price
            premium = context.current_price * delta * iv * np.sqrt(T) * (moneyness * 0.5 - 0.5)
        
        return max(premium, 0.01)
    
    def _estimate_assignment_probability(
        self,
        dte: int,
        context: DTEMarketContext,
        right: str,
        iv: float
    ) -> float:
        """Estimate probability of assignment using simplified delta approximation."""
        
        delta = 0.30
        
        if right == "P":
            strike = context.current_price * (1 - delta)
        else:
            strike = context.current_price * (1 + delta)
        
        T = dte / 365.0
        
        # ITM probability approximation (delta ≈ ITM probability for OTM options)
        moneyness = strike / context.current_price
        itm_prob = delta * (1 + 0.5 * (1 - moneyness) / (iv * np.sqrt(T))) if right == "P" else delta * (1 + 0.5 * (moneyness - 1) / (iv * np.sqrt(T)))
        
        # Adjust for regime
        regime_adj = {
            'bull': 1.2 if right == "C" else 0.8,
            'bear': 1.2 if right == "P" else 0.8,
            'neutral': 1.0,
            'high_vol': 0.9
        }
        
        return min(1.0, itm_prob * regime_adj.get(context.market_regime, 1.0))
    
    def _calculate_risk_score(
        self,
        dte: int,
        context: DTEMarketContext,
        right: str
    ) -> float:
        """Calculate risk score (0-1, where 1 is highest risk)."""
        
        # Base risk from DTE
        dte_risk = min(1.0, (dte - self.config.dte_min) / 
                       (self.config.dte_max - self.config.dte_min))
        
        # Adjust for market conditions
        if context.market_regime == "high_vol":
            dte_risk *= 1.3
        elif context.market_regime == "bear" and right == "P":
            dte_risk *= 1.1
        elif context.market_regime == "bull" and right == "C":
            dte_risk *= 1.1
        
        # Adjust for liquidity
        dte_risk *= (2.0 - context.option_liquidity)
        
        return min(1.0, dte_risk)
    
    def update_performance(
        self,
        dte: int,
        symbol: str,
        context: DTEMarketContext,
        actual_pnl: float,
        actual_assignment: bool
    ):
        """Update Q-table with performance data."""
        
        state_key = f"{symbol}_{context.market_regime}_{context.strategy_phase}"
        action_key = f"dte_{dte}"
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Calculate reward
        reward = actual_pnl
        if actual_assignment and actual_pnl > 0:
            reward *= 1.2
        
        # Q-learning update
        if action_key in self.q_table[state_key]:
            old_value = self.q_table[state_key][action_key]
            self.q_table[state_key][action_key] = old_value + self.config.learning_rate * (
                reward - old_value
            )
        else:
            self.q_table[state_key][action_key] = reward
        
        # Update model
        key = f"{symbol}_{dte}_{context.market_regime}_{context.strategy_phase}"
        if key not in self.model['dte_performance']:
            self.model['dte_performance'][key] = []
        
        self.model['dte_performance'][key].append(actual_pnl)
        
        if len(self.model['dte_performance'][key]) > 100:
            self.model['dte_performance'][key] = self.model['dte_performance'][key][-100:]
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'dte': dte,
            'symbol': symbol,
            'regime': context.market_regime,
            'phase': context.strategy_phase,
            'pnl': actual_pnl,
            'assignment': actual_assignment
        })
        
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def pretrain_with_history(
        self,
        symbol: str,
        historical_bars: List[Dict],
        iv_estimate: float = 0.25,
        right: str = "P",
        strategy_phase: str = "SP"
    ) -> Dict[str, Any]:
        """Pretrain model using historical data."""
        
        if len(historical_bars) < 60:
            return {"status": "skipped", "reason": "insufficient_data"}
        
        candidate_dtes = list(range(self.config.dte_min, self.config.dte_max + 1, self.config.dte_step))
        
        stats = {
            "total_simulations": 0,
            "regimes_tested": set(),
            "phases_tested": set(),
            "best_dte_by_regime": {}
        }
        
        train_size = int(len(historical_bars) * 0.3)
        train_bars = historical_bars[-train_size:]
        
        for i in range(len(train_bars) - self.config.dte_max - 1):
            entry_bar = train_bars[i]
            
            for dte in candidate_dtes:
                if i + dte >= len(train_bars):
                    continue
                
                exit_bar = train_bars[i + dte]
                entry_price = entry_bar['close']
                exit_price = exit_bar['close']
                
                # Simulate trade
                pnl, assigned = self._simulate_option_trade(
                    entry_price, exit_price, dte, right, iv_estimate, strategy_phase
                )
                
                # Determine regime
                lookback = min(30, i)
                recent_bars = train_bars[max(0, i-lookback):i+1]
                closes = [b['close'] for b in recent_bars]
                volatility = np.std(np.diff(closes) / np.array(closes[:-1])) * np.sqrt(252) if len(closes) > 1 else 0.25
                momentum_5d = (entry_price - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
                momentum_20d = (entry_price - closes[0]) / closes[0] if len(closes) > 0 else 0
                
                regime = self._determine_market_regime(volatility, momentum_5d, momentum_20d)
                stats["regimes_tested"].add(regime)
                stats["phases_tested"].add(strategy_phase)
                
                # Create context and update
                context = DTEMarketContext(
                    symbol=symbol,
                    current_price=entry_price,
                    cost_basis=entry_price,
                    volatility_20d=volatility,
                    volatility_30d=volatility,
                    momentum_5d=momentum_5d,
                    momentum_20d=momentum_20d,
                    pe_ratio=25.0,
                    iv_rank=50.0,
                    market_regime=regime,
                    days_to_earnings=30,
                    option_liquidity=0.5,
                    strategy_phase=strategy_phase
                )
                
                self.update_performance(dte, symbol, context, pnl, assigned)
                stats["total_simulations"] += 1
        
        stats["regimes_tested"] = list(stats["regimes_tested"])
        stats["phases_tested"] = list(stats["phases_tested"])
        
        self.last_retrain = datetime.now()
        
        return stats
    
    def _simulate_option_trade(
        self,
        entry_price: float,
        exit_price: float,
        dte: int,
        right: str,
        iv: float,
        strategy_phase: str
    ) -> Tuple[float, bool]:
        """Simulate an option trade outcome."""
        
        T = dte / 365.0
        
        # Calculate delta
        if strategy_phase == "SP" or strategy_phase == "CC+SP":
            delta = 0.30
        else:
            delta = 0.30
        
        if right == "P":
            strike = entry_price * (1 - delta)
            moneyness = strike / entry_price
            entry_premium = entry_price * delta * iv * np.sqrt(T) * (1 - moneyness * 0.5)
            assigned = exit_price < strike
            if assigned:
                stock_pnl = exit_price - strike
                pnl = (entry_premium + stock_pnl) / strike
            else:
                pnl = entry_premium / strike
        else:
            strike = entry_price * (1 + delta)
            moneyness = strike / entry_price
            entry_premium = entry_price * delta * iv * np.sqrt(T) * (moneyness * 0.5 - 0.5)
            assigned = exit_price > strike
            if assigned:
                stock_pnl = strike - exit_price
                pnl = (entry_premium + stock_pnl) / strike
            else:
                pnl = entry_premium / strike
        
        return max(pnl, -0.5), assigned  # Cap max loss at 50%
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights about optimization performance."""
        
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        recent = self.performance_history[-100:]
        
        avg_pnl = np.mean([p['pnl'] for p in recent])
        win_rate = sum(1 for p in recent if p['pnl'] > 0) / len(recent)
        
        # Best performing DTEs
        dte_performance = {}
        for p in recent:
            dte = p['dte']
            if dte not in dte_performance:
                dte_performance[dte] = []
            dte_performance[dte].append(p['pnl'])
        
        best_dtes = sorted(
            [(dte, np.mean(perfs)) for dte, perfs in dte_performance.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "average_pnl": avg_pnl,
            "win_rate": win_rate,
            "total_trades": len(self.performance_history),
            "best_performing_dtes": best_dtes,
            "last_retrain": str(self.last_retrain) if self.last_retrain else None,
            "exploration_rate": self.config.exploration_rate
        }