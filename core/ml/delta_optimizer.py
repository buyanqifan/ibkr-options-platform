"""
ML-based Delta Optimizer for Options Strategies.

This module provides intelligent delta selection using machine learning models
that optimize for risk-adjusted returns across different market conditions.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from core.ml.features.volatility import VolatilityFeatures
from core.ml.features.technical import TechnicalFeatures
from core.backtesting.pricing import OptionsPricer

logger = logging.getLogger("delta_optimizer")


@dataclass
class DeltaOptimizationConfig:
    """Configuration for ML Delta optimization."""
    # Model training parameters
    model_path: str = "data/models/delta_optimizer.pkl"
    retrain_interval_days: int = 30
    min_training_samples: int = 1000
    
    # Optimization parameters
    exploration_rate: float = 0.1  # 10% exploration
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    
    # Delta search space
    delta_min: float = 0.05
    delta_max: float = 0.40
    delta_step: float = 0.05
    
    # Risk parameters
    max_loss_threshold: float = 0.20  # 20% max loss per trade
    min_premium_threshold: float = 0.01  # 1% minimum premium


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
    days_to_earnings: int
    option_liquidity: float  # 0-1 scale


@dataclass
class OptimizationResult:
    """Result of delta optimization."""
    optimal_delta: float
    expected_premium: float
    expected_probability_assignment: float
    risk_score: float
    confidence: float  # 0-1 scale
    reasoning: str


class DeltaOptimizerML:
    """
    Machine Learning Delta Optimizer for Options Strategies.
    
    Uses Q-learning and prediction models to optimize delta selection
    based on market conditions, risk tolerance, and historical performance.
    """
    
    def __init__(self, config: DeltaOptimizationConfig):
        self.config = config
        self.model = None
        self.q_table = {}
        self.performance_history = []
        self.last_retrain = None
        self._load_model()
    
    def _load_model(self):
        """Load existing model or initialize new one."""
        if os.path.exists(self.config.model_path):
            try:
                with open(self.config.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.q_table = data.get('q_table', {})
                    self.performance_history = data.get('performance_history', [])
                    self.last_retrain = data.get('last_retrain')
                logger.info("ML Delta optimizer model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self._initialize_model()
        else:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize ML model with default values."""
        self.model = {
            'delta_performance': {},  # delta -> historical performance
            'market_regime_patterns': {},  # regime -> optimal delta range
            'symbol_specific': {}  # symbol -> custom delta preferences
        }
        self.q_table = {}
        logger.info("Initialized new ML Delta optimizer")
    
    def pretrain_with_history(self, 
                              symbol: str,
                              historical_bars: List[Dict],
                              iv_estimate: float = 0.25,
                              right: str = "P",
                              training_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Pretrain the model using historical price data.
        
        This simulates trades across different market conditions and delta values
        to build initial Q-table knowledge before live trading.
        
        Args:
            symbol: Stock symbol
            historical_bars: List of historical price bars with 'date', 'close', 'high', 'low'
            iv_estimate: Estimated implied volatility for option pricing
            right: Option type ('P' for put, 'C' for call)
            training_ratio: Fraction of data to use for training (0.0-1.0)
            
        Returns:
            Dict with training statistics
        """
        if len(historical_bars) < 60:
            logger.warning(f"Insufficient history for pretraining: {len(historical_bars)} bars")
            return {"status": "skipped", "reason": "insufficient_data"}
        
        logger.info(f"Starting pretraining for {symbol} with {len(historical_bars)} bars...")
        
        # Use portion of data for training
        train_size = int(len(historical_bars) * training_ratio)
        train_bars = historical_bars[-train_size:]
        
        # Candidate deltas to test
        candidate_deltas = np.arange(0.10, 0.40, 0.05)
        
        # Default DTE for simulation
        dte = 30
        T = dte / 365.0
        
        stats = {
            "total_simulations": 0,
            "regimes_tested": set(),
            "best_delta_by_regime": {}
        }
        
        # Slide through historical data simulating trades
        for i in range(len(train_bars) - dte - 1):
            entry_bar = train_bars[i]
            exit_bar = train_bars[min(i + dte, len(train_bars) - 1)]
            
            entry_price = entry_bar['close']
            exit_price = exit_bar['close']
            entry_date = entry_bar.get('date', str(i))
            
            # Determine market regime at entry
            lookback = min(30, i)
            recent_bars = train_bars[max(0, i-lookback):i+1]
            
            volatility = self._calculate_simple_volatility(recent_bars)
            momentum_5d = (entry_price - train_bars[max(0, i-5)]['close']) / train_bars[max(0, i-5)]['close']
            momentum_20d = (entry_price - train_bars[max(0, i-20)]['close']) / train_bars[max(0, i-20)]['close'] if i >= 20 else momentum_5d
            
            regime = self._determine_market_regime(volatility, momentum_5d, momentum_20d)
            stats["regimes_tested"].add(regime)
            
            # Create market context
            context = MarketContext(
                symbol=symbol,
                current_price=entry_price,
                cost_basis=entry_price,  # Assume entry at current price
                volatility_20d=volatility,
                volatility_30d=volatility,
                momentum_5d=momentum_5d,
                momentum_20d=momentum_20d,
                pe_ratio=25.0,
                iv_rank=50.0,
                market_regime=regime,
                days_to_earnings=30,
                option_liquidity=0.5
            )
            
            # Test each delta and record performance
            for delta in candidate_deltas:
                # Simulate trade outcome
                pnl, assigned = self._simulate_option_trade(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    delta=delta,
                    right=right,
                    iv=iv_estimate,
                    T=T
                )
                
                # Update Q-table with simulated result
                self.update_performance(
                    delta=delta,
                    symbol=symbol,
                    context=context,
                    actual_pnl=pnl,
                    actual_assignment=assigned
                )
                
                stats["total_simulations"] += 1
        
        # Convert set to list for JSON serialization
        stats["regimes_tested"] = list(stats["regimes_tested"])
        
        # Calculate best delta by regime from accumulated data
        for regime in stats["regimes_tested"]:
            best_delta, avg_pnl = self._find_best_delta_for_regime(symbol, regime)
            stats["best_delta_by_regime"][regime] = {
                "delta": best_delta,
                "avg_pnl": avg_pnl
            }
        
        self.last_retrain = datetime.now()
        logger.info(f"Pretraining complete: {stats['total_simulations']} simulations, "
                   f"regimes: {stats['regimes_tested']}")
        
        return stats
    
    def _calculate_simple_volatility(self, bars: List[Dict]) -> float:
        """Calculate simple historical volatility from price bars."""
        if len(bars) < 2:
            return 0.25
        
        prices = [b['close'] for b in bars]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        if not returns:
            return 0.25
        
        # Annualized volatility
        std = np.std(returns)
        return std * np.sqrt(252)
    
    def _simulate_option_trade(self,
                               entry_price: float,
                               exit_price: float,
                               delta: float,
                               right: str,
                               iv: float,
                               T: float) -> Tuple[float, bool]:
        """
        Simulate an option trade outcome.
        
        Returns:
            Tuple of (pnl, was_assigned)
        """
        # Calculate strike based on delta
        if right == "P":
            strike = entry_price * (1 - delta)  # OTM put
        else:
            strike = entry_price * (1 + delta)  # OTM call
        
        # Calculate entry premium using Black-Scholes approximation
        if right == "P":
            entry_premium = OptionsPricer.put_price(entry_price, strike, T, iv)
        else:
            entry_premium = OptionsPricer.call_price(entry_price, strike, T, iv)
        
        # Determine if assigned (ITM at expiry)
        if right == "P":
            assigned = exit_price < strike
            if assigned:
                # Assignment: buy stock at strike, current value is exit_price
                stock_pnl = exit_price - strike  # Negative if below strike
                pnl = entry_premium + stock_pnl  # Premium + stock loss
            else:
                # Expires worthless: keep full premium
                pnl = entry_premium
        else:
            assigned = exit_price > strike
            if assigned:
                # Assignment: sell stock at strike, could have sold at exit_price
                stock_pnl = strike - exit_price  # Negative if above strike
                pnl = entry_premium + stock_pnl
            else:
                pnl = entry_premium
        
        # Normalize to percentage of strike
        pnl_pct = pnl / strike
        
        return pnl_pct, assigned
    
    def _find_best_delta_for_regime(self, symbol: str, regime: str) -> Tuple[float, float]:
        """Find the best performing delta for a given regime."""
        delta_performance = {}
        
        for key, performances in self.model['delta_performance'].items():
            if f"{symbol}_" in key and regime in key:
                # Extract delta from key
                parts = key.split('_')
                if len(parts) >= 2:
                    try:
                        delta = float(parts[1])
                        avg_pnl = np.mean(performances) if performances else 0
                        delta_performance[delta] = avg_pnl
                    except (ValueError, IndexError):
                        continue
        
        if delta_performance:
            best_delta = max(delta_performance, key=delta_performance.get)
            return best_delta, delta_performance[best_delta]
        
        return 0.30, 0.0  # Default
    
    def extract_market_context(self, 
                             symbol: str,
                             current_price: float,
                             cost_basis: float,
                             bars: List[Dict],
                             options_data: List[Dict],
                             fundamentals: Dict = None) -> MarketContext:
        """Extract market context features from data."""
        
        if len(bars) < 30:
            # Fallback to simplified market context when insufficient data
            logger.warning(f"Insufficient bars ({len(bars)}), using simplified market context")
            return self._create_simplified_context(symbol, current_price, cost_basis)
        
        # Convert to DataFrame for feature calculation
        df = pd.DataFrame(bars)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Calculate technical features
        vol_features = VolatilityFeatures.calculate_volatility_features(df)
        tech_features = TechnicalFeatures.calculate_technical_features(df)
        
        # Extract key metrics
        current_vol = vol_features.get('volatility_20d', 0.2)
        momentum_5d = tech_features.get('momentum_5d', 0.0)
        momentum_20d = tech_features.get('momentum_20d', 0.0)
        
        # Determine market regime
        market_regime = self._determine_market_regime(
            current_vol, momentum_5d, momentum_20d
        )
        
        # Calculate days to earnings (approximate)
        days_to_earnings = self._estimate_days_to_earnings(symbol, df)
        
        # Calculate option liquidity
        option_liquidity = self._calculate_option_liquidity(options_data)
        
        # Get fundamentals if available
        pe_ratio = fundamentals.get('pe_ratio', 25.0) if fundamentals else 25.0
        iv_rank = options_data[0].get('iv_rank', 50.0) if options_data else 50.0
        
        return MarketContext(
            symbol=symbol,
            current_price=current_price,
            cost_basis=cost_basis,
            volatility_20d=current_vol,
            volatility_30d=vol_features.get('volatility_30d', 0.25),
            momentum_5d=momentum_5d,
            momentum_20d=momentum_20d,
            pe_ratio=pe_ratio,
            iv_rank=iv_rank,
            market_regime=market_regime,
            days_to_earnings=days_to_earnings,
            option_liquidity=option_liquidity
        )
    
    def _create_simplified_context(self, symbol: str, current_price: float, cost_basis: float) -> MarketContext:
        """Create simplified market context when insufficient historical data."""
        
        # Determine regime based on cost basis vs current price
        if cost_basis > 0:
            price_ratio = current_price / cost_basis
            if price_ratio < 0.95:
                market_regime = "bear"  # Price dropped below cost
            elif price_ratio > 1.05:
                market_regime = "bull"  # Price above cost
            else:
                market_regime = "neutral"
        else:
            market_regime = "neutral"
        
        return MarketContext(
            symbol=symbol,
            current_price=current_price,
            cost_basis=cost_basis,
            volatility_20d=0.25,  # Default moderate volatility
            volatility_30d=0.25,
            momentum_5d=0.0,
            momentum_20d=0.0,
            pe_ratio=25.0,  # Default PE
            iv_rank=50.0,  # Default mid IV rank
            market_regime=market_regime,
            days_to_earnings=30,
            option_liquidity=0.5  # Moderate liquidity
        )
    
    def _determine_market_regime(self, volatility: float, momentum_5d: float, momentum_20d: float) -> str:
        """Determine current market regime."""
        
        if volatility > 0.30:  # High volatility
            return "high_vol"
        elif momentum_5d > 0.05 and momentum_20d > 0.10:  # Strong upward trend
            return "bull"
        elif momentum_5d < -0.05 and momentum_20d < -0.10:  # Strong downward trend
            return "bear"
        else:
            return "neutral"
    
    def _estimate_days_to_earnings(self, symbol: str, df: pd.DataFrame) -> int:
        """Estimate days to next earnings announcement."""
        # Simplified estimation - in production would use actual earnings calendar
        price_change_rate = abs(df['close'].pct_change().mean())
        if price_change_rate > 0.02:  # High volatility before earnings
            return 7  # 7 days before typical earnings
        else:
            return 30  # Normal cycle
    
    def _calculate_option_liquidity(self, options_data: List[Dict]) -> float:
        """Calculate option liquidity score based on bid-ask spread and volume."""
        if not options_data:
            return 0.0
        
        total_volume = sum(opt.get('volume', 0) for opt in options_data)
        avg_spread = np.mean([
            abs(opt.get('ask', 0) - opt.get('bid', 0)) 
            for opt in options_data 
            if opt.get('ask') and opt.get('bid')
        ])
        
        # Liquidity score: higher volume and lower spread = better liquidity
        liquidity_score = min(1.0, total_volume / 10000) * max(0.0, 1.0 - avg_spread)
        return liquidity_score
    
    def optimize_delta(self, 
                      market_context: MarketContext,
                      right: str,  # "P" or "C"
                      iv: float,
                      time_to_expiry: float,
                      current_positions: List[Dict] = None) -> OptimizationResult:
        """
        Optimize delta selection using ML models and Q-learning.
        
        Returns the optimal delta with confidence and reasoning.
        """
        
        # Generate candidate delta values
        candidate_deltas = np.arange(
            self.config.delta_min, 
            self.config.delta_max + self.config.delta_step, 
            self.config.delta_step
        )
        
        # Score each candidate delta
        delta_scores = []
        
        for delta in candidate_deltas:
            score = self._score_delta_candidate(
                delta, market_context, right, iv, time_to_expiry, current_positions
            )
            delta_scores.append((delta, score))
        
        # Select optimal delta with exploration
        optimal_delta, confidence, reasoning = self._select_optimal_delta(delta_scores, market_context)
        
        # Calculate expected metrics
        expected_premium = self._estimate_expected_premium(
            optimal_delta, market_context, right, iv, time_to_expiry
        )
        
        expected_prob_assignment = self._estimate_assignment_probability(
            optimal_delta, market_context, right, iv, time_to_expiry
        )
        
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
    
    def _score_delta_candidate(self, 
                             delta: float,
                             context: MarketContext,
                             right: str,
                             iv: float,
                             time_to_expiry: float,
                             positions: List[Dict] = None) -> float:
        """Score a delta candidate using multiple factors."""
        
        score = 0.0
        
        # 1. Historical performance for this delta
        hist_score = self._get_historical_score(delta, context.symbol, context.market_regime)
        score += hist_score * 0.3
        
        # 2. Market regime suitability
        regime_score = self._get_regime_suitability(delta, context.market_regime, right)
        score += regime_score * 0.25
        
        # 3. Risk-adjusted return
        risk_score = self._calculate_risk_adjusted_return(delta, context, right, iv, time_to_expiry)
        score += risk_score * 0.25
        
        # 4. Position-specific optimization
        if positions:
            pos_score = self._optimize_for_positions(delta, positions, context)
            score += pos_score * 0.2
        
        return score
    
    def _get_historical_score(self, delta: float, symbol: str, regime: str) -> float:
        """Get historical performance score for a delta value."""
        key = f"{symbol}_{delta}_{regime}"
        
        if key in self.model['delta_performance']:
            performance = self.model['delta_performance'][key]
            # Return normalized performance (0-1 scale)
            return min(1.0, max(0.0, performance + 0.5))  # Shift to positive range
        else:
            # Return base score for unseen delta-regime combinations
            return 0.5  # Neutral baseline
    
    def _get_regime_suitability(self, delta: float, regime: str, right: str) -> float:
        """Get suitability score for current market regime."""
        
        regime_patterns = {
            'bull': {'P': 0.3, 'C': 0.4},  # In bull market, slightly OTM calls
            'bear': {'P': 0.2, 'C': 0.3},  # In bear market, more protective puts
            'neutral': {'P': 0.3, 'C': 0.3},  # Neutral: standard OTM
            'high_vol': {'P': 0.25, 'C': 0.2}  # High vol: more conservative
        }
        
        optimal_delta = regime_patterns.get(regime, {}).get(right, 0.3)
        
        # Score based on distance from optimal delta
        distance = abs(delta - optimal_delta)
        return max(0.0, 1.0 - distance / 0.2)  # Normalize to 0-1
    
    def _calculate_risk_adjusted_return(self, 
                                     delta: float,
                                     context: MarketContext,
                                     right: str,
                                     iv: float,
                                     time_to_expiry: float) -> float:
        """Calculate risk-adjusted return score for delta candidate."""
        
        # Estimate premium for this delta
        if right == "P":
            strike = context.current_price * (1 - delta)  # Approximate
            premium = OptionsPricer.put_price(context.current_price, strike, time_to_expiry, iv)
        else:
            strike = context.current_price * (1 + delta)  # Approximate
            premium = OptionsPricer.call_price(context.current_price, strike, time_to_expiry, iv)
        
        # Calculate risk metrics
        assignment_risk = 0.0
        protection_value = 0.0
        
        if right == "P" and context.cost_basis > 0:
            # For puts: risk if assigned at strike below cost
            assignment_risk = max(0, context.cost_basis - strike) / context.cost_basis
        elif right == "C" and context.cost_basis > 0:
            # For calls: protection if price drops below cost
            protection_value = max(0, strike - context.cost_basis) / context.cost_basis
        
        # Risk-adjusted score: prefer higher premiums with lower risk
        risk_score = premium * (1.0 - assignment_risk) * (1.0 + protection_value)
        return min(1.0, risk_score / 0.1)  # Normalize to 0-1
    
    def _optimize_for_positions(self, delta: float, positions: List[Dict], context: MarketContext) -> float:
        """Optimize delta based on current position exposure."""
        
        if not positions:
            return 1.0  # No position constraints
        
        total_exposure = sum(abs(p.get('delta', 0)) for p in positions)
        
        # Avoid excessive total delta exposure
        if total_exposure + delta > 1.5:  # Max 1.5 total delta exposure
            return 0.3  # Low score for high exposure
        elif total_exposure + delta > 1.0:  # Moderate exposure
            return 0.7  # Medium score
        else:
            return 1.0  # Good score for low exposure
    
    def _select_optimal_delta(self, 
                            delta_scores: List[Tuple[float, float]], 
                            context: MarketContext) -> Tuple[float, float, str]:
        """Select optimal delta with exploration mechanism."""
        
        # Sort by score
        delta_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Exploration: occasionally try random delta
        if np.random.random() < self.config.exploration_rate:
            selected_delta = np.random.choice([d[0] for d in delta_scores])
            confidence = 0.5  # Lower confidence for exploration
            reasoning = f"Exploring delta {selected_delta:.2f} for learning purposes"
        else:
            # Select best delta
            selected_delta = delta_scores[0][0]
            # Scale confidence with minimum baseline of 0.7 for consistent results
            confidence = max(0.7, min(1.0, delta_scores[0][1] * 2))
            
            reasoning = f"Optimal delta {selected_delta:.2f} based on {context.market_regime} market regime and {context.symbol} characteristics"
        
        return selected_delta, confidence, reasoning
    
    def _estimate_expected_premium(self, 
                                 delta: float,
                                 context: MarketContext,
                                 right: str,
                                 iv: float,
                                 time_to_expiry: float) -> float:
        """Estimate expected premium for selected delta."""
        
        if right == "P":
            strike = context.current_price * (1 - delta)
            premium = OptionsPricer.put_price(context.current_price, strike, time_to_expiry, iv)
        else:
            strike = context.current_price * (1 + delta)
            premium = OptionsPricer.call_price(context.current_price, strike, time_to_expiry, iv)
        
        return premium
    
    def _estimate_assignment_probability(self, 
                                       delta: float,
                                       context: MarketContext,
                                       right: str,
                                       iv: float,
                                       time_to_expiry: float) -> float:
        """Estimate probability of option assignment."""
        
        # Simplified model: assignment probability increases with delta
        base_prob = delta
        
        # Adjust for market regime
        regime_adjustment = {
            'bull': 1.2 if right == "C" else 0.8,  # Higher call assignment in bull
            'bear': 1.2 if right == "P" else 0.8,  # Higher put assignment in bear
            'neutral': 1.0,
            'high_vol': 0.7  # Lower assignment in high volatility
        }
        
        adjusted_prob = base_prob * regime_adjustment.get(context.market_regime, 1.0)
        return min(1.0, adjusted_prob)
    
    def _calculate_risk_score(self, 
                            delta: float,
                            context: MarketContext,
                            right: str) -> float:
        """Calculate risk score (0-1, where 1 is highest risk)."""
        
        risk_score = delta  # Base risk on delta
        
        # Adjust for market conditions
        if context.market_regime == "high_vol":
            risk_score *= 1.3  # Higher risk in high volatility
        elif context.market_regime == "bear" and right == "P":
            risk_score *= 1.2  # Higher put assignment risk in bear market
        elif context.market_regime == "bull" and right == "C":
            risk_score *= 1.1  # Higher call assignment risk in bull market
        
        # Adjust for liquidity
        risk_score *= (2.0 - context.option_liquidity)  # Lower liquidity = higher risk
        
        return min(1.0, risk_score)
    
    def update_performance(self, 
                          delta: float,
                          symbol: str,
                          context: MarketContext,
                          actual_pnl: float,
                          actual_assignment: bool):
        """Update performance metrics for learning."""
        
        # Update Q-table
        state_key = f"{symbol}_{context.market_regime}_{int(context.current_price)}"
        action_key = f"delta_{delta}"
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Q-learning update
        reward = actual_pnl
        if actual_assignment:
            reward *= 1.2 if reward > 0 else 0.8  # Bonus for successful assignment
        
        if action_key in self.q_table[state_key]:
            old_value = self.q_table[state_key][action_key]
            self.q_table[state_key][action_key] = old_value + self.config.learning_rate * (
                reward - old_value
            )
        else:
            self.q_table[state_key][action_key] = reward
        
        # Update historical performance
        key = f"{symbol}_{delta}_{context.market_regime}"
        if key not in self.model['delta_performance']:
            self.model['delta_performance'][key] = []
        
        self.model['delta_performance'][key].append(actual_pnl)
        
        # Keep only recent performance (last 100 trades)
        if len(self.model['delta_performance'][key]) > 100:
            self.model['delta_performance'][key] = self.model['delta_performance'][key][-100:]
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'delta': delta,
            'symbol': symbol,
            'regime': context.market_regime,
            'pnl': actual_pnl,
            'assignment': actual_assignment
        })
        
        # Limit history size
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        logger.info(f"Updated performance for delta {delta}: P&L = {actual_pnl:.3f}")
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if self.last_retrain is None:
            return True
        
        days_since_retrain = (datetime.now() - self.last_retrain).days
        if days_since_retrain >= self.config.retrain_interval_days:
            return True
        
        if len(self.performance_history) >= self.config.min_training_samples:
            return True
        
        return False
    
    def save_model(self):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'q_table': self.q_table,
            'performance_history': self.performance_history,
            'last_retrain': datetime.now(),
            'config': self.config
        }
        
        with open(self.config.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"ML Delta optimizer model saved to {self.config.model_path}")
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights about optimization performance."""
        
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        recent_performance = self.performance_history[-100:]  # Last 100 trades
        
        avg_pnl = np.mean([p['pnl'] for p in recent_performance])
        win_rate = sum(1 for p in recent_performance if p['pnl'] > 0) / len(recent_performance)
        
        # Best performing deltas
        delta_performance = {}
        for p in recent_performance:
            delta = p['delta']
            if delta not in delta_performance:
                delta_performance[delta] = []
            delta_performance[delta].append(p['pnl'])
        
        best_deltas = sorted(
            [(delta, np.mean(perfs)) for delta, perfs in delta_performance.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        return {
            "average_pnl": avg_pnl,
            "win_rate": win_rate,
            "total_trades": len(self.performance_history),
            "best_performing_deltas": best_deltas,
            "last_retrain": self.last_retrain,
            "exploration_rate": self.config.exploration_rate
        }