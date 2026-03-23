"""
ML-based DTE (Days to Expiration) Optimizer for Options Strategies.

This module provides intelligent DTE selection using machine learning models
that optimize for risk-adjusted returns across different market conditions.

Supports three strategy phases:
- "SP": Sell Put phase (standard)
- "CC": Covered Call phase (standard)
- "CC+SP": Simultaneous mode (binbingod策略优化 - CC阶段可同时开SP)
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

logger = logging.getLogger("dte_optimizer")


@dataclass
class DTEOptimizationConfig:
    """Configuration for ML DTE optimization."""
    # Model training parameters
    model_path: str = "data/models/dte_optimizer.pkl"
    retrain_interval_days: int = 30
    min_training_samples: int = 1000
    persist_model: bool = False  # Disabled - each backtest starts fresh
    
    # Optimization parameters
    exploration_rate: float = 0.1  # 10% exploration
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    
    # DTE search space
    dte_min: int = 21
    dte_max: int = 60
    dte_step: int = 5  # 5-day intervals
    
    # Risk parameters
    max_loss_threshold: float = 0.20  # 20% max loss per trade
    min_premium_threshold: float = 0.01  # 1% minimum premium


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
    market_regime: str  # "bull", "bear", "neutral", "high_vol"
    days_to_earnings: int
    option_liquidity: float  # 0-1 scale
    strategy_phase: str  # "SP" for Sell Put, "CC" for Covered Call, "CC+SP" for simultaneous mode


@dataclass
class DTEOptimizationResult:
    """Result of DTE optimization."""
    optimal_dte_min: int
    optimal_dte_max: int
    expected_premium: float
    expected_probability_assignment: float
    risk_score: float
    confidence: float  # 0-1 scale
    reasoning: str


class DTEOptimizerML:
    """
    Machine Learning DTE Optimizer for Options Strategies.
    
    Uses Q-learning and prediction models to optimize DTE selection
    based on market conditions, risk tolerance, and historical performance.
    """

    def __init__(self, config: DTEOptimizationConfig):
        self.config = config
        self.model = None
        self.q_table = {}
        self.performance_history = []
        self.last_retrain = None
        # Set random seed for reproducible results
        np.random.seed(42)  # Fixed seed for reproducibility
        self._load_model()

    def _load_model(self):
        """Load existing model or initialize new one.
        
        Note: Model persistence is disabled by default (persist_model=False).
        Each backtest starts with a fresh model for consistent results.
        """
        # Check if model persistence is enabled
        if not getattr(self.config, 'persist_model', False):
            logger.info("Model persistence disabled - initializing fresh model")
            self._initialize_model()
            return
        
        if os.path.exists(self.config.model_path):
            try:
                with open(self.config.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.q_table = data.get('q_table', {})
                    self.performance_history = data.get('performance_history', [])
                    self.last_retrain = data.get('last_retrain')
                logger.info("ML DTE optimizer model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self._initialize_model()
        else:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize ML model with default values."""
        self.model = {
            'dte_performance': {},  # dte -> historical performance
            'market_regime_patterns': {},  # regime -> optimal dte range
            'symbol_specific': {},  # symbol -> custom dte preferences
            'strategy_phase_preferences': {}  # phase -> optimal dte range
        }
        self.q_table = {}
        logger.info("Initialized new ML DTE optimizer")

    def pretrain_with_history(self, 
                              symbol: str,
                              historical_bars: List[Dict],
                              iv_estimate: float = 0.25,
                              right: str = "P",
                              strategy_phase: str = "SP",
                              training_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Pretrain the model using historical price data.
        
        This simulates trades across different market conditions and DTE values
        to build initial Q-table knowledge before live trading.
        
        Args:
            symbol: Stock symbol
            historical_bars: List of historical price bars with 'date', 'close', 'high', 'low'
            iv_estimate: Estimated implied volatility for option pricing
            right: Option type ('P' for put, 'C' for call)
            strategy_phase: Strategy phase ('SP' for Sell Put, 'CC' for Covered Call)
            training_ratio: Fraction of data to use for training (0.0-1.0)
            
        Returns:
            Dict with training statistics
        """
        if len(historical_bars) < 60:
            logger.warning(f"Insufficient history for pretraining: {len(historical_bars)} bars")
            return {"status": "skipped", "reason": "insufficient_data"}

        logger.info(f"Starting DTE pretraining for {symbol} with {len(historical_bars)} bars...")

        # Use portion of data for training
        train_size = int(len(historical_bars) * training_ratio)
        train_bars = historical_bars[-train_size:]

        # Candidate DTEs to test
        candidate_dtes = list(range(self.config.dte_min, self.config.dte_max + 1, self.config.dte_step))

        # Context for each simulation
        stats = {
            "total_simulations": 0,
            "regimes_tested": set(),
            "phases_tested": set(),
            "best_dte_by_regime": {}
        }

        # Slide through historical data simulating trades
        for i in range(len(train_bars) - self.config.dte_max - 1):
            entry_bar = train_bars[i]

            # Test each DTE value
            for dte in candidate_dtes:
                if i + dte >= len(train_bars):
                    continue  # Skip if not enough data to simulate

                exit_bar = train_bars[i + dte]
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
                stats["phases_tested"].add(strategy_phase)

                # Create market context
                context = DTEMarketContext(
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
                    option_liquidity=0.5,
                    strategy_phase=strategy_phase
                )

                # Simulate trade outcome
                pnl, assigned = self._simulate_option_trade(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    dte=dte,
                    right=right,
                    iv=iv_estimate,
                    strategy_phase=strategy_phase
                )

                # Update Q-table with simulated result
                self.update_performance(
                    dte=dte,
                    symbol=symbol,
                    context=context,
                    actual_pnl=pnl,
                    actual_assignment=assigned
                )

                stats["total_simulations"] += 1

        # Convert sets to lists for JSON serialization
        stats["regimes_tested"] = list(stats["regimes_tested"])
        stats["phases_tested"] = list(stats["phases_tested"])

        # Calculate best DTE by regime from accumulated data
        for regime in stats["regimes_tested"]:
            best_dte, avg_pnl = self._find_best_dte_for_regime(symbol, regime, strategy_phase)
            stats["best_dte_by_regime"][regime] = {
                "dte": best_dte,
                "avg_pnl": avg_pnl
            }

        self.last_retrain = datetime.now()
        logger.info(f"DTE pretraining complete: {stats['total_simulations']} simulations, "
                   f"regimes: {stats['regimes_tested']}, phases: {stats['phases_tested']}")

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
                               dte: int,
                               right: str,
                               iv: float,
                               strategy_phase: str) -> Tuple[float, bool]:
        """
        Simulate an option trade outcome.
        
        Returns:
            Tuple of (pnl, was_assigned)
        """
        T = dte / 365.0
        
        # Calculate delta based on market conditions and strategy phase
        if strategy_phase == "SP":  # Sell Put
            delta = self._calculate_put_delta(entry_price, iv, T)
        else:  # Covered Call
            delta = self._calculate_call_delta(entry_price, iv, T)
        
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

    def _calculate_put_delta(self, price: float, iv: float, T: float) -> float:
        """Calculate typical put delta based on market conditions."""
        # Use a default delta that's typically used for Sell Put strategies
        base_delta = 0.30
        
        # Adjust based on volatility and time to expiration
        vol_adjustment = max(0.1, min(0.4, base_delta + (iv - 0.25) * 0.1))
        time_adjustment = max(0.1, min(0.4, base_delta + (T * 365 - 30) * 0.001))
        
        return (vol_adjustment + time_adjustment) / 2

    def _calculate_call_delta(self, price: float, iv: float, T: float) -> float:
        """Calculate typical call delta based on market conditions."""
        # Use a default delta that's typically used for Covered Call strategies
        base_delta = 0.30
        
        # Adjust based on volatility and time to expiration
        vol_adjustment = max(0.1, min(0.4, base_delta + (iv - 0.25) * 0.1))
        time_adjustment = max(0.1, min(0.4, base_delta + (T * 365 - 30) * 0.001))
        
        return (vol_adjustment + time_adjustment) / 2

    def _find_best_dte_for_regime(self, symbol: str, regime: str, strategy_phase: str) -> Tuple[int, float]:
        """Find the best performing DTE for a given regime and strategy phase."""
        dte_performance = {}

        for key, performances in self.model['dte_performance'].items():
            if f"{symbol}_" in key and regime in key and strategy_phase in key:
                # Extract DTE from key
                parts = key.split('_')
                if len(parts) >= 2:
                    try:
                        dte = int(parts[1])
                        avg_pnl = np.mean(performances) if performances else 0
                        dte_performance[dte] = avg_pnl
                    except (ValueError, IndexError):
                        continue

        if dte_performance:
            best_dte = max(dte_performance, key=dte_performance.get)
            return best_dte, dte_performance[best_dte]

        return 30, 0.0  # Default DTE

    def extract_market_context(self, 
                             symbol: str,
                             current_price: float,
                             cost_basis: float,
                             bars: List[Dict],
                             options_data: List[Dict],
                             fundamentals: Dict = None,
                             strategy_phase: str = "SP") -> DTEMarketContext:
        """Extract market context features from data."""

        if len(bars) < 30:
            # Fallback to simplified market context when insufficient data
            # This is a normal condition early in backtest, use debug level
            logger.debug(f"Insufficient bars ({len(bars)}), using simplified market context")
            return self._create_simplified_context(symbol, current_price, cost_basis, strategy_phase)

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

        return DTEMarketContext(
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
            option_liquidity=option_liquidity,
            strategy_phase=strategy_phase
        )

    def _create_simplified_context(self, symbol: str, current_price: float, cost_basis: float, strategy_phase: str) -> DTEMarketContext:
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

        return DTEMarketContext(
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
            option_liquidity=0.5,  # Moderate liquidity
            strategy_phase=strategy_phase
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

    def optimize_dte_range(self, 
                          market_context: DTEMarketContext,
                          right: str,  # "P" or "C"
                          iv: float,
                          current_positions: List[Dict] = None) -> DTEOptimizationResult:
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
        optimal_dte, confidence, reasoning = self._select_optimal_dte(dte_scores, market_context)
        
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
        
        # Calculate a reasonable DTE range around the optimal DTE
        # Typically, a range of ±5-10 days around optimal
        dte_range_buffer = max(5, self.config.dte_step)
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

    def _score_dte_candidate(self, 
                             dte: int,
                             context: DTEMarketContext,
                             right: str,
                             iv: float,
                             positions: List[Dict] = None) -> float:
        """Score a DTE candidate using multiple factors."""
        
        score = 0.0
        
        # 1. Historical performance for this DTE
        hist_score = self._get_historical_score(dte, context.symbol, context.market_regime, context.strategy_phase)
        score += hist_score * 0.3
        
        # 2. Market regime suitability
        regime_score = self._get_regime_suitability(dte, context.market_regime, right, context.strategy_phase)
        score += regime_score * 0.25
        
        # 3. Risk-adjusted return
        risk_score = self._calculate_risk_adjusted_return(dte, context, right, iv)
        score += risk_score * 0.25
        
        # 4. Strategy phase optimization
        phase_score = self._optimize_for_strategy_phase(dte, context.strategy_phase, context)
        score += phase_score * 0.2
        
        return score

    def _get_historical_score(self, dte: int, symbol: str, regime: str, strategy_phase: str) -> float:
        """Get historical performance score for a DTE value."""
        key = f"{symbol}_{dte}_{regime}_{strategy_phase}"
        
        if key in self.model['dte_performance']:
            performance = self.model['dte_performance'][key]
            # Return normalized performance (0-1 scale)
            return min(1.0, max(0.0, np.mean(performance) + 0.5))  # Shift to positive range
        else:
            # Return base score for unseen dte-regime-phase combinations
            return 0.5  # Neutral baseline

    def _get_regime_suitability(self, dte: int, regime: str, right: str, strategy_phase: str) -> float:
        """Get suitability score for current market regime and strategy phase."""
        
        # Different regimes and strategy phases have different optimal DTE ranges
        regime_patterns = {
            'bull': {
                'SP': 25,  # In bull market, shorter DTE for sell puts (less risk)
                'CC': 40   # In bull market, longer DTE for covered calls (more premium)
            },
            'bear': {
                'SP': 40,  # In bear market, longer DTE for sell puts (more premium)
                'CC': 25   # In bear market, shorter DTE for covered calls (less assignment risk)
            },
            'neutral': {
                'SP': 30,  # Standard DTE for neutral market
                'CC': 35   # Standard DTE for neutral market
            },
            'high_vol': {
                'SP': 21,  # Very short DTE in high volatility for rapid premium collection
                'CC': 25   # Short DTE in high volatility for covered calls
            }
        }
        
        optimal_dte = regime_patterns.get(regime, {}).get(strategy_phase, 30)
        
        # Score based on distance from optimal DTE
        distance = abs(dte - optimal_dte)
        return max(0.0, 1.0 - distance / 20.0)  # Normalize to 0-1

    def _calculate_risk_adjusted_return(self, 
                                       dte: int,
                                       context: DTEMarketContext,
                                       right: str,
                                       iv: float) -> float:
        """Calculate risk-adjusted return score for DTE candidate."""
        
        T = dte / 365.0
        
        # Calculate a typical delta for this strategy
        if right == "P":
            delta = 0.30  # Typical put delta
        else:
            delta = 0.30  # Typical call delta
            
        # Estimate premium for this DTE and delta
        if right == "P":
            strike = context.current_price * (1 - delta)  # Approximate
            premium = OptionsPricer.put_price(context.current_price, strike, T, iv)
        else:
            strike = context.current_price * (1 + delta)  # Approximate
            premium = OptionsPricer.call_price(context.current_price, strike, T, iv)
        
        # Calculate risk metrics
        assignment_risk = 0.0
        time_decay_factor = min(1.0, T * 365 / 45)  # Time decay benefit
        
        if right == "P" and context.cost_basis > 0:
            # For puts: risk if assigned at strike below cost
            assignment_risk = max(0, context.cost_basis - strike) / context.cost_basis
        elif right == "C" and context.cost_basis > 0:
            # For calls: risk if assigned at strike above cost (missing out on gains)
            assignment_risk = max(0, strike - context.cost_basis) / context.cost_basis
        
        # Risk-adjusted score: prefer higher premiums with lower risk and good time decay
        risk_score = premium * (1.0 + time_decay_factor) * (1.0 - assignment_risk)
        return min(1.0, risk_score / 0.1)  # Normalize to 0-1

    def _optimize_for_strategy_phase(self, dte: int, strategy_phase: str, context: DTEMarketContext) -> float:
        """Optimize DTE based on strategy phase.

        CC+SP模式使用SP的逻辑，因为在CC阶段开SP时实际是卖Put。
        """

        # Different strategy phases have different optimal DTE preferences
        if strategy_phase == "SP" or strategy_phase == "CC+SP":  # Sell Put (包括CC+SP模式)
            # Prefer moderate DTE to balance premium collection and assignment risk
            if 25 <= dte <= 40:
                return 1.0
            elif 21 <= dte <= 45:
                return 0.8
            else:
                return 0.3
        elif strategy_phase == "CC":  # Covered Call
            # Prefer moderate to longer DTE to maximize premium while managing assignment risk
            if 30 <= dte <= 45:
                return 1.0
            elif 25 <= dte <= 50:
                return 0.8
            else:
                return 0.3
        else:
            return 0.5  # Neutral for unknown phases

    def _select_optimal_dte(self, 
                           dte_scores: List[Tuple[int, float]], 
                           context: DTEMarketContext) -> Tuple[int, float, str]:
        """Select optimal DTE with exploration mechanism."""
        
        # Sort by score
        dte_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Exploration: occasionally try random DTE
        if np.random.random() < self.config.exploration_rate:
            selected_dte = np.random.choice([d[0] for d in dte_scores])
            confidence = 0.5  # Lower confidence for exploration
            reasoning = f"Exploring DTE {selected_dte} for learning purposes in {context.strategy_phase} phase"
        else:
            # Select best DTE
            selected_dte = dte_scores[0][0]
            # Scale confidence with minimum baseline of 0.7 for consistent results
            confidence = max(0.7, min(1.0, dte_scores[0][1] * 2))
            
            reasoning = f"Optimal DTE {selected_dte} based on {context.market_regime} market regime, {context.strategy_phase} phase, and {context.symbol} characteristics"

        return selected_dte, confidence, reasoning

    def _estimate_expected_premium(self, 
                                   dte: int,
                                   context: DTEMarketContext,
                                   right: str,
                                   iv: float) -> float:
        """Estimate expected premium for selected DTE."""
        
        T = dte / 365.0
        
        # Calculate a typical delta for this strategy
        if right == "P":
            delta = 0.30  # Typical put delta
        else:
            delta = 0.30  # Typical call delta
            
        if right == "P":
            strike = context.current_price * (1 - delta)
            premium = OptionsPricer.put_price(context.current_price, strike, T, iv)
        else:
            strike = context.current_price * (1 + delta)
            premium = OptionsPricer.call_price(context.current_price, strike, T, iv)
        
        return premium

    def _estimate_assignment_probability(self,
                                         dte: int,
                                         context: DTEMarketContext,
                                         right: str,
                                         iv: float) -> float:
        """Estimate probability of option assignment using optionlab.

        Uses Black-Scholes ITM probability with market regime adjustments.
        """
        if dte <= 0 or iv <= 0:
            return 0.0

        try:
            # Calculate strike from typical delta
            delta = 0.30  # Typical delta for strategy
            if right == "P":
                strike = context.current_price * (1 - delta)  # OTM put
            else:
                strike = context.current_price * (1 + delta)  # OTM call

            T = dte / 365.0

            # Use optionlab's ITM probability
            itm_prob = OptionsPricer.itm_probability(
                S=context.current_price,
                K=strike,
                T=T,
                sigma=iv,
                right=right,
            )

            # Adjust for market regime (empirical adjustments)
            regime_adjustment = {
                'bull': 1.2 if right == "C" else 0.8,  # Higher call assignment in bull
                'bear': 1.2 if right == "P" else 0.8,  # Higher put assignment in bear
                'neutral': 1.0,
                'high_vol': 0.9  # Slightly lower assignment in high volatility
            }

            # Adjust for strategy phase
            phase_adjustment = 1.0
            if (context.strategy_phase == "SP" or context.strategy_phase == "CC+SP") and right == "P":
                phase_adjustment = 1.0  # Normal assignment for Sell Put (包括CC+SP模式)
            elif context.strategy_phase == "CC" and right == "C":
                phase_adjustment = 1.1  # Slightly higher assignment risk for Covered Call

            adjusted_prob = itm_prob * regime_adjustment.get(context.market_regime, 1.0) * phase_adjustment
            return min(1.0, adjusted_prob)
        except Exception as e:
            logger.debug(f"ITM probability calculation failed: {e}")
            # Fallback to delta approximation
            return min(1.0, 0.30)

    def _calculate_risk_score(self, 
                              dte: int,
                              context: DTEMarketContext,
                              right: str) -> float:
        """Calculate risk score (0-1, where 1 is highest risk)."""
        
        # Base risk on DTE - longer DTE has more uncertainty
        dte_risk = min(1.0, (dte - self.config.dte_min) / (self.config.dte_max - self.config.dte_min))
        
        # Adjust for market conditions
        if context.market_regime == "high_vol":
            dte_risk *= 1.3  # Higher risk in high volatility regardless of DTE
        elif context.market_regime == "bear" and right == "P":
            dte_risk *= 1.1  # Higher put assignment risk in bear market
        elif context.market_regime == "bull" and right == "C":
            dte_risk *= 1.1  # Higher call assignment risk in bull market
        
        # Adjust for strategy phase
        if context.strategy_phase == "CC" and right == "C":
            # Covered calls have assignment risk that may be desired
            dte_risk *= 0.8  # Slightly reduce risk for covered calls
        elif (context.strategy_phase == "SP" or context.strategy_phase == "CC+SP") and right == "P":
            # Sell puts have assignment risk that may be desired (包括CC+SP模式)
            dte_risk *= 0.9  # Slightly reduce risk for sell puts

        # Adjust for liquidity
        dte_risk *= (2.0 - context.option_liquidity)  # Lower liquidity = higher risk
        
        return min(1.0, dte_risk)

    def update_performance(self, 
                          dte: int,
                          symbol: str,
                          context: DTEMarketContext,
                          actual_pnl: float,
                          actual_assignment: bool):
        """Update performance metrics for learning."""
        
        # Update Q-table
        state_key = f"{symbol}_{context.market_regime}_{context.strategy_phase}_{int(context.current_price)}"
        action_key = f"dte_{dte}"
        
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
        key = f"{symbol}_{dte}_{context.market_regime}_{context.strategy_phase}"
        if key not in self.model['dte_performance']:
            self.model['dte_performance'][key] = []
        
        self.model['dte_performance'][key].append(actual_pnl)
        
        # Keep only recent performance (last 100 trades)
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
        
        # Limit history size
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

        logger.info(f"Updated performance for DTE {dte}: P&L = {actual_pnl:.3f}")

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
        """Save trained model to disk.
        
        Note: Model persistence is disabled by default (persist_model=False).
        Override by setting config.persist_model=True to enable saving.
        """
        # Check if model persistence is enabled
        if not getattr(self.config, 'persist_model', False):
            logger.debug("Model persistence disabled - skipping save")
            return
        
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

        logger.info(f"ML DTE optimizer model saved to {self.config.model_path}")

    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights about optimization performance."""
        
        if not self.performance_history:
            return {"message": "No performance data available"}

        recent_performance = self.performance_history[-100:]  # Last 100 trades

        avg_pnl = np.mean([p['pnl'] for p in recent_performance])
        win_rate = sum(1 for p in recent_performance if p['pnl'] > 0) / len(recent_performance)

        # Best performing DTEs
        dte_performance = {}
        for p in recent_performance:
            dte = p['dte']
            if dte not in dte_performance:
                dte_performance[dte] = []
            dte_performance[dte].append(p['pnl'])

        best_dtes = sorted(
            [(dte, np.mean(perfs)) for dte, perfs in dte_performance.items()],
            key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "average_pnl": avg_pnl,
            "win_rate": win_rate,
            "total_trades": len(self.performance_history),
            "best_performing_dtes": best_dtes,
            "last_retrain": self.last_retrain,
            "exploration_rate": self.config.exploration_rate
        }