"""
ML Position Size Optimizer for QuantConnect
============================================

Machine learning model that optimizes position size based on:
- Market conditions (IV Rank, VIX, volatility regime)
- Strategy phase (SP/CC/CC+SP simultaneous)
- Portfolio state (drawdown, margin utilization)
- Historical performance patterns

binbingod策略优化: 支持CC阶段同时开SP
- strategy_phase: "SP" = 卖Put阶段
                  "CC" = 卖Covered Call阶段
                  "CC+SP" = 同时操作模式

Output: Optimal number of contracts and risk-adjusted position multiplier.
"""

from AlgorithmImports import *
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class PositionRecommendation:
    """ML recommendation for position sizing."""
    num_contracts: int
    confidence: float
    position_multiplier: float
    expected_return: float
    expected_risk: float
    kelly_fraction: float
    reasoning: str
    var_95: float
    max_loss_scenario: float


class MLPositionOptimizer:
    """
    Machine learning model for optimizing position sizing in Wheel strategy.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.is_trained = False
        
        self.feature_names = [
            'iv_rank', 'iv_percentile', 'historical_volatility',
            'vix', 'vix_percentile', 'vix_rank', 'vix_change_pct', 'vix_term_structure',
            'market_regime',
            'available_margin_pct', 'margin_utilization', 'current_drawdown', 'portfolio_concentration',
            'strategy_phase',
            'strike_distance_pct', 'dte', 'delta', 'premium', 'premium_yield', 'theta_per_day',
            'momentum_5d', 'momentum_10d', 'vs_ma20', 'vs_ma50',
            'assignment_probability', 'break_even_distance_pct',
        ]
        
        # Risk management parameters
        self.max_position_multiplier = 1.5
        self.min_position_multiplier = 0.3
        self.kelly_fraction = 0.25
    
    def build_features(
        self,
        market_data: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        strategy_phase: str,
        option_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Build feature vector for position sizing decision."""
        
        features = {}
        
        # === Volatility Features ===
        iv = market_data.get('iv', 0.3)
        features['iv_rank'] = market_data.get('iv_rank', 50)
        features['iv_percentile'] = market_data.get('iv_percentile', 50)
        features['historical_volatility'] = market_data.get('historical_volatility', iv * 100)
        
        # === VIX Features ===
        vix_data = market_data.get('vix', 20)
        if isinstance(vix_data, dict):
            features['vix'] = vix_data.get('vix', 20)
            features['vix_percentile'] = vix_data.get('vix_percentile', 50)
            features['vix_rank'] = vix_data.get('vix_rank', 50)
            features['vix_change_pct'] = vix_data.get('vix_change_pct', 0)
            features['vix_term_structure'] = vix_data.get('vix_term_structure', 0)
        else:
            features['vix'] = vix_data
            features['vix_percentile'] = 50
            features['vix_rank'] = 50
            features['vix_change_pct'] = 0
            features['vix_term_structure'] = 0
        
        # === Market Regime ===
        if features['iv_rank'] < 20 and features['vix'] < 15:
            features['market_regime'] = 0
        elif features['iv_rank'] > 50 or features['vix'] > 25:
            features['market_regime'] = 2
        else:
            features['market_regime'] = 1
        
        # === Portfolio State ===
        total_capital = portfolio_state.get('total_capital', 100000)
        available_margin = portfolio_state.get('available_margin', total_capital)
        margin_used = portfolio_state.get('margin_used', 0)
        
        features['available_margin_pct'] = available_margin / total_capital if total_capital > 0 else 0
        features['margin_utilization'] = margin_used / total_capital if total_capital > 0 else 0
        features['current_drawdown'] = portfolio_state.get('drawdown', 0)
        
        # Portfolio concentration
        positions = portfolio_state.get('positions', [])
        if positions:
            position_values = [abs(p.get('market_value', 0)) for p in positions]
            total_value = sum(position_values)
            if total_value > 0:
                weights = [v / total_value for v in position_values]
                features['portfolio_concentration'] = sum(w ** 2 for w in weights)
            else:
                features['portfolio_concentration'] = 0
        else:
            features['portfolio_concentration'] = 0
        
        # === Strategy Phase ===
        if strategy_phase == "SP":
            features['strategy_phase'] = 0
        elif strategy_phase == "CC":
            features['strategy_phase'] = 1
        else:  # CC+SP simultaneous mode
            features['strategy_phase'] = 2
        
        # === Option Features ===
        underlying_price = option_info.get('underlying_price', market_data.get('price', 100))
        strike = option_info.get('strike', underlying_price * 0.95)
        delta = option_info.get('delta', 0.3)
        premium = option_info.get('premium', 1.0)
        dte = option_info.get('dte', 30)
        
        # Strike distance
        if strategy_phase == "SP" or strategy_phase == "CC+SP":
            features['strike_distance_pct'] = (underlying_price - strike) / underlying_price * 100
        else:
            features['strike_distance_pct'] = (strike - underlying_price) / underlying_price * 100
        
        features['dte'] = dte
        features['delta'] = abs(delta)
        features['premium'] = premium
        
        # Premium yield
        if strategy_phase == "SP" or strategy_phase == "CC+SP":
            margin_required = strike * 100
        else:
            margin_required = 0
        features['premium_yield'] = premium / margin_required * 100 if margin_required > 0 else premium * 100
        features['theta_per_day'] = premium / dte if dte > 0 else 0
        
        # === Momentum Features ===
        momentum = market_data.get('momentum', {})
        features['momentum_5d'] = momentum.get('momentum_5d', 0)
        features['momentum_10d'] = momentum.get('momentum_10d', 0)
        features['vs_ma20'] = momentum.get('vs_ma20', 0)
        features['vs_ma50'] = momentum.get('vs_ma50', 0)
        
        # === Risk Features ===
        features['assignment_probability'] = self._estimate_assignment_probability(
            underlying_price, strike, dte, iv,
            'P' if strategy_phase in ("SP", "CC+SP") else 'C'
        )
        
        if strategy_phase == "SP" or strategy_phase == "CC+SP":
            breakeven = strike - premium
            features['break_even_distance_pct'] = (underlying_price - breakeven) / underlying_price * 100
        else:
            cost_basis = portfolio_state.get('cost_basis', underlying_price)
            features['break_even_distance_pct'] = (underlying_price - cost_basis) / underlying_price * 100
        
        return features
    
    def _estimate_assignment_probability(
        self,
        underlying_price: float,
        strike: float,
        dte: int,
        iv: float,
        right: str = 'P'
    ) -> float:
        """Estimate probability of assignment using simplified approximation."""
        
        if dte <= 0 or iv <= 0:
            return 0.0
        
        T = dte / 365.0
        
        try:
            # Simplified ITM probability approximation
            moneyness = strike / underlying_price
            delta_approx = abs(underlying_price - strike) / underlying_price
            
            # Adjust for time and volatility
            time_factor = np.sqrt(T)
            vol_factor = iv * time_factor
            
            if right == 'P':
                # Put ITM when price < strike
                itm_prob = delta_approx * (1 + vol_factor) if moneyness > 1 else delta_approx * 0.5
            else:
                # Call ITM when price > strike
                itm_prob = delta_approx * (1 + vol_factor) if moneyness < 1 else delta_approx * 0.5
            
            return float(max(0.05, min(0.95, itm_prob)))
        except Exception:
            delta_approx = abs(underlying_price - strike) / underlying_price
            return max(0, min(1, delta_approx))
    
    def predict_position_size(
        self,
        market_data: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        strategy_phase: str,
        option_info: Dict[str, Any],
        base_position: int = 1,
        max_position: int = 10
    ) -> PositionRecommendation:
        """
        Predict optimal position size using ML model.
        
        Args:
            market_data: Market data
            portfolio_state: Portfolio state
            strategy_phase: "SP", "CC", or "CC+SP"
            option_info: Option details
            base_position: Base position size
            max_position: Maximum position size
            
        Returns:
            PositionRecommendation with optimal size and risk metrics
        """
        
        features = self.build_features(market_data, portfolio_state, strategy_phase, option_info)
        
        if self.model is not None and self.is_trained:
            return self._ml_predict(features, base_position, max_position, strategy_phase)
        else:
            return self._rule_based_predict(features, base_position, max_position, strategy_phase)
    
    def _rule_based_predict(
        self,
        features: Dict[str, float],
        base_position: int,
        max_position: int,
        strategy_phase: str
    ) -> PositionRecommendation:
        """Rule-based prediction (fallback when model not trained)."""
        
        multiplier = 1.0
        reasoning_parts = []
        
        # === IV-based adjustment ===
        iv_rank = features['iv_rank']
        if iv_rank > 70:
            multiplier *= 1.3
            reasoning_parts.append(f"High IV ({iv_rank:.0f}): +30% position")
        elif iv_rank > 50:
            multiplier *= 1.15
            reasoning_parts.append(f"Elevated IV ({iv_rank:.0f}): +15% position")
        elif iv_rank < 30:
            multiplier *= 0.7
            reasoning_parts.append(f"Low IV ({iv_rank:.0f}): -30% position")
        
        # === VIX-based adjustment ===
        vix = features['vix']
        if vix > 30:
            multiplier *= 0.6
            reasoning_parts.append(f"High VIX ({vix:.1f}): -40% position")
        elif vix > 25:
            multiplier *= 0.8
            reasoning_parts.append(f"Elevated VIX ({vix:.1f}): -20% position")
        elif vix < 15:
            multiplier *= 1.2
            reasoning_parts.append(f"Low VIX ({vix:.1f}): +20% position")
        
        # === Drawdown adjustment ===
        drawdown = features['current_drawdown']
        if drawdown > 15:
            multiplier *= 0.5
            reasoning_parts.append(f"Large drawdown ({drawdown:.1f}%): -50% position")
        elif drawdown > 10:
            multiplier *= 0.7
            reasoning_parts.append(f"Moderate drawdown ({drawdown:.1f}%): -30% position")
        elif drawdown > 5:
            multiplier *= 0.85
            reasoning_parts.append(f"Small drawdown ({drawdown:.1f}%): -15% position")
        
        # === Margin utilization adjustment ===
        margin_util = features['margin_utilization']
        if margin_util > 0.7:
            multiplier *= 0.6
            reasoning_parts.append(f"High margin use ({margin_util*100:.0f}%): -40% position")
        elif margin_util > 0.5:
            multiplier *= 0.8
            reasoning_parts.append(f"Moderate margin ({margin_util*100:.0f}%): -20% position")
        
        # === Strategy phase adjustment ===
        if strategy_phase == "SP":
            multiplier *= 0.9
            reasoning_parts.append("SP phase: -10% (cash reserve)")
        elif strategy_phase == "CC":
            reasoning_parts.append("CC phase: standard (stock collateral)")
        else:  # CC+SP simultaneous mode
            multiplier *= 0.85
            reasoning_parts.append("CC+SP simultaneous: -15% (dual margin usage)")
        
        # === Assignment probability adjustment ===
        assign_prob = features['assignment_probability']
        if assign_prob > 0.5:
            multiplier *= 0.8
            reasoning_parts.append(f"High assignment risk ({assign_prob*100:.0f}%): -20%")
        
        # Clamp multiplier
        multiplier = max(self.min_position_multiplier, 
                        min(self.max_position_multiplier, multiplier))
        
        # Calculate position
        num_contracts = max(1, min(int(base_position * multiplier), max_position))
        
        # Calculate metrics
        expected_return = self._calculate_expected_return(features, strategy_phase)
        expected_risk = self._calculate_expected_risk(features, strategy_phase)
        kelly = self._calculate_kelly_fraction(features, strategy_phase)
        var_95 = self._calculate_var(features, strategy_phase, num_contracts)
        max_loss = self._calculate_max_loss(features, strategy_phase, num_contracts)
        
        # Confidence
        confidence = 0.5
        if iv_rank > 60 or iv_rank < 30:
            confidence += 0.1
        if vix > 25 or vix < 15:
            confidence += 0.1
        
        return PositionRecommendation(
            num_contracts=num_contracts,
            confidence=min(confidence, 0.8),
            position_multiplier=multiplier,
            expected_return=expected_return * num_contracts,
            expected_risk=expected_risk * num_contracts,
            kelly_fraction=kelly,
            reasoning=" | ".join(reasoning_parts) if reasoning_parts else "Standard position sizing",
            var_95=var_95,
            max_loss_scenario=max_loss
        )
    
    def _ml_predict(
        self,
        features: Dict[str, float],
        base_position: int,
        max_position: int,
        strategy_phase: str
    ) -> PositionRecommendation:
        """ML-based prediction (when model is trained)."""
        
        try:
            import pandas as pd
            
            X = pd.DataFrame([features])[self.feature_names]
            multiplier = self.model.predict(X)[0]
            multiplier = max(self.min_position_multiplier,
                           min(self.max_position_multiplier, multiplier))
            
            num_contracts = max(1, min(int(base_position * multiplier), max_position))
            
            # Calculate metrics
            expected_return = self._calculate_expected_return(features, strategy_phase)
            expected_risk = self._calculate_expected_risk(features, strategy_phase)
            kelly = self._calculate_kelly_fraction(features, strategy_phase)
            var_95 = self._calculate_var(features, strategy_phase, num_contracts)
            max_loss = self._calculate_max_loss(features, strategy_phase, num_contracts)
            
            confidence = 1 - abs(multiplier - 1) / (self.max_position_multiplier - 1)
            
            return PositionRecommendation(
                num_contracts=num_contracts,
                confidence=confidence,
                position_multiplier=multiplier,
                expected_return=expected_return * num_contracts,
                expected_risk=expected_risk * num_contracts,
                kelly_fraction=kelly,
                reasoning=f"ML optimized: {multiplier:.2f}x base position",
                var_95=var_95,
                max_loss_scenario=max_loss
            )
            
        except Exception as e:
            return self._rule_based_predict(features, base_position, max_position, strategy_phase)
    
    def _calculate_expected_return(self, features: Dict[str, float], strategy_phase: str) -> float:
        """Calculate expected return per contract."""
        
        premium = features['premium']
        assignment_prob = features['assignment_probability']
        
        expected_return = premium * 100 * (1 - assignment_prob * 0.5)
        return expected_return
    
    def _calculate_expected_risk(self, features: Dict[str, float], strategy_phase: str) -> float:
        """Calculate expected maximum drawdown per contract."""
        
        strike_distance = features['strike_distance_pct'] / 100
        strike = 100  # Approximate
        
        expected_risk = strike * 100 * strike_distance
        return expected_risk
    
    def _calculate_kelly_fraction(self, features: Dict[str, float], strategy_phase: str) -> float:
        """Calculate Kelly optimal fraction."""
        
        iv_rank = features['iv_rank']
        delta = features['delta']
        
        win_prob = 1 - delta
        premium = features['premium']
        
        max_loss = features['strike_distance_pct'] / 100 * 100 if features['strike_distance_pct'] > 0 else premium * 2
        win_loss_ratio = premium / max_loss if max_loss > 0 else 1
        
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio if win_loss_ratio > 0 else 0
        
        return max(0, min(kelly * 0.25, 0.25))
    
    def _calculate_var(self, features: Dict[str, float], strategy_phase: str, num_contracts: int) -> float:
        """Calculate Value at Risk (95% confidence)."""
        
        iv = features['historical_volatility'] / 100
        strike = 100  # Approximate
        
        position_value = strike * 100 * num_contracts
        var_95 = 1.65 * iv * position_value
        
        return var_95
    
    def _calculate_max_loss(self, features: Dict[str, float], strategy_phase: str, num_contracts: int) -> float:
        """Calculate worst-case loss scenario."""
        
        premium = features['premium']
        strike_distance = features['strike_distance_pct'] / 100
        strike = 100  # Approximate
        
        if strategy_phase == "SP" or strategy_phase == "CC+SP":
            return (strike - premium) * 100 * num_contracts
        else:
            return premium * 100 * num_contracts
    
    def train(
        self,
        historical_trades: List[Dict[str, Any]],
        optimize_for: str = "sharpe"
    ) -> Dict[str, Any]:
        """Train the position optimization model."""
        
        if len(historical_trades) < 50:
            return {"status": "failed", "reason": "insufficient_data"}
        
        try:
            import pandas as pd
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import cross_val_score
            
            X_samples = []
            y_samples = []
            
            for trade in historical_trades:
                market_data = trade.get('market_data_at_entry', {})
                portfolio_state = trade.get('portfolio_state_at_entry', {})
                option_info = trade.get('option_info', {})
                
                is_put = trade.get('trade_type', '').endswith('PUT')
                is_cc_phase_sp = trade.get('cc_phase_sp', False)
                
                if is_put and is_cc_phase_sp:
                    strategy_phase = "CC+SP"
                elif is_put:
                    strategy_phase = "SP"
                else:
                    strategy_phase = "CC"
                
                features = self.build_features(market_data, portfolio_state, strategy_phase, option_info)
                
                actual_return = trade.get('pnl', 0)
                max_drawdown = trade.get('max_drawdown', 0)
                contracts = trade.get('quantity', 1)
                
                if contracts == 0:
                    continue
                
                if max_drawdown > 0:
                    risk_adjusted_return = actual_return / max_drawdown
                else:
                    risk_adjusted_return = actual_return
                
                if optimize_for == "sharpe":
                    multiplier = max(0.5, min(1.5, 1 + risk_adjusted_return / 1000))
                elif optimize_for == "return":
                    return_per_contract = actual_return / abs(contracts)
                    multiplier = max(0.5, min(1.5, 1 + return_per_contract / 500))
                else:
                    if max_drawdown < 500:
                        multiplier = 1.2
                    elif max_drawdown < 1000:
                        multiplier = 1.0
                    else:
                        multiplier = 0.7
                
                X_samples.append([features.get(f, 0) for f in self.feature_names])
                y_samples.append(multiplier)
            
            if len(X_samples) == 0:
                return {"status": "failed", "reason": "no_valid_samples"}
            
            X = pd.DataFrame(X_samples, columns=self.feature_names)
            y = np.array(y_samples)
            
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
            
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
            self.model.fit(X, y)
            self.is_trained = True
            
            return {
                "status": "success",
                "n_samples": len(X),
                "cv_r2_mean": cv_scores.mean(),
                "cv_r2_std": cv_scores.std(),
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}


class WheelPositionIntegration:
    """Integration layer between MLPositionOptimizer and Wheel/BinbinGod strategies."""
    
    def __init__(
        self,
        optimizer: Optional[MLPositionOptimizer] = None,
        enabled: bool = True,
        fallback_to_rules: bool = True
    ):
        self.optimizer = optimizer or MLPositionOptimizer()
        self.enabled = enabled
        self.fallback_to_rules = fallback_to_rules
    
    def get_position_size(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        strategy_phase: str,
        option_info: Dict[str, Any],
        base_position: int = 1,
        max_position: int = 10
    ) -> Tuple[int, PositionRecommendation]:
        """Get ML-optimized position size."""
        
        if not self.enabled:
            features = self.optimizer.build_features(
                market_data, portfolio_state, strategy_phase, option_info
            )
            recommendation = self.optimizer._rule_based_predict(
                features, base_position, max_position, strategy_phase
            )
            return recommendation.num_contracts, recommendation
        
        try:
            recommendation = self.optimizer.predict_position_size(
                market_data=market_data,
                portfolio_state=portfolio_state,
                strategy_phase=strategy_phase,
                option_info=option_info,
                base_position=base_position,
                max_position=max_position
            )
            return recommendation.num_contracts, recommendation
        except Exception:
            if self.fallback_to_rules:
                features = self.optimizer.build_features(
                    market_data, portfolio_state, strategy_phase, option_info
                )
                recommendation = self.optimizer._rule_based_predict(
                    features, base_position, max_position, strategy_phase
                )
                return recommendation.num_contracts, recommendation
            else:
                return base_position, None