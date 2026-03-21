"""ML-based position size optimizer for Wheel strategy.

Uses machine learning to predict optimal position size based on:
- Market conditions (IV Rank, VIX, volatility regime)
- Strategy phase (SP/CC)
- Portfolio state (drawdown, margin utilization)
- Historical performance patterns

Output: Optimal number of contracts and risk-adjusted position multiplier.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PositionRecommendation:
    """ML recommendation for position sizing."""
    num_contracts: int  # Recommended number of contracts
    confidence: float  # 0.0 - 1.0
    position_multiplier: float  # Risk adjustment factor (0.5-1.5)
    expected_return: float  # Expected return per contract
    expected_risk: float  # Expected max drawdown per contract
    kelly_fraction: float  # Kelly optimal fraction
    reasoning: str

    # Risk metrics
    var_95: float  # Value at Risk (95% confidence)
    max_loss_scenario: float  # Worst case loss


@dataclass
class TrainingSample:
    """Training sample for position optimization model."""
    # Market features
    iv_rank: float
    vix: float
    vix_percentile: float
    market_regime: int  # 0=low vol, 1=normal, 2=high vol

    # Portfolio features
    available_margin: float
    margin_utilization: float
    current_drawdown: float
    portfolio_beta: float

    # Strategy features
    strategy_phase: int  # 0=SP, 1=CC
    shares_held: int
    cost_basis: float

    # Option features
    strike: float
    underlying_price: float
    dte: int
    delta: float
    premium: float

    # Label: What was the optimal position size in hindsight?
    # This is calculated from historical trades
    optimal_contracts: int
    actual_return: float
    max_drawdown: float


class MLPositionOptimizer:
    """Machine learning model for optimizing position sizing in Wheel strategy."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize ML position optimizer.

        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.model = None
        self.model_path = model_path
        self.is_trained = False

        # Feature names for model
        self.feature_names = [
            # Volatility features (high importance)
            'iv_rank',
            'iv_percentile',
            'historical_volatility',

            # Fear/Greed index (VIX)
            'vix',
            'vix_percentile',
            'vix_rank',
            'vix_change_pct',
            'vix_term_structure',

            # Market regime
            'market_regime',

            # Portfolio state
            'available_margin_pct',  # Available margin / total capital
            'margin_utilization',
            'current_drawdown',
            'portfolio_concentration',  # Herfindahl index of positions

            # Strategy phase
            'strategy_phase',  # 0=SP, 1=CC

            # Stock/Option features
            'strike_distance_pct',  # OTM percentage
            'dte',
            'delta',
            'premium',  # Option premium (required for feature extraction)
            'premium_yield',  # Premium / margin required
            'theta_per_day',  # Daily time decay value

            # Momentum features
            'momentum_5d',
            'momentum_10d',
            'vs_ma20',
            'vs_ma50',

            # Risk features
            'assignment_probability',  # Estimated probability of assignment
            'break_even_distance_pct',  # Distance to break-even
        ]

        # Risk management parameters
        self.max_position_multiplier = 1.5  # Max increase over base
        self.min_position_multiplier = 0.3  # Min decrease
        self.kelly_fraction = 0.25  # Use 25% Kelly (conservative)

        # Load pre-trained model if available
        if model_path:
            self.load_model(model_path)

    def build_features(
        self,
        market_data: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        strategy_phase: str,
        option_info: Dict[str, Any],
    ) -> pd.DataFrame:
        """Build feature vector for position sizing decision.

        Args:
            market_data: Market data (price, IV, VIX, etc.)
            portfolio_state: Portfolio state (capital, margin, drawdown)
            strategy_phase: "SP" or "CC"
            option_info: Option details (strike, delta, premium, DTE)

        Returns:
            Feature DataFrame
        """
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
        # 0=low vol (complacent), 1=normal, 2=high vol (fear)
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

        # Portfolio concentration (Herfindahl index)
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
        features['strategy_phase'] = 0 if strategy_phase == "SP" else 1

        # === Option Features ===
        underlying_price = option_info.get('underlying_price', market_data.get('price', 100))
        strike = option_info.get('strike', underlying_price * 0.95)
        delta = option_info.get('delta', 0.3)
        premium = option_info.get('premium', 1.0)
        dte = option_info.get('dte', 30)

        # Strike distance (OTM percentage)
        if strategy_phase == "SP":  # Put: strike < price
            features['strike_distance_pct'] = (underlying_price - strike) / underlying_price * 100
        else:  # Call: strike > price
            features['strike_distance_pct'] = (strike - underlying_price) / underlying_price * 100

        features['dte'] = dte
        features['delta'] = abs(delta)
        features['premium'] = premium

        # Premium yield: premium / margin required
        if strategy_phase == "SP":
            margin_required = strike * 100  # Cash-secured put
        else:
            margin_required = 0  # Covered call (stock is collateral)
        features['premium_yield'] = premium / margin_required * 100 if margin_required > 0 else premium * 100

        # Theta per day (approximation)
        features['theta_per_day'] = premium / dte if dte > 0 else 0

        # === Momentum Features ===
        momentum = market_data.get('momentum', {})
        features['momentum_5d'] = momentum.get('momentum_5d', 0)
        features['momentum_10d'] = momentum.get('momentum_10d', 0)
        features['vs_ma20'] = momentum.get('vs_ma20', 0)
        features['vs_ma50'] = momentum.get('vs_ma50', 0)

        # === Risk Features ===
        # Assignment probability (approximation based on delta and DTE)
        # Higher delta and lower DTE = higher assignment probability
        features['assignment_probability'] = self._estimate_assignment_probability(
            abs(delta), dte, features['strike_distance_pct']
        )

        # Break-even distance
        # For SP: break-even = strike - premium
        # For CC: break-even = cost_basis (if we have it)
        if strategy_phase == "SP":
            breakeven = strike - premium
            features['break_even_distance_pct'] = (underlying_price - breakeven) / underlying_price * 100
        else:
            cost_basis = portfolio_state.get('cost_basis', underlying_price)
            features['break_even_distance_pct'] = (underlying_price - cost_basis) / underlying_price * 100

        # Convert to DataFrame
        df = pd.DataFrame([features])
        return df[self.feature_names]

    def _estimate_assignment_probability(
        self,
        delta: float,
        dte: int,
        strike_distance_pct: float,
    ) -> float:
        """Estimate probability of assignment.

        Args:
            delta: Option delta
            dte: Days to expiry
            strike_distance_pct: Distance from ATM (%)

        Returns:
            Estimated assignment probability (0-1)
        """
        # Base probability from delta (delta ≈ probability of ITM at expiry)
        base_prob = delta

        # Adjust for DTE (more time = more uncertainty)
        dte_factor = 1 + (dte - 30) / 100  # Normalize around 30 DTE

        # Adjust for strike distance
        if strike_distance_pct > 5:  # > 5% OTM
            distance_factor = 0.7
        elif strike_distance_pct > 2:
            distance_factor = 0.9
        else:
            distance_factor = 1.0

        prob = base_prob * dte_factor * distance_factor
        return max(0, min(1, prob))

    def predict_position_size(
        self,
        market_data: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        strategy_phase: str,
        option_info: Dict[str, Any],
        base_position: int = 1,
        max_position: int = 10,
    ) -> PositionRecommendation:
        """Predict optimal position size using ML model.

        Args:
            market_data: Market data
            portfolio_state: Portfolio state
            strategy_phase: "SP" or "CC"
            option_info: Option details
            base_position: Base position size
            max_position: Maximum position size

        Returns:
            PositionRecommendation with optimal size and risk metrics
        """
        # Build features
        features = self.build_features(market_data, portfolio_state, strategy_phase, option_info)

        if self.model is not None and self.is_trained:
            # Use ML model for prediction
            return self._ml_predict(features, base_position, max_position, strategy_phase)
        else:
            # Use rule-based fallback
            return self._rule_based_predict(features, base_position, max_position, strategy_phase)

    def _ml_predict(
        self,
        features: pd.DataFrame,
        base_position: int,
        max_position: int,
        strategy_phase: str,
    ) -> PositionRecommendation:
        """ML-based prediction (when model is trained)."""
        try:
            # Get position multiplier from model
            multiplier = self.model.predict(features)[0]
            multiplier = np.clip(multiplier, self.min_position_multiplier, self.max_position_multiplier)

            # Calculate number of contracts
            num_contracts = int(base_position * multiplier)
            num_contracts = max(1, min(num_contracts, max_position))

            # Get confidence (if model supports prediction intervals)
            if hasattr(self.model, 'predict_proba'):
                # For classification models
                probs = self.model.predict_proba(features)[0]
                confidence = max(probs)
            else:
                # For regression models, use distance from extremes
                confidence = 1 - abs(multiplier - 1) / (self.max_position_multiplier - 1)

            # Calculate expected metrics
            expected_return = self._calculate_expected_return(features, strategy_phase)
            expected_risk = self._calculate_expected_risk(features, strategy_phase)

            # Kelly fraction
            kelly = self._calculate_kelly_fraction(features, strategy_phase)

            # Risk metrics
            var_95 = self._calculate_var(features, strategy_phase, num_contracts)
            max_loss = self._calculate_max_loss(features, strategy_phase, num_contracts)

            return PositionRecommendation(
                num_contracts=num_contracts,
                confidence=confidence,
                position_multiplier=multiplier,
                expected_return=expected_return * num_contracts,
                expected_risk=expected_risk * num_contracts,
                kelly_fraction=kelly,
                reasoning=self._generate_reasoning(features, multiplier, strategy_phase),
                var_95=var_95,
                max_loss_scenario=max_loss,
            )

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._rule_based_predict(features, base_position, max_position, strategy_phase)

    def _rule_based_predict(
        self,
        features: pd.DataFrame,
        base_position: int,
        max_position: int,
        strategy_phase: str,
    ) -> PositionRecommendation:
        """Rule-based prediction (fallback when model not trained)."""
        multiplier = 1.0
        reasoning_parts = []

        # === IV-based adjustment ===
        iv_rank = features['iv_rank'].values[0]
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
        vix = features['vix'].values[0]
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
        drawdown = features['current_drawdown'].values[0]
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
        margin_util = features['margin_utilization'].values[0]
        if margin_util > 0.7:
            multiplier *= 0.6
            reasoning_parts.append(f"High margin use ({margin_util*100:.0f}%): -40% position")
        elif margin_util > 0.5:
            multiplier *= 0.8
            reasoning_parts.append(f"Moderate margin ({margin_util*100:.0f}%): -20% position")

        # === Strategy phase adjustment ===
        if strategy_phase == "SP":
            # SP phase: more conservative (cash required)
            multiplier *= 0.9
            reasoning_parts.append("SP phase: -10% (cash reserve)")
        else:
            # CC phase: can be slightly more aggressive (stock collateral)
            reasoning_parts.append("CC phase: standard (stock collateral)")

        # === Assignment probability adjustment ===
        assign_prob = features['assignment_probability'].values[0]
        if assign_prob > 0.5:
            multiplier *= 0.8
            reasoning_parts.append(f"High assignment risk ({assign_prob*100:.0f}%): -20%")

        # Clamp multiplier
        multiplier = np.clip(multiplier, self.min_position_multiplier, self.max_position_multiplier)

        # Calculate position
        num_contracts = int(base_position * multiplier)
        num_contracts = max(1, min(num_contracts, max_position))

        # Calculate metrics
        expected_return = self._calculate_expected_return(features, strategy_phase)
        expected_risk = self._calculate_expected_risk(features, strategy_phase)
        kelly = self._calculate_kelly_fraction(features, strategy_phase)
        var_95 = self._calculate_var(features, strategy_phase, num_contracts)
        max_loss = self._calculate_max_loss(features, strategy_phase, num_contracts)

        # Confidence based on feature clarity
        confidence = 0.5  # Base confidence for rule-based
        if iv_rank > 60 or iv_rank < 30:
            confidence += 0.1  # Clear IV signal
        if vix > 25 or vix < 15:
            confidence += 0.1  # Clear VIX signal

        return PositionRecommendation(
            num_contracts=num_contracts,
            confidence=min(confidence, 0.8),  # Cap at 0.8 for rule-based
            position_multiplier=multiplier,
            expected_return=expected_return * num_contracts,
            expected_risk=expected_risk * num_contracts,
            kelly_fraction=kelly,
            reasoning=" | ".join(reasoning_parts) if reasoning_parts else "Standard position sizing",
            var_95=var_95,
            max_loss_scenario=max_loss,
        )

    def _calculate_expected_return(self, features: pd.DataFrame, strategy_phase: str) -> float:
        """Calculate expected return per contract."""
        premium = features['premium'].values[0]
        premium_yield = features['premium_yield'].values[0]
        assignment_prob = features['assignment_probability'].values[0]

        # Expected return = premium * (1 - assignment_prob) + stock_pnl * assignment_prob
        # For simplicity, assume stock P&L averages to 0 over many cycles
        expected_return = premium * 100 * (1 - assignment_prob * 0.5)  # Per contract

        return expected_return

    def _calculate_expected_risk(self, features: pd.DataFrame, strategy_phase: str) -> float:
        """Calculate expected maximum drawdown per contract."""
        strike = features.get('strike', 100).values[0] if 'strike' in features.columns else 100
        iv = features['historical_volatility'].values[0] / 100  # Convert to decimal

        # Expected max move: 2 standard deviations
        max_move = 2 * iv
        expected_risk = strike * 100 * max_move  # Per contract

        return expected_risk

    def _calculate_kelly_fraction(self, features: pd.DataFrame, strategy_phase: str) -> float:
        """Calculate Kelly optimal fraction.

        Kelly = (p * b - q) / b
        where p = win probability, q = loss probability, b = win/loss ratio
        """
        # Estimate from features
        iv_rank = features['iv_rank'].values[0]
        delta = features['delta'].values[0]

        # Win probability (option expires worthless)
        win_prob = 1 - delta  # Approximation

        # Win/loss ratio
        premium = features['premium'].values[0]
        max_loss = features['strike_distance_pct'].values[0] / 100 * features.get('strike', 100).values[0] if 'strike' in features.columns else premium * 2
        win_loss_ratio = premium / max_loss if max_loss > 0 else 1

        # Kelly fraction
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio if win_loss_ratio > 0 else 0

        # Use fractional Kelly (25%)
        return max(0, min(kelly * 0.25, 0.25))

    def _calculate_var(self, features: pd.DataFrame, strategy_phase: str, num_contracts: int) -> float:
        """Calculate Value at Risk (95% confidence)."""
        iv = features['historical_volatility'].values[0] / 100
        strike = features.get('strike', 100).values[0] if 'strike' in features.columns else 100

        # 95% VaR = 1.65 * sigma * position_value
        position_value = strike * 100 * num_contracts
        var_95 = 1.65 * iv * position_value

        return var_95

    def _calculate_max_loss(self, features: pd.DataFrame, strategy_phase: str, num_contracts: int) -> float:
        """Calculate worst-case loss scenario."""
        strike = features.get('strike', 100).values[0] if 'strike' in features.columns else 100
        premium = features['premium'].values[0]

        # Max loss for put assignment = strike * 100 - premium * 100
        if strategy_phase == "SP":
            max_loss = (strike - premium) * 100 * num_contracts
        else:
            # For CC, max loss is opportunity cost (stock called away below potential highs)
            max_loss = premium * 100 * num_contracts  # Conservative estimate

        return max_loss

    def _generate_reasoning(self, features: pd.DataFrame, multiplier: float, strategy_phase: str) -> str:
        """Generate human-readable reasoning for the recommendation."""
        parts = []

        iv_rank = features['iv_rank'].values[0]
        vix = features['vix'].values[0]
        drawdown = features['current_drawdown'].values[0]

        if multiplier > 1.1:
            parts.append(f"Favorable conditions: IV={iv_rank:.0f}, VIX={vix:.1f}")
        elif multiplier < 0.9:
            parts.append(f"Conservative sizing: IV={iv_rank:.0f}, VIX={vix:.1f}, DD={drawdown:.1f}%")
        else:
            parts.append(f"Standard sizing: IV={iv_rank:.0f}, VIX={vix:.1f}")

        return " | ".join(parts)

    def train(
        self,
        historical_trades: List[Dict[str, Any]],
        optimize_for: str = "sharpe",  # "sharpe", "return", "min_drawdown"
    ) -> Dict[str, Any]:
        """Train the position optimization model.

        Args:
            historical_trades: List of historical trade records with outcomes
            optimize_for: Optimization objective

        Returns:
            Training metrics
        """
        if len(historical_trades) < 50:
            logger.warning("Insufficient data for training (need at least 50 trades)")
            return {"status": "failed", "reason": "insufficient_data"}

        # Build training dataset
        X, y = self._build_training_data(historical_trades, optimize_for)

        if len(X) == 0:
            return {"status": "failed", "reason": "no_valid_samples"}

        # Train model
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score

        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )

        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')

        # Fit final model
        self.model.fit(X, y)
        self.is_trained = True

        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

        logger.info(f"Model trained. CV R2: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        return {
            "status": "success",
            "n_samples": len(X),
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "top_features": top_features,
        }

    def _build_training_data(
        self,
        historical_trades: List[Dict[str, Any]],
        optimize_for: str,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build training data from historical trades.

        Label = optimal position multiplier in hindsight
        """
        X_samples = []
        y_samples = []

        for trade in historical_trades:
            try:
                # Extract features at trade entry
                market_data = trade.get('market_data_at_entry', {})
                portfolio_state = trade.get('portfolio_state_at_entry', {})
                option_info = trade.get('option_info', {})
                strategy_phase = "SP" if trade.get('trade_type', '').endswith('PUT') else "CC"

                features = self.build_features(market_data, portfolio_state, strategy_phase, option_info)

                # Calculate optimal position multiplier in hindsight
                # Based on actual outcome vs expected outcome
                actual_return = trade.get('pnl', 0)
                max_drawdown = trade.get('max_drawdown', 0)
                contracts = trade.get('quantity', 1)

                if contracts == 0:
                    continue

                return_per_contract = actual_return / abs(contracts)

                # Calculate what the optimal position would have been
                # Optimal = maximize return / risk
                if max_drawdown > 0:
                    risk_adjusted_return = actual_return / max_drawdown
                else:
                    risk_adjusted_return = actual_return

                # Normalize to multiplier (1.0 = base position)
                # Higher return/risk = higher multiplier
                if optimize_for == "sharpe":
                    # Scale based on risk-adjusted return
                    multiplier = np.clip(1 + risk_adjusted_return / 1000, 0.5, 1.5)
                elif optimize_for == "return":
                    multiplier = np.clip(1 + return_per_contract / 500, 0.5, 1.5)
                else:  # min_drawdown
                    if max_drawdown < 500:
                        multiplier = 1.2
                    elif max_drawdown < 1000:
                        multiplier = 1.0
                    else:
                        multiplier = 0.7

                X_samples.append(features.iloc[0].values)
                y_samples.append(multiplier)

            except Exception as e:
                logger.debug(f"Skipping trade due to error: {e}")
                continue

        if len(X_samples) == 0:
            return pd.DataFrame(), np.array([])

        X = pd.DataFrame(X_samples, columns=self.feature_names)
        y = np.array(y_samples)

        return X, y

    def save_model(self, path: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
            }, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load pre-trained model from disk."""
        if not Path(path).exists():
            logger.warning(f"Model file not found: {path}")
            return

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_trained = data.get('is_trained', True)

        logger.info(f"Model loaded from {path}")


class WheelPositionIntegration:
    """Integration layer between MLPositionOptimizer and Wheel/BinbinGod strategies."""

    def __init__(
        self,
        optimizer: Optional[MLPositionOptimizer] = None,
        enabled: bool = True,
        fallback_to_rules: bool = True,
    ):
        """Initialize integration.

        Args:
            optimizer: ML position optimizer instance
            enabled: Whether ML optimization is enabled
            fallback_to_rules: Fall back to rules if ML fails
        """
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
        max_position: int = 10,
    ) -> Tuple[int, PositionRecommendation]:
        """Get ML-optimized position size.

        Args:
            symbol: Stock symbol
            market_data: Market data
            portfolio_state: Portfolio state
            strategy_phase: "SP" or "CC"
            option_info: Option details
            base_position: Base position size
            max_position: Maximum position size

        Returns:
            Tuple of (position_size, recommendation)
        """
        if not self.enabled:
            # Use simple rule-based sizing
            recommendation = self.optimizer._rule_based_predict(
                self.optimizer.build_features(market_data, portfolio_state, strategy_phase, option_info),
                base_position,
                max_position,
                strategy_phase,
            )
            return recommendation.num_contracts, recommendation

        try:
            recommendation = self.optimizer.predict_position_size(
                market_data=market_data,
                portfolio_state=portfolio_state,
                strategy_phase=strategy_phase,
                option_info=option_info,
                base_position=base_position,
                max_position=max_position,
            )

            return recommendation.num_contracts, recommendation

        except Exception as e:
            logger.error(f"ML position optimization failed: {e}")

            if self.fallback_to_rules:
                recommendation = self.optimizer._rule_based_predict(
                    self.optimizer.build_features(market_data, portfolio_state, strategy_phase, option_info),
                    base_position,
                    max_position,
                    strategy_phase,
                )
                return recommendation.num_contracts, recommendation
            else:
                return base_position, None