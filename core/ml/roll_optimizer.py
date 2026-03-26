"""ML-based Roll decision optimizer for Wheel strategy.

Instead of traditional profit targets / stop losses, Wheel strategy benefits
from intelligent roll management:

- Roll Forward: Close early, open new position to capture more premium
- Roll Out: Extend to later expiry when near assignment risk
- Let Expire: Hold to expiry when premium capture is optimal

ML model predicts:
1. Should we roll now or continue holding?
2. What type of roll (forward/out/close) is optimal?
3. Expected P&L improvement from rolling vs holding
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RollRecommendation:
    """ML recommendation for roll decision."""
    action: str  # "ROLL_FORWARD", "ROLL_OUT", "LET_EXPIRE", "CLOSE_EARLY"
    confidence: float  # 0.0 - 1.0
    expected_pnl_improvement: float  # Expected P&L improvement from action vs hold
    optimal_dte: Optional[int]  # Target DTE for roll
    optimal_delta: Optional[float]  # Target delta for roll
    reasoning: str


class MLRollOptimizer:
    """Machine learning model for optimizing roll decisions in Wheel strategy."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize ML roll optimizer.

        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.model = None
        self.model_path = model_path

        self.feature_names = [
            # Volatility features
            'iv_rank',
            'historical_volatility',
            'iv_percentile',

            # Fear/Greed index (VIX)
            'vix',
            'vix_percentile',
            'vix_term_structure',

            # Time features
            'dte',
            'dte_percent',  # DTE / original_DTE
            'days_held',
            'theta_capture_rate',  # Premium captured per day

            # Premium features
            'premium_received',
            'premium_remaining',
            'premium_capture_pct',  # (entry_premium - current_premium) / entry_premium

            # Option features
            'delta',
            'delta_change_ratio',  # Current delta / entry delta
            'strike_distance_pct',  # Distance from ATM

            # Position features
            'current_pnl_pct',
            'underlying_price_change_pct',  # Price change since entry

            # Market regime
            'market_regime',  # 0=low vol, 1=normal, 2=high vol

            # Phase features
            'strategy_phase',  # 0=SP, 1=CC, 2=CC+SP (simultaneous mode)
        ]

        if model_path:
            self.load_model(model_path)

    def build_features(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        current_date: str,
    ) -> pd.DataFrame:
        """Build feature vector for roll decision.

        Args:
            position: Position dictionary with entry info
            market_data: Current market data
            current_date: Current date (YYYY-MM-DD)

        Returns:
            Feature DataFrame
        """
        # Time calculations
        try:
            entry_date = datetime.strptime(position.get('entry_date', ''), '%Y-%m-%d')
            curr_date = datetime.strptime(current_date, '%Y-%m-%d')
            days_held = (curr_date - entry_date).days
        except (ValueError, TypeError):
            days_held = 0

        try:
            expiry_date = datetime.strptime(position.get('expiry', ''), '%Y%m%d')
            dte = (expiry_date - curr_date).days
            original_dte = (expiry_date - entry_date).days
            dte_percent = dte / original_dte if original_dte > 0 else 0
        except (ValueError, TypeError, UnboundLocalError):
            dte = 0
            dte_percent = 0

        # Premium calculations
        entry_premium = abs(position.get('entry_price', 0))
        current_premium = market_data.get('option_price', entry_premium)
        premium_captured = entry_premium - current_premium
        premium_capture_pct = premium_captured / entry_premium if entry_premium > 0 else 0
        theta_capture_rate = premium_captured / max(days_held, 1)

        # Price and volatility
        underlying_price = market_data.get('price', position.get('underlying_price', 0))
        entry_underlying = position.get('underlying_price', underlying_price)
        price_change_pct = (underlying_price - entry_underlying) / entry_underlying * 100 if entry_underlying > 0 else 0

        iv = market_data.get('iv', 0.3)
        iv_rank = market_data.get('iv_rank', 50)

        # Delta change
        entry_delta = position.get('delta_at_entry', 0.3)
        current_delta = market_data.get('delta', entry_delta)
        delta_change_ratio = abs(current_delta / entry_delta) if entry_delta != 0 else 1

        # Strike distance
        strike = position.get('strike', underlying_price)
        if position.get('right') == 'P':
            strike_distance_pct = (underlying_price - strike) / underlying_price * 100
        else:
            strike_distance_pct = (strike - underlying_price) / underlying_price * 100

        # P&L
        quantity = position.get('quantity', -1)
        if quantity < 0:  # Short position
            current_pnl_pct = premium_capture_pct * 100
        else:
            current_pnl_pct = (current_premium - entry_premium) / entry_premium * 100

        # Market regime
        if iv_rank < 20:
            market_regime = 0  # Low vol
        elif iv_rank > 50:
            market_regime = 2  # High vol
        else:
            market_regime = 1  # Normal

        # Strategy phase - 支持CC+SP模式
        phase = position.get('strategy_phase', 'SP')
        if phase == 'SP':
            strategy_phase = 0
        elif phase == 'CC':
            strategy_phase = 1
        else:  # CC+SP simultaneous mode
            strategy_phase = 2

        features = {
            'iv_rank': iv_rank,
            'historical_volatility': market_data.get('historical_volatility', 30),
            'iv_percentile': market_data.get('iv_percentile', iv_rank),

            'vix': market_data.get('vix', 20.0),
            'vix_percentile': market_data.get('vix_percentile', 50.0),
            'vix_term_structure': market_data.get('vix_term_structure', 0.0),

            'dte': dte,
            'dte_percent': dte_percent,
            'days_held': days_held,
            'theta_capture_rate': theta_capture_rate,

            'premium_received': entry_premium * 100,
            'premium_remaining': current_premium * 100,
            'premium_capture_pct': premium_capture_pct * 100,

            'delta': abs(current_delta),
            'delta_change_ratio': delta_change_ratio,
            'strike_distance_pct': strike_distance_pct,

            'current_pnl_pct': current_pnl_pct,
            'underlying_price_change_pct': price_change_pct,

            'market_regime': market_regime,
            'strategy_phase': strategy_phase,
        }

        return pd.DataFrame([features])

    def predict_roll_decision(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        current_date: str,
        base_dte_range: Tuple[int, int] = (30, 45),
        base_delta: float = 0.30,
    ) -> RollRecommendation:
        """Predict optimal roll decision using ML or rule-based fallback.

        Args:
            position: Position info
            market_data: Current market data
            current_date: Current date
            base_dte_range: Default DTE range for rolls
            base_delta: Default delta for rolls

        Returns:
            RollRecommendation with action and parameters
        """
        features = self.build_features(position, market_data, current_date)
        feat = features.iloc[0]

        # If ML model available, use it
        if self.model is not None:
            try:
                return self._ml_prediction(features, feat, base_dte_range, base_delta)
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}, using rule-based")

        # Rule-based roll decision
        return self._rule_based_decision(feat, position, market_data, base_dte_range, base_delta)

    def _rule_based_decision(
        self,
        feat: pd.Series,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        base_dte_range: Tuple[int, int],
        base_delta: float,
    ) -> RollRecommendation:
        """Rule-based roll decision logic.

        Wheel strategy roll rules:
        1. Premium capture > 80% + DTE > 7: Roll forward (capture more premium)
        2. Delta doubled + DTE <= 14: Roll out (avoid unwanted assignment)
        3. DTE <= 2: Let expire (final theta capture)
        4. Premium capture < 50% + DTE > 21: Continue holding
        """

        dte = feat['dte']
        premium_capture = feat['premium_capture_pct']
        delta_ratio = feat['delta_change_ratio']
        iv_rank = feat['iv_rank']
        phase = position.get('strategy_phase', 'SP')

        # Rule 1: High premium capture - Roll forward to maximize returns
        if premium_capture >= 80 and dte > 7:
            action = "ROLL_FORWARD"
            confidence = 0.85
            expected_improvement = self._estimate_roll_pnl_improvement(
                feat, "ROLL_FORWARD", base_dte_range, base_delta
            )

            # Roll to same delta, standard DTE
            return RollRecommendation(
                action=action,
                confidence=confidence,
                expected_pnl_improvement=expected_improvement,
                optimal_dte=base_dte_range[1],  # Target longer DTE
                optimal_delta=base_delta,
                reasoning=f"Premium capture {premium_capture:.0f}% - roll to capture more theta"
            )

        # Rule 2: Delta spike near expiry - Roll out to avoid assignment
        # IMPORTANT: Only ROLL_OUT if we're still profitable (buyback cost < premium received)
        if delta_ratio >= 2.0 and dte <= 14:
            premium_received = feat['premium_received']
            premium_remaining = feat['premium_remaining']  # Current buyback cost
            
            # Check if rolling out is profitable
            # If buyback cost > premium received, we'd lock in a loss
            # Better to let it expire/assign and keep full premium
            if premium_remaining > premium_received:
                # Call is ITM, buyback would cost more than premium received
                # Let it assign - we sell shares at strike price and keep premium
                action = "LET_EXPIRE"
                confidence = 0.90
                expected_improvement = 0

                return RollRecommendation(
                    action=action,
                    confidence=confidence,
                    expected_pnl_improvement=expected_improvement,
                    optimal_dte=None,
                    optimal_delta=None,
                    reasoning=f"Delta {delta_ratio:.1f}x, buyback ${premium_remaining:.0f} > premium ${premium_received:.0f} - let assign, keep premium"
                )
            
            # Still profitable to roll out
            action = "ROLL_OUT"
            confidence = 0.80
            expected_improvement = self._estimate_roll_pnl_improvement(
                feat, "ROLL_OUT", base_dte_range, base_delta
            )

            # Roll to further out expiry
            return RollRecommendation(
                action=action,
                confidence=confidence,
                expected_pnl_improvement=expected_improvement,
                optimal_dte=min(dte + 30, 60),  # Extend by 30 days
                optimal_delta=min(base_delta * 0.9, 0.25),  # Slightly more conservative
                reasoning=f"Delta {delta_ratio:.1f}x entry, DTE {dte} - roll out to avoid assignment"
            )

        # Rule 3: Near expiry with good capture - Let expire
        if dte <= 2 and premium_capture >= 50:
            action = "LET_EXPIRE"
            confidence = 0.90
            expected_improvement = 0  # No action needed

            return RollRecommendation(
                action=action,
                confidence=confidence,
                expected_pnl_improvement=expected_improvement,
                optimal_dte=None,
                optimal_delta=None,
                reasoning=f"DTE {dte}, capture {premium_capture:.0f}% - let expire for max theta"
            )

        # Rule 4: Near expiry with poor capture - Consider early close
        if dte <= 5 and premium_capture < 30:
            action = "CLOSE_EARLY"
            confidence = 0.60
            expected_improvement = self._estimate_close_improvement(feat)

            return RollRecommendation(
                action=action,
                confidence=confidence,
                expected_pnl_improvement=expected_improvement,
                optimal_dte=None,
                optimal_delta=None,
                reasoning=f"DTE {dte}, low capture {premium_capture:.0f}% - close to free margin"
            )

        # Rule 5: High IV environment - Continue holding (capture more premium)
        if iv_rank > 60 and dte > 14:
            action = "LET_EXPIRE"
            confidence = 0.70
            expected_improvement = 0

            return RollRecommendation(
                action=action,
                confidence=confidence,
                expected_pnl_improvement=expected_improvement,
                optimal_dte=None,
                optimal_delta=None,
                reasoning=f"High IV rank {iv_rank:.0f} - continue capturing premium"
            )

        # Default: Continue holding
        action = "LET_EXPIRE"
        confidence = 0.50
        expected_improvement = 0

        return RollRecommendation(
            action=action,
            confidence=confidence,
            expected_pnl_improvement=expected_improvement,
            optimal_dte=None,
            optimal_delta=None,
            reasoning=f"Continue holding - DTE {dte}, capture {premium_capture:.0f}%"
        )

    def _ml_prediction(
        self,
        features: pd.DataFrame,
        feat: pd.Series,
        base_dte_range: Tuple[int, int],
        base_delta: float,
    ) -> RollRecommendation:
        """Use ML model for roll prediction."""

        # Get model prediction
        prediction = self.model.predict_proba(features)[0]

        # Actions: 0=LET_EXPIRE, 1=ROLL_FORWARD, 2=ROLL_OUT, 3=CLOSE_EARLY
        action_idx = np.argmax(prediction)
        confidence = prediction[action_idx]

        actions = ["LET_EXPIRE", "ROLL_FORWARD", "ROLL_OUT", "CLOSE_EARLY"]
        action = actions[action_idx]

        # CRITICAL: In SP phase, only allow ROLL_FORWARD or LET_EXPIRE
        # ROLL_OUT and CLOSE_EARLY would break Wheel cycle by avoiding assignment
        strategy_phase = feat.get('strategy_phase', 0)  # 0=SP, 1=CC
        if strategy_phase == 0 and action in ["ROLL_OUT", "CLOSE_EARLY"]:
            # Override: In SP phase, let the option expire for assignment
            action = "LET_EXPIRE"
            confidence = 0.90

        # Estimate improvement
        expected_improvement = self._estimate_roll_pnl_improvement(
            feat, action, base_dte_range, base_delta
        )

        return RollRecommendation(
            action=action,
            confidence=confidence,
            expected_pnl_improvement=expected_improvement,
            optimal_dte=base_dte_range[1] if action in ["ROLL_FORWARD", "ROLL_OUT"] else None,
            optimal_delta=base_delta if action in ["ROLL_FORWARD", "ROLL_OUT"] else None,
            reasoning=f"ML predicted {action} with {confidence:.0%} confidence"
        )

    def _estimate_roll_pnl_improvement(
        self,
        feat: pd.Series,
        action: str,
        base_dte_range: Tuple[int, int],
        base_delta: float,
    ) -> float:
        """Estimate expected P&L improvement from roll action vs holding."""

        if action == "LET_EXPIRE":
            return 0.0

        premium_remaining = feat['premium_remaining']
        dte = feat['dte']
        iv_rank = feat['iv_rank']

        if action == "ROLL_FORWARD":
            # Close current, open new
            # Benefit: New premium received
            # Cost: Buyback current position + Slippage

            # Estimate new premium (based on IV and DTE)
            new_dte = base_dte_range[1]
            iv_multiplier = 1 + (iv_rank - 50) / 100  # Higher IV = more premium
            estimated_new_premium = base_delta * 100 * iv_multiplier * (new_dte / 45)

            # Net benefit: new premium - buyback cost - slippage
            slippage = premium_remaining * 0.05  # 5% slippage estimate
            improvement = estimated_new_premium - premium_remaining - slippage

            return max(0, improvement)

        elif action == "ROLL_OUT":
            # Extend expiry to avoid assignment
            # Benefit: Avoid potentially unwanted assignment
            # Cost: Additional premium paid + slippage

            delta_ratio = feat['delta_change_ratio']

            # Higher delta ratio = higher assignment risk = more benefit from roll
            risk_reduction = min(delta_ratio * 20, 50)  # Up to $50 per contract

            # Cost of roll
            roll_cost = premium_remaining * 0.1  # 10% cost estimate

            improvement = risk_reduction - roll_cost
            return max(0, improvement)

        elif action == "CLOSE_EARLY":
            # Close position to free margin
            return self._estimate_close_improvement(feat)

        return 0.0

    def _estimate_close_improvement(self, feat: pd.Series) -> float:
        """Estimate improvement from closing position early."""
        # Benefit: Free margin, avoid further risk
        # Cost: Forego remaining theta

        premium_remaining = feat['premium_remaining']
        dte = feat['dte']

        # If DTE is very low, remaining theta is minimal
        theta_remaining = premium_remaining * (dte / 45) * 0.5  # Estimate

        # Close early saves margin and risk
        margin_benefit = 10  # $10 per contract for margin flexibility

        return margin_benefit - theta_remaining

    def should_roll(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        current_date: str,
        min_confidence: float = 0.6,
    ) -> Tuple[bool, RollRecommendation]:
        """Determine if position should be rolled.

        Args:
            position: Position info
            market_data: Market data
            current_date: Current date
            min_confidence: Minimum confidence threshold

        Returns:
            (should_roll, recommendation) tuple
        """

        recommendation = self.predict_roll_decision(position, market_data, current_date)

        should_roll = (
            recommendation.action in ["ROLL_FORWARD", "ROLL_OUT", "CLOSE_EARLY"]
            and recommendation.confidence >= min_confidence
            and recommendation.expected_pnl_improvement > 0
        )

        return should_roll, recommendation

    def train_model(self, historical_trades: pd.DataFrame) -> Dict[str, float]:
        """Train ML model on historical roll decisions.

        Args:
            historical_trades: DataFrame with features and optimal_action column

        Returns:
            Training metrics
        """
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            import xgboost as xgb

            X = historical_trades[self.feature_names]
            y = historical_trades['optimal_action']  # 0-3 for actions

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            self.model = model
            logger.info(f"ML Roll Optimizer trained with accuracy: {accuracy:.2f}")

            return {
                'accuracy': accuracy,
                'n_samples': len(historical_trades),
                'n_features': len(self.feature_names),
            }

        except ImportError:
            logger.warning("ML libraries not available, using rule-based only")
            return {'error': 'ML libraries not available'}
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'error': str(e)}

    def save_model(self, path: str):
        """Save trained model to disk."""
        if self.model:
            import joblib
            joblib.dump(self.model, path)
            logger.info(f"Roll model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk."""
        try:
            import joblib
            self.model = joblib.load(path)
            logger.info(f"Roll model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None