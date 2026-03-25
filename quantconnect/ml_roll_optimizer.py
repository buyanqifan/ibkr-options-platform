"""
ML Roll Optimizer for QuantConnect
===================================

Machine learning model that optimizes roll decisions for Wheel strategy:

- Roll Forward: Close early, open new position to capture more premium
- Roll Out: Extend to later expiry when near assignment risk
- Let Expire: Hold to expiry when premium capture is optimal
- Close Early: Close position to free margin

ML model predicts:
1. Should we roll now or continue holding?
2. What type of roll is optimal?
3. Expected P&L improvement from rolling vs holding
"""

from AlgorithmImports import *
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class RollRecommendation:
    """ML recommendation for roll decision."""
    action: str  # "ROLL_FORWARD", "ROLL_OUT", "LET_EXPIRE", "CLOSE_EARLY"
    confidence: float  # 0.0 - 1.0
    expected_pnl_improvement: float  # Expected P&L improvement
    optimal_dte: Optional[int]
    optimal_delta: Optional[float]
    reasoning: str


class MLRollOptimizer:
    """
    Machine learning model for optimizing roll decisions in Wheel strategy.
    
    Wheel strategy roll rules:
    1. Premium capture > 80% + DTE > 7: Roll forward (capture more premium)
    2. Delta doubled + DTE <= 14: Roll out (avoid unwanted assignment)
    3. DTE <= 2: Let expire (final theta capture)
    4. Premium capture < 50% + DTE > 21: Continue holding
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        
        self.feature_names = [
            'iv_rank', 'historical_volatility', 'iv_percentile',
            'vix', 'vix_percentile', 'vix_term_structure',
            'dte', 'dte_percent', 'days_held', 'theta_capture_rate',
            'premium_received', 'premium_remaining', 'premium_capture_pct',
            'delta', 'delta_change_ratio', 'strike_distance_pct',
            'current_pnl_pct', 'underlying_price_change_pct',
            'market_regime',
            'strategy_phase',
        ]
    
    def build_features(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        current_date: str
    ) -> Dict[str, float]:
        """Build feature vector for roll decision."""
        
        # Time calculations
        try:
            entry_date = datetime.strptime(position.get('entry_date', ''), '%Y-%m-%d')
            curr_date = datetime.strptime(current_date, '%Y-%m-%d')
            days_held = (curr_date - entry_date).days
        except (ValueError, TypeError):
            days_held = 0
        
        # DTE calculation - handle multiple formats
        # 1. Use pre-calculated dte from build_position_data (preferred)
        # 2. If expiry is datetime object, calculate directly
        # 3. If expiry is string, parse it
        dte = position.get('dte')  # Use pre-calculated DTE if available
        if dte is None:
            expiry = position.get('expiry')
            if hasattr(expiry, '__sub__') and hasattr(expiry, 'year'):
                # expiry is datetime/date object
                try:
                    dte = (expiry - curr_date).days
                except:
                    dte = 0
            elif isinstance(expiry, str):
                # expiry is string, try parsing
                try:
                    expiry_date = datetime.strptime(expiry, '%Y%m%d')
                    dte = (expiry_date - curr_date).days
                except (ValueError, TypeError):
                    dte = 0
            else:
                dte = 0
        
        # Calculate dte_percent for theta capture rate
        try:
            original_dte = dte + days_held if dte and days_held else 45
            dte_percent = dte / original_dte if original_dte > 0 else 0
        except:
            dte_percent = 0
        
        # Premium calculations
        entry_premium = abs(position.get('entry_price', 0))
        current_premium = market_data.get('option_price', entry_premium)
        premium_captured = entry_premium - current_premium
        premium_capture_pct = premium_captured / entry_premium * 100 if entry_premium > 0 else 0
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
        
        # Strike distance - protect against underlying_price=0
        strike = position.get('strike', underlying_price)
        if underlying_price > 0:
            if position.get('right') == 'P':
                strike_distance_pct = (underlying_price - strike) / underlying_price * 100
            else:
                strike_distance_pct = (strike - underlying_price) / underlying_price * 100
        else:
            strike_distance_pct = 0
        
        # P&L
        quantity = position.get('quantity', -1)
        if quantity < 0:  # Short position
            current_pnl_pct = premium_capture_pct
        else:
            current_pnl_pct = (current_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
        
        # Market regime
        if iv_rank < 20:
            market_regime = 0
        elif iv_rank > 50:
            market_regime = 2
        else:
            market_regime = 1
        
        # Strategy phase
        phase = position.get('strategy_phase', 'SP')
        if phase == 'SP':
            strategy_phase = 0
        elif phase == 'CC':
            strategy_phase = 1
        else:
            strategy_phase = 2
        
        return {
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
            'premium_capture_pct': premium_capture_pct,
            'delta': abs(current_delta),
            'delta_change_ratio': delta_change_ratio,
            'strike_distance_pct': strike_distance_pct,
            'current_pnl_pct': current_pnl_pct,
            'underlying_price_change_pct': price_change_pct,
            'market_regime': market_regime,
            'strategy_phase': strategy_phase,
        }
    
    def predict_roll_decision(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        current_date: str,
        base_dte_range: Tuple[int, int] = (30, 45),
        base_delta: float = 0.30
    ) -> RollRecommendation:
        """
        Predict optimal roll decision using ML or rule-based fallback.
        
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
        
        # If ML model available, use it
        if self.model is not None:
            try:
                return self._ml_prediction(features, base_dte_range, base_delta)
            except Exception:
                pass
        
        # Rule-based roll decision
        return self._rule_based_decision(features, position, market_data, base_dte_range, base_delta)
    
    def _rule_based_decision(
        self,
        features: Dict[str, float],
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        base_dte_range: Tuple[int, int],
        base_delta: float
    ) -> RollRecommendation:
        """Rule-based roll decision logic."""
        
        dte = features['dte']
        premium_capture = features['premium_capture_pct']
        delta_ratio = features['delta_change_ratio']
        iv_rank = features['iv_rank']
        strategy_phase = position.get('strategy_phase', 'SP')
        
        # Rule 1: High premium capture - Roll forward
        if premium_capture >= 80 and dte > 7:
            action = "ROLL_FORWARD"
            confidence = 0.85
            expected_improvement = self._estimate_roll_pnl_improvement(
                features, "ROLL_FORWARD", base_dte_range, base_delta
            )
            
            return RollRecommendation(
                action=action,
                confidence=confidence,
                expected_pnl_improvement=expected_improvement,
                optimal_dte=base_dte_range[1],
                optimal_delta=base_delta,
                reasoning=f"Premium capture {premium_capture:.0f}% - roll to capture more theta"
            )
        
        # Rule 2: Delta spike near expiry - Roll out
        # DISABLED for SP phase: Wheel strategy accepts assignment in SP phase
        # Only roll out in CC phase when we want to protect gains on held shares
        # IMPORTANT: Only ROLL_OUT if we're still profitable (buyback cost < premium received)
        if delta_ratio >= 2.0 and dte <= 14 and strategy_phase == 'CC':
            premium_received = features['premium_received']
            premium_remaining = features['premium_remaining']  # Current buyback cost
            
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
                features, "ROLL_OUT", base_dte_range, base_delta
            )
            
            return RollRecommendation(
                action=action,
                confidence=confidence,
                expected_pnl_improvement=expected_improvement,
                optimal_dte=min(dte + 30, 60),
                optimal_delta=min(base_delta * 0.9, 0.25),
                reasoning=f"Delta {delta_ratio:.1f}x entry, DTE {dte} - roll out to avoid assignment"
            )
        
        # Rule 3: Near expiry with good capture - Let expire
        if dte <= 2 and premium_capture >= 50:
            action = "LET_EXPIRE"
            confidence = 0.90
            expected_improvement = 0
            
            return RollRecommendation(
                action=action,
                confidence=confidence,
                expected_pnl_improvement=expected_improvement,
                optimal_dte=None,
                optimal_delta=None,
                reasoning=f"DTE {dte}, capture {premium_capture:.0f}% - let expire for max theta"
            )
        
        # Rule 4: Near expiry with poor capture - DISABLED for SP phase
        # Wheel strategy should hold to expiry for assignment in SP phase
        # Only consider early close in CC phase when stock price >> cost basis
        if dte <= 5 and premium_capture < 30 and strategy_phase == 'CC':
            # In CC phase, close early only if we can redeploy capital better
            action = "CLOSE_EARLY"
            confidence = 0.60
            expected_improvement = self._estimate_close_improvement(features)
            
            return RollRecommendation(
                action=action,
                confidence=confidence,
                expected_pnl_improvement=expected_improvement,
                optimal_dte=None,
                optimal_delta=None,
                reasoning=f"CC phase, DTE {dte}, low capture {premium_capture:.0f}% - close to redeploy capital"
            )
        
        # Rule 5: High IV environment - Continue holding
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
        features: Dict[str, float],
        base_dte_range: Tuple[int, int],
        base_delta: float
    ) -> RollRecommendation:
        """Use ML model for roll prediction."""
        
        import pandas as pd
        
        X = pd.DataFrame([features])[self.feature_names]
        prediction = self.model.predict_proba(X)[0]
        
        action_idx = np.argmax(prediction)
        confidence = prediction[action_idx]
        
        actions = ["LET_EXPIRE", "ROLL_FORWARD", "ROLL_OUT", "CLOSE_EARLY"]
        action = actions[action_idx]
        
        expected_improvement = self._estimate_roll_pnl_improvement(
            features, action, base_dte_range, base_delta
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
        features: Dict[str, float],
        action: str,
        base_dte_range: Tuple[int, int],
        base_delta: float
    ) -> float:
        """Estimate expected P&L improvement from roll action."""
        
        if action == "LET_EXPIRE":
            return 0.0
        
        premium_remaining = features['premium_remaining']
        dte = features['dte']
        iv_rank = features['iv_rank']
        
        if action == "ROLL_FORWARD":
            new_dte = base_dte_range[1]
            iv_multiplier = 1 + (iv_rank - 50) / 100
            estimated_new_premium = base_delta * 100 * iv_multiplier * (new_dte / 45)
            slippage = premium_remaining * 0.05
            improvement = premium_remaining + estimated_new_premium - slippage
            return max(0, improvement)
        
        elif action == "ROLL_OUT":
            delta_ratio = features['delta_change_ratio']
            risk_reduction = min(delta_ratio * 20, 50)
            roll_cost = premium_remaining * 0.1
            improvement = risk_reduction - roll_cost
            return max(0, improvement)
        
        elif action == "CLOSE_EARLY":
            return self._estimate_close_improvement(features)
        
        return 0.0
    
    def _estimate_close_improvement(self, features: Dict[str, float]) -> float:
        """Estimate improvement from closing position early."""
        
        premium_remaining = features['premium_remaining']
        dte = features['dte']
        
        theta_remaining = premium_remaining * (dte / 45) * 0.5
        margin_benefit = 10
        
        return margin_benefit - theta_remaining
    
    def should_roll(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        current_date: str,
        min_confidence: float = 0.6
    ) -> Tuple[bool, RollRecommendation]:
        """
        Determine if position should be rolled.
        
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
    
    def train_model(self, historical_trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train ML model on historical roll decisions."""
        
        try:
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # Build training data
            X_samples = []
            y_samples = []
            
            for trade in historical_trades:
                features = self.build_features(
                    trade.get('position', {}),
                    trade.get('market_data', {}),
                    trade.get('date', '')
                )
                X_samples.append([features.get(f, 0) for f in self.feature_names])
                y_samples.append(trade.get('optimal_action', 0))
            
            if len(X_samples) < 50:
                return {"status": "failed", "reason": "insufficient_data"}
            
            X = pd.DataFrame(X_samples, columns=self.feature_names)
            y = np.array(y_samples)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            try:
                import xgboost as xgb
                
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.model = model
            
            return {
                'status': 'success',
                'accuracy': accuracy,
                'n_samples': len(X_samples),
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}