"""ML-based profit target and stop loss optimizer.

Uses XGBoost to predict optimal exit points based on:
- Market volatility (IV Rank, HV)
- Time decay (DTE, theta)
- Price momentum
- Delta changes
- Historical win/loss patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MLExitOptimizer:
    """Machine learning model for optimizing profit targets and stop losses."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize ML exit optimizer.
        
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
            'vix',  # Current VIX level
            'vix_percentile',  # Where VIX sits in recent history
            'vix_rank',  # VIX rank (0-100)
            'vix_change_pct',  # 5-day VIX change
            'vix_5d_ma',  # 5-day VIX MA
            'vix_20d_ma',  # 20-day VIX MA
            'vix_term_structure',  # Contango/backwardation
            
            # Time features
            'dte',
            'dte_percent',  # DTE / original_DTE
            'theta_decay',  # Expected daily decay
            
            # Price features
            'underlying_price',
            'price_momentum_5d',
            'price_momentum_10d',
            'price_vs_ma20',
            'price_vs_ma50',
            
            # Option-specific features
            'delta',
            'gamma',
            'vega',
            'delta_change_ratio',  # Current delta / entry delta
            
            # Profit/Loss features
            'current_pnl_pct',
            'days_held',
            'premium_received',
            'strike_distance_pct',  # (price - strike) / price
            
            # Market regime
            'market_regime',  # 0=low vol, 1=normal, 2=high vol
        ]
        
        # Load pre-trained model if available
        if model_path:
            self.load_model(model_path)
    
    def build_features(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        current_date: str,
    ) -> pd.DataFrame:
        """Build feature vector for a single position.
        
        Args:
            position: Position dictionary with entry info
            market_data: Current market data
            current_date: Current date (YYYY-MM-DD)
            
        Returns:
            Feature DataFrame
        """
        try:
            entry_date = datetime.strptime(position.get('entry_date', ''), '%Y-%m-%d')
            curr_date = datetime.strptime(current_date, '%Y-%m-%d')
            days_held = (curr_date - entry_date).days
        except (ValueError, TypeError):
            days_held = 0
        
        # Time calculations
        try:
            expiry_date = datetime.strptime(position.get('expiry', ''), '%Y%m%d')
            dte = (expiry_date - curr_date).days
            original_dte = (expiry_date - entry_date).days
            dte_percent = dte / original_dte if original_dte > 0 else 0
        except (ValueError, TypeError, UnboundLocalError):
            dte = 0
            dte_percent = 0
        
        # Price and volatility
        underlying_price = market_data.get('price', position.get('underlying_price', 0))
        iv = market_data.get('iv', position.get('iv_at_entry', 0.3))
        hv = market_data.get('historical_volatility', iv)
        iv_rank = market_data.get('iv_rank', 50)
        
        # Entry details
        entry_price = position.get('entry_price', 0)
        entry_delta = position.get('delta_at_entry', 0.3)
        strike = position.get('strike', underlying_price)
        premium = entry_price * 100  # Option premium in dollars
        
        # Current P&L
        current_price = market_data.get('option_price', entry_price)
        quantity = position.get('quantity', -1)
        if quantity < 0:  # Short position
            current_pnl_pct = (entry_price - current_price) / entry_price * 100
        else:
            current_pnl_pct = (current_price - entry_price) / entry_price * 100
        
        # Delta change
        current_delta = market_data.get('delta', entry_delta)
        delta_change_ratio = abs(current_delta / entry_delta) if entry_delta != 0 else 1
        
        # Price momentum
        price_history = market_data.get('price_history', [])
        if len(price_history) >= 10:
            price_5d_ago = price_history[-5] if len(price_history) >= 5 else price_history[0]
            price_10d_ago = price_history[-10] if len(price_history) >= 10 else price_history[0]
            momentum_5d = (underlying_price - price_5d_ago) / price_5d_ago * 100
            momentum_10d = (underlying_price - price_10d_ago) / price_10d_ago * 100
        else:
            momentum_5d = 0
            momentum_10d = 0
        
        # Moving averages
        ma20 = market_data.get('ma20', underlying_price)
        ma50 = market_data.get('ma50', underlying_price)
        price_vs_ma20 = (underlying_price - ma20) / ma20 * 100
        price_vs_ma50 = (underlying_price - ma50) / ma50 * 100
        
        # Strike distance
        if position.get('right') == 'P':
            strike_distance_pct = (underlying_price - strike) / underlying_price * 100
        else:
            strike_distance_pct = (strike - underlying_price) / underlying_price * 100
        
        # Theta decay (simplified)
        theta_decay = premium * 0.05 / dte if dte > 0 else 0  # ~5% daily decay assumption
        
        # Market regime
        if iv_rank < 20:
            market_regime = 0  # Low vol
        elif iv_rank > 50:
            market_regime = 2  # High vol
        else:
            market_regime = 1  # Normal
        
        # Build feature vector
        features = {
            # Volatility
            'iv_rank': iv_rank,
            'historical_volatility': hv * 100,
            'iv_percentile': market_data.get('iv_percentile', iv_rank),
            
            # Fear/Greed (VIX)
            'vix': market_data.get('vix', 20.0),
            'vix_percentile': market_data.get('vix_percentile', 50.0),
            'vix_rank': market_data.get('vix_rank', 50.0),
            'vix_change_pct': market_data.get('vix_change_pct', 0.0),
            'vix_5d_ma': market_data.get('vix_5d_ma', 20.0),
            'vix_20d_ma': market_data.get('vix_20d_ma', 20.0),
            'vix_term_structure': market_data.get('vix_term_structure', 0.0),
            
            # Time
            'dte': dte,
            'dte_percent': dte_percent,
            'theta_decay': theta_decay,
            
            # Price
            'underlying_price': underlying_price,
            'price_momentum_5d': momentum_5d,
            'price_momentum_10d': momentum_10d,
            'price_vs_ma20': price_vs_ma20,
            'price_vs_ma50': price_vs_ma50,
            
            # Option
            'delta': abs(current_delta),
            'gamma': market_data.get('gamma', 0.05),
            'vega': market_data.get('vega', 0.2),
            'delta_change_ratio': delta_change_ratio,
            
            # P&L
            'current_pnl_pct': current_pnl_pct,
            'days_held': days_held,
            'premium_received': premium,
            'strike_distance_pct': strike_distance_pct,
            
            # Regime
            'market_regime': market_regime,
        }
        
        return pd.DataFrame([features])
    
    def predict_optimal_exits(
        self,
        features: pd.DataFrame,
        base_profit_target: float = 50.0,
        base_stop_loss: float = 200.0,
    ) -> Tuple[float, float]:
        """Predict optimal profit target and stop loss.
        
        Args:
            features: Feature DataFrame
            base_profit_target: Base profit target (%)
            base_stop_loss: Base stop loss (%)
            
        Returns:
            (profit_target, stop_loss) tuple
        """
        if self.model is None:
            # Fallback to rule-based optimization
            return self.rule_based_optimization(features, base_profit_target, base_stop_loss)
        
        try:
            # Get model prediction (probability of profit)
            prob_profit = self.model.predict_proba(features)[0][1]
            
            # Adjust targets based on predicted probability
            if prob_profit > 0.7:
                # High confidence: aggressive profit target, wide stop
                profit_target = base_profit_target * 0.7  # 35%
                stop_loss = base_stop_loss * 1.3  # 260%
            elif prob_profit > 0.5:
                # Medium confidence: standard targets
                profit_target = base_profit_target  # 50%
                stop_loss = base_stop_loss  # 200%
            else:
                # Low confidence: quick profit, tight stop
                profit_target = base_profit_target * 1.5  # 75%
                stop_loss = base_stop_loss * 0.7  # 140%
            
            return profit_target, stop_loss
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self.rule_based_optimization(features, base_profit_target, base_stop_loss)
    
    def rule_based_optimization(
        self,
        features: pd.DataFrame,
        base_profit_target: float = 50.0,
        base_stop_loss: float = 200.0,
    ) -> Tuple[float, float]:
        """Rule-based fallback for dynamic profit/stop adjustment.
        
        Rules based on:
        - IV Rank: High IV = higher profit target
        - DTE: Low DTE = reduce targets
        - Delta change: Delta翻倍 = 止损
        - Days held: Time-based adjustment
        """
        feat = features.iloc[0]
        
        profit_target = base_profit_target
        stop_loss = base_stop_loss
        
        # 1. IV Rank adjustment
        iv_rank = feat['iv_rank']
        if iv_rank > 50:  # High IV
            profit_target *= 1.5  # 75%
            stop_loss *= 0.8  # 160%
        elif iv_rank < 20:  # Low IV
            profit_target *= 0.7  # 35%
            stop_loss *= 1.3  # 260%
        
        # 2. DTE adjustment (last week)
        dte = feat['dte']
        if dte <= 7:
            profit_target *= 0.7  # Reduce to capture remaining premium
            stop_loss *= 1.3  # Widen to avoid whipsaw
        
        # 3. Delta change ratio (risk control)
        delta_ratio = feat['delta_change_ratio']
        if delta_ratio > 2.0:  # Delta doubled
            stop_loss = min(stop_loss, 100)  # Tight stop
        
        # 4. Days held adjustment
        days_held = feat['days_held']
        if days_held > 20:
            profit_target *= 0.6  # Just take profit
        elif days_held > 10:
            profit_target *= 0.8
        
        # 5. Market regime
        regime = feat['market_regime']
        if regime == 2:  # High vol
            profit_target *= 1.3
            stop_loss *= 0.9
        elif regime == 0:  # Low vol
            profit_target *= 0.8
            stop_loss *= 1.2
        
        return profit_target, stop_loss
    
    def should_exit_early(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        current_date: str,
    ) -> Tuple[bool, str]:
        """Determine if position should be closed early.
        
        Args:
            position: Position info
            market_data: Current market data
            current_date: Current date
            
        Returns:
            (should_exit, reason) tuple
        """
        features = self.build_features(position, market_data, current_date)
        feat = features.iloc[0]
        
        # Emergency stop conditions
        if feat['delta_change_ratio'] > 3.0:
            return True, "DELTA_SPIKE"  # Delta 翻 3 倍，立即止损
        
        if feat['current_pnl_pct'] < -150 and feat['dte'] > 14:
            return True, "EARLY_STOP_LOSS"  # 亏损 150% 且还有时间
        
        # Take profit conditions
        if feat['current_pnl_pct'] > 80 and feat['dte'] <= 3:
            return True, "EARLY_PROFIT"  # 已赚 80% 且临近到期
        
        # Time-based exit
        if feat['dte'] <= 2 and feat['current_pnl_pct'] > 20:
            return True, "TIME_EXIT"  # 只剩 2 天且盈利，提前平仓
        
        return False, ""
    
    def train_model(self, historical_trades: pd.DataFrame) -> Dict[str, float]:
        """Train the ML model on historical trade data.
        
        Args:
            historical_trades: DataFrame with columns:
                - All feature columns
                - 'outcome': 1=profitable, 0=loss
                - 'optimal_exit': Best exit point (for regression)
                
        Returns:
            Training metrics
        """
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            import xgboost as xgb
            
            # Prepare data
            X = historical_trades[self.feature_names]
            y = historical_trades['outcome']  # Binary classification
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.model = model
            logger.info(f"ML Exit Optimizer trained with accuracy: {accuracy:.2f}")
            
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
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk."""
        try:
            import joblib
            self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    @staticmethod
    def prepare_training_data(
        trades_history: list[dict],
        market_data: dict,
        use_real_greeks: bool = True
    ) -> pd.DataFrame:
        """Prepare training data from historical trades.
        
        Args:
            trades_history: List of completed trade dictionaries
            market_data: Historical market data for feature calculation
            use_real_greeks: If True, calculate real Greeks using Black-Scholes
            
        Returns:
            DataFrame with features and outcome labels
        """
        if not trades_history:
            return pd.DataFrame()
        
        from core.ml.market_data import MarketDataCalculator
        
        training_data = []
        
        for trade in trades_history:
            # Skip if missing essential data
            if not all(k in trade for k in ['entry_date', 'exit_date', 'symbol', 'pnl']):
                continue
            
            # Build feature vector at entry time
            entry_date = trade['entry_date']
            symbol = trade['symbol']
            
            # Get market data at entry
            market_at_entry = market_data.get(symbol, {}).get(entry_date, {})
            
            # Calculate Greeks if enabled
            if use_real_greeks:
                greeks = MarketDataCalculator.calculate_option_greeks_for_trade(
                    trade=trade,
                    market_data=market_at_entry,
                    current_date=entry_date
                )
                delta = abs(greeks.get('delta', 0.3))
                gamma = greeks.get('gamma', 0.05)
                vega = greeks.get('vega', 0.2)
            else:
                # Fallback to estimates
                delta = abs(trade.get('delta_at_entry', 0.3))
                gamma = 0.05
                vega = 0.2
            
            # Calculate features with real data
            features = {
                # Volatility - now with real IV Rank
                'iv_rank': market_at_entry.get('iv_rank', 50),
                'historical_volatility': market_at_entry.get('historical_volatility', 30),
                'iv_percentile': market_at_entry.get('iv_percentile', 50),
                
                # Time
                'dte': trade.get('days_to_expiry', 30),
                'dte_percent': 1.0,  # At entry
                'theta_decay': trade.get('premium', 0) * 0.05 / max(trade.get('days_to_expiry', 30), 1),
                
                # Price - now with real momentum
                'underlying_price': trade.get('underlying_entry', 0),
                'price_momentum_5d': market_at_entry.get('momentum_5d', 0),
                'price_momentum_10d': market_at_entry.get('momentum_10d', 0),
                'price_vs_ma20': market_at_entry.get('vs_ma20', 0),
                'price_vs_ma50': market_at_entry.get('vs_ma50', 0),
                
                # Option - now with real Greeks
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'delta_change_ratio': 1.0,  # At entry
                
                # P&L
                'current_pnl_pct': 0,  # At entry
                'days_held': 0,  # At entry
                'premium_received': trade.get('premium', 0) * 100,
                'strike_distance_pct': trade.get('strike_distance_pct', 0),
                
                # Regime
                'market_regime': market_at_entry.get('market_regime', 1),
            }
            
            # Label: 1 = profitable trade, 0 = loss
            outcome = 1 if trade.get('pnl', 0) > 0 else 0
            
            # Add to dataset
            row = {**features, 'outcome': outcome}
            training_data.append(row)
        
        return pd.DataFrame(training_data)
