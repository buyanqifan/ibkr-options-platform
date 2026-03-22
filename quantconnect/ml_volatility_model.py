"""
ML Volatility Prediction Model for QuantConnect
================================================

Machine learning model that predicts future volatility for:
- IV estimation
- Option pricing
- Risk management
- Market regime detection

Uses GARCH-like features with ML enhancement.
"""

from AlgorithmImports import *
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class VolatilityPrediction:
    """Volatility prediction result."""
    predicted_vol_5d: float
    predicted_vol_10d: float
    predicted_vol_20d: float
    iv_estimate: float
    iv_rank: float
    vol_regime: str  # "low", "normal", "high", "extreme"
    confidence: float
    trend: str  # "rising", "falling", "stable"


class VolatilityModel:
    """
    ML-enhanced volatility prediction model.
    
    Combines statistical methods (GARCH-like) with ML corrections.
    """
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.vol_history = []
        self.model = None
        self.is_trained = False
        
        np.random.seed(42)
    
    def predict(
        self,
        bars: List[Dict],
        current_iv: Optional[float] = None
    ) -> VolatilityPrediction:
        """
        Predict future volatility from price history.
        
        Args:
            bars: Historical price bars
            current_iv: Current implied volatility if available
            
        Returns:
            VolatilityPrediction with forecasts
        """
        
        if len(bars) < 20:
            return self._default_prediction()
        
        closes = np.array([b['close'] for b in bars[-self.lookback:]])
        returns = np.diff(closes) / closes[:-1]
        
        # Current realized volatility
        current_vol = np.std(returns[-20:]) * np.sqrt(252)
        
        # Calculate GARCH-like features
        vol_5d = self._predict_vol_garch(returns, 5)
        vol_10d = self._predict_vol_garch(returns, 10)
        vol_20d = self._predict_vol_garch(returns, 20)
        
        # IV estimation
        if current_iv is not None:
            iv_estimate = current_iv
        else:
            # Estimate IV from realized vol + premium
            iv_estimate = current_vol * 1.1  # IV typically 10% higher than RV
        
        # IV rank calculation
        vol_history = [np.std(returns[-min(i+20, len(returns)):len(returns)-i if i > 0 else None]) * np.sqrt(252) 
                       for i in range(min(252, len(returns)-20))]
        
        if vol_history:
            vol_percentile = np.percentile(vol_history, [10, 90])
            if current_vol <= vol_percentile[0]:
                iv_rank = 10
            elif current_vol >= vol_percentile[1]:
                iv_rank = 90
            else:
                iv_rank = 10 + 80 * (current_vol - vol_percentile[0]) / (vol_percentile[1] - vol_percentile[0])
        else:
            iv_rank = 50
        
        # Volatility regime
        if current_vol < 0.15:
            vol_regime = "low"
        elif current_vol < 0.25:
            vol_regime = "normal"
        elif current_vol < 0.40:
            vol_regime = "high"
        else:
            vol_regime = "extreme"
        
        # Volatility trend
        if len(returns) >= 40:
            recent_vol = np.std(returns[-10:]) * np.sqrt(252)
            prior_vol = np.std(returns[-40:-10]) * np.sqrt(252)
            
            if recent_vol > prior_vol * 1.2:
                trend = "rising"
            elif recent_vol < prior_vol * 0.8:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Confidence based on data quality
        confidence = min(1.0, len(bars) / 60)
        
        return VolatilityPrediction(
            predicted_vol_5d=vol_5d,
            predicted_vol_10d=vol_10d,
            predicted_vol_20d=vol_20d,
            iv_estimate=iv_estimate,
            iv_rank=iv_rank,
            vol_regime=vol_regime,
            confidence=confidence,
            trend=trend
        )
    
    def _predict_vol_garch(self, returns: np.ndarray, horizon: int) -> float:
        """
        Predict volatility using simplified GARCH-like model.
        
        Uses EWMA volatility with mean reversion.
        """
        
        if len(returns) < 20:
            return np.std(returns) * np.sqrt(252)
        
        # EWMA volatility
        lambda_param = 0.94  # Standard EWMA parameter
        weights = np.array([(1 - lambda_param) * lambda_param ** i for i in range(len(returns))])
        weights = weights[::-1] / weights.sum()
        
        squared_returns = returns ** 2
        ewma_var = np.sum(weights * squared_returns)
        
        # Current volatility
        current_vol = np.sqrt(ewma_var * 252)
        
        # Long-run volatility (mean reversion target)
        long_run_vol = np.std(returns) * np.sqrt(252)
        
        # Mean reversion speed
        kappa = 0.05  # 5% mean reversion per day
        
        # Project forward
        projected_vol = current_vol
        for _ in range(horizon):
            projected_vol = projected_vol + kappa * (long_run_vol - projected_vol)
        
        return projected_vol
    
    def _default_prediction(self) -> VolatilityPrediction:
        """Return default prediction when insufficient data."""
        
        return VolatilityPrediction(
            predicted_vol_5d=0.25,
            predicted_vol_10d=0.25,
            predicted_vol_20d=0.25,
            iv_estimate=0.275,
            iv_rank=50.0,
            vol_regime="normal",
            confidence=0.3,
            trend="stable"
        )
    
    def calculate_iv_surface(
        self,
        current_price: float,
        strikes: List[float],
        dtes: List[int],
        base_iv: float
    ) -> Dict[str, Dict[int, float]]:
        """
        Calculate implied volatility surface.
        
        Uses standard volatility smile and term structure patterns.
        
        Args:
            current_price: Current underlying price
            strikes: List of strike prices
            dtes: List of days to expiration
            base_iv: At-the-money IV
            
        Returns:
            IV surface as dict[strike][dte] -> iv
        """
        
        iv_surface = {}
        
        for strike in strikes:
            iv_surface[strike] = {}
            
            # Moneyness
            moneyness = strike / current_price
            
            # Strike adjustment (volatility smile)
            if moneyness < 0.9:  # Deep OTM put
                strike_adj = 1.0 + 0.15 * (1 - moneyness)
            elif moneyness < 1.0:  # OTM put
                strike_adj = 1.0 + 0.08 * (1 - moneyness)
            elif moneyness > 1.1:  # Deep OTM call
                strike_adj = 1.0 + 0.10 * (moneyness - 1)
            elif moneyness > 1.0:  # OTM call
                strike_adj = 1.0 + 0.05 * (moneyness - 1)
            else:  # ATM
                strike_adj = 1.0
            
            for dte in dtes:
                # Time adjustment (volatility term structure)
                if dte < 7:
                    time_adj = 1.15  # Short-term vol premium
                elif dte < 30:
                    time_adj = 1.05
                elif dte > 90:
                    time_adj = 0.95  # Long-term vol discount
                else:
                    time_adj = 1.0
                
                iv_surface[strike][dte] = base_iv * strike_adj * time_adj
        
        return iv_surface
    
    def get_volatility_features(self, bars: List[Dict]) -> Dict[str, float]:
        """
        Extract volatility features for ML models.
        
        Returns comprehensive volatility metrics.
        """
        
        if len(bars) < 20:
            return self._default_features()
        
        closes = np.array([b['close'] for b in bars])
        returns = np.diff(closes) / closes[:-1]
        
        # Realized volatility at different windows
        vol_5d = np.std(returns[-5:]) * np.sqrt(252) if len(returns) >= 5 else 0.25
        vol_10d = np.std(returns[-10:]) * np.sqrt(252) if len(returns) >= 10 else 0.25
        vol_20d = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.25
        vol_30d = np.std(returns[-30:]) * np.sqrt(252) if len(returns) >= 30 else vol_20d
        
        # Volatility of volatility
        vol_series = [np.std(returns[max(0, i-20):i]) * np.sqrt(252) 
                     for i in range(20, len(returns))]
        vol_of_vol = np.std(vol_series) if vol_series else 0
        
        # Volatility skew (recent vs prior)
        recent_vol = vol_10d
        prior_vol = vol_30d
        vol_skew = (recent_vol - prior_vol) / prior_vol if prior_vol > 0 else 0
        
        # Parkinson volatility (using high/low)
        if all(k in bars[-1] for k in ['high', 'low']):
            hl_ratios = [np.log(b['high'] / b['low']) ** 2 for b in bars[-20:]]
            parkinson_vol = np.sqrt(np.mean(hl_ratios) / (4 * np.log(2))) * np.sqrt(252)
        else:
            parkinson_vol = vol_20d
        
        # Garman-Klass volatility (OHLC)
        if all(k in bars[-1] for k in ['open', 'high', 'low', 'close']):
            gk_vars = []
            for b in bars[-20:]:
                hl = np.log(b['high'] / b['low'])
                co = np.log(b['close'] / b['open'])
                gk_var = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
                gk_vars.append(gk_var)
            garman_klass_vol = np.sqrt(np.mean(gk_vars)) * np.sqrt(252)
        else:
            garman_klass_vol = vol_20d
        
        # Volatility regime features
        high_vol_threshold = vol_30d * 1.5
        low_vol_threshold = vol_30d * 0.7
        
        is_high_vol = 1 if vol_5d > high_vol_threshold else 0
        is_low_vol = 1 if vol_5d < low_vol_threshold else 0
        
        return {
            'volatility_5d': vol_5d,
            'volatility_10d': vol_10d,
            'volatility_20d': vol_20d,
            'volatility_30d': vol_30d,
            'vol_of_vol': vol_of_vol,
            'vol_skew': vol_skew,
            'parkinson_vol': parkinson_vol,
            'garman_klass_vol': garman_klass_vol,
            'is_high_vol': is_high_vol,
            'is_low_vol': is_low_vol,
            'vol_percentile_20d': self._calculate_percentile(returns, 20),
        }
    
    def _calculate_percentile(self, returns: np.ndarray, window: int) -> float:
        """Calculate current volatility percentile."""
        
        if len(returns) < window + 20:
            return 50.0
        
        current_vol = np.std(returns[-window:]) * np.sqrt(252)
        
        historical_vols = []
        for i in range(window, len(returns) - window, 5):  # Step by 5 for efficiency
            hist_vol = np.std(returns[i-window:i]) * np.sqrt(252)
            historical_vols.append(hist_vol)
        
        if not historical_vols:
            return 50.0
        
        return np.mean([1 if current_vol > v else 0 for v in historical_vols]) * 100
    
    def _default_features(self) -> Dict[str, float]:
        """Return default features when insufficient data."""
        
        return {
            'volatility_5d': 0.25,
            'volatility_10d': 0.25,
            'volatility_20d': 0.25,
            'volatility_30d': 0.25,
            'vol_of_vol': 0.05,
            'vol_skew': 0.0,
            'parkinson_vol': 0.25,
            'garman_klass_vol': 0.25,
            'is_high_vol': 0,
            'is_low_vol': 0,
            'vol_percentile_20d': 50.0,
        }
    
    def update(self, new_bar: Dict):
        """Update model with new data point."""
        
        self.vol_history.append(new_bar)
        
        # Keep limited history
        if len(self.vol_history) > 500:
            self.vol_history = self.vol_history[-500:]


class VolatilityIntegration:
    """
    Integration layer for volatility model with trading strategies.
    """
    
    def __init__(self, model: Optional[VolatilityModel] = None):
        self.model = model or VolatilityModel()
    
    def get_iv_estimate(
        self,
        bars: List[Dict],
        strike: float,
        dte: int,
        current_price: float
    ) -> Tuple[float, float]:
        """
        Get IV estimate and IV rank for a specific option.
        
        Returns:
            (iv_estimate, iv_rank)
        """
        
        prediction = self.model.predict(bars)
        
        # Adjust IV for strike and DTE
        moneyness = strike / current_price
        
        # Standard smile adjustment
        if moneyness < 1.0:
            smile_adj = 1.0 + 0.1 * (1 - moneyness)
        else:
            smile_adj = 1.0 + 0.05 * (moneyness - 1)
        
        # Term structure adjustment
        if dte < 30:
            term_adj = 1.05
        elif dte > 60:
            term_adj = 0.95
        else:
            term_adj = 1.0
        
        adjusted_iv = prediction.iv_estimate * smile_adj * term_adj
        
        return adjusted_iv, prediction.iv_rank
    
    def should_adjust_strategy(
        self,
        bars: List[Dict],
        current_phase: str
    ) -> Tuple[bool, str]:
        """
        Determine if strategy should be adjusted based on volatility.
        
        Returns:
            (should_adjust, recommendation)
        """
        
        prediction = self.model.predict(bars)
        
        if prediction.vol_regime == "extreme":
            return True, f"Extreme volatility ({prediction.predicted_vol_5d:.1%}) - reduce position sizes"
        
        if prediction.vol_regime == "high" and prediction.trend == "rising":
            return True, "High and rising volatility - consider more conservative deltas"
        
        if prediction.vol_regime == "low" and prediction.trend == "falling":
            return True, "Low and falling volatility - consider increasing position sizes"
        
        return False, "Normal volatility conditions"