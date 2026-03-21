"""Market data calculator for ML features.

Calculates real-time market metrics including:
- IV Rank (from options chain)
- Historical volatility
- Greeks (Delta, Gamma, Vega, Theta)
- Price momentum indicators
- VIX (fear/greed index) and term structure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class MarketDataCalculator:
    """Calculate market data features for ML exit optimizer."""
    
    @staticmethod
    def calculate_iv_rank(
        current_iv: float,
        iv_52w_high: float,
        iv_52w_low: float
    ) -> float:
        """Calculate IV Rank (0-100).
        
        IV Rank = (Current IV - IV Low) / (IV High - IV Low) * 100
        
        Args:
            current_iv: Current implied volatility
            iv_52w_high: 52-week IV high
            iv_52w_low: 52-week IV low
            
        Returns:
            IV Rank (0-100)
        """
        if iv_52w_high == iv_52w_low:
            return 50.0
        
        iv_rank = (current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100
        return max(0.0, min(100.0, iv_rank))
    
    @staticmethod
    def calculate_historical_volatility(
        prices: List[float],
        window: int = 20,
        annualization: int = 252
    ) -> float:
        """Calculate historical volatility (annualized).
        
        Args:
            prices: List of closing prices
            window: Rolling window size (default: 20 days)
            annualization: Annualization factor (default: 252 trading days)
            
        Returns:
            Historical volatility (percentage)
        """
        if len(prices) < window + 1:
            return 0.3  # Default 30%
        
        # Calculate log returns
        prices_array = np.array(prices[-window-1:])
        returns = np.log(prices_array[1:] / prices_array[:-1])
        
        # Calculate standard deviation
        std_dev = np.std(returns, ddof=1)
        
        # Annualize
        hv = std_dev * np.sqrt(annualization) * 100
        
        return hv
    
    @staticmethod
    def calculate_greeks(
        underlying_price: float,
        strike: float,
        time_to_expiry: float,
        iv: float,
        risk_free_rate: float = 0.05,
        option_type: str = "P"
    ) -> Dict[str, float]:
        """Calculate option Greeks using Black-Scholes model.
        
        Args:
            underlying_price: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            iv: Implied volatility
            risk_free_rate: Risk-free interest rate
            option_type: 'P' for put, 'C' for call
            
        Returns:
            Dictionary with delta, gamma, vega, theta
        """
        if time_to_expiry <= 0 or iv <= 0:
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0
            }
        
        # Black-Scholes components
        d1 = (
            np.log(underlying_price / strike) +
            (risk_free_rate + 0.5 * iv**2) * time_to_expiry
        ) / (iv * np.sqrt(time_to_expiry))
        
        d2 = d1 - iv * np.sqrt(time_to_expiry)
        
        # Calculate Greeks
        sqrt_t = np.sqrt(time_to_expiry)
        
        # Gamma (same for both calls and puts)
        gamma = norm.pdf(d1) / (underlying_price * iv * sqrt_t)
        
        # Vega (same for both calls and puts)
        vega = underlying_price * sqrt_t * norm.pdf(d1) / 100  # Per 1% change
        
        # Theta (per day)
        term1 = -underlying_price * norm.pdf(d1) * iv / (2 * sqrt_t * 365)
        term2 = risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) / 365
        
        if option_type.upper() == "C":
            theta = term1 - term2 * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:  # Put
            theta = term1 + term2 * norm.cdf(-d2)
            delta = -norm.cdf(-d1)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }
    
    @staticmethod
    def calculate_momentum_indicators(
        prices: List[float],
        current_idx: int
    ) -> Dict[str, float]:
        """Calculate momentum indicators.
        
        Args:
            prices: List of closing prices
            current_idx: Current index in price series
            
        Returns:
            Dictionary with momentum metrics
        """
        if current_idx < 10:
            return {
                'momentum_5d': 0.0,
                'momentum_10d': 0.0,
                'vs_ma20': 0.0,
                'vs_ma50': 0.0
            }
        
        current_price = prices[current_idx]
        
        # 5-day momentum
        price_5d = prices[current_idx - 5] if current_idx >= 5 else prices[0]
        momentum_5d = (current_price - price_5d) / price_5d * 100
        
        # 10-day momentum
        price_10d = prices[current_idx - 10] if current_idx >= 10 else prices[0]
        momentum_10d = (current_price - price_10d) / price_10d * 100
        
        # 20-day MA
        ma20_start = max(0, current_idx - 19)
        ma20 = np.mean(prices[ma20_start:current_idx + 1])
        vs_ma20 = (current_price - ma20) / ma20 * 100
        
        # 50-day MA
        ma50_start = max(0, current_idx - 49)
        ma50 = np.mean(prices[ma50_start:current_idx + 1])
        vs_ma50 = (current_price - ma50) / ma50 * 100
        
        return {
            'momentum_5d': momentum_5d,
            'momentum_10d': momentum_10d,
            'vs_ma20': vs_ma20,
            'vs_ma50': vs_ma50
        }
    
    @staticmethod
    def calculate_iv_percentile(
        current_iv: float,
        historical_ivs: List[float],
        window: int = 252
    ) -> float:
        """Calculate IV Percentile.
        
        IV Percentile = percentage of days in past year where IV was lower
        
        Args:
            current_iv: Current implied volatility
            historical_ivs: List of historical IV values
            window: Lookback window (default: 252 days = 1 year)
            
        Returns:
            IV Percentile (0-100)
        """
        if not historical_ivs:
            return 50.0
        
        # Use last 'window' days
        ivs = historical_ivs[-window:] if len(historical_ivs) >= window else historical_ivs
        
        # Count days with lower IV
        days_lower = sum(1 for iv in ivs if iv < current_iv)
        
        iv_percentile = days_lower / len(ivs) * 100
        
        return iv_percentile
    
    @staticmethod
    def calculate_vix_features(
        current_vix: Optional[float] = None,
        vix_history: Optional[List[float]] = None,
        vix9d: Optional[float] = None,
        vix3m: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate VIX (fear index) related features.
        
        Args:
            current_vix: Current VIX level
            vix_history: Historical VIX values (last 20-90 days)
            vix9d: 9-day forward VIX (if available)
            vix3m: 3-month forward VIX (if available)
            
        Returns:
            Dictionary with VIX features
        """
        if current_vix is None:
            # Default values when VIX data not available
            return {
                'vix': 20.0,  # Long-term average
                'vix_percentile': 50.0,
                'vix_rank': 50.0,
                'vix_change_pct': 0.0,
                'vix_5d_ma': 20.0,
                'vix_20d_ma': 20.0,
                'vix_term_structure': 0.0,
            }
        
        # VIX percentile (where current VIX sits in recent history)
        if vix_history and len(vix_history) > 0:
            vix_percentile = MarketDataCalculator.calculate_iv_percentile(
                current_vix, vix_history
            )
            vix_rank = MarketDataCalculator.calculate_iv_rank(
                current_vix,
                max(vix_history) if vix_history else current_vix * 1.5,
                min(vix_history) if vix_history else current_vix * 0.5
            )
        else:
            vix_percentile = 50.0
            vix_rank = 50.0
        
        # VIX change (5-day)
        vix_change_pct = 0.0
        if vix_history and len(vix_history) >= 5:
            vix_5d_ago = vix_history[-5]
            if vix_5d_ago > 0:
                vix_change_pct = (current_vix - vix_5d_ago) / vix_5d_ago * 100
        
        # VIX moving averages
        vix_5d_ma = current_vix
        vix_20d_ma = current_vix
        if vix_history:
            if len(vix_history) >= 5:
                vix_5d_ma = np.mean(vix_history[-5:])
            if len(vix_history) >= 20:
                vix_20d_ma = np.mean(vix_history[-20:])
        
        # VIX term structure (contango/backwardation)
        # Positive = contango (normal), Negative = backwardation (fear)
        vix_term_structure = 0.0
        if vix9d is not None and current_vix > 0:
            vix_term_structure = (vix9d - current_vix) / current_vix * 100
        
        return {
            'vix': current_vix,
            'vix_percentile': vix_percentile,
            'vix_rank': vix_rank,
            'vix_change_pct': vix_change_pct,
            'vix_5d_ma': vix_5d_ma,
            'vix_20d_ma': vix_20d_ma,
            'vix_term_structure': vix_term_structure,
        }
    
    @staticmethod
    def build_market_data_snapshot(
        prices: List[float],
        current_idx: int,
        iv_data: Optional[Dict[str, float]] = None,
        vix_data: Optional[Dict[str, float]] = None,
        risk_free_rate: float = 0.05
    ) -> Dict[str, float]:
        """Build complete market data snapshot for ML features.
        
        Args:
            prices: List of closing prices
            current_idx: Current index in price series
            iv_data: Optional IV data {'current_iv', 'iv_52w_high', 'iv_52w_low', 'historical_ivs'}
            vix_data: Optional VIX data {'current_vix', 'vix_history', 'vix9d', 'vix3m'}
            risk_free_rate: Risk-free interest rate
            
        Returns:
            Complete market data dictionary with VIX features
        """
        current_price = prices[current_idx]
        
        # Calculate momentum indicators
        momentum = MarketDataCalculator.calculate_momentum_indicators(prices, current_idx)
        
        # Calculate historical volatility
        hv = MarketDataCalculator.calculate_historical_volatility(prices)
        
        # Use IV data if available, otherwise estimate from HV
        if iv_data:
            current_iv = iv_data.get('current_iv', hv / 100)
            iv_rank = MarketDataCalculator.calculate_iv_rank(
                current_iv,
                iv_data.get('iv_52w_high', current_iv * 1.5),
                iv_data.get('iv_52w_low', current_iv * 0.5)
            )
            iv_percentile = MarketDataCalculator.calculate_iv_percentile(
                current_iv,
                iv_data.get('historical_ivs', [])
            )
        else:
            # Fallback: estimate IV from HV
            current_iv = hv / 100
            iv_rank = 50.0
            iv_percentile = 50.0
        
        # Calculate VIX features
        vix_features = MarketDataCalculator.calculate_vix_features(
            current_vix=vix_data.get('current_vix') if vix_data else None,
            vix_history=vix_data.get('vix_history') if vix_data else None,
            vix9d=vix_data.get('vix9d') if vix_data else None,
            vix3m=vix_data.get('vix3m') if vix_data else None
        )
        
        # Determine market regime (now considers both IV and VIX)
        if iv_rank < 20 and vix_features['vix_percentile'] < 30:
            market_regime = 0  # Low vol / complacent
        elif iv_rank > 50 or vix_features['vix_percentile'] > 70:
            market_regime = 2  # High vol / fear
        else:
            market_regime = 1  # Normal
        
        return {
            'price': current_price,
            'historical_volatility': hv,
            'iv_rank': iv_rank,
            'iv_percentile': iv_percentile,
            'current_iv': current_iv,
            **momentum,
            **vix_features,  # VIX features
            'market_regime': market_regime,
            'risk_free_rate': risk_free_rate
        }
    
    @staticmethod
    def calculate_option_greeks_for_trade(
        trade: Dict[str, any],
        market_data: Dict[str, float],
        current_date: str
    ) -> Dict[str, float]:
        """Calculate current Greeks for a specific trade.
        
        Args:
            trade: Trade dictionary with option details
            market_data: Current market data
            current_date: Current date (YYYY-MM-DD)
            
        Returns:
            Greeks dictionary
        """
        try:
            # Extract trade details
            strike = trade.get('strike', 0)
            expiry = trade.get('expiry', '')
            right = trade.get('right', 'P')
            entry_price = trade.get('entry_price', 0)
            
            # Calculate time to expiry
            try:
                expiry_date = datetime.strptime(expiry, '%Y%m%d')
                curr_date = datetime.strptime(current_date, '%Y-%m-%d')
                days_to_expiry = (expiry_date - curr_date).days
                time_to_expiry = days_to_expiry / 365.0
            except (ValueError, TypeError):
                time_to_expiry = 0.01
            
            # Get current IV (use entry IV or estimate)
            current_iv = trade.get('current_iv', market_data.get('current_iv', 0.3))
            
            # Calculate Greeks
            greeks = MarketDataCalculator.calculate_greeks(
                underlying_price=market_data.get('price', trade.get('underlying_price', 0)),
                strike=strike,
                time_to_expiry=time_to_expiry,
                iv=current_iv,
                risk_free_rate=market_data.get('risk_free_rate', 0.05),
                option_type=right
            )
            
            return greeks
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0
            }
