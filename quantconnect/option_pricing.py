"""
Black-Scholes Option Pricing Module for QuantConnect
=====================================================

Provides option pricing and Greeks calculation using the Black-Scholes model.
"""

from AlgorithmImports import *
from math import log, sqrt, exp
from typing import Optional


class BlackScholes:
    """
    Black-Scholes option pricing model.
    
    Provides static methods for option pricing and Greeks calculation.
    """
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        return d1_val - sigma * sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate call option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (in years)
            r: Risk-free interest rate
            sigma: Implied volatility
            
        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0.0)
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        d2_val = BlackScholes.d2(S, K, T, r, sigma)
        
        from scipy.stats import norm
        call = S * norm.cdf(d1_val) - K * exp(-r * T) * norm.cdf(d2_val)
        return max(call, 0.01)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate put option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (in years)
            r: Risk-free interest rate
            sigma: Implied volatility
            
        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0.0)
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        d2_val = BlackScholes.d2(S, K, T, r, sigma)
        
        from scipy.stats import norm
        put = K * exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)
        return max(put, 0.01)
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate option delta.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (in years)
            r: Risk-free interest rate
            sigma: Implied volatility
            option_type: 'C' for call, 'P' for put
            
        Returns:
            Option delta
        """
        if T <= 0:
            if option_type == 'C':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        from scipy.stats import norm
        
        if option_type == 'C':
            return norm.cdf(d1_val)
        else:
            return norm.cdf(d1_val) - 1
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate option gamma (same for calls and puts).
        
        Returns:
            Option gamma
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        from scipy.stats import norm
        
        return norm.pdf(d1_val) / (S * sigma * sqrt(T))
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate option theta (time decay).
        
        Returns:
            Option theta (per day)
        """
        if T <= 0:
            return 0.0
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        d2_val = BlackScholes.d2(S, K, T, r, sigma)
        
        from scipy.stats import norm
        
        first_term = -S * norm.pdf(d1_val) * sigma / (2 * sqrt(T))
        
        if option_type == 'C':
            second_term = -r * K * exp(-r * T) * norm.cdf(d2_val)
            theta = (first_term + second_term) / 365  # Per day
        else:
            second_term = r * K * exp(-r * T) * norm.cdf(-d2_val)
            theta = (first_term + second_term) / 365  # Per day
        
        return theta
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate option vega (sensitivity to volatility).
        
        Returns:
            Option vega (per 1% change in volatility)
        """
        if T <= 0:
            return 0.0
        
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        from scipy.stats import norm
        
        return S * norm.pdf(d1_val) * sqrt(T) / 100  # Per 1% IV change
    
    @staticmethod
    def implied_volatility(
        option_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str,
        max_iterations: int = 100,
        tolerance: float = 0.0001
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            option_price: Observed option price
            S: Current stock price
            K: Strike price
            T: Time to expiry (in years)
            r: Risk-free interest rate
            option_type: 'C' for call, 'P' for put
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility or None if not converged
        """
        if T <= 0:
            return None
        
        # Initial guess
        sigma = 0.3
        
        for _ in range(max_iterations):
            if option_type == 'C':
                price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                price = BlackScholes.put_price(S, K, T, r, sigma)
            
            vega = BlackScholes.vega(S, K, T, r, sigma) * 100  # Full vega
            
            if abs(vega) < 1e-10:
                break
            
            diff = option_price - price
            
            if abs(diff) < tolerance:
                return sigma
            
            sigma = sigma + diff / vega
            
            # Bound sigma to reasonable range
            sigma = max(0.01, min(5.0, sigma))
        
        return sigma
    
    @staticmethod
    def strike_from_delta(
        S: float,
        T: float,
        r: float,
        sigma: float,
        target_delta: float,
        option_type: str
    ) -> float:
        """
        Find strike price that gives target delta.
        
        Uses binary search to find the strike.
        
        Args:
            S: Current stock price
            T: Time to expiry (in years)
            r: Risk-free interest rate
            sigma: Implied volatility
            target_delta: Target delta value
            option_type: 'C' for call, 'P' for put
            
        Returns:
            Strike price that gives approximately target delta
        """
        if T <= 0:
            return S  # ATM at expiry
        
        # Set search bounds based on option type
        if option_type == 'C':
            # For calls: higher strike = lower delta
            # Target delta is positive (0.3 means OTM)
            low = S * 0.8
            high = S * 1.2
            target = abs(target_delta)
        else:
            # For puts: lower strike = less negative delta (closer to 0)
            # Target delta is negative (-0.3 means OTM)
            low = S * 0.8
            high = S * 1.2
            target = target_delta  # Negative value
        
        # Binary search
        for _ in range(50):
            mid = (low + high) / 2
            delta = BlackScholes.delta(S, mid, T, r, sigma, option_type)
            
            if option_type == 'C':
                if delta > target:
                    low = mid
                else:
                    high = mid
            else:
                if delta < target:  # More negative
                    high = mid
                else:
                    low = mid
            
            if abs(delta - target) < 0.005:
                break
        
        # Round to nearest dollar or half-dollar
        if S > 100:
            return round(mid)
        else:
            return round(mid * 2) / 2


class OptionsPricer:
    """
    Convenience class for option pricing with default parameters.
    """
    
    DEFAULT_RISK_FREE_RATE = 0.05
    
    @classmethod
    def set_risk_free_rate(cls, rate: float):
        """Set default risk-free rate."""
        cls.DEFAULT_RISK_FREE_RATE = rate
    
    @classmethod
    def call_price(cls, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate call price with default risk-free rate."""
        return BlackScholes.call_price(S, K, T, cls.DEFAULT_RISK_FREE_RATE, sigma)
    
    @classmethod
    def put_price(cls, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate put price with default risk-free rate."""
        return BlackScholes.put_price(S, K, T, cls.DEFAULT_RISK_FREE_RATE, sigma)
    
    @classmethod
    def delta(cls, S: float, K: float, T: float, sigma: float, option_type: str) -> float:
        """Calculate delta with default risk-free rate."""
        return BlackScholes.delta(S, K, T, cls.DEFAULT_RISK_FREE_RATE, sigma, option_type)
    
    @classmethod
    def strike_from_delta(cls, S: float, T: float, sigma: float, target_delta: float, option_type: str) -> float:
        """Find strike from target delta with default risk-free rate."""
        return BlackScholes.strike_from_delta(S, T, cls.DEFAULT_RISK_FREE_RATE, sigma, target_delta, option_type)