"""Options pricing using Black-Scholes model."""

import math
from scipy.stats import norm


class OptionsPricer:
    """Black-Scholes options pricing and Greeks calculator."""

    RISK_FREE_RATE = 0.05  # 5% default

    @classmethod
    def call_price(cls, S: float, K: float, T: float, sigma: float, r: float | None = None) -> float:
        """Calculate European call option price.

        S: underlying price, K: strike, T: time to expiry in years,
        sigma: implied volatility, r: risk-free rate.
        """
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or sigma <= 0:
            return max(0, S - K)
        d1, d2 = cls._d1d2(S, K, T, sigma, r)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    @classmethod
    def put_price(cls, S: float, K: float, T: float, sigma: float, r: float | None = None) -> float:
        """Calculate European put option price."""
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or sigma <= 0:
            return max(0, K - S)
        d1, d2 = cls._d1d2(S, K, T, sigma, r)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @classmethod
    def delta(cls, S: float, K: float, T: float, sigma: float, right: str, r: float | None = None) -> float:
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or sigma <= 0:
            if right == "C":
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        d1, _ = cls._d1d2(S, K, T, sigma, r)
        if right == "C":
            return norm.cdf(d1)
        return norm.cdf(d1) - 1

    @classmethod
    def gamma(cls, S: float, K: float, T: float, sigma: float, r: float | None = None) -> float:
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1, _ = cls._d1d2(S, K, T, sigma, r)
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))

    @classmethod
    def theta(cls, S: float, K: float, T: float, sigma: float, right: str, r: float | None = None) -> float:
        """Theta per calendar day."""
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, d2 = cls._d1d2(S, K, T, sigma, r)
        common = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
        if right == "C":
            return (common - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
        return (common + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

    @classmethod
    def vega(cls, S: float, K: float, T: float, sigma: float, r: float | None = None) -> float:
        """Vega per 1% move in IV."""
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or sigma <= 0:
            return 0.0
        d1, _ = cls._d1d2(S, K, T, sigma, r)
        return S * math.sqrt(T) * norm.pdf(d1) / 100

    @classmethod
    def implied_volatility(
        cls, price: float, S: float, K: float, T: float, right: str,
        r: float | None = None, tol: float = 1e-6, max_iter: int = 100,
    ) -> float:
        """Calculate implied volatility from option price using Newton-Raphson."""
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or price <= 0:
            return 0.0

        sigma = 0.3  # initial guess
        price_fn = cls.call_price if right == "C" else cls.put_price

        for _ in range(max_iter):
            calc_price = price_fn(S, K, T, sigma, r)
            v = cls.vega(S, K, T, sigma, r) * 100  # vega is per 1%
            if abs(v) < 1e-12:
                break
            diff = calc_price - price
            if abs(diff) < tol:
                return sigma
            sigma -= diff / v
            sigma = max(0.001, min(sigma, 5.0))

        return sigma

    @classmethod
    def _d1d2(cls, S: float, K: float, T: float, sigma: float, r: float) -> tuple[float, float]:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    @classmethod
    def strike_from_delta(
        cls,
        S: float,
        T: float,
        sigma: float,
        target_delta: float,
        right: str,
        r: float | None = None,
        tol: float = 0.001,
        max_iter: int = 100,
    ) -> float:
        """Find strike that gives the target delta.

        Args:
            S: Underlying price
            T: Time to expiry in years
            sigma: Implied volatility
            target_delta: Target delta (positive for calls, negative for puts)
            right: 'C' for call, 'P' for put
            r: Risk-free rate

        Returns:
            Strike price that achieves the target delta
        """
        r = r if r is not None else cls.RISK_FREE_RATE

        if T <= 0 or sigma <= 0:
            # At expiry, return ATM
            return S

        # Initial guess: use delta approximation
        # For OTM put with delta -0.30, strike should be below S
        # For OTM call with delta 0.30, strike should be above S
        if right == 'P':
            # Put delta is negative
            # For OTM put, strike < S
            # norm.ppf(|delta|) gives us the z-score
            # For OTM put (|delta| < 0.5), we need strike below S
            target_abs_delta = abs(target_delta)
            # Use abs to ensure we get strike < S
            K = S * math.exp(-abs(norm.ppf(target_abs_delta)) * sigma * math.sqrt(T))
        else:
            # Call delta is positive
            # For OTM call, strike > S
            # norm.ppf(delta) is negative when delta < 0.5
            # We need to use |norm.ppf(delta)| to get strike > S
            K = S * math.exp(abs(norm.ppf(target_delta)) * sigma * math.sqrt(T))

        # Newton-Raphson iteration
        for _ in range(max_iter):
            current_delta = cls.delta(S, K, T, sigma, right, r)

            # Difference from target (use absolute value for puts)
            if right == 'P':
                diff = current_delta - target_delta  # both should be negative
            else:
                diff = current_delta - target_delta

            if abs(diff) < tol:
                return K

            # Approximate dDelta/dK
            # For calls: dDelta/dK ≈ -norm.pdf(d1) * e^(-rT) / (S * sigma * sqrt(T))
            # Use numerical derivative for simplicity
            dK = K * 0.01
            delta_up = cls.delta(S, K + dK, T, sigma, right, r)
            delta_down = cls.delta(S, K - dK, T, sigma, right, r)
            ddelta_dK = (delta_up - delta_down) / (2 * dK)

            if abs(ddelta_dK) < 1e-12:
                break

            # Update strike
            K = K - diff / ddelta_dK
            K = max(S * 0.5, min(K, S * 2.0))  # Bound the strike

        return K
