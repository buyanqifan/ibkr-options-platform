"""Options pricing using optionlab Black-Scholes implementation."""

import math
from scipy.stats import norm
from optionlab.black_scholes import (
    get_option_price,
    get_delta,
    get_gamma,
    get_theta,
    get_vega,
    get_implied_vol,
    get_d1,
    get_d2,
    get_bs_info,
    get_probability_of_touch,
)


class OptionsPricer:
    """Black-Scholes options pricing and Greeks calculator using optionlab."""

    RISK_FREE_RATE = 0.05  # 5% default

    @classmethod
    def call_price(cls, S: float, K: float, T: float, sigma: float, r: float | None = None) -> float:
        """Calculate European call option price.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            r: Risk-free rate (default: 5%)

        Returns:
            Call option price
        """
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or sigma <= 0:
            return max(0, S - K)
        d1 = get_d1(S, K, r, sigma, T)
        d2 = get_d2(S, K, r, sigma, T)
        return float(get_option_price('call', S, K, r, T, d1, d2))

    @classmethod
    def put_price(cls, S: float, K: float, T: float, sigma: float, r: float | None = None) -> float:
        """Calculate European put option price.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            r: Risk-free rate (default: 5%)

        Returns:
            Put option price
        """
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or sigma <= 0:
            return max(0, K - S)
        d1 = get_d1(S, K, r, sigma, T)
        d2 = get_d2(S, K, r, sigma, T)
        return float(get_option_price('put', S, K, r, T, d1, d2))

    @classmethod
    def delta(cls, S: float, K: float, T: float, sigma: float, right: str, r: float | None = None) -> float:
        """Calculate option delta.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            right: Option type ('C' for call, 'P' for put)
            r: Risk-free rate (default: 5%)

        Returns:
            Option delta
        """
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or sigma <= 0:
            if right == "C":
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        d1 = get_d1(S, K, r, sigma, T)
        opt_type = 'call' if right == 'C' else 'put'
        return float(get_delta(opt_type, d1, T))

    @classmethod
    def gamma(cls, S: float, K: float, T: float, sigma: float, r: float | None = None) -> float:
        """Calculate option gamma.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            r: Risk-free rate (default: 5%)

        Returns:
            Option gamma
        """
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = get_d1(S, K, r, sigma, T)
        return float(get_gamma(S, sigma, T, d1))

    @classmethod
    def theta(cls, S: float, K: float, T: float, sigma: float, right: str, r: float | None = None) -> float:
        """Calculate option theta per calendar day.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            right: Option type ('C' for call, 'P' for put)
            r: Risk-free rate (default: 5%)

        Returns:
            Option theta per day
        """
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = get_d1(S, K, r, sigma, T)
        d2 = get_d2(S, K, r, sigma, T)
        opt_type = 'call' if right == 'C' else 'put'
        # optionlab returns annual theta, convert to per day
        return float(get_theta(opt_type, S, K, r, sigma, T, d1, d2) / 365)

    @classmethod
    def vega(cls, S: float, K: float, T: float, sigma: float, r: float | None = None) -> float:
        """Calculate option vega per 1% move in IV.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            r: Risk-free rate (default: 5%)

        Returns:
            Option vega per 1% IV move
        """
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = get_d1(S, K, r, sigma, T)
        # optionlab returns vega, convert to per 1%
        return float(get_vega(S, T, d1) / 100)

    @classmethod
    def implied_volatility(
        cls, price: float, S: float, K: float, T: float, right: str,
        r: float | None = None, tol: float = 1e-6, max_iter: int = 100,
    ) -> float:
        """Calculate implied volatility from option price.

        Args:
            price: Option price
            S: Underlying price
            K: Strike price
            T: Time to expiry in years
            right: Option type ('C' for call, 'P' for put)
            r: Risk-free rate (default: 5%)
            tol: Tolerance (not used, optionlab has its own)
            max_iter: Max iterations (not used, optionlab has its own)

        Returns:
            Implied volatility
        """
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or price <= 0:
            return 0.0
        opt_type = 'call' if right == 'C' else 'put'
        return float(get_implied_vol(opt_type, price, S, K, r, T))

    @classmethod
    def _d1d2(cls, S: float, K: float, T: float, sigma: float, r: float) -> tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes model.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            r: Risk-free rate

        Returns:
            Tuple of (d1, d2)
        """
        d1 = get_d1(S, K, r, sigma, T)
        d2 = get_d2(S, K, r, sigma, T)
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

        Uses Newton-Raphson iteration with optionlab's delta function.

        Args:
            S: Underlying price
            T: Time to expiry in years
            sigma: Implied volatility
            target_delta: Target delta (positive for calls, negative for puts)
            right: 'C' for call, 'P' for put
            r: Risk-free rate (default: 5%)
            tol: Tolerance for convergence
            max_iter: Maximum iterations

        Returns:
            Strike price that achieves the target delta
        """
        r = r if r is not None else cls.RISK_FREE_RATE

        if T <= 0 or sigma <= 0:
            # At expiry, return ATM
            return S

        # Initial guess: use delta approximation from Black-Scholes
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
            d = cls.delta(S, K, T, sigma, right, r)
            diff = d - target_delta

            if abs(diff) < tol:
                return K

            # Numerical derivative dDelta/dK
            dK = K * 0.01
            d_up = cls.delta(S, K + dK, T, sigma, right, r)
            d_down = cls.delta(S, K - dK, T, sigma, right, r)
            ddelta_dK = (d_up - d_down) / (2 * dK)

            if abs(ddelta_dK) < 1e-12:
                break

            # Update strike
            K = K - diff / ddelta_dK
            K = max(S * 0.5, min(K, S * 2.0))  # Bound the strike

        return K

    @classmethod
    def itm_probability(cls, S: float, K: float, T: float, sigma: float, right: str, r: float | None = None) -> float:
        """Calculate probability of option being in-the-money at expiry.

        Uses optionlab's get_bs_info for accurate ITM probability calculation.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            right: Option type ('C' for call, 'P' for put)
            r: Risk-free rate (default: 5%)

        Returns:
            Probability of being ITM at expiry (0.0 to 1.0)
        """
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0:
            # At expiry, ITM probability is 1 if ITM, 0 otherwise
            if right == "C":
                return 1.0 if S > K else 0.0
            return 1.0 if S < K else 0.0
        if sigma <= 0:
            return 0.0

        info = get_bs_info(S, K, r, sigma, T)
        if right == "C":
            return float(info.call_itm_prob)
        return float(info.put_itm_prob)

    @classmethod
    def probability_of_touch(cls, S: float, K: float, T: float, sigma: float, right: str, r: float | None = None) -> float:
        """Calculate probability of underlying price touching strike before expiry.

        Uses optionlab's get_probability_of_touch for accurate calculation.
        This is different from ITM probability - it measures the chance that
        the price will reach the strike at any point before expiry.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            right: Option type ('C' for call, 'P' for put)
            r: Risk-free rate (default: 5%)

        Returns:
            Probability of touching strike (0.0 to 1.0)
        """
        r = r if r is not None else cls.RISK_FREE_RATE
        if T <= 0 or sigma <= 0:
            return 0.0

        opt_type = 'call' if right == 'C' else 'put'
        return float(get_probability_of_touch(opt_type, S, K, r, sigma, T))

    @classmethod
    def get_all_info(cls, S: float, K: float, T: float, sigma: float, r: float | None = None) -> dict:
        """Get all option info in one call for better performance.

        Uses optionlab's get_bs_info to get all Greeks and probabilities at once.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            r: Risk-free rate (default: 5%)

        Returns:
            Dictionary with all option info:
            - call_price, put_price
            - call_delta, put_delta
            - gamma, vega
            - call_theta, put_theta (annual)
            - call_itm_prob, put_itm_prob
            - call_prob_of_touch, put_prob_of_touch
        """
        r = r if r is not None else cls.RISK_FREE_RATE

        if T <= 0 or sigma <= 0:
            # Return intrinsic values at expiry
            return {
                'call_price': max(0, S - K),
                'put_price': max(0, K - S),
                'call_delta': 1.0 if S > K else 0.0,
                'put_delta': -1.0 if S < K else 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'call_theta': 0.0,
                'put_theta': 0.0,
                'call_itm_prob': 1.0 if S > K else 0.0,
                'put_itm_prob': 1.0 if S < K else 0.0,
                'call_prob_of_touch': 0.0,
                'put_prob_of_touch': 0.0,
            }

        info = get_bs_info(S, K, r, sigma, T)
        return {
            'call_price': float(info.call_price),
            'put_price': float(info.put_price),
            'call_delta': float(info.call_delta),
            'put_delta': float(info.put_delta),
            'gamma': float(info.gamma),
            'vega': float(info.vega),
            'call_theta': float(info.call_theta),
            'put_theta': float(info.put_theta),
            'call_itm_prob': float(info.call_itm_prob),
            'put_itm_prob': float(info.put_itm_prob),
            'call_prob_of_touch': float(info.call_prob_of_touch),
            'put_prob_of_touch': float(info.put_prob_of_touch),
        }