"""Test helpers shared across the suite."""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass, field

from scipy.stats import norm


def _install_optionlab_stub() -> None:
    """Provide a minimal in-process optionlab stub for local unit tests."""
    if "optionlab" in sys.modules and "optionlab.black_scholes" in sys.modules:
        return

    optionlab_module = types.ModuleType("optionlab")
    black_scholes_module = types.ModuleType("optionlab.black_scholes")
    models_module = types.ModuleType("optionlab.models")

    @dataclass
    class Inputs:
        stock_price: float
        volatility: float
        interest_rate: float
        min_stock: float
        max_stock: float
        strategy: list
        days_to_target_date: int

    @dataclass
    class Option:
        type: str
        strike: float
        premium: float
        action: str
        n: int = 1
        prev_pos: str | None = None
        expiration: str | None = None

    @dataclass
    class Stock:
        type: str
        n: int
        action: str
        prev_pos: str | None = None

    @dataclass
    class Outputs:
        probability_of_profit: float = 0.5
        profit_ranges: list = field(default_factory=lambda: [(0.0, 0.0)])
        expected_profit_if_profitable: float = 0.0
        expected_loss_if_unprofitable: float = 0.0
        maximum_return_in_the_domain: float = 0.0
        minimum_return_in_the_domain: float = 0.0
        break_even_points: list = field(default_factory=list)
        strategy_cost: float = 0.0
        data: dict = field(default_factory=dict)

    def get_d1(S, K, r, sigma, T):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

    def get_d2(S, K, r, sigma, T):
        return get_d1(S, K, r, sigma, T) - sigma * math.sqrt(T)

    def get_option_price(option_type, S, K, r, T, d1, d2):
        if option_type == "call":
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def get_delta(option_type, d1, T):
        if option_type == "call":
            return norm.cdf(d1)
        return norm.cdf(d1) - 1

    def get_gamma(S, sigma, T, d1):
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))

    def get_theta(option_type, S, K, r, sigma, T, d1, d2):
        if T <= 0 or sigma <= 0:
            return 0.0
        front = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
        if option_type == "call":
            return front - r * K * math.exp(-r * T) * norm.cdf(d2)
        return front + r * K * math.exp(-r * T) * norm.cdf(-d2)

    def get_vega(S, T, d1):
        if T <= 0 or S <= 0:
            return 0.0
        return S * norm.pdf(d1) * math.sqrt(T)

    def get_implied_vol(option_type, price, S, K, r, T):
        # Basic bounded search is enough for unit tests.
        low, high = 0.01, 3.0
        for _ in range(60):
            mid = (low + high) / 2
            d1 = get_d1(S, K, r, mid, T)
            d2 = get_d2(S, K, r, mid, T)
            model = get_option_price(option_type, S, K, r, T, d1, d2)
            if model > price:
                high = mid
            else:
                low = mid
        return (low + high) / 2

    def get_bs_info(*args, **kwargs):
        return {}

    def get_probability_of_touch(*args, **kwargs):
        return 0.0

    def run_strategy(inputs):
        premiums = []
        for leg in getattr(inputs, "strategy", []):
            premium = getattr(leg, "premium", 0.0) * getattr(leg, "n", 1) * 100
            if getattr(leg, "action", "buy") == "sell":
                premiums.append(premium)
            else:
                premiums.append(-premium)
        strategy_cost = -sum(premiums)
        return Outputs(
            probability_of_profit=0.5,
            profit_ranges=[(inputs.min_stock, inputs.max_stock)],
            expected_profit_if_profitable=max(strategy_cost, 0.0),
            expected_loss_if_unprofitable=min(strategy_cost, 0.0),
            maximum_return_in_the_domain=max(strategy_cost, 0.0),
            minimum_return_in_the_domain=min(strategy_cost, 0.0),
            break_even_points=[],
            strategy_cost=strategy_cost,
            data={"stock_price_array": [inputs.min_stock, inputs.max_stock], "strategy_profit": [0.0, 0.0]},
        )

    black_scholes_module.get_d1 = get_d1
    black_scholes_module.get_d2 = get_d2
    black_scholes_module.get_option_price = get_option_price
    black_scholes_module.get_delta = get_delta
    black_scholes_module.get_gamma = get_gamma
    black_scholes_module.get_theta = get_theta
    black_scholes_module.get_vega = get_vega
    black_scholes_module.get_implied_vol = get_implied_vol
    black_scholes_module.get_bs_info = get_bs_info
    black_scholes_module.get_probability_of_touch = get_probability_of_touch

    optionlab_module.Inputs = Inputs
    optionlab_module.run_strategy = run_strategy
    optionlab_module.black_scholes = black_scholes_module
    models_module.Option = Option
    models_module.Stock = Stock
    models_module.Outputs = Outputs
    sys.modules["optionlab"] = optionlab_module
    sys.modules["optionlab.black_scholes"] = black_scholes_module
    sys.modules["optionlab.models"] = models_module


_install_optionlab_stub()
