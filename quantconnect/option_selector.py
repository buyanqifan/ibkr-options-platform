"""Option selection functions for BinbinGod Strategy."""
from typing import Dict, List, Optional
from AlgorithmImports import OptionRight
from option_utils import filter_option_by_itm_protection, estimate_delta_from_moneyness, build_option_result
from scoring import calculate_historical_vol
from option_pricing import BlackScholes

RISK_FREE_RATE = 0.05


def bs_put_price(S, K, T, sigma):
    return BlackScholes.put_price(S, K, T, RISK_FREE_RATE, sigma)


def bs_call_price(S, K, T, sigma):
    return BlackScholes.call_price(S, K, T, RISK_FREE_RATE, sigma)


def _find_option_by_constraints(
    algo,
    symbol: str,
    equity_symbol,
    target_right,
    target_delta: float,
    dte_min: int,
    dte_max: int,
    delta_tolerance: float,
    min_strike: float | None,
) -> Optional[Dict]:
    underlying_price = algo.Securities[equity_symbol].Price
    min_premium = (
        getattr(algo, "sp_min_option_premium", 0.05)
        if target_right == OptionRight.Put
        else 0.10
    )
    option_chain = algo.OptionChainProvider.GetOptionContractList(equity_symbol, algo.Time)
    if not option_chain:
        algo._last_option_selection_stats = {
            "symbol": symbol,
            "target_right": str(target_right),
            "target_delta": target_delta,
            "dte_min": dte_min,
            "dte_max": dte_max,
            "delta_tolerance": delta_tolerance,
            "min_strike": min_strike,
            "min_premium": min_premium,
            "total_chain": 0,
            "stats": {"right": 0, "dte": 0, "min_strike": 0, "itm": 0, "delta_none": 0, "tolerance": 0, "premium": 0},
        }
        return None
    suitable = []
    stats = {'right': 0, 'dte': 0, 'min_strike': 0, 'itm': 0, 'delta_none': 0, 'tolerance': 0, 'premium': 0}
    for option_symbol in option_chain:
        if option_symbol.ID.OptionRight != target_right: 
            continue
        stats['right'] += 1
        dte = (option_symbol.ID.Date - algo.Time).days
        if not (dte_min <= dte <= dte_max): 
            continue
        stats['dte'] += 1
        strike = option_symbol.ID.StrikePrice
        if min_strike and strike < min_strike: 
            stats['min_strike'] += 1
            continue
        if not filter_option_by_itm_protection(strike, underlying_price, target_right): 
            stats['itm'] += 1
            continue
        delta = estimate_delta_from_moneyness(strike, underlying_price, target_right)
        iv = calculate_historical_vol(algo.price_history.get(symbol, []))
        if delta is None: 
            stats['delta_none'] += 1
            continue
        if abs(delta - target_delta) > delta_tolerance: 
            stats['tolerance'] += 1
            continue
        T = dte / 365.0
        if target_right == OptionRight.Put:
            premium = bs_put_price(underlying_price, strike, T, iv)
        else:
            premium = bs_call_price(underlying_price, strike, T, iv)
        if premium <= min_premium:
            stats['premium'] += 1
            continue
        suitable.append(build_option_result(option_symbol, strike, option_symbol.ID.Date, dte,
            delta, iv, premium, abs(delta - target_delta), premium * 0.99, premium * 1.01))
    algo._last_option_selection_stats = {
        "symbol": symbol,
        "target_right": str(target_right),
        "target_delta": target_delta,
        "dte_min": dte_min,
        "dte_max": dte_max,
        "delta_tolerance": delta_tolerance,
        "min_strike": min_strike,
        "min_premium": min_premium,
        "total_chain": len(option_chain),
        "stats": stats,
        "suitable_count": len(suitable),
    }
    if not suitable:
        return None
    
    suitable.sort(key=lambda x: x['delta_diff'])
    return suitable[0]


def find_option_by_greeks(
    algo,
    symbol: str,
    equity_symbol,
    target_right,
    target_delta: float,
    dte_min: int,
    dte_max: int,
    delta_tolerance: float = 0.10,
    min_strike: float = None,
    selection_tiers: Optional[List[Dict]] = None,
) -> Optional[Dict]:
    tiers = selection_tiers or [
        {
            "label": "primary",
            "delta_tolerance": delta_tolerance,
            "dte_min": dte_min,
            "dte_max": dte_max,
            "min_strike": min_strike,
        }
    ]

    for tier in tiers:
        selected = _find_option_by_constraints(
            algo,
            symbol=symbol,
            equity_symbol=equity_symbol,
            target_right=target_right,
            target_delta=target_delta,
            dte_min=int(tier.get("dte_min", dte_min)),
            dte_max=int(tier.get("dte_max", dte_max)),
            delta_tolerance=float(tier.get("delta_tolerance", delta_tolerance)),
            min_strike=tier.get("min_strike", min_strike),
        )
        if selected:
            selected["selection_tier"] = str(tier.get("label", "primary"))
            return selected

    return None
