"""Option selection functions for BinbinGod Strategy."""
from typing import Dict, Optional
from AlgorithmImports import OptionRight
from option_utils import filter_option_by_itm_protection, estimate_delta_from_moneyness, build_option_result
from scoring import calculate_historical_vol
from option_pricing import BlackScholes

RISK_FREE_RATE = 0.05


def bs_put_price(S, K, T, sigma):
    return BlackScholes.put_price(S, K, T, RISK_FREE_RATE, sigma)


def bs_call_price(S, K, T, sigma):
    return BlackScholes.call_price(S, K, T, RISK_FREE_RATE, sigma)


def find_option_by_greeks(algo, symbol: str, equity_symbol, target_right, target_delta: float,
                            dte_min: int, dte_max: int, delta_tolerance: float = 0.10, min_strike: float = None) -> Optional[Dict]:
    underlying_price = algo.Securities[equity_symbol].Price
    option_chain = algo.OptionChainProvider.GetOptionContractList(equity_symbol, algo.Time)
    if not option_chain:
        return None
    suitable = []
    stats = {'right': 0, 'dte': 0, 'min_strike': 0, 'itm': 0, 'tolerance': 0}
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
            continue
        if abs(delta - target_delta) > delta_tolerance: 
            stats['tolerance'] += 1
            continue
        T = dte / 365.0
        if target_right == OptionRight.Put:
            premium = bs_put_price(underlying_price, strike, T, iv)
        else:
            premium = bs_call_price(underlying_price, strike, T, iv)
        if premium <= 0.10:
            continue
        suitable.append(build_option_result(option_symbol, strike, option_symbol.ID.Date, dte,
            delta, iv, premium, abs(delta - target_delta), premium * 0.99, premium * 1.01))
    
    if not suitable:
        return None
    
    suitable.sort(key=lambda x: x['delta_diff'])
    return suitable[0]