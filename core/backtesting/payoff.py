"""Payoff calculator using optionlab for options strategy analysis."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from optionlab import run_strategy, Inputs
from optionlab.models import Option, Stock, Outputs


@dataclass
class PayoffResult:
    """Result of payoff calculation for an options strategy."""
    profit_probability: float          # Probability of profit
    profit_ranges: List[Tuple[float, float]]  # Price ranges where strategy is profitable
    expected_profit: float             # Expected profit if profitable
    expected_loss: float               # Expected loss if unprofitable
    max_profit: float                  # Maximum profit
    max_loss: float                    # Maximum loss
    break_even_points: List[float]     # Break-even stock prices
    strategy_cost: float               # Net cost/credit of strategy


class PayoffCalculator:
    """Calculate options strategy payoff using optionlab.
    
    This class provides a unified interface for calculating:
    - Option prices and Greeks (via OptionsPricer)
    - Strategy-level payoff analysis (via optionlab.run_strategy)
    - Break-even points
    - Maximum profit/loss
    - Probability of profit
    """

    DEFAULT_RISK_FREE_RATE = 0.05
    DEFAULT_PRICE_RANGE_PCT = 0.30  # 30% above/below current price

    @classmethod
    def calculate_strategy_payoff(
        cls,
        strategy_legs: List[Dict[str, Any]],
        stock_price: float,
        volatility: float,
        days_to_expiry: int,
        interest_rate: float = None,
        min_stock: float = None,
        max_stock: float = None,
    ) -> PayoffResult:
        """Calculate comprehensive payoff analysis for an options strategy.
        
        Args:
            strategy_legs: List of strategy legs, each with:
                - type: 'call', 'put', or 'stock'
                - strike: Strike price (for options)
                - premium: Premium per share (for options)
                - action: 'buy' or 'sell'
                - n: Number of contracts
            stock_price: Current underlying price
            volatility: Implied volatility (decimal, e.g., 0.25 for 25%)
            days_to_expiry: Days to expiration
            interest_rate: Risk-free rate (default: 5%)
            min_stock: Minimum stock price for analysis (default: 70% of current)
            max_stock: Maximum stock price for analysis (default: 130% of current)
        
        Returns:
            PayoffResult with all payoff metrics
        """
        r = interest_rate or cls.DEFAULT_RISK_FREE_RATE
        
        # Set default price range
        if min_stock is None:
            min_stock = stock_price * (1 - cls.DEFAULT_PRICE_RANGE_PCT)
        if max_stock is None:
            max_stock = stock_price * (1 + cls.DEFAULT_PRICE_RANGE_PCT)
        
        # Convert strategy legs to optionlab format
        ol_strategy = []
        for leg in strategy_legs:
            leg_type = leg.get('type', '').lower()
            
            if leg_type == 'stock':
                ol_strategy.append(Stock(
                    type='stock',
                    n=leg.get('n', 1),
                    action=leg.get('action', 'buy'),
                    prev_pos=leg.get('prev_pos'),
                ))
            else:
                # Option leg
                ol_strategy.append(Option(
                    type=leg_type,  # 'call' or 'put'
                    strike=leg.get('strike', stock_price),
                    premium=leg.get('premium', 0),
                    action=leg.get('action', 'sell'),
                    n=leg.get('n', 1),
                    prev_pos=leg.get('prev_pos'),
                    expiration=leg.get('expiration'),
                ))
        
        # Create inputs
        inputs = Inputs(
            stock_price=stock_price,
            volatility=volatility,
            interest_rate=r,
            min_stock=min_stock,
            max_stock=max_stock,
            strategy=ol_strategy,
            days_to_target_date=days_to_expiry,
        )
        
        # Run strategy analysis
        outputs = run_strategy(inputs)
        
        # Extract break-even points from profit data
        break_even_points = cls._extract_breakeven_points(outputs)
        
        return PayoffResult(
            profit_probability=outputs.probability_of_profit,
            profit_ranges=outputs.profit_ranges,
            expected_profit=outputs.expected_profit_if_profitable,
            expected_loss=outputs.expected_loss_if_unprofitable,
            max_profit=outputs.maximum_return_in_the_domain,
            max_loss=outputs.minimum_return_in_the_domain,  # This is negative for losses
            break_even_points=break_even_points,
            strategy_cost=outputs.strategy_cost,
        )

    @classmethod
    def calculate_single_option_payoff(
        cls,
        option_type: str,
        strike: float,
        premium: float,
        action: str,
        stock_price: float,
        volatility: float,
        days_to_expiry: int,
        n: int = 1,
        interest_rate: float = None,
    ) -> PayoffResult:
        """Calculate payoff for a single option position.
        
        Args:
            option_type: 'call' or 'put'
            strike: Strike price
            premium: Premium per share
            action: 'buy' or 'sell'
            stock_price: Current underlying price
            volatility: Implied volatility
            days_to_expiry: Days to expiration
            n: Number of contracts
            interest_rate: Risk-free rate
        
        Returns:
            PayoffResult with payoff metrics
        """
        strategy_legs = [{
            'type': option_type,
            'strike': strike,
            'premium': premium,
            'action': action,
            'n': n,
        }]
        
        return cls.calculate_strategy_payoff(
            strategy_legs=strategy_legs,
            stock_price=stock_price,
            volatility=volatility,
            days_to_expiry=days_to_expiry,
            interest_rate=interest_rate,
        )

    @classmethod
    def get_profit_at_price(
        cls,
        strategy_legs: List[Dict[str, Any]],
        stock_price: float,
        target_price: float,
        volatility: float,
        days_to_expiry: int,
        interest_rate: float = None,
    ) -> float:
        """Get profit/loss at a specific target stock price.
        
        Args:
            strategy_legs: List of strategy legs
            stock_price: Current underlying price
            target_price: Target stock price to evaluate
            volatility: Implied volatility
            days_to_expiry: Days to expiration
            interest_rate: Risk-free rate
        
        Returns:
            Profit/loss at the target price
        """
        result = cls.calculate_strategy_payoff(
            strategy_legs=strategy_legs,
            stock_price=stock_price,
            volatility=volatility,
            days_to_expiry=days_to_expiry,
            interest_rate=interest_rate,
            min_stock=target_price * 0.99,
            max_stock=target_price * 1.01,
        )
        
        # For precise profit at target price, use intrinsic value calculation
        # when we're close to expiration, or run_strategy's data
        total_pnl = 0.0
        
        for leg in strategy_legs:
            leg_type = leg.get('type', '').lower()
            action = leg.get('action', 'sell')
            n = leg.get('n', 1)
            premium = leg.get('premium', 0)
            strike = leg.get('strike', stock_price)
            
            if leg_type == 'stock':
                # Stock P&L
                pnl = (target_price - stock_price) * n
                if action == 'sell':
                    pnl = -pnl
            elif leg_type == 'put':
                # Put intrinsic value at target
                intrinsic = max(0, strike - target_price)
                if action == 'sell':
                    pnl = (premium - intrinsic) * n * 100
                else:
                    pnl = (intrinsic - premium) * n * 100
            elif leg_type == 'call':
                # Call intrinsic value at target
                intrinsic = max(0, target_price - strike)
                if action == 'sell':
                    pnl = (premium - intrinsic) * n * 100
                else:
                    pnl = (intrinsic - premium) * n * 100
            else:
                pnl = 0
            
            total_pnl += pnl
        
        return total_pnl

    @classmethod
    def calculate_breakeven(
        cls,
        strategy_legs: List[Dict[str, Any]],
        stock_price: float,
        volatility: float,
        days_to_expiry: int,
        interest_rate: float = None,
    ) -> List[float]:
        """Calculate break-even points for a strategy.
        
        Args:
            strategy_legs: List of strategy legs
            stock_price: Current underlying price
            volatility: Implied volatility
            days_to_expiry: Days to expiration
            interest_rate: Risk-free rate
        
        Returns:
            List of break-even stock prices
        """
        result = cls.calculate_strategy_payoff(
            strategy_legs=strategy_legs,
            stock_price=stock_price,
            volatility=volatility,
            days_to_expiry=days_to_expiry,
            interest_rate=interest_rate,
        )
        return result.break_even_points

    @classmethod
    def calculate_max_profit_loss(
        cls,
        strategy_legs: List[Dict[str, Any]],
        stock_price: float,
        volatility: float,
        days_to_expiry: int,
        interest_rate: float = None,
    ) -> Tuple[float, float]:
        """Calculate maximum profit and loss for a strategy.
        
        Args:
            strategy_legs: List of strategy legs
            stock_price: Current underlying price
            volatility: Implied volatility
            days_to_expiry: Days to expiration
            interest_rate: Risk-free rate
        
        Returns:
            Tuple of (max_profit, max_loss) where max_loss is positive
        """
        result = cls.calculate_strategy_payoff(
            strategy_legs=strategy_legs,
            stock_price=stock_price,
            volatility=volatility,
            days_to_expiry=days_to_expiry,
            interest_rate=interest_rate,
        )
        
        # max_loss from optionlab is negative (representing a loss)
        # Convert to positive for consistency with expectations
        max_loss = abs(result.max_loss) if result.max_loss < 0 else 0
        
        return result.max_profit, max_loss

    @classmethod
    def calculate_probability_of_profit(
        cls,
        strategy_legs: List[Dict[str, Any]],
        stock_price: float,
        volatility: float,
        days_to_expiry: int,
        interest_rate: float = None,
    ) -> float:
        """Calculate probability of profit for a strategy.
        
        Args:
            strategy_legs: List of strategy legs
            stock_price: Current underlying price
            volatility: Implied volatility
            days_to_expiry: Days to expiration
            interest_rate: Risk-free rate
        
        Returns:
            Probability of profit (0.0 to 1.0)
        """
        result = cls.calculate_strategy_payoff(
            strategy_legs=strategy_legs,
            stock_price=stock_price,
            volatility=volatility,
            days_to_expiry=days_to_expiry,
            interest_rate=interest_rate,
        )
        return result.profit_probability

    @staticmethod
    def _extract_breakeven_points(outputs: Outputs) -> List[float]:
        """Extract break-even points from optionlab outputs.
        
        Break-even points are where profit transitions from negative to positive.
        """
        break_evens = []
        
        if outputs.data.strategy_profit is None or outputs.data.stock_price_array is None:
            return break_evens
        
        profits = outputs.data.strategy_profit
        prices = outputs.data.stock_price_array
        
        # Find where profit crosses zero
        for i in range(len(profits) - 1):
            if profits[i] < 0 and profits[i + 1] >= 0:
                # Transition from loss to profit
                break_evens.append(float(prices[i + 1]))
            elif profits[i] > 0 and profits[i + 1] <= 0:
                # Transition from profit to loss
                break_evens.append(float(prices[i]))
        
        return break_evens


# Convenience functions for common use cases

def get_sell_put_payoff(
    strike: float,
    premium: float,
    stock_price: float,
    volatility: float,
    days_to_expiry: int,
    n: int = 1,
) -> PayoffResult:
    """Get payoff analysis for selling a put.
    
    Args:
        strike: Strike price
        premium: Premium received per share
        stock_price: Current underlying price
        volatility: Implied volatility
        days_to_expiry: Days to expiration
        n: Number of contracts
    
    Returns:
        PayoffResult with payoff metrics
    """
    return PayoffCalculator.calculate_single_option_payoff(
        option_type='put',
        strike=strike,
        premium=premium,
        action='sell',
        stock_price=stock_price,
        volatility=volatility,
        days_to_expiry=days_to_expiry,
        n=n,
    )


def get_sell_call_payoff(
    strike: float,
    premium: float,
    stock_price: float,
    volatility: float,
    days_to_expiry: int,
    n: int = 1,
) -> PayoffResult:
    """Get payoff analysis for selling a call.
    
    Args:
        strike: Strike price
        premium: Premium received per share
        stock_price: Current underlying price
        volatility: Implied volatility
        days_to_expiry: Days to expiration
        n: Number of contracts
    
    Returns:
        PayoffResult with payoff metrics
    """
    return PayoffCalculator.calculate_single_option_payoff(
        option_type='call',
        strike=strike,
        premium=premium,
        action='sell',
        stock_price=stock_price,
        volatility=volatility,
        days_to_expiry=days_to_expiry,
        n=n,
    )


def get_covered_call_payoff(
    stock_price: float,
    strike: float,
    premium: float,
    volatility: float,
    days_to_expiry: int,
    shares: int = 100,
) -> PayoffResult:
    """Get payoff analysis for a covered call position.
    
    Args:
        stock_price: Current underlying price (and cost basis)
        strike: Call strike price
        premium: Premium received per share
        volatility: Implied volatility
        days_to_expiry: Days to expiration
        shares: Number of shares held (default: 100)
    
    Returns:
        PayoffResult with payoff metrics
    """
    strategy_legs = [
        {'type': 'stock', 'n': shares, 'action': 'buy'},
        {'type': 'call', 'strike': strike, 'premium': premium, 'action': 'sell', 'n': shares // 100},
    ]
    
    return PayoffCalculator.calculate_strategy_payoff(
        strategy_legs=strategy_legs,
        stock_price=stock_price,
        volatility=volatility,
        days_to_expiry=days_to_expiry,
    )