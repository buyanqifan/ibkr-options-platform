"""Trading cost model for realistic backtesting."""

from dataclasses import dataclass


@dataclass
class TradingCostModel:
    """Model for calculating trading costs including commissions and slippage.
    
    Attributes:
        commission_per_contract: Commission fee per option contract (default $0.65)
        commission_min: Minimum commission per order (default $1.00)
        commission_max: Maximum commission per order (optional cap)
        slippage_per_contract: Slippage cost per contract (default $0.05)
        exercise_fee: Fee for option exercise/assignment (default $0.00)
    """
    commission_per_contract: float = 0.65  # IBKR standard rate
    commission_min: float = 1.00  # Minimum per order
    commission_max: float | None = None  # Optional cap
    slippage_per_contract: float = 0.05  # Bid-ask spread impact
    exercise_fee: float = 0.00  # Exercise/assignment fee
    
    def calculate_commission(self, quantity: int) -> float:
        """Calculate commission for a trade.
        
        Args:
            quantity: Number of contracts (positive for long, negative for short)
            
        Returns:
            Commission amount (always positive as it's a cost)
        """
        num_contracts = abs(quantity)
        commission = num_contracts * self.commission_per_contract
        
        # Apply minimum commission
        commission = max(commission, self.commission_min)
        
        # Apply maximum commission cap if set
        if self.commission_max is not None:
            commission = min(commission, self.commission_max)
        
        return commission
    
    def calculate_slippage(self, quantity: int) -> float:
        """Calculate slippage cost for a trade.
        
        Args:
            quantity: Number of contracts
            
        Returns:
            Slippage cost (always positive as it's a cost)
        """
        num_contracts = abs(quantity)
        return num_contracts * self.slippage_per_contract
    
    def calculate_total_cost(self, quantity: int, include_slippage: bool = True) -> float:
        """Calculate total trading cost.
        
        Args:
            quantity: Number of contracts
            include_slippage: Whether to include slippage in calculation
            
        Returns:
            Total cost (commission + optional slippage)
        """
        commission = self.calculate_commission(quantity)
        slippage = self.calculate_slippage(quantity) if include_slippage else 0.0
        
        return commission + slippage
    
    def calculate_entry_adjustment(self, premium: float, quantity: int) -> float:
        """Adjust entry price to account for trading costs.
        
        For short positions (sell): Reduces effective premium received
        For long positions (buy): Increases effective premium paid
        
        Args:
            premium: Option premium per contract
            quantity: Number of contracts (negative for short, positive for long)
            
        Returns:
            Adjusted premium considering trading costs
        """
        total_cost = self.calculate_total_cost(quantity)
        cost_per_contract = total_cost / abs(quantity) if quantity != 0 else 0
        
        if quantity < 0:  # Short position (selling)
            # Reduce premium received by costs
            adjusted_premium = premium - cost_per_contract
        else:  # Long position (buying)
            # Increase premium paid by costs
            adjusted_premium = premium + cost_per_contract
        
        return adjusted_premium
    
    def calculate_exit_adjustment(self, exit_premium: float, quantity: int, 
                                   entry_premium: float) -> float:
        """Calculate adjusted P&L accounting for round-trip costs.
        
        Args:
            exit_premium: Exit premium per contract
            quantity: Number of contracts
            entry_premium: Entry premium per contract
            
        Returns:
            Adjusted P&L after all costs
        """
        # Calculate raw P&L
        if quantity < 0:  # Short position
            raw_pnl = (entry_premium - exit_premium) * abs(quantity) * 100
        else:  # Long position
            raw_pnl = (exit_premium - entry_premium) * abs(quantity) * 100
        
        # Subtract round-trip costs (entry + exit)
        entry_cost = self.calculate_total_cost(quantity)
        exit_cost = self.calculate_total_cost(-quantity)  # Opposite sign for closing trade
        
        adjusted_pnl = raw_pnl - entry_cost - exit_cost
        
        return adjusted_pnl
