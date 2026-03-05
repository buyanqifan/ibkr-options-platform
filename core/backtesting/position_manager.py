"""Position manager for backtesting: handles capital allocation, margin tracking, and leverage."""

from dataclasses import dataclass, field
from typing import Dict, List
from utils.logger import setup_logger

logger = setup_logger("position_manager")


@dataclass
class CapitalAllocation:
    """Tracks capital allocation for a single position."""
    strategy: str
    symbol: str
    entry_date: str
    allocated_margin: float  # Margin reserved for this position
    released: bool = False  # Whether margin has been released
    
    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "entry_date": self.entry_date,
            "allocated_margin": self.allocated_margin,
            "released": self.released,
        }


class PositionManager:
    """
    Manages position sizing, margin allocation, and leverage control for backtesting.
    
    Key features:
    - Tracks total capital and available capital
    - Allocates margin for each open position
    - Supports leverage with interest calculation
    - Prevents over-allocation
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_leverage: float = 1.0,
        position_percentage: float = 0.10,
        margin_interest_rate: float = 0.05,
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_leverage = max_leverage
        self.position_percentage = position_percentage
        self.margin_interest_rate = margin_interest_rate
        
        # Capital tracking
        self.total_margin_used = 0.0  # Total margin allocated to open positions
        self.cumulative_pnl = 0.0  # Realized P&L from closed positions
        
        # Leverage tracking
        self.borrowed_amount = 0.0  # Amount borrowed for leverage
        self.accrued_interest = 0.0  # Accumulated margin interest
        
        # Position tracking
        self.allocations: Dict[str, CapitalAllocation] = {}  # key: position_id
        
    @property
    def gross_capital(self) -> float:
        """Total capital including leverage."""
        return self.initial_capital * self.max_leverage
    
    @property
    def net_capital(self) -> float:
        """Net equity (initial capital + realized P&L)."""
        return self.initial_capital + self.cumulative_pnl
    
    @property
    def available_margin(self) -> float:
        """Available margin for new positions."""
        # Available = (Net Capital * Max Leverage) - Used Margin
        max_available = self.net_capital * self.max_leverage
        return max_available - self.total_margin_used
    
    @property
    def margin_utilization(self) -> float:
        """Percentage of available margin being used."""
        if self.gross_capital == 0:
            return 0.0
        return self.total_margin_used / self.gross_capital * 100
    
    def calculate_position_size(
        self,
        margin_per_contract: float,
        max_positions: int = 1,
    ) -> int:
        """
        Calculate optimal position size based on available margin and constraints.
        
        Args:
            margin_per_contract: Required margin per contract (e.g., strike * 100 for cash-secured put)
            max_positions: Maximum number of contracts allowed
            
        Returns:
            Number of contracts that can be opened (0 if insufficient capital)
        """
        if margin_per_contract <= 0:
            return 0
        
        # Calculate maximum contracts by available margin (primary constraint)
        max_by_margin = int(self.available_margin / margin_per_contract)
        
        # Take minimum of margin and max_positions constraints
        num_contracts = min(max_positions, max_by_margin)
        
        # Return 0 if insufficient capital (no forced position sizing)
        # This ensures realistic backtest results - can't trade with no money!
        if num_contracts <= 0:
            logger.debug(
                f"Insufficient capital for position: "
                f"available_margin={self.available_margin:.2f}, "
                f"margin_per_contract={margin_per_contract:.2f}, "
                f"max_by_margin={max_by_margin}, max_by_capital={max_by_capital}"
            )
            return 0
        
        return num_contracts
    
    def allocate_margin(
        self,
        position_id: str,
        strategy: str,
        symbol: str,
        entry_date: str,
        margin_amount: float,
    ) -> bool:
        """
        Allocate margin for a new position.
        
        Args:
            position_id: Unique identifier for the position
            strategy: Strategy name
            symbol: Underlying symbol
            entry_date: Entry date
            margin_amount: Margin to allocate
            
        Returns:
            True if allocation successful, False if insufficient margin
        """
        if margin_amount > self.available_margin:
            logger.warning(
                f"Insufficient margin: requested {margin_amount:.2f}, available {self.available_margin:.2f}"
            )
            return False
        
        if position_id in self.allocations and not self.allocations[position_id].released:
            logger.warning(f"Position {position_id} already exists")
            return False
        
        self.allocations[position_id] = CapitalAllocation(
            strategy=strategy,
            symbol=symbol,
            entry_date=entry_date,
            allocated_margin=margin_amount,
        )
        self.total_margin_used += margin_amount
        
        logger.info(
            f"Allocated ${margin_amount:.2f} margin for {position_id} "
            f"(utilization: {self.margin_utilization:.1f}%)"
        )
        return True
    
    def release_margin(self, position_id: str, pnl: float = 0.0) -> bool:
        """
        Release margin when a position is closed.
        
        Args:
            position_id: Position identifier
            pnl: Realized P&L from closing the position
            
        Returns:
            True if release successful
        """
        if position_id not in self.allocations:
            logger.warning(f"Position {position_id} not found")
            return False
        
        allocation = self.allocations[position_id]
        if allocation.released:
            logger.warning(f"Position {position_id} already released")
            return False
        
        # Release margin
        self.total_margin_used -= allocation.allocated_margin
        allocation.released = True
        
        # Update cumulative P&L
        self.cumulative_pnl += pnl
        
        logger.info(
            f"Released ${allocation.allocated_margin:.2f} margin for {position_id}, "
            f"PnL: ${pnl:.2f}"
        )
        return True
    
    def apply_daily_interest(self):
        """Apply daily margin interest on borrowed funds."""
        if self.borrowed_amount <= 0:
            return 0.0
        
        daily_rate = self.margin_interest_rate / 365
        daily_interest = self.borrowed_amount * daily_rate
        self.accrued_interest += daily_interest
        
        # Interest reduces net capital
        self.cumulative_pnl -= daily_interest
        
        return daily_interest
    
    def use_leverage(self, amount: float) -> bool:
        """
        Borrow funds to increase available capital.
        
        Args:
            amount: Amount to borrow
            
        Returns:
            True if borrowing successful
        """
        if amount <= 0:
            return False
        
        max_borrowable = self.net_capital * (self.max_leverage - 1)
        if self.borrowed_amount + amount > max_borrowable:
            logger.warning(
                f"Cannot borrow {amount:.2f}. Max borrowable: {max_borrowable - self.borrowed_amount:.2f}"
            )
            return False
        
        self.borrowed_amount += amount
        self.current_capital += amount
        
        logger.info(f"Borrowed ${amount:.2f} (total: ${self.borrowed_amount:.2f})")
        return True
    
    def repay_leverage(self, amount: float) -> bool:
        """
        Repay borrowed funds.
        
        Args:
            amount: Amount to repay
            
        Returns:
            True if repayment successful
        """
        if amount <= 0 or amount > self.borrowed_amount:
            return False
        
        self.borrowed_amount -= amount
        self.current_capital -= amount
        
        logger.info(f"Repaid ${amount:.2f} (remaining: ${self.borrowed_amount:.2f})")
        return True
    
    def get_portfolio_summary(self) -> dict:
        """Get comprehensive portfolio summary."""
        open_positions = sum(1 for a in self.allocations.values() if not a.released)
        
        return {
            "initial_capital": self.initial_capital,
            "net_capital": self.net_capital,
            "gross_capital": self.gross_capital,
            "total_margin_used": self.total_margin_used,
            "available_margin": self.available_margin,
            "margin_utilization": self.margin_utilization,
            "borrowed_amount": self.borrowed_amount,
            "accrued_interest": self.accrued_interest,
            "cumulative_pnl": self.cumulative_pnl,
            "open_positions": open_positions,
        }
    
    def reset(self):
        """Reset position manager to initial state."""
        self.current_capital = self.initial_capital
        self.total_margin_used = 0.0
        self.cumulative_pnl = 0.0
        self.borrowed_amount = 0.0
        self.accrued_interest = 0.0
        self.allocations.clear()
