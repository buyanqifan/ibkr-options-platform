"""Global service registry for sharing services between callbacks.

Includes position management and leverage controls."""

# Global service registry
_services: dict | None = None
_position_manager: 'PositionManager' = None


def get_services() -> dict | None:
    """Access shared service instances from any callback."""
    return _services


def set_services(services: dict) -> None:
    """Initialize the global service registry."""
    global _services
    _services = services


class PositionManager:
    """Manages position sizing and leverage control."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_leverage = 1.0  # Default to no leverage
        self.margin_interest_rate = 0.05  # 5% annual interest rate
        self.borrowed_funds = 0.0
        self.position_percentage = 0.10  # Default to 10% of capital per trade
    
    def set_position_percentage(self, percentage: float):
        """Set the percentage of capital to use per position."""
        self.position_percentage = percentage
    
    def set_leverage(self, leverage: float):
        """Set the maximum leverage allowed."""
        self.max_leverage = leverage
    
    def calculate_position_size(self, asset_price: float, strike_price: float = None, 
                               option_premium: float = None) -> int:
        """Calculate position size based on available capital and leverage."""
        available_capital = self.current_capital * self.position_percentage
        
        if option_premium:
            # For options strategies, use premium and margin requirements
            if strike_price:
                # Cash-secured put: need to reserve strike_price * 100 per contract
                margin_per_contract = strike_price * 100
            else:
                # Other strategies: use asset price as reference
                margin_per_contract = asset_price * 100
            
            # Apply leverage if available
            leveraged_capital = available_capital * self.max_leverage
            position_size = int(leveraged_capital / margin_per_contract)
            return max(1, position_size)
        else:
            # For stock positions
            shares_per_position = int(available_capital / asset_price)
            return max(1, shares_per_position)
    
    def borrow_funds(self, amount: float):
        """Borrow funds for leverage, applying interest charges."""
        if amount <= 0:
            return
        
        max_borrowable = self.current_capital * (self.max_leverage - 1)
        if self.borrowed_funds + amount > max_borrowable:
            raise ValueError(f"Cannot borrow {amount}. Max borrowable: {max_borrowable - self.borrowed_funds}")
        
        self.borrowed_funds += amount
        self.current_capital += amount
    
    def apply_daily_margin_interest(self):
        """Apply daily margin interest to borrowed funds."""
        if self.borrowed_funds > 0:
            daily_interest_rate = self.margin_interest_rate / 365
            daily_interest = self.borrowed_funds * daily_interest_rate
            self.current_capital -= daily_interest
            return daily_interest
        return 0.0
    
    def get_available_capital(self):
        """Get available capital for trading (including leverage)."""
        return self.current_capital * self.max_leverage
    
    def get_total_portfolio_value(self, unrealized_pnl: float = 0):
        """Get total portfolio value including unrealized PnL."""
        return self.current_capital + unrealized_pnl

def get_position_manager() -> PositionManager:
    """Get the position manager instance."""
    global _position_manager
    if _position_manager is None:
        # Initialize with default capital if services are available
        initial_capital = 100000
        services = get_services()
        if services and 'data_client' in services:
            from config.settings import settings
            initial_capital = getattr(settings, 'INITIAL_CAPITAL', 100000)
        _position_manager = PositionManager(initial_capital)
    return _position_manager