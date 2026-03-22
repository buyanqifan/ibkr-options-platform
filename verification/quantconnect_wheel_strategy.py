"""
QuantConnect Wheel Strategy Verification Script

This script is designed to run on QuantConnect platform to verify
the P&L calculations from the custom backtest engine.

Usage:
1. Copy this file to QuantConnect cloud
2. Set the same parameters as your local backtest
3. Run and compare the results

The strategy implements the Wheel strategy:
- Phase SP: Sell OTM puts
- Phase CC: Sell covered calls after assignment
"""

from AlgorithmImports import *


class WheelStrategyVerification(QCAlgorithm):
    """Wheel strategy for verifying custom backtest calculations."""
    
    def Initialize(self):
        # === PARAMETERS (Match with local backtest) ===
        self.SetStartDate(2025, 1, 1)
        self.SetEndDate(2025, 3, 31)  # Short period for verification
        self.SetCash(100000)
        
        # Strategy parameters
        self.symbol = self.AddEquity("NVDA", Resolution.Daily).Symbol
        self.put_delta = 0.30  # Target delta for puts
        self.call_delta = 0.30  # Target delta for calls
        self.dte_min = 30
        self.dte_max = 45
        self.profit_target_pct = 0.50  # 50% profit target
        self.stop_loss_pct = 2.00  # 200% stop loss
        
        # State tracking
        self.phase = "SP"  # SP or CC
        self.stock_holding = 0
        self.stock_cost_basis = 0.0
        self.total_premium_collected = 0.0
        
        # Trade tracking for comparison
        self.trades = []
        self.daily_pnl = []
        
        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.EveryDay(self.symbol),
            self.TimeRules.AfterMarketOpen(self.symbol, 30),
            self.Rebalance
        )
        
        # Warm up for indicators
        self.SetWarmUp(20)
        
    def Rebalance(self):
        """Main rebalancing logic."""
        if self.IsWarmingUp:
            return
            
        # Close any profitable positions
        self.CloseProfitablePositions()
        
        # Open new positions based on phase
        if self.phase == "SP":
            self.SellPut()
        else:
            self.SellCoveredCall()
    
    def SellPut(self):
        """Sell an OTM put."""
        # Get current option chain
        option_chain = self.GetOptionChain()
        if not option_chain:
            return
            
        # Find ATM price
        underlying_price = self.Securities[self.symbol].Price
        
        # Find OTM put with target delta
        target_strike = underlying_price * (1 - self.put_delta * 0.5)  # Approximation
        expiry = self.GetTargetExpiry()
        
        # Filter puts
        puts = [x for x in option_chain if x.Right == OptionRight.Put 
                and x.Expiry == expiry
                and x.Strike < underlying_price]
        
        if not puts:
            self.Debug(f"No puts found for {expiry}")
            return
            
        # Find put closest to target delta
        put = min(puts, key=lambda x: abs(x.Greeks.Delta + self.put_delta))
        
        if put:
            # Sell the put
            self.Sell(put.Symbol, 1)
            self.total_premium_collected += put.BidPrice * 100
            self.Debug(f"Sold Put: {put.Symbol} @ {put.BidPrice}")
    
    def SellCoveredCall(self):
        """Sell a covered call against held shares."""
        if self.stock_holding < 100:
            self.phase = "SP"
            return
            
        # Get current option chain
        option_chain = self.GetOptionChain()
        if not option_chain:
            return
            
        underlying_price = self.Securities[self.symbol].Price
        expiry = self.GetTargetExpiry()
        
        # Filter calls
        calls = [x for x in option_chain if x.Right == OptionRight.Call 
                 and x.Expiry == expiry
                 and x.Strike > underlying_price]
        
        if not calls:
            return
            
        # Find call closest to target delta
        call = min(calls, key=lambda x: abs(x.Greeks.Delta - self.call_delta))
        
        if call:
            # Sell the call
            self.Sell(call.Symbol, 1)
            self.total_premium_collected += call.BidPrice * 100
            self.Debug(f"Sold Call: {call.Symbol} @ {call.BidPrice}")
    
    def CloseProfitablePositions(self):
        """Close positions that hit profit target."""
        for holding in self.Portfolio.Values:
            if holding.Invested and holding.Type == SecurityType.Option:
                # Check profit target (50% of premium)
                if holding.UnrealizedProfitPercent >= self.profit_target_pct:
                    self.Liquidate(holding.Symbol)
                    self.Debug(f"Closed {holding.Symbol} at profit target: {holding.UnrealizedProfitPercent:.1%}")
    
    def GetTargetExpiry(self):
        """Get target expiration date."""
        # Find the first expiry within our DTE range
        option_chain = self.GetOptionChain()
        if not option_chain:
            return None
            
        today = self.Time
        expiries = sorted(set(x.Expiry for x in option_chain))
        
        for expiry in expiries:
            dte = (expiry - today).days
            if self.dte_min <= dte <= self.dte_max:
                return expiry
                
        return expiries[0] if expiries else None
    
    def GetOptionChain(self):
        """Get the option chain for the symbol."""
        canonical = self.AddOption(self.symbol)
        canonical.SetFilter(-2, 2, timedelta(days=self.dte_min), timedelta(days=self.dte_max))
        
        option_chain = self.CurrentOptionChain(self.symbol)
        if option_chain:
            return [x for x in option_chain]
        return []
    
    def OnOrderEvent(self, orderEvent):
        """Handle order events for assignment tracking."""
        if orderEvent.Status == OrderStatus.Filled:
            security = self.Securities[orderEvent.Symbol]
            
            if security.Type == SecurityType.Option:
                option = security
                if option.IsAutoExercise or (option.Right == OptionRight.Put and 
                    self.Securities[self.symbol].Price < option.Strike):
                    # Put assignment
                    shares = 100 * abs(orderEvent.FillQuantity)
                    self.stock_holding += shares
                    self.stock_cost_basis = (
                        (self.stock_cost_basis * (self.stock_holding - shares) + 
                         option.Strike * shares) / self.stock_holding
                    )
                    self.phase = "CC"
                    self.Debug(f"Put assigned: acquired {shares} shares @ ${option.Strike}")
                    
                elif option.Right == OptionRight.Call and self.Securities[self.symbol].Price > option.Strike:
                    # Call assignment
                    shares = 100 * abs(orderEvent.FillQuantity)
                    self.stock_holding -= shares
                    if self.stock_holding <= 0:
                        self.phase = "SP"
                        self.stock_holding = 0
                        self.stock_cost_basis = 0
                    self.Debug(f"Call assigned: sold {shares} shares @ ${option.Strike}")
    
    def OnEndOfAlgorithm(self):
        """Print summary at end of algorithm."""
        self.Debug("=" * 50)
        self.Debug("WHEEL STRATEGY VERIFICATION SUMMARY")
        self.Debug("=" * 50)
        self.Debug(f"Total Premium Collected: ${self.total_premium_collected:.2f}")
        self.Debug(f"Stock Holdings: {self.stock_holding} shares")
        self.Debug(f"Stock Cost Basis: ${self.stock_cost_basis:.2f}")
        self.Debug(f"Final Phase: {self.phase}")
        self.Debug(f"Total Trades: {len(self.trades)}")
        self.Debug("=" * 50)
        
        # Output detailed trade log for comparison
        for trade in self.trades:
            self.Debug(f"Trade: {trade}")