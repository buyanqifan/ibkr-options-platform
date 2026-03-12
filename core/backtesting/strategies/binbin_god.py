"""Binbin God Strategy: Dynamic MAG7 stock selection with full Wheel strategy logic.

This strategy intelligently selects the best stock from MAG7 universe based on:
- P/E Ratio (20% weight) - Value stocks preferred
- Option IV (40% weight) - Higher premium income  
- Momentum (20% weight) - Positive trend
- Stability (20% weight) - Risk management

Then executes FULL Wheel strategy logic (both Sell Put AND Covered Call phases).

Phase 1 (SP): Sell OTM puts, collect premium
  - If expires worthless: keep premium, continue selling puts
  - If assigned: buy shares at strike, switch to Phase 2

Phase 2 (CC): Sell OTM calls against owned shares
  - If expires worthless: keep premium + shares, continue selling calls
  - If assigned: sell shares at strike, return to Phase 1
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any
from core.backtesting.strategies.base import BaseStrategy, Signal
from core.backtesting.pricing import OptionsPricer
import logging


logger = logging.getLogger("binbin_god")


# MAG7 Universe
MAG7_STOCKS = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]


@dataclass
class StockScore:
    """Score for a single stock."""
    symbol: str
    pe_ratio: float
    iv_rank: float
    momentum: float
    stability: float
    total_score: float
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "pe_ratio": self.pe_ratio,
            "iv_rank": self.iv_rank,
            "momentum": self.momentum,
            "stability": self.stability,
            "total_score": round(self.total_score, 2),
        }


@dataclass
class StockHolding:
    """Tracks stock position from put assignment."""
    shares: int = 0
    cost_basis: float = 0.0  # average cost per share
    total_premium_collected: float = 0.0  # cumulative premium from both phases


class BinbinGodStrategy(BaseStrategy):
    """
    Binbin God Strategy - Intelligent stock selection + Full Wheel logic.
    
    Implements BOTH Sell Put and Covered Call phases.
    """
    
    @property
    def name(self) -> str:
        return "binbin_god"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.symbol = config.get("symbol", "MAG7_AUTO")
        self.dte_min = config.get("dte_min", 30)
        self.dte_max = config.get("dte_max", 45)
        self.delta_target = config.get("delta_target", 0.30)
        self.profit_target_pct = config.get("profit_target_pct", 50)
        self.stop_loss_pct = config.get("stop_loss_pct", 200)
        self.max_positions = config.get("max_positions", 10)
        self.use_synthetic_data = config.get("use_synthetic_data", False)
        
        # Wheel-specific parameters
        self.put_delta = config.get("put_delta", 0.30)
        self.call_delta = config.get("call_delta", 0.30)
        
        # Scoring weights
        self.weights = {
            "pe_ratio": 0.20,
            "iv_rank": 0.40,
            "momentum": 0.20,
            "stability": 0.20,
        }
        
        # Phase tracking
        self.phase = "SP"  # Start with Sell Put phase
        self.stock_holding = StockHolding()
        
        # Stock pool for selection (default: MAG7)
        self.stock_pool = config.get("stock_pool", MAG7_STOCKS.copy())
        
        # Storage for analysis
        self.mag7_analysis = {
            "ranked_stocks": [],
            "best_pick": None,
        }
        
        # Check if profit target/stop loss are disabled
        self._profit_target_disabled = self.profit_target_pct >= 999999
        self._stop_loss_disabled = self.stop_loss_pct >= 999999
    
    def _score_stocks(self, market_data: Dict[str, Any]) -> List[StockScore]:
        """Score all stocks in the pool based on metrics."""
        scores = []
        
        # Use configurable stock pool instead of hardcoded MAG7
        stock_pool = getattr(self, 'stock_pool', MAG7_STOCKS)
        
        for symbol in stock_pool:
            # Get market data for this symbol
            if symbol not in market_data:
                continue
                
            data = market_data[symbol]
            
            # Check if this is backtest mode (data is list of bars)
            if isinstance(data, list):
                # Backtest mode: calculate metrics from price bars
                # Use minimum 5 days for early dates in backtest
                if len(data) < 5:
                    continue  # Need at least 5 days for basic calculations
                
                # Extract latest bar
                latest_bar = data[-1]
                current_price = latest_bar["close"]
                
                # Calculate momentum from recent returns (different for each stock)
                prev_20_price = data[-20]["close"] if len(data) >= 20 else data[0]["close"]
                momentum = ((current_price - prev_20_price) / prev_20_price) * 100
                
                # Calculate IV proxy from recent volatility (unique per stock)
                prices = [bar["close"] for bar in data[-30:]]
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                if returns and len(returns) > 1:
                    import statistics
                    # Higher volatility = higher IV (use as IV proxy)
                    vol = statistics.stdev(returns)
                    iv_rank = min(100, max(0, vol * 200))  # Scale volatility to 0-100
                    stability = 100 - (vol * 100)  # Lower vol = higher stability
                else:
                    iv_rank = 50.0
                    stability = 50.0
                
                # PE ratio: estimate from price trend (simplified proxy)
                # Higher recent return = assumed lower PE (growth stock)
                # Normalize: -20% to +50% return maps to PE 50 to 15
                pe_ratio = max(5, min(60, 35 - momentum * 0.3))
                
            else:
                # Real-time mode: use provided fundamentals
                pe_ratio = data.get("fundamentals", {}).get("pe_ratio", 25.0)
                iv_rank = data.get("options", {}).get("iv_rank", 50.0)
                momentum = data.get("technical", {}).get("momentum_score", 50.0)
                stability = data.get("technical", {}).get("stability_score", 50.0)
            
            # Normalize PE ratio (lower is better, invert the score)
            pe_score = max(0, min(100, 100 - pe_ratio))
            
            # Calculate weighted total score
            total_score = (
                pe_score * self.weights["pe_ratio"] +
                iv_rank * self.weights["iv_rank"] +
                momentum * self.weights["momentum"] +
                stability * self.weights["stability"]
            )
            
            scores.append(StockScore(
                symbol=symbol,
                pe_ratio=pe_ratio,
                iv_rank=iv_rank,
                momentum=momentum,
                stability=stability,
                total_score=total_score,
            ))
        
        # Sort by total score (descending)
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores
    
    def _select_best_stock(self, market_data: Dict[str, Any]) -> str:
        """Select the best stock from MAG7 based on scoring."""
        scores = self._score_stocks(market_data)
        
        if not scores:
            logger.warning("No stocks scored, defaulting to NVDA")
            return "NVDA"
        
        # Store analysis results
        self.mag7_analysis["ranked_stocks"] = [s.to_dict() for s in scores]
        self.mag7_analysis["best_pick"] = scores[0].to_dict()
        
        best_symbol = scores[0].symbol
        logger.info(f"Selected {best_symbol} with score {scores[0].total_score:.1f}")
        
        return best_symbol
    
    def generate_signals(
        self,
        current_date: str,
        underlying_price: float,
        iv: float,
        open_positions: list,
        position_mgr=None,
    ) -> list[Signal]:
        """Generate signals for backtesting (standard interface).
        
        This method adapts the real-time generate_signal() to the backtesting interface.
        
        Key logic:
        - Always re-select the best stock when entering SP phase
        - If holding shares, continue using that stock for CC phase
        - Can switch stocks between different SP cycles
        """
        from datetime import datetime
        
        # Check if we already have max positions
        wheel_positions = [
            p for p in open_positions 
            if p.trade_type in ("BINBIN_PUT", "BINBIN_CALL")
        ]
        if len(wheel_positions) >= self.max_positions:
            return []
        
        # Select best stock dynamically based on phase
        if self.phase == "SP":
            # In SP phase: always re-select the best stock for new puts
            if "AUTO" in self.symbol:
                # Get all stock pool data for scoring
                pool_data = getattr(self, 'mag7_data', {})
                stock_pool = getattr(self, 'stock_pool', MAG7_STOCKS)
                
                # Build market_data with FULL bars data for each stock
                # _score_stocks expects: market_data[symbol] = list of bars (backtest mode)
                market_data = {}
                for sym in stock_pool:
                    bars = pool_data.get(sym, [])
                    if bars and len(bars) > 0:
                        # Filter bars up to current_date
                        filtered_bars = [bar for bar in bars if bar["date"][:10] <= current_date]
                        if filtered_bars:
                            # Pass full bars list for scoring (backtest mode)
                            market_data[sym] = filtered_bars
                
                # Select best stock based on current metrics
                if market_data:
                    actual_symbol = self._select_best_stock(market_data)
                    logger.info(f"SP phase: Selected {actual_symbol} for new put position")
                else:
                    # Fallback: use first stock in pool
                    actual_symbol = stock_pool[0] if stock_pool else "MSFT"
                    logger.warning(f"No market data available, using fallback: {actual_symbol}")
            else:
                actual_symbol = self.symbol
            
            return self._generate_backtest_put_signal(
                actual_symbol, current_date, underlying_price, iv, position_mgr
            )
        else:  # CC phase
            # In CC phase: use the stock we already hold shares of
            # Don't switch stocks mid-cycle
            if hasattr(self, '_current_cc_stock'):
                actual_symbol = self._current_cc_stock
            elif self.symbol == "MAG7_AUTO":
                # Fallback: select a stock (shouldn't happen normally)
                mag7_data = getattr(self, 'mag7_data', {})
                market_data = {}
                for sym, bars in mag7_data.items():
                    if bars and len(bars) > 0:
                        current_bar = None
                        for bar in bars:
                            if bar["date"][:10] <= current_date:
                                current_bar = bar
                        if current_bar:
                            market_data[sym] = {
                                'current_date': current_date,
                                'underlying_price': current_bar["close"],
                                'iv': iv,
                            }
                actual_symbol = self._select_best_stock(market_data)
                self._current_cc_stock = actual_symbol
            else:
                actual_symbol = self.symbol
            
            return self._generate_backtest_call_signal(
                actual_symbol, current_date, underlying_price, iv, position_mgr
            )
    
    def _generate_backtest_put_signal(
        self,
        symbol: str,
        current_date: str,
        underlying_price: float,
        iv: float,
        position_mgr=None,
    ) -> list[Signal]:
        """Generate Sell Put signal for backtesting."""
        from datetime import timedelta
        from core.backtesting.pricing import OptionsPricer
        
        T = self.dte_max / 365.0
        dte_days = int(self.dte_max)
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=dte_days)
        expiry_str = expiry_date.strftime("%Y%m%d")
        
        # Use put-specific delta
        original_delta = self.delta_target
        self.delta_target = self.put_delta
        strike = self.select_strike(underlying_price, iv, T, "P")
        self.delta_target = original_delta
        
        premium = OptionsPricer.put_price(underlying_price, strike, T, iv)
        delta = OptionsPricer.delta(underlying_price, strike, T, iv, "P")
        
        # Position sizing using position manager
        if position_mgr:
            max_contracts = position_mgr.calculate_position_size(
                margin_per_contract=strike * 100,
                max_positions=self.max_positions,
            )
        else:
            # Fallback: 1 contract per $10k
            max_contracts = min(int(self.initial_capital / 10000), self.max_positions)
        
        if max_contracts <= 0:
            return []
        
        quantity = -max_contracts  # Sell
        
        return [Signal(
            symbol=symbol,
            trade_type="BINBIN_PUT",
            right="P",
            strike=strike,
            expiry=expiry_str,
            quantity=quantity,
            iv=iv,
            delta=delta,
            premium=premium,
            underlying_price=underlying_price,
            margin_requirement=strike * 100,
        )]
    
    def _generate_backtest_call_signal(
        self,
        symbol: str,
        current_date: str,
        underlying_price: float,
        iv: float,
        position_mgr=None,
    ) -> list[Signal]:
        """Generate Covered Call signal for backtesting."""
        from datetime import timedelta
        from core.backtesting.pricing import OptionsPricer
        
        # Can only sell calls for shares we own
        if self.stock_holding.shares <= 0:
            # Fallback to SP phase
            self.phase = "SP"
            return self._generate_backtest_put_signal(
                symbol, current_date, underlying_price, iv, position_mgr
            )
        
        T = self.dte_max / 365.0
        dte_days = int(self.dte_max)
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=dte_days)
        expiry_str = expiry_date.strftime("%Y%m%d")
        
        # Use call-specific delta
        original_delta = self.delta_target
        self.delta_target = self.call_delta
        strike = self.select_strike(underlying_price, iv, T, "C")
        self.delta_target = original_delta
        
        premium = OptionsPricer.call_price(underlying_price, strike, T, iv)
        delta = OptionsPricer.delta(underlying_price, strike, T, iv, "C")
        
        # Covered call: 1 contract per 100 shares owned
        max_contracts = min(self.stock_holding.shares // 100, self.max_positions)
        
        if max_contracts <= 0:
            return []
        
        quantity = -max_contracts  # Sell
        
        return [Signal(
            symbol=symbol,
            trade_type="BINBIN_CALL",
            right="C",
            strike=strike,
            expiry=expiry_str,
            quantity=quantity,
            iv=iv,
            delta=delta,
            premium=premium,
            underlying_price=underlying_price,
            margin_requirement=0.0,  # No additional margin - shares are collateral
        )]
    
    def generate_signal(
        self,
        symbol: str,
        current_dt: datetime,
        bars: List[Dict],
        contracts: List[Dict],
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Signal | None:
        """Generate trading signal for Binbin God strategy (real-time interface)."""
        
        # If symbol is MAG7_AUTO, select the best stock
        if self.symbol == "MAG7_AUTO":
            actual_symbol = self._select_best_stock(market_data)
        else:
            actual_symbol = self.symbol
        
        # Check phase and generate appropriate signal
        if self.phase == "SP":
            return self._generate_put_signal(
                actual_symbol, current_dt, bars, contracts, portfolio, market_data
            )
        else:  # CC phase
            return self._generate_call_signal(
                actual_symbol, current_dt, bars, contracts, portfolio, market_data
            )
    
    def _generate_put_signal(
        self,
        symbol: str,
        current_dt: datetime,
        bars: List[Dict],
        contracts: List[Dict],
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Signal | None:
        """Generate Sell Put signal (Phase 1)."""
        
        # Check if we already have max positions
        current_positions = len(portfolio.get("positions", []))
        if current_positions >= self.max_positions:
            logger.debug(f"Max positions ({self.max_positions}) reached")
            return None
        
        # Filter for puts with target DTE and delta
        suitable_contracts = []
        
        for contract in contracts:
            if contract.get("right", "") != "P":
                continue
            
            # Check DTE
            expiry = contract.get("expiry")
            if expiry:
                if isinstance(expiry, datetime):
                    dte = (expiry.date() - current_dt.date()).days
                else:
                    dte = 0
            else:
                dte = 0
                
            if not (self.dte_min <= dte <= self.dte_max):
                continue
            
            # Check delta (absolute value)
            delta = abs(contract.get("delta", 0))
            if 0.25 <= delta <= 0.35:  # Allow some flexibility around target
                suitable_contracts.append((contract, abs(delta - self.put_delta)))
        
        if not suitable_contracts:
            logger.debug(f"No suitable put contracts found for {symbol}")
            return None
        
        # Select contract closest to target delta
        suitable_contracts.sort(key=lambda x: x[1])
        selected_contract = suitable_contracts[0][0]
        
        # Calculate position size (1 contract per $10k capital as rough guide)
        capital = portfolio.get("cash", 100000)
        max_contracts_by_capital = int(capital / 10000)
        max_contracts_by_limit = self.max_positions - current_positions
        quantity = min(max_contracts_by_capital, max_contracts_by_limit, 10)
        
        if quantity <= 0:
            return None
        
        logger.info(f"Selling {quantity} put(s) on {symbol} @ ${selected_contract.get('bid', 0):.2f}")
        
        return Signal(
            action="SELL_PUT",
            symbol=symbol,
            contract=selected_contract,
            quantity=quantity,
            price=selected_contract.get("bid", 0),
        )
    
    def _generate_call_signal(
        self,
        symbol: str,
        current_dt: datetime,
        bars: List[Dict],
        contracts: List[Dict],
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Signal | None:
        """Generate Covered Call signal (Phase 2)."""
        
        # Check if we have shares to sell calls against
        if self.stock_holding.shares <= 0:
            logger.warning("In CC phase but no shares held, switching back to SP")
            self.phase = "SP"
            return None
        
        # Calculate how many call contracts we can sell (1 contract per 100 shares)
        max_contracts = self.stock_holding.shares // 100
        if max_contracts <= 0:
            return None
        
        # Filter for calls with target DTE and delta
        suitable_contracts = []
        
        for contract in contracts:
            if contract.get("right", "") != "C":
                continue
            
            # Check DTE
            expiry = contract.get("expiry")
            if expiry:
                if isinstance(expiry, datetime):
                    dte = (expiry.date() - current_dt.date()).days
                else:
                    dte = 0
            else:
                dte = 0
                
            if not (self.dte_min <= dte <= self.dte_max):
                continue
            
            # Check delta (absolute value)
            delta = abs(contract.get("delta", 0))
            if 0.25 <= delta <= 0.35:  # Allow some flexibility around target
                suitable_contracts.append((contract, abs(delta - self.call_delta)))
        
        if not suitable_contracts:
            logger.debug(f"No suitable call contracts found for {symbol}")
            return None
        
        # Select contract closest to target delta
        suitable_contracts.sort(key=lambda x: x[1])
        selected_contract = suitable_contracts[0][0]
        
        # Sell calls against shares (limited by shares owned)
        quantity = min(max_contracts, 10)
        
        if quantity <= 0:
            return None
        
        logger.info(f"Selling {quantity} call(s) on {symbol} @ ${selected_contract.get('bid', 0):.2f}")
        
        return Signal(
            action="SELL_CALL",
            symbol=symbol,
            contract=selected_contract,
            quantity=quantity,
            price=selected_contract.get("bid", 0),
        )
    
    def should_exit_position(
        self,
        position: Dict[str, Any],
        current_price: float,
        entry_price: float,
        current_dt: datetime,
    ) -> tuple[bool, str]:
        """Check if position should be exited."""
        
        # Calculate P&L
        pnl = current_price - entry_price
        pnl_pct = (pnl / entry_price) * 100 if entry_price > 0 else 0
        
        # Profit target exit
        if not self._profit_target_disabled:
            profit_threshold = self.profit_target_pct / 100.0 * abs(entry_price)
            if abs(pnl) >= profit_threshold and pnl < 0:  # Premium decayed enough
                return True, "PROFIT_TARGET"
        
        # Stop loss exit
        if not self._stop_loss_disabled:
            loss_threshold = self.stop_loss_pct / 100.0 * abs(entry_price)
            if pnl >= loss_threshold:  # Loss exceeded threshold
                return True, "STOP_LOSS"
        
        # Expiry exit
        expiry = position.get("expiry")
        if expiry and current_dt >= expiry:
            return True, "EXPIRY"
        
        return False, ""
    
    def on_assignment(self, position: Dict[str, Any]):
        """Called when option is assigned/exercised."""
        right = position.get("right", "")
        quantity = abs(position.get("quantity", 0))
        strike = position.get("strike", 0)
        
        if right == "P":
            # Put assignment: we bought shares
            shares_acquired = quantity * 100
            self.stock_holding.shares += shares_acquired
            self.stock_holding.cost_basis = strike
            self.phase = "CC"  # Switch to Covered Call phase
            logger.info(f"Put assigned: Bought {shares_acquired} shares @ ${strike}, switching to CC phase")
        
        elif right == "C":
            # Call assignment: we sold shares
            shares_sold = quantity * 100
            if shares_sold <= self.stock_holding.shares:
                # Calculate realized stock P&L
                stock_cost_basis = self.stock_holding.cost_basis * shares_sold
                stock_proceeds = strike * shares_sold
                stock_pnl = stock_proceeds - stock_cost_basis
               
                # Log complete P&L breakdown
                option_pnl = position.get("pnl", 0)
                total_trade_pnl = option_pnl + stock_pnl
                logger.info(
                    f"Call assigned: Option P&L=${option_pnl:+.2f}, Stock P&L=${stock_pnl:+.2f}, "
                    f"Total=${total_trade_pnl:+.2f} (bought at ${self.stock_holding.cost_basis:.2f}, "
                    f"sold at ${strike:.2f}, {shares_sold} shares)"
                )
               
                self.stock_holding.shares -= shares_sold
                if self.stock_holding.shares == 0:
                    self.phase = "SP"  # Switch back to Sell Put phase
            else:
                logger.warning(f"Call assignment error: Trying to sell {shares_sold} shares but only have {self.stock_holding.shares}")
    
    def on_trade_closed(self, trade: dict):
        """Called by engine when a trade is closed. Updates internal state and tracks performance.
        
        Returns:
            float: Additional P&L from stock position (e.g., when call is assigned).
                   This should be added to cumulative_pnl by the engine.
        """
        # Track additional stock P&L to return to engine
        additional_stock_pnl = 0.0
        
        if trade.get("exit_reason") == "ASSIGNMENT":
            right = trade.get("right", "")
            quantity = abs(trade.get("quantity", 0))
            strike = trade.get("strike", 0)
            
            if right == "P":
                # Put assignment: we bought shares
                shares_acquired = quantity * 100
                self.stock_holding.shares += shares_acquired
                self.stock_holding.cost_basis = strike
                self.phase = "CC"  # Switch to Covered Call phase
                
                # IMPORTANT: Record which stock we're holding for CC phase
                # The symbol comes from the trade (the stock underlying the option)
                assigned_symbol = trade.get("symbol", "UNKNOWN")
                self._current_cc_stock = assigned_symbol
                
                logger.info(f"Put assigned: Bought {shares_acquired} shares @ ${strike} on {assigned_symbol}, switching to CC phase")
            
            elif right == "C":
                # Call assignment: we sold shares
                shares_sold = quantity * 100
                if shares_sold <= self.stock_holding.shares:
                    # Calculate realized stock P&L
                    stock_cost_basis = self.stock_holding.cost_basis * shares_sold
                    stock_proceeds = strike * shares_sold
                    stock_pnl = stock_proceeds - stock_cost_basis
                    
                    # IMPORTANT: Record stock P&L to be added to cumulative_pnl
                    additional_stock_pnl = stock_pnl
                   
                    # Log complete P&L breakdown
                    option_pnl = trade.get("pnl", 0)
                    total_trade_pnl = option_pnl + stock_pnl
                    logger.info(
                        f"Call assigned: Option P&L=${option_pnl:+.2f}, Stock P&L=${stock_pnl:+.2f}, "
                        f"Total=${total_trade_pnl:+.2f} (bought at ${self.stock_holding.cost_basis:.2f}, "
                        f"sold at ${strike:.2f}, {shares_sold} shares)"
                    )
                   
                    self.stock_holding.shares -= shares_sold
                    if self.stock_holding.shares == 0:
                        self.phase = "SP"  # Switch back to Sell Put phase
                        
                        # Clear the CC stock tracking since we no longer hold shares
                        if hasattr(self, '_current_cc_stock'):
                            del self._current_cc_stock
                else:
                    logger.warning(f"Call assignment error: Trying to sell {shares_sold} shares but only have {self.stock_holding.shares}")
        
        # Return additional stock P&L for engine to add to cumulative_pnl
        return additional_stock_pnl
    
    def get_mag7_analysis(self) -> Dict[str, Any]:
        """Get MAG7 analysis results."""
        return self.mag7_analysis
