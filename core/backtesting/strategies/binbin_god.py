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
    technical_score: float  # RSI + MA position (replaces stability)
    liquidity_score: float  # Volume-based liquidity
    total_score: float
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "pe_ratio": self.pe_ratio,
            "iv_rank": self.iv_rank,
            "momentum": self.momentum,
            "technical_score": self.technical_score,
            "liquidity_score": self.liquidity_score,
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
        # Don't call super().__init__ to avoid double ML initialization
        # Instead, manually set base class attributes
        self.params = config
        self.dte_min = config.get("dte_min", 30)
        self.dte_max = config.get("dte_max", 45)
        self.delta_target = config.get("delta_target", 0.30)
        self.profit_target_pct = config.get("profit_target_pct", 50)
        self.stop_loss_pct = config.get("stop_loss_pct", 200)
        self.initial_capital = config.get("initial_capital", 100000)
        self.max_risk_per_trade = config.get("max_risk_per_trade", 0.02)
        self.max_leverage = config.get("max_leverage", 1.0)
        
        self.config = config
        self.symbol = config.get("symbol", "MAG7_AUTO")
        self.max_positions = config.get("max_positions", 10)
        self.use_synthetic_data = config.get("use_synthetic_data", False)
        
        # Wheel-specific parameters
        self.put_delta = config.get("put_delta", 0.30)
        self.call_delta = config.get("call_delta", 0.30)
        
        # CC optimization parameters
        self.cc_optimization_enabled = config.get("cc_optimization_enabled", True)
        self.cc_min_delta_cost = config.get("cc_min_delta_cost", 0.15)  # Min delta when cost > price
        self.cc_cost_basis_threshold = config.get("cc_cost_basis_threshold", 0.05)  # 5% below cost to trigger optimization
        self.cc_min_strike_premium = config.get("cc_min_strike_premium", 0.02)  # Min premium as % of cost basis
        
        # ML delta optimization parameters - use BaseStrategy's implementation
        self.ml_delta_optimization = config.get("ml_delta_optimization", False)
        self.ml_adoption_rate = config.get("ml_adoption_rate", 0.5)
        self.ml_integration = None
        self.logger = logging.getLogger("binbin_god")
        
        # Initialize ML integration if enabled (use BaseStrategy's pretrain_ml_model)
        if self.ml_delta_optimization:
            try:
                from core.ml.delta_strategy_integration import BinGodDeltaIntegration, AdaptiveDeltaStrategy
                self.ml_integration = BinGodDeltaIntegration(
                    ml_optimization_enabled=True,
                    fallback_delta=self.call_delta,
                    config=config.get("ml_config")
                )
                self.adaptive_strategy = AdaptiveDeltaStrategy(
                    ml_integration=self.ml_integration,
                    adoption_rate=self.ml_adoption_rate
                )
                self.logger.info("ML Delta optimizer initialized for BinbinGod")
            except ImportError as e:
                self.logger.warning(f"ML Delta optimization not available: {e}")
                self.ml_delta_optimization = False
            except Exception as e:
                self.logger.warning(f"ML Delta optimization initialization failed: {e}")
                self.ml_delta_optimization = False
        
        # Scoring weights - optimized to avoid correlation
        # IV Rank is most important for options premium (35%)
        # Technical factors provide trend confirmation (25%)
        # Momentum and Value provide balance (20% each)
        self.weights = {
            "iv_rank": 0.35,       # Higher IV = better premium income
            "technical": 0.25,     # RSI + MA position (trend quality)
            "momentum": 0.20,      # Price trend strength
            "pe_score": 0.20,      # Value factor (inverse of momentum proxy)
        }
        
        # Selection memory: avoid frequent switching
        self._last_selected_stock = None
        self._selection_count = 0
        self._min_hold_cycles = 3  # Minimum cycles before switching
        
        # Selection history: track stock switches for visualization
        self.selection_history = []  # List of {"date": date, "from": symbol, "to": symbol}
        
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
                # Use minimum 20 days for technical indicators
                if len(data) < 20:
                    continue  # Need at least 20 days for meaningful calculations
                
                # Extract latest bar
                latest_bar = data[-1]
                current_price = latest_bar["close"]
                
                # === 1. Momentum (20-day price change, normalized 0-100) ===
                prev_20_price = data[-20]["close"]
                raw_momentum = ((current_price - prev_20_price) / prev_20_price) * 100
                momentum = max(0, min(100, (raw_momentum + 20) / 70 * 100))
                
                # === 2. IV Rank (volatility-based, 30-day window) ===
                prices_30 = [bar["close"] for bar in data[-30:]]
                returns = [(prices_30[i] - prices_30[i-1]) / prices_30[i-1] 
                          for i in range(1, len(prices_30))]
                if returns and len(returns) > 1:
                    import statistics
                    vol = statistics.stdev(returns)
                    iv_rank = min(100, max(0, vol * 200))
                else:
                    iv_rank = 50.0
                
                # === 3. Technical Score (RSI + MA Position) ===
                # RSI: Prefer stocks not overbought (RSI < 70)
                if len(data) >= 14:
                    deltas = [data[i]["close"] - data[i-1]["close"] for i in range(-14, 0)]
                    gains = [d if d > 0 else 0 for d in deltas]
                    losses = [-d if d < 0 else 0 for d in deltas]
                    avg_gain = sum(gains) / 14
                    avg_loss = sum(losses) / 14
                    rs = avg_gain / avg_loss if avg_loss > 0 else 100
                    rsi = 100 - (100 / (1 + rs))
                    # RSI score: 50-70 is ideal (trending but not overbought)
                    # Score = 100 - |RSI - 60| * 2
                    rsi_score = max(0, min(100, 100 - abs(rsi - 60) * 2))
                else:
                    rsi_score = 50.0
                
                # MA Position: Price vs 20-day SMA
                sma_20 = sum(bar["close"] for bar in data[-20:]) / 20
                ma_position = (current_price - sma_20) / sma_20 * 100  # % above/below MA
                # Prefer stocks slightly above MA (0-10%)
                ma_score = max(0, min(100, 50 + ma_position * 5))
                
                # Combined technical score
                technical_score = (rsi_score * 0.6 + ma_score * 0.4)
                
                # === 4. Liquidity Score (volume-based) ===
                if len(data) >= 10 and "volume" in data[-1]:
                    recent_vol = sum(bar.get("volume", 0) for bar in data[-5:]) / 5
                    prior_vol = sum(bar.get("volume", 0) for bar in data[-10:-5]) / 5
                    if prior_vol > 0:
                        vol_change = recent_vol / prior_vol
                        # Higher recent volume = better liquidity
                        liquidity_score = min(100, vol_change * 50)
                    else:
                        liquidity_score = 50.0
                else:
                    # Default for synthetic data without volume
                    liquidity_score = 70.0
                
                # === 5. PE Ratio (value factor, inverse of momentum) ===
                pe_ratio = max(5, min(60, 35 + raw_momentum * 0.3))
                
            else:
                # Real-time mode: use provided fundamentals
                pe_ratio = data.get("fundamentals", {}).get("pe_ratio", 25.0)
                iv_rank = data.get("options", {}).get("iv_rank", 50.0)
                momentum = data.get("technical", {}).get("momentum_score", 50.0)
                technical_score = data.get("technical", {}).get("technical_score", 50.0)
                liquidity_score = data.get("technical", {}).get("liquidity_score", 70.0)
            
            # Normalize PE ratio (lower is better, invert the score)
            pe_score = max(0, min(100, 100 - pe_ratio))
            
            # Calculate weighted total score
            total_score = (
                iv_rank * self.weights["iv_rank"] +
                technical_score * self.weights["technical"] +
                momentum * self.weights["momentum"] +
                pe_score * self.weights["pe_score"]
            )
            
            # Apply liquidity penalty for very low liquidity
            if liquidity_score < 30:
                total_score *= 0.8  # 20% penalty for low liquidity
            
            scores.append(StockScore(
                symbol=symbol,
                pe_ratio=pe_ratio,
                iv_rank=iv_rank,
                momentum=momentum,
                technical_score=technical_score,
                liquidity_score=liquidity_score,
                total_score=total_score,
            ))
        
        # Sort by total score (descending)
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores
    
    def _select_best_stock(self, market_data: Dict[str, Any], current_date: str = None) -> str:
        """Select the best stock from MAG7 based on scoring with memory.
        
        Implements a minimum hold period to avoid excessive switching.
        Will stick with previous selection unless a significantly better option exists.
        
        Args:
            market_data: Market data for all stocks
            current_date: Current backtest date for tracking selection history
        """
        scores = self._score_stocks(market_data)
        
        if not scores:
            logger.warning("No stocks scored, defaulting to NVDA")
            return "NVDA"
        
        # Store analysis results
        self.mag7_analysis["ranked_stocks"] = [s.to_dict() for s in scores]
        self.mag7_analysis["best_pick"] = scores[0].to_dict()
        
        best_symbol = scores[0].symbol
        best_score = scores[0].total_score
        
        # Selection memory: avoid frequent switching
        if self._last_selected_stock is not None:
            self._selection_count += 1
            
            # Find previous stock's current score
            prev_stock_score = None
            for s in scores:
                if s.symbol == self._last_selected_stock:
                    prev_stock_score = s.total_score
                    break
            
            # Only switch if:
            # 1. Held for minimum cycles, OR
            # 2. New stock is significantly better (>10% score improvement)
            if prev_stock_score is not None:
                score_improvement = (best_score - prev_stock_score) / prev_stock_score * 100
                
                if self._selection_count < self._min_hold_cycles and score_improvement < 10:
                    # Stick with previous selection
                    logger.info(
                        f"Keeping {self._last_selected_stock} (score: {prev_stock_score:.1f}) "
                        f"vs {best_symbol} (score: {best_score:.1f}), "
                        f"improvement: {score_improvement:.1f}%"
                    )
                    return self._last_selected_stock
        
        # Update selection memory
        if self._last_selected_stock != best_symbol:
            # Record the selection change
            if current_date:
                self.selection_history.append({
                    "date": current_date,
                    "from": self._last_selected_stock,
                    "to": best_symbol,
                    "score": best_score
                })
            logger.info(f"Switching from {self._last_selected_stock} to {best_symbol}")
            self._selection_count = 0
        self._last_selected_stock = best_symbol
        
        logger.info(f"Selected {best_symbol} with score {best_score:.1f}")
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
                    actual_symbol = self._select_best_stock(market_data, current_date)
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
                actual_symbol = self._select_best_stock(market_data, current_date)
                self._current_cc_stock = actual_symbol
            else:
                actual_symbol = self.symbol
            
            # CRITICAL: Check if we already have Call positions covering shares
            # This prevents over-selling calls beyond share holdings
            existing_call_contracts = sum(
                abs(p.quantity) for p in wheel_positions 
                if p.trade_type == "BINBIN_CALL"
            )
            shares_already_covered = existing_call_contracts * 100
            shares_available = self.stock_holding.shares - shares_already_covered
            
            if shares_available <= 0:
                logger.info(
                    f"All shares already covered by existing calls. "
                    f"Shares: {self.stock_holding.shares}, Covered: {shares_already_covered}"
                )
                return []
            
            return self._generate_backtest_call_signal(
                actual_symbol, current_date, underlying_price, iv, position_mgr, shares_available
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
        shares_available: int = None,
    ) -> list[Signal]:
        """Generate Covered Call signal for backtesting.
        
        Args:
            shares_available: Maximum shares that can be covered by new calls.
                              If None, use all shares held.
        """
        from datetime import timedelta
        from core.backtesting.pricing import OptionsPricer
        
        # Determine shares available for covering new calls
        if shares_available is None:
            shares_available = self.stock_holding.shares
        
        # Can only sell calls for shares we own
        if shares_available <= 0:
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
        
        # Check if we need CC optimization
        call_delta_target = self.call_delta
        additional_constraints = {}
        
        # ML Delta optimization
        ml_result = None
        if self.ml_delta_optimization and self.ml_integration:
            try:
                ml_result = self.ml_integration.optimize_call_delta(
                    symbol=symbol,
                    current_price=underlying_price,
                    cost_basis=self.stock_holding.cost_basis,
                    bars=[],  # Will be populated by real-time interface
                    options_data=[],  # Will be populated by real-time interface
                    iv=iv
                )
                logger.info(f"ML optimized delta: {ml_result.optimal_delta:.3f} (confidence: {ml_result.confidence:.2f})")
            except Exception as e:
                logger.warning(f"ML optimization failed: {e}")
                ml_result = None
        
        if self.cc_optimization_enabled and self.stock_holding.cost_basis > 0:
            # Check if current price is below cost basis (loss position)
            price_cost_ratio = underlying_price / self.stock_holding.cost_basis
            
            if price_cost_ratio < (1 - self.cc_cost_basis_threshold):
                # We're in a loss position, optimize for higher strike price
                logger.info(
                    f"CC optimization: Stock price ${underlying_price:.2f} below cost basis "
                    f"${self.stock_holding.cost_basis:.2f} (ratio: {price_cost_ratio:.2f}). "
                )
                
                # Combine CC optimization with ML if available
                if ml_result and ml_result.confidence > 0.7:
                    # Use ML result with CC protective adjustments
                    call_delta_target = max(self.cc_min_delta_cost, ml_result.optimal_delta)
                    logger.info(f"Combined ML + CC optimization: delta = {call_delta_target:.3f}")
                else:
                    # Traditional CC optimization
                    call_delta_target = self.cc_min_delta_cost
                    
                # Add minimum strike constraint: try to get strike close to cost basis
                min_strike_desired = self.stock_holding.cost_basis * (1 - self.cc_min_strike_premium)
                additional_constraints["min_strike"] = min_strike_desired
                logger.info(f"CC optimization: Setting minimum strike target to ${min_strike_desired:.2f}")
        
        # Apply ML adaptive strategy if available
        if self.ml_delta_optimization and self.ml_integration and ml_result:
            # Use adaptive strategy to combine traditional and ML approaches
            final_delta = self.adaptive_strategy.select_call_delta(
                traditional_delta=call_delta_target,
                ml_result=ml_result
            )
            logger.info(f"Adaptive final delta: {final_delta:.3f}")
        else:
            final_delta = call_delta_target
        
        # Use optimized call-specific delta
        original_delta = self.delta_target
        self.delta_target = final_delta
        strike = self.select_strike_with_constraints(underlying_price, iv, T, "C", additional_constraints)
        self.delta_target = original_delta
        
        premium = OptionsPricer.call_price(underlying_price, strike, T, iv)
        delta = OptionsPricer.delta(underlying_price, strike, T, iv, "C")
        
        # Covered call: Calculate contracts based on shares available
        # generate_signals() already checks for existing call positions to prevent over-selling
        max_by_shares = shares_available // 100
        max_contracts = min(max_by_shares, self.max_positions)
        
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
        
        # CRITICAL: Check existing Call positions to prevent over-selling
        # Count existing call contracts from portfolio
        existing_call_contracts = 0
        for pos in portfolio.get("positions", []):
            if pos.get("trade_type") == "BINBIN_CALL" or (
                pos.get("right") == "C" and pos.get("quantity", 0) < 0
            ):
                existing_call_contracts += abs(pos.get("quantity", 0))
        
        shares_already_covered = existing_call_contracts * 100
        shares_available = self.stock_holding.shares - shares_already_covered
        
        if shares_available <= 0:
            logger.info(
                f"All shares already covered by existing calls. "
                f"Shares: {self.stock_holding.shares}, Covered: {shares_already_covered}"
            )
            return None
        
        # Calculate how many call contracts we can sell (1 contract per 100 shares)
        max_contracts = shares_available // 100
        if max_contracts <= 0:
            return None
        
        # Filter for calls with target DTE and delta (with optimization if needed)
        suitable_contracts = []
        
        # Determine target delta range based on optimization
        if self.cc_optimization_enabled and self.stock_holding.cost_basis > 0:
            # Check if current price is below cost basis (loss position)
            current_price = bars[-1]["close"] if bars else underlying_price
            price_cost_ratio = current_price / self.stock_holding.cost_basis
            
            if price_cost_ratio < (1 - self.cc_cost_basis_threshold):
                # Use reduced delta range for optimization
                min_delta = self.cc_min_delta_cost
                max_delta = 0.35  # Keep upper bound reasonable
                logger.info(
                    f"Real-time CC optimization: Stock price ${current_price:.2f} below cost basis "
                    f"${self.stock_holding.cost_basis:.2f}, using delta range [{min_delta:.2f}, {max_delta:.2f}]"
                )
            else:
                min_delta, max_delta = 0.25, 0.35  # Normal range
        else:
            min_delta, max_delta = 0.25, 0.35  # Normal range
        
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
            
            # Check delta (absolute value) with dynamic range
            delta = abs(contract.get("delta", 0))
            if min_delta <= delta <= max_delta:
                # Use original call_delta as reference for sorting, even with optimization
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
            
            # CRITICAL: Defensive check - if no shares held, this is an error
            if self.stock_holding.shares <= 0 or self.stock_holding.cost_basis <= 0:
                logger.warning(
                    f"Call assigned but no shares held! This indicates a bug. "
                    f"shares={self.stock_holding.shares}, cost_basis={self.stock_holding.cost_basis:.2f}"
                )
                return
            
            # Limit shares_sold to actual shares held
            actual_shares_sold = min(shares_sold, self.stock_holding.shares)
            if actual_shares_sold != shares_sold:
                logger.warning(
                    f"Call assignment for {shares_sold} shares but only {self.stock_holding.shares} held. "
                    f"Adjusting to {actual_shares_sold} shares."
                )
                shares_sold = actual_shares_sold
            
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
                
                # Clear the CC stock tracking since we no longer hold shares
                if hasattr(self, '_current_cc_stock'):
                    del self._current_cc_stock
    
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
                # Ensure shares acquired is a multiple of 100 to maintain proper lot sizes
                shares_acquired = quantity * 100  # This should already be multiple of 100, but ensure it
                
                # Calculate weighted average cost basis (same logic as wheel.py)
                total_stock_cost = self.stock_holding.shares * self.stock_holding.cost_basis
                total_stock_cost += shares_acquired * strike
                self.stock_holding.shares += shares_acquired
                if self.stock_holding.shares > 0:
                    self.stock_holding.cost_basis = total_stock_cost / self.stock_holding.shares
                
                self.phase = "CC"  # Switch to Covered Call phase
                
                # IMPORTANT: Record which stock we're holding for CC phase
                # The symbol comes from the trade (the stock underlying the option)
                assigned_symbol = trade.get("symbol", "UNKNOWN")
                self._current_cc_stock = assigned_symbol
                
                logger.info(f"Put assigned: Bought {shares_acquired} shares @ ${strike} on {assigned_symbol}, switching to CC phase")
            
            elif right == "C":
                # Call assignment: we sold shares
                shares_sold = quantity * 100
                
                # CRITICAL: Defensive check - if no shares held, this is an error
                # This should not happen after the fix to generate_signals, but we guard against it
                if self.stock_holding.shares <= 0 or self.stock_holding.cost_basis <= 0:
                    logger.warning(
                        f"Call assigned but no shares held! This indicates a bug. "
                        f"shares={self.stock_holding.shares}, cost_basis={self.stock_holding.cost_basis:.2f}"
                    )
                    # Return 0 to avoid incorrect PnL
                    return 0.0
                
                # Ensure shares_sold is a multiple of 100 to maintain proper lot sizes
                # But also ensure we don't sell more shares than we hold
                actual_shares_sold = min(shares_sold, self.stock_holding.shares)
                
                # Make sure we're selling in lots of 100 shares (or all remaining shares if less than 100)
                if actual_shares_sold >= 100:
                    # Round down to nearest 100 to maintain lot size
                    actual_shares_sold = (actual_shares_sold // 100) * 100
                # If less than 100 shares held, we'll sell all of them
                
                if actual_shares_sold != shares_sold:
                    logger.info(
                        f"Adjusted call assignment from {shares_sold} to {actual_shares_sold} shares "
                        f"to maintain proper lot size (100-share multiples)."
                    )
                    shares_sold = actual_shares_sold
                
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
        
        # Return additional stock P&L for engine to add to cumulative_pnl
        return additional_stock_pnl
    
    def select_strike_with_constraints(
        self,
        underlying_price: float,
        iv: float,
        T: float,
        right: str,
        constraints: Dict[str, Any] = None,
    ) -> float:
        """Select strike price with additional constraints."""
        from core.backtesting.pricing import OptionsPricer
        
        if constraints is None:
            constraints = {}
        
        # Binary search for the strike that gives target delta
        if right == "P":
            target_delta = -abs(self.delta_target)
            # For put options: when S > K (OTM), delta approaches 0
            # When S < K (ITM), delta approaches -1
            # So to get delta near -0.3 (slightly OTM), we want K slightly less than S
            low = underlying_price * 0.9  # e.g. 135 for S=150
            high = underlying_price * 1.05  # e.g. 157.5 for S=150
        else:
            target_delta = abs(self.delta_target)
            # For call options: when S > K (ITM), delta approaches 1
            # When S < K (OTM), delta approaches 0
            # So to get delta near 0.3 (slightly OTM for covered calls), we want K > S
            low = underlying_price * 0.8  # Wider range for better convergence
            high = underlying_price * 1.2
            
            # Apply minimum strike constraint if provided
            if "min_strike" in constraints:
                min_strike = constraints["min_strike"]
                
                # CRITICAL FIX: Check if min_strike is within reasonable range
                # If min_strike > high (underlying_price * 1.2), the constraint is impossible
                # This happens when stock price drops significantly below cost basis
                if min_strike > high:
                    logger.warning(
                        f"min_strike constraint (${min_strike:.2f}) exceeds reasonable range "
                        f"(max ${high:.2f}). Stock price ${underlying_price:.2f} is far below "
                        f"cost basis. Relaxing constraint to allow valid strike selection."
                    )
                    # Don't apply the constraint - let algorithm find best available strike
                    # This is the correct behavior: we can't sell calls above cost basis
                    # if the stock price is too low
                else:
                    low = max(low, min_strike)
                    logger.info(f"Applying minimum strike constraint: strike >= ${min_strike:.2f}")
                    
                    # If minimum strike is very high, we may need to adjust target delta
                    # to ensure we find a valid strike
                    test_delta = OptionsPricer.delta(underlying_price, min_strike, T, iv, right)
                    logger.info(f"Test delta at minimum strike ${min_strike:.2f}: {test_delta:.3f}")

        for _ in range(50):
            mid = (low + high) / 2
            d = OptionsPricer.delta(underlying_price, mid, T, iv, right)
            if right == "P":
                # For puts: increasing strike makes delta more negative
                # So if d < target_delta (too negative), decrease strike (high = mid)
                # If d > target_delta (too positive), increase strike (low = mid)
                if d < target_delta:
                    high = mid
                else:
                    low = mid
            else:
                # For calls: increasing strike makes delta decrease (more negative)
                # So if d > target_delta (too high), increase strike (low = mid)
                # If d < target_delta (too low), decrease strike (high = mid)
                if d > target_delta:
                    low = mid
                else:
                    high = mid
            if abs(d - target_delta) < 0.005:
                break

        # Round to nearest 0.5 or 1.0
        if underlying_price > 100:
            return round(mid)
        return round(mid * 2) / 2
    
    def get_mag7_analysis(self) -> Dict[str, Any]:
        """Get MAG7 analysis results."""
        return self.mag7_analysis
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report with selection history."""
        return {
            "strategy": "binbin_god",
            # current_state: for compatibility with Wheel's monitoring dashboard
            "current_state": {
                "phase": self.phase,
                "shares_held": self.stock_holding.shares,
                "cost_basis": round(self.stock_holding.cost_basis, 2),
                "total_premium_collected": round(self.stock_holding.total_premium_collected, 2),
            },
            # performance_metrics: placeholder for monitoring dashboard
            "performance_metrics": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "avg_pnl_per_trade": 0.0,
                "max_drawdown": 0.0,
                "phase_transitions": 0,
            },
            # Top-level fields for easy access
            "phase": self.phase,
            "shares_held": self.stock_holding.shares,
            "cost_basis": round(self.stock_holding.cost_basis, 2),
            "total_premium_collected": round(self.stock_holding.total_premium_collected, 2),
            "selection_history": self.selection_history,
            "mag7_analysis": self.mag7_analysis,
            # For UI compatibility with Wheel strategy monitoring
            "trade_history": [],  # Would need to track in on_trade_closed
            "phase_history": [],  # Would need to track on phase transitions
        }
