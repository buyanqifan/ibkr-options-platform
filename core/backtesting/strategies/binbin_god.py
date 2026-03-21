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

binbingod策略优化 - CC阶段可同时开SP:
  - SP和CC不是对立的，可以同时操作
  - 在CC阶段，条件允许（有足够margin）的情况下也可以开SP
  - 配置参数:
    - allow_sp_in_cc_phase: 是否允许CC阶段开SP（默认True）
    - sp_in_cc_margin_threshold: margin使用上限阈值（默认0.5）
    - sp_in_cc_max_positions: CC阶段最多开几个SP仓位（默认3）
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from core.backtesting.strategies.base import BaseStrategy, Signal
from core.backtesting.pricing import OptionsPricer
from core.ml.dte_optimizer import DTEOptimizerML, DTEOptimizationConfig, DTEOptimizationResult
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
    """Tracks stock position from put assignment.
    
    Supports multi-stock holdings: each stock is tracked separately.
    """
    shares: int = 0  # Total shares (for backward compatibility)
    cost_basis: float = 0.0  # average cost per share (for backward compatibility)
    total_premium_collected: float = 0.0  # cumulative premium from both phases
    symbol: str = ""  # Primary stock symbol (for backward compatibility)
    
    # Multi-stock support: {symbol: {"shares": int, "cost_basis": float}}
    holdings: dict = None
    
    def __post_init__(self):
        if self.holdings is None:
            self.holdings = {}
    
    def add_shares(self, symbol: str, shares: int, cost_basis: float):
        """Add shares of a stock to holdings."""
        if symbol not in self.holdings:
            self.holdings[symbol] = {"shares": 0, "cost_basis": 0.0}
        
        # Calculate weighted average cost basis
        existing = self.holdings[symbol]
        total_shares = existing["shares"] + shares
        if total_shares > 0:
            total_cost = existing["shares"] * existing["cost_basis"] + shares * cost_basis
            existing["cost_basis"] = total_cost / total_shares
        existing["shares"] = total_shares
        
        # Update legacy fields for backward compatibility
        self._update_legacy_fields()
    
    def remove_shares(self, symbol: str, shares: int) -> int:
        """Remove shares of a stock. Returns actual shares removed."""
        if symbol not in self.holdings:
            return 0
        
        existing = self.holdings[symbol]
        removed = min(shares, existing["shares"])
        existing["shares"] -= removed
        
        # Clean up if no shares left
        if existing["shares"] <= 0:
            del self.holdings[symbol]
        
        # Update legacy fields
        self._update_legacy_fields()
        return removed
    
    def get_shares(self, symbol: str) -> int:
        """Get shares for a specific symbol."""
        if symbol in self.holdings:
            return self.holdings[symbol]["shares"]
        return 0
    
    def get_symbols(self) -> list:
        """Get list of symbols with holdings."""
        return list(self.holdings.keys())
    
    def _update_legacy_fields(self):
        """Update legacy fields for backward compatibility."""
        total_shares = sum(h["shares"] for h in self.holdings.values())
        self.shares = total_shares
        
        # Set primary symbol (largest holding)
        if self.holdings:
            primary = max(self.holdings.items(), key=lambda x: x[1]["shares"])
            self.symbol = primary[0]
            self.cost_basis = primary[1]["cost_basis"]
        else:
            self.symbol = ""
            self.cost_basis = 0.0


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

        # SP in CC phase parameters (binbingod策略优化：CC阶段可同时开SP)
        self.allow_sp_in_cc_phase = config.get("allow_sp_in_cc_phase", True)  # 允许CC阶段开SP
        self.sp_in_cc_margin_threshold = config.get("sp_in_cc_margin_threshold", 0.5)  # CC阶段开SP的margin使用上限阈值
        self.sp_in_cc_max_positions = config.get("sp_in_cc_max_positions", 3)  # CC阶段最多开几个SP仓位
        
        # ML delta optimization parameters - use BaseStrategy's implementation
        self.ml_delta_optimization = config.get("ml_delta_optimization", False)
        self.ml_dte_optimization = config.get("ml_dte_optimization", False)  # Add DTE optimization flag
        self.ml_adoption_rate = config.get("ml_adoption_rate", 0.5)
        self.ml_integration = None
        self.ml_dte_optimizer = None  # Add DTE optimizer

        # ML roll optimization parameters (replaces traditional stop loss / profit target)
        self.ml_roll_optimization = config.get("ml_roll_optimization", False)
        self.ml_roll_optimizer = None
        self.ml_roll_confidence_threshold = config.get("ml_roll_confidence_threshold", 0.6)

        # ML Position Optimization
        self.ml_position_optimization = config.get("ml_position_optimization", False)
        self.ml_position_optimizer = None
        self.position_recommendations = []  # Track ML recommendations for analysis

        self.logger = logging.getLogger("binbin_god")
        
        # Initialize ML integration if enabled (use BaseStrategy's pretrain_ml_model)
        if self.ml_delta_optimization or self.ml_dte_optimization:
            try:
                from core.ml.delta_strategy_integration import BinGodDeltaIntegration, AdaptiveDeltaStrategy
                self.ml_integration = BinGodDeltaIntegration(
                    ml_optimization_enabled=self.ml_delta_optimization,
                    ml_dte_optimization_enabled=self.ml_dte_optimization,
                    fallback_delta=self.call_delta,
                    config=config.get("ml_config")
                )
                self.adaptive_strategy = AdaptiveDeltaStrategy(
                    ml_integration=self.ml_integration,
                    adoption_rate=self.ml_adoption_rate
                )
                self.logger.info("ML optimizers initialized for BinbinGod (Delta: {}, DTE: {})".format(
                    self.ml_delta_optimization, self.ml_dte_optimization))
            except ImportError as e:
                self.logger.warning(f"ML optimization not available: {e}")
                self.ml_delta_optimization = False
                self.ml_dte_optimization = False
            except Exception as e:
                self.logger.warning(f"ML optimization initialization failed: {e}")
                self.ml_delta_optimization = False
                self.ml_dte_optimization = False

        # Initialize ML Roll Optimizer if enabled
        if self.ml_roll_optimization:
            try:
                from core.ml.roll_optimizer import MLRollOptimizer
                self.ml_roll_optimizer = MLRollOptimizer(
                    model_path=config.get("ml_roll_model_path")
                )
                self.logger.info("ML Roll Optimizer initialized for intelligent roll management")
            except ImportError as e:
                self.logger.warning(f"ML Roll optimization not available: {e}")
                self.ml_roll_optimization = False
            except Exception as e:
                self.logger.warning(f"ML Roll optimization initialization failed: {e}")
                self.ml_roll_optimization = False

        # Initialize ML Position Optimizer if enabled
        if self.ml_position_optimization:
            try:
                from core.ml.position_optimizer import MLPositionOptimizer, WheelPositionIntegration
                self.ml_position_optimizer = WheelPositionIntegration(
                    optimizer=MLPositionOptimizer(model_path=config.get("ml_position_model_path")),
                    enabled=True,
                    fallback_to_rules=True,
                )
                self.logger.info("ML Position Optimizer initialized for BinbinGod strategy")
            except ImportError as e:
                self.logger.warning(f"ML Position optimization not available: {e}")
                self.ml_position_optimization = False
            except Exception as e:
                self.logger.warning(f"ML Position optimization initialization failed: {e}")
                self.ml_position_optimization = False

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

    def _calculate_ml_position_size(
        self,
        symbol: str,
        underlying_price: float,
        iv: float,
        strike: float,
        premium: float,
        dte: int,
        delta: float,
        position_mgr,
        strategy_phase: str,
        shares_available: int = None,
    ) -> int:
        """Calculate position size using ML optimizer for BinbinGod strategy.

        Args:
            symbol: Stock symbol
            underlying_price: Current stock price
            iv: Implied volatility
            strike: Option strike
            premium: Option premium
            dte: Days to expiry
            delta: Option delta
            position_mgr: Position manager
            strategy_phase: "SP" or "CC"
            shares_available: For CC phase, max shares that can be covered

        Returns:
            Recommended number of contracts
        """
        # Base position calculation differs by phase
        if strategy_phase == "CC":
            # For CC: base position limited by shares available
            if shares_available and shares_available > 0:
                base_position = min(shares_available // 100, self.max_positions)
            else:
                shares = self.stock_holding.get_shares(symbol)
                base_position = min(shares // 100, self.max_positions) if shares > 0 else 0
        else:
            # For SP: use margin-based calculation
            if position_mgr:
                base_position = position_mgr.calculate_position_size(
                    margin_per_contract=strike * 100,
                    max_positions=self.max_positions,
                )
            else:
                base_position = self.max_positions

        if base_position <= 0:
            return 0

        # If ML optimization disabled, return base position
        if not self.ml_position_optimization or self.ml_position_optimizer is None:
            return base_position

        try:
            # Build market data for ML
            market_data = {
                'price': underlying_price,
                'iv': iv,
                'iv_rank': getattr(self, '_current_iv_rank', 50),
                'iv_percentile': getattr(self, '_current_iv_percentile', 50),
                'historical_volatility': iv * 100,
                'vix': getattr(self, '_current_vix', 20),
                'momentum': getattr(self, '_current_momentum', {}),
            }

            # Build portfolio state
            portfolio_state = {
                'total_capital': self.initial_capital,
                'available_margin': position_mgr.available_margin if position_mgr else self.initial_capital,
                'margin_used': position_mgr.total_margin_used if position_mgr else 0,
                'drawdown': getattr(self, '_current_drawdown', 0),
                'positions': [],
                'cost_basis': self.stock_holding.cost_basis,
            }

            # Build option info
            option_info = {
                'underlying_price': underlying_price,
                'strike': strike,
                'premium': premium,
                'dte': dte,
                'delta': delta,
            }

            # Get ML recommendation
            max_position = self.max_positions
            if strategy_phase == "CC":
                # For CC only, limit by shares held
                shares = self.stock_holding.get_shares(symbol)
                max_position = min(max_position, shares // 100) if shares > 0 else 0
            # For SP or CC+SP mode, max_position is already self.max_positions
            # CC+SP模式开的是SP，不需要shares限制

            num_contracts, recommendation = self.ml_position_optimizer.get_position_size(
                symbol=symbol,
                market_data=market_data,
                portfolio_state=portfolio_state,
                strategy_phase=strategy_phase,
                option_info=option_info,
                base_position=base_position,
                max_position=max_position,
            )

            # Log recommendation for analysis
            if recommendation:
                self.position_recommendations.append({
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'symbol': symbol,
                    'phase': strategy_phase,
                    'contracts': num_contracts,
                    'multiplier': recommendation.position_multiplier,
                    'confidence': recommendation.confidence,
                    'reasoning': recommendation.reasoning,
                })

                self.logger.info(
                    f"ML Position for {symbol}: {num_contracts} contracts ({recommendation.position_multiplier:.2f}x, "
                    f"confidence: {recommendation.confidence:.0%}) - {recommendation.reasoning}"
                )

            return num_contracts

        except Exception as e:
            self.logger.warning(f"ML position optimization failed, using base: {e}")
            return base_position

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
        
        # CRITICAL DEBUG: Always log phase and holdings state
        logger.info(f"generate_signals START: date={current_date}, phase={self.phase}, "
                   f"total_shares={self.stock_holding.shares}, "
                   f"held_symbols={self.stock_holding.get_symbols()}, "
                   f"holdings={self.stock_holding.holdings}")
        
        # Check if we already have max positions
        wheel_positions = [
            p for p in open_positions 
            if p.trade_type in ("BINBIN_PUT", "BINBIN_CALL")
        ]
        
        # Debug log for CC phase
        if self.phase == "CC":
            logger.info(f"generate_signals CC: date={current_date}, wheel_positions={len(wheel_positions)}, "
                       f"max_positions={self.max_positions}, phase={self.phase}, "
                       f"held_symbols={self.stock_holding.get_symbols()}")
        
        if len(wheel_positions) >= self.max_positions:
            logger.info(f"Max positions reached: {len(wheel_positions)} >= {self.max_positions}")
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
                        # Filter bars up to current_date (ensure date is string for comparison)
                        filtered_bars = []
                        for bar in bars:
                            bar_date_str = str(bar["date"])[:10] if bar["date"] else ""
                            if bar_date_str <= current_date:
                                filtered_bars.append(bar)
                        if filtered_bars:
                            # Pass full bars list for scoring (backtest mode)
                            market_data[sym] = filtered_bars
                
                # Select best stock based on current metrics
                if market_data:
                    actual_symbol = self._select_best_stock(market_data, current_date)
                    logger.info(f"SP phase: Selected {actual_symbol} for new put position")
                    
                    # CRITICAL FIX: Get the actual underlying price AND IV for the selected stock
                    # This ensures correct option pricing and strike selection
                    actual_underlying_price = underlying_price  # Default to passed price
                    actual_iv = iv  # Default to passed IV
                    
                    # Get stock_hv dictionary (set by engine.py)
                    stock_hv = getattr(self, 'stock_hv', {})
                    
                    if actual_symbol in pool_data:
                        bars = pool_data[actual_symbol]
                        bar_index = -1
                        for idx, bar in enumerate(bars):
                            bar_date_str = str(bar["date"])[:10] if bar["date"] else ""
                            if bar_date_str <= current_date:
                                actual_underlying_price = bar["close"]
                                bar_index = idx
                        
                        # Get the correct IV for this stock at current date
                        if actual_symbol in stock_hv and bar_index >= 0:
                            sym_hv = stock_hv[actual_symbol]
                            if bar_index < len(sym_hv) and sym_hv[bar_index] > 0.01:
                                actual_iv = sym_hv[bar_index]
                                logger.info(f"Using {actual_symbol} price: ${actual_underlying_price:.2f}, IV: {actual_iv:.3f}")
                            else:
                                logger.warning(f"Invalid IV for {actual_symbol} at index {bar_index}, using fallback")
                        else:
                            logger.warning(f"No IV data for {actual_symbol}, using fallback IV: {actual_iv:.3f}")
                    else:
                        logger.warning(f"No price data for {actual_symbol}, using fallback price")
                else:
                    # Fallback: use first stock in pool
                    actual_symbol = stock_pool[0] if stock_pool else "MSFT"
                    actual_underlying_price = underlying_price
                    actual_iv = iv
                    logger.warning(f"No market data available, using fallback: {actual_symbol}")
            else:
                actual_symbol = self.symbol
                actual_underlying_price = underlying_price
                actual_iv = iv
            
            return self._generate_backtest_put_signal(
                actual_symbol, current_date, actual_underlying_price, actual_iv, position_mgr
            )
        else:  # CC phase
            # In CC phase: generate Call signals for EACH stock we hold shares of
            # Support multi-stock holdings
            
            held_symbols = self.stock_holding.get_symbols()
            logger.info(f"CC phase check: held_symbols={held_symbols}, holdings={self.stock_holding.holdings}")
            
            if not held_symbols:
                logger.warning("CC phase but no stocks held, returning to SP phase")
                self.phase = "SP"
                return []
            
            logger.info(f"CC phase: Holding shares for {held_symbols}")
            
            # Get stock_hv dictionary (set by engine.py)
            stock_hv = getattr(self, 'stock_hv', {})
            pool_data = getattr(self, 'mag7_data', {})
            
            # DEBUG: Log pool_data availability
            logger.info(f"CC phase: pool_data keys={list(pool_data.keys()) if pool_data else 'None'}")
            logger.info(f"CC phase: stock_hv keys={list(stock_hv.keys()) if stock_hv else 'None'}")
            
            all_signals = []
            
            # Calculate existing Call coverage per stock
            existing_call_coverage = {}  # {symbol: contracts}
            for p in wheel_positions:
                if p.trade_type == "BINBIN_CALL":
                    sym = p.symbol
                    existing_call_coverage[sym] = existing_call_coverage.get(sym, 0) + abs(p.quantity)
            
            # DEBUG: Log existing call coverage
            logger.info(f"CC phase: existing_call_coverage={existing_call_coverage}")
            
            # Generate Call signals for each held stock
            for stock_symbol in held_symbols:
                shares_held = self.stock_holding.get_shares(stock_symbol)
                stock_cost_basis = self.stock_holding.holdings.get(stock_symbol, {}).get("cost_basis", 0)
                logger.info(f"CC phase: {stock_symbol} shares_held={shares_held}, cost_basis={stock_cost_basis}")
                if shares_held <= 0:
                    continue
                
                # Check how many shares are already covered by existing Calls
                covered_contracts = existing_call_coverage.get(stock_symbol, 0)
                shares_covered = covered_contracts * 100
                shares_available = shares_held - shares_covered
                
                if shares_available <= 0:
                    logger.info(f"{stock_symbol}: All {shares_held} shares already covered by {covered_contracts} Call contracts")
                    continue
                
                # Get the correct price and IV for this stock
                actual_underlying_price = underlying_price
                actual_iv = iv
                
                # CRITICAL DEBUG: Log price retrieval process
                logger.info(f"CC phase: {stock_symbol} - pool_data has data: {stock_symbol in pool_data}")
                
                if stock_symbol in pool_data:
                    bars = pool_data[stock_symbol]
                    bar_index = -1
                    logger.info(f"CC phase: {stock_symbol} - total bars: {len(bars)}")
                    
                    for idx, bar in enumerate(bars):
                        bar_date_str = str(bar["date"])[:10] if bar["date"] else ""
                        if bar_date_str <= current_date:
                            actual_underlying_price = bar["close"]
                            bar_index = idx
                    
                    logger.info(f"CC phase: {stock_symbol} - bar_index: {bar_index}, price: ${actual_underlying_price:.2f}")
                    
                    # Get the correct IV for this stock at current date
                    if stock_symbol in stock_hv and bar_index >= 0:
                        sym_hv = stock_hv[stock_symbol]
                        if bar_index < len(sym_hv) and sym_hv[bar_index] > 0.01:
                            actual_iv = sym_hv[bar_index]
                            logger.info(f"CC phase: {stock_symbol} - IV: {actual_iv:.3f} (from stock_hv)")
                        else:
                            logger.warning(f"CC phase: {stock_symbol} - IV not available (bar_index={bar_index}, hv_len={len(sym_hv) if sym_hv else 0})")
                    else:
                        logger.warning(f"CC phase: {stock_symbol} - stock_hv not available or bar_index < 0")
                else:
                    logger.warning(f"CC phase: {stock_symbol} NOT in pool_data! Using fallback price ${actual_underlying_price:.2f}")
                
                # Generate Call signal for this stock with its specific cost basis
                signals = self._generate_backtest_call_signal(
                    stock_symbol, current_date, actual_underlying_price, actual_iv,
                    position_mgr, shares_available, stock_cost_basis
                )
                if signals:
                    all_signals.extend(signals)
                    logger.info(f"CC phase: Generated {len(signals)} Call signal(s) for {stock_symbol}")

            # ========== binbingod策略优化: CC阶段条件允许时开SP ==========
            # SP和CC不是对立的，可以同时操作
            # 条件: 1) allow_sp_in_cc_phase开启
            #       2) margin使用率低于阈值
            #       3) 当前SP positions数量未达到上限
            if self.allow_sp_in_cc_phase:
                sp_signals = self._generate_sp_in_cc_phase(
                    current_date, underlying_price, iv, wheel_positions, position_mgr,
                    held_symbols, pool_data, stock_hv
                )
                if sp_signals:
                    all_signals.extend(sp_signals)
                    logger.info(f"CC phase: Generated {len(sp_signals)} SP signal(s) (simultaneous CC+SP mode)")

            return all_signals

    def _generate_sp_in_cc_phase(
        self,
        current_date: str,
        underlying_price: float,
        iv: float,
        wheel_positions: list,
        position_mgr,
        held_symbols: list,
        pool_data: dict,
        stock_hv: dict,
    ) -> list[Signal]:
        """在CC阶段生成SP信号（binbingod策略优化）。

        SP和CC不是对立的，可以同时操作。条件允许时，在持有股票卖CC的同时，
        也可以开新的SP仓位，增加资金利用效率和收益。

        Args:
            current_date: 当前日期
            underlying_price: 基础价格（fallback）
            iv: 隐含波动率（fallback）
            wheel_positions: 当前持有的Wheel策略仓位
            position_mgr: 仓位管理器
            held_symbols: 当前持有的股票列表
            pool_data: 股票池数据
            stock_hv: 股票历史波动率字典

        Returns:
            SP信号列表（可能为空）
        """
        from datetime import timedelta
        from core.backtesting.pricing import OptionsPricer

        # 条件1: 检查margin使用率是否低于阈值
        if position_mgr:
            margin_utilization = position_mgr.total_margin_used / position_mgr.net_capital if position_mgr.net_capital > 0 else 0
            if margin_utilization > self.sp_in_cc_margin_threshold:
                logger.info(
                    f"SP in CC phase: margin utilization {margin_utilization:.1%} > threshold {self.sp_in_cc_margin_threshold:.1%}, "
                    f"skipping SP"
                )
                return []
        else:
            # 没有position_mgr时，保守起见不允许开SP
            logger.debug("SP in CC phase: no position_mgr, skipping SP")
            return []

        # 条件2: 检查当前SP positions数量
        current_sp_positions = sum(
            1 for p in wheel_positions
            if p.trade_type == "BINBIN_PUT"
        )

        if current_sp_positions >= self.sp_in_cc_max_positions:
            logger.info(
                f"SP in CC phase: already have {current_sp_positions} SP positions, "
                f"max allowed is {self.sp_in_cc_max_positions}"
            )
            return []

        # 条件3: 检查总仓位数量
        total_wheel_positions = len(wheel_positions)
        if total_wheel_positions >= self.max_positions:
            logger.info(f"SP in CC phase: max positions {self.max_positions} reached")
            return []

        # 选择最佳股票开SP
        # 优先选择当前未持有的股票（分散风险）
        stock_pool = getattr(self, 'stock_pool', MAG7_STOCKS)
        available_stocks = [s for s in stock_pool if s not in held_symbols]

        # 如果所有股票都已持有，则选择持有的股票中得分最高的
        if not available_stocks:
            available_stocks = stock_pool

        # 构建market_data用于评分
        market_data = {}
        for sym in available_stocks:
            bars = pool_data.get(sym, [])
            if bars:
                filtered_bars = [bar for bar in bars if str(bar.get("date", ""))[:10] <= current_date]
                if filtered_bars:
                    market_data[sym] = filtered_bars

        if not market_data:
            logger.warning("SP in CC phase: no market data available for stock selection")
            return []

        # 选择最佳股票
        best_symbol = self._select_best_stock(market_data, current_date)

        # 获取选中股票的价格和IV
        actual_underlying_price = underlying_price
        actual_iv = iv

        if best_symbol in pool_data:
            bars = pool_data[best_symbol]
            bar_index = -1
            for idx, bar in enumerate(bars):
                bar_date_str = str(bar.get("date", ""))[:10] if bar.get("date") else ""
                if bar_date_str <= current_date:
                    actual_underlying_price = bar["close"]
                    bar_index = idx

            if best_symbol in stock_hv and bar_index >= 0:
                sym_hv = stock_hv[best_symbol]
                if bar_index < len(sym_hv) and sym_hv[bar_index] > 0.01:
                    actual_iv = sym_hv[bar_index]

        # 生成SP信号
        logger.info(
            f"SP in CC phase: Selected {best_symbol} for new SP position "
            f"(price: ${actual_underlying_price:.2f}, IV: {actual_iv:.3f})"
        )

        # 使用_generate_backtest_put_signal生成信号，传入"CC+SP"模式
        sp_signals = self._generate_backtest_put_signal(
            best_symbol, current_date, actual_underlying_price, actual_iv, position_mgr,
            strategy_phase="CC+SP"  # 标记为同时操作模式
        )

        if sp_signals:
            logger.info(
                f"SP in CC phase: Successfully generated {len(sp_signals)} SP signal(s) for {best_symbol} "
                f"(simultaneous with CC positions)"
            )

        return sp_signals

    def _generate_backtest_put_signal(
        self,
        symbol: str,
        current_date: str,
        underlying_price: float,
        iv: float,
        position_mgr=None,
        strategy_phase: str = "SP",  # 支持SP或CC+SP模式
    ) -> list[Signal]:
        """Generate Sell Put signal for backtesting.
        
        Args:
            symbol: Stock symbol
            current_date: Current date string
            underlying_price: Current stock price
            iv: Implied volatility
            position_mgr: Position manager
            strategy_phase: "SP" for pure SP mode, "CC+SP" for simultaneous mode
        """
        from datetime import timedelta
        from core.backtesting.pricing import OptionsPricer
        
        # Default DTE values
        dte_days = int(self.dte_max)
        T = dte_days / 365.0
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=dte_days)
        expiry_str = expiry_date.strftime("%Y%m%d")
        
        # Get historical bars for ML optimization (backtest mode)
        pool_data = getattr(self, 'mag7_data', {})
        ml_bars = pool_data.get(symbol, [])
        # Filter bars up to current_date
        if ml_bars:
            ml_bars = [bar for bar in ml_bars if str(bar.get("date", ""))[:10] <= current_date]
        
        # ML Delta optimization
        ml_result = None
        if self.ml_delta_optimization and self.ml_integration:
            try:
                ml_result = self.ml_integration.optimize_put_delta(
                    symbol=symbol,
                    current_price=underlying_price,
                    cost_basis=self.stock_holding.cost_basis,
                    bars=ml_bars,  # Pass actual bars for backtest
                    options_data=[],  # Will be populated by real-time interface
                    iv=iv
                )
                logger.info(f"ML optimized delta: {ml_result.optimal_delta:.3f} (confidence: {ml_result.confidence:.2f})")
            except Exception as e:
                logger.warning(f"ML optimization failed: {e}")
                ml_result = None
        
        # ML DTE optimization for Sell Puts
        ml_dte_result = None
        if self.ml_dte_optimization and self.ml_integration:
            try:
                ml_dte_result = self.ml_integration.optimize_put_dte(
                    symbol=symbol,
                    current_price=underlying_price,
                    cost_basis=self.stock_holding.cost_basis,
                    bars=ml_bars,  # Pass actual bars for backtest
                    options_data=[],  # Will be populated by real-time interface
                    iv=iv,
                    strategy_phase="SP"
                )
                logger.info(f"ML optimized DTE for SP: {ml_dte_result.optimal_dte_min}-{ml_dte_result.optimal_dte_max} days "
                           f"(confidence: {ml_dte_result.confidence:.2f})")
                
                # Update DTE values based on ML recommendation
                if ml_dte_result.confidence > 0.6:  # Only apply if confidence is reasonable
                    dte_days = int((ml_dte_result.optimal_dte_min + ml_dte_result.optimal_dte_max) / 2)
                    T = dte_days / 365.0
                    expiry_date = entry + timedelta(days=dte_days)
                    expiry_str = expiry_date.strftime("%Y%m%d")
                    logger.info(f"Updated DTE to {dte_days} days based on ML recommendation")
            except Exception as e:
                logger.warning(f"ML DTE optimization failed: {e}")
                ml_dte_result = None
        
        # Use put-specific delta
        original_delta = self.delta_target
        self.delta_target = self.put_delta
        
        # Apply ML delta if available and confident
        effective_delta = self.put_delta
        if ml_result and ml_result.confidence >= 0.8:
            effective_delta = ml_result.optimal_delta
            logger.info(f"Using ML optimized put delta: {effective_delta:.3f}")
        elif ml_result:
            logger.info(f"ML delta confidence {ml_result.confidence:.2f} < 0.8, using fallback delta {self.put_delta}")
        
        strike = self.select_strike_with_delta(underlying_price, iv, T, "P", effective_delta)
        self.delta_target = original_delta

        premium = OptionsPricer.put_price(underlying_price, strike, T, iv)
        delta = OptionsPricer.delta(underlying_price, strike, T, iv, "P")

        # SAFETY CHECK: Skip if premium is too low (indicates strike selection error)
        if premium < 0.01:
            logger.warning(
                f"Skipping put signal for {symbol}: premium too low ({premium:.4f}), "
                f"strike={strike:.2f}, underlying={underlying_price:.2f}"
            )
            return []

        # Position sizing: use ML optimizer if enabled, otherwise traditional
        if self.ml_position_optimization and self.ml_position_optimizer:
            max_contracts = self._calculate_ml_position_size(
                symbol=symbol,
                underlying_price=underlying_price,
                iv=iv,
                strike=strike,
                premium=premium,
                dte=dte_days,
                delta=delta,
                position_mgr=position_mgr,
                strategy_phase=strategy_phase,  # 使用传入的strategy_phase (SP或CC+SP)
            )
        else:
            # Traditional position sizing using position manager
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
            strategy_phase=strategy_phase,  # SP or CC+SP
        )]
    
    def _generate_backtest_call_signal(
        self,
        symbol: str,
        current_date: str,
        underlying_price: float,
        iv: float,
        position_mgr=None,
        shares_available: int = None,
        cost_basis: float = None,
    ) -> list[Signal]:
        """Generate Covered Call signal for backtesting.
        
        Args:
            symbol: Stock symbol
            current_date: Current date string
            underlying_price: Current stock price
            iv: Implied volatility
            position_mgr: Position manager
            shares_available: Maximum shares that can be covered by new calls.
            cost_basis: Cost basis for this specific stock. If None, get from holdings.
        """
        from datetime import timedelta
        from core.backtesting.pricing import OptionsPricer
        
        logger.info(f"_generate_backtest_call_signal: symbol={symbol}, date={current_date}, "
                   f"price={underlying_price:.2f}, iv={iv:.3f}, shares_avail={shares_available}, cost={cost_basis}")
        
        # Get cost basis for this specific stock
        if cost_basis is None:
            cost_basis = self.stock_holding.holdings.get(symbol, {}).get("cost_basis", 0)
        
        # Determine shares available for covering new calls
        if shares_available is None:
            shares_available = self.stock_holding.get_shares(symbol)
        
        # Can only sell calls for shares we own
        if shares_available <= 0:
            # Fallback to SP phase
            self.phase = "SP"
            return self._generate_backtest_put_signal(
                symbol, current_date, underlying_price, iv, position_mgr
            )
        
        # Default DTE values
        dte_days = int(self.dte_max)
        T = dte_days / 365.0
        entry = datetime.strptime(current_date, "%Y-%m-%d")
        expiry_date = entry + timedelta(days=dte_days)
        expiry_str = expiry_date.strftime("%Y%m%d")
        
        # Get historical bars for ML optimization (backtest mode)
        pool_data = getattr(self, 'mag7_data', {})
        ml_bars = pool_data.get(symbol, [])
        # Filter bars up to current_date
        if ml_bars:
            ml_bars = [bar for bar in ml_bars if str(bar.get("date", ""))[:10] <= current_date]
        
        # Check if we need CC optimization
        call_delta_target = self.call_delta
        additional_constraints = {}
        
        # ML DTE optimization for Covered Calls
        ml_dte_result = None
        if self.ml_dte_optimization and self.ml_integration:
            try:
                ml_dte_result = self.ml_integration.optimize_call_dte(
                    symbol=symbol,
                    current_price=underlying_price,
                    cost_basis=cost_basis,
                    bars=ml_bars,  # Pass actual bars for backtest
                    options_data=[],  # Will be populated by real-time interface
                    iv=iv,
                    strategy_phase="CC"
                )
                logger.info(f"ML optimized DTE for CC: {ml_dte_result.optimal_dte_min}-{ml_dte_result.optimal_dte_max} days "
                           f"(confidence: {ml_dte_result.confidence:.2f})")
                
                # Update DTE values based on ML recommendation
                if ml_dte_result.confidence > 0.6:  # Only apply if confidence is reasonable
                    dte_days = int((ml_dte_result.optimal_dte_min + ml_dte_result.optimal_dte_max) / 2)
                    T = dte_days / 365.0
                    expiry_date = entry + timedelta(days=dte_days)
                    expiry_str = expiry_date.strftime("%Y%m%d")
                    logger.info(f"Updated DTE to {dte_days} days based on ML recommendation")
            except Exception as e:
                logger.warning(f"ML DTE optimization failed: {e}")
                ml_dte_result = None
        
        # ML Delta optimization
        ml_result = None
        if self.ml_delta_optimization and self.ml_integration:
            try:
                ml_result = self.ml_integration.optimize_call_delta(
                    symbol=symbol,
                    current_price=underlying_price,
                    cost_basis=cost_basis,
                    bars=ml_bars,  # Pass actual bars for backtest
                    options_data=[],  # Will be populated by real-time interface
                    iv=iv
                )
                logger.info(f"ML optimized delta: {ml_result.optimal_delta:.3f} (confidence: {ml_result.confidence:.2f})")
            except Exception as e:
                logger.warning(f"ML optimization failed: {e}")
                ml_result = None
        
        if self.cc_optimization_enabled and cost_basis > 0:
            # Check if current price is below cost basis (loss position)
            price_cost_ratio = underlying_price / cost_basis
            
            if price_cost_ratio < (1 - self.cc_cost_basis_threshold):
                # We're in a loss position, optimize for higher strike price
                logger.info(
                    f"CC optimization: Stock price ${underlying_price:.2f} below cost basis "
                    f"${cost_basis:.2f} (ratio: {price_cost_ratio:.2f}). "
                )
                
                # Combine CC optimization with ML if available
                if ml_result and ml_result.confidence >= 0.8:
                    # Use ML result with CC protective adjustments
                    call_delta_target = max(self.cc_min_delta_cost, ml_result.optimal_delta)
                    logger.info(f"Combined ML + CC optimization: delta = {call_delta_target:.3f}")
                else:
                    # Traditional CC optimization
                    call_delta_target = self.cc_min_delta_cost
                    if ml_result:
                        logger.info(f"ML delta confidence {ml_result.confidence:.2f} < 0.8, using CC fallback delta")
                    
                # Add minimum strike constraint: try to get strike close to cost basis
                min_strike_desired = cost_basis * (1 - self.cc_min_strike_premium)
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

        # SAFETY CHECK: Skip if premium is too low (indicates strike selection error)
        if premium < 0.01:
            logger.warning(
                f"Skipping call signal for {symbol}: premium too low ({premium:.4f}), "
                f"strike={strike:.2f}, underlying={underlying_price:.2f}"
            )
            return []

        # Position sizing: use ML optimizer if enabled, otherwise traditional
        if self.ml_position_optimization and self.ml_position_optimizer:
            max_contracts = self._calculate_ml_position_size(
                symbol=symbol,
                underlying_price=underlying_price,
                iv=iv,
                strike=strike,
                premium=premium,
                dte=dte_days,
                delta=delta,
                position_mgr=position_mgr,
                strategy_phase="CC",
                shares_available=shares_available,
            )
            # For CC, ensure we don't exceed shares held
            max_by_shares = shares_available // 100
            max_contracts = min(max_contracts, max_by_shares)
        else:
            # Traditional: Covered call: Calculate contracts based on shares available
            # generate_signals() already checks for existing call positions to prevent over-selling
            max_by_shares = shares_available // 100
            max_contracts = min(max_by_shares, self.max_positions)

        if max_contracts <= 0:
            logger.warning(f"CC signal: {symbol} max_contracts={max_contracts} (shares_available={shares_available}, max_positions={self.max_positions})")
            return []
        
        quantity = -max_contracts  # Sell
        
        logger.info(f"CC signal: {symbol} - Generated {max_contracts} contracts, strike=${strike:.2f}, premium=${premium:.2f}, delta={delta:.3f}")
        
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
            strategy_phase="CC",  # Covered Call phase
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
        """Generate trading signal for Binbin God strategy (real-time interface).
        
        Note: Real-time interface currently supports single-stock focus per call.
        For multi-stock holdings in CC phase, generates signals for the first held stock.
        """
        
        # Check phase and generate appropriate signal
        if self.phase == "SP":
            # In SP phase: select the best stock for new puts
            if self.symbol == "MAG7_AUTO":
                actual_symbol = self._select_best_stock(market_data)
            else:
                actual_symbol = self.symbol
            
            return self._generate_put_signal(
                actual_symbol, current_dt, bars, contracts, portfolio, market_data
            )
        else:  # CC phase
            # In CC phase: generate signals for held stocks
            held_symbols = self.stock_holding.get_symbols()
            if not held_symbols:
                logger.warning("CC phase but no stocks held, switching to SP phase")
                self.phase = "SP"
                return None
            
            # For real-time interface, focus on the first held stock
            # (In practice, this method would be called multiple times for different stocks)
            actual_symbol = held_symbols[0]
            
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
        """Generate Covered Call signal (Phase 2).
        
        Args:
            symbol: The stock symbol to generate call signal for
        """
        
        # Get shares and cost basis for this specific stock
        shares_held = self.stock_holding.get_shares(symbol)
        stock_cost_basis = self.stock_holding.holdings.get(symbol, {}).get("cost_basis", 0)
        
        # Check if we have shares to sell calls against
        if shares_held <= 0:
            logger.warning(f"In CC phase but no {symbol} shares held, switching back to SP")
            self.phase = "SP"
            return None
        
        # CRITICAL: Check existing Call positions for THIS stock to prevent over-selling
        existing_call_contracts = 0
        for pos in portfolio.get("positions", []):
            if pos.get("symbol") == symbol and (
                pos.get("trade_type") == "BINBIN_CALL" or 
                (pos.get("right") == "C" and pos.get("quantity", 0) < 0)
            ):
                existing_call_contracts += abs(pos.get("quantity", 0))
        
        shares_already_covered = existing_call_contracts * 100
        shares_available = shares_held - shares_already_covered
        
        if shares_available <= 0:
            logger.info(
                f"All {symbol} shares already covered by existing calls. "
                f"Shares: {shares_held}, Covered: {shares_already_covered}"
            )
            return None
        
        # Calculate how many call contracts we can sell (1 contract per 100 shares)
        max_contracts = shares_available // 100
        if max_contracts <= 0:
            return None
        
        # Filter for calls with target DTE and delta (with optimization if needed)
        suitable_contracts = []
        
        # Determine target delta range based on optimization
        if self.cc_optimization_enabled and stock_cost_basis > 0:
            # Check if current price is below cost basis (loss position)
            current_price = bars[-1]["close"] if bars else 0
            price_cost_ratio = current_price / stock_cost_basis
            
            if price_cost_ratio < (1 - self.cc_cost_basis_threshold):
                # Use reduced delta range for optimization
                min_delta = self.cc_min_delta_cost
                max_delta = 0.35  # Keep upper bound reasonable
                logger.info(
                    f"Real-time CC optimization: {symbol} price ${current_price:.2f} below cost basis "
                    f"${stock_cost_basis:.2f}, using delta range [{min_delta:.2f}, {max_delta:.2f}]"
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
        market_data: Dict[str, Any] = None,
    ) -> tuple[bool, str]:
        """Check if position should be closed (for Wheel strategy, this means roll).

        For Wheel strategy, we don't use traditional stop loss. Instead:
        - Roll Forward: Premium captured > 80%, roll to new position
        - Roll Out: Near assignment risk, extend expiry
        - Let Expire: Hold to expiry for max theta
        - Close Early: Only for margin management

        Args:
            position: Position dictionary
            current_price: Current option price
            entry_price: Entry option price
            current_dt: Current datetime
            market_data: Current market data (for ML features)

        Returns:
            (should_close, reason) tuple
        """

        # Calculate premium capture
        pnl = current_price - entry_price
        premium_captured_pct = abs(pnl) / abs(entry_price) * 100 if entry_price else 0

        # ML-based roll optimization
        if self.ml_roll_optimization and self.ml_roll_optimizer and market_data:
            try:
                current_date = current_dt.strftime('%Y-%m-%d') if isinstance(current_dt, datetime) else str(current_dt)[:10]

                # Add strategy phase to position for ML features
                position_with_phase = {**position, 'strategy_phase': self.phase}

                # Get roll recommendation
                should_roll, recommendation = self.ml_roll_optimizer.should_roll(
                    position=position_with_phase,
                    market_data=market_data,
                    current_date=current_date,
                    min_confidence=self.ml_roll_confidence_threshold
                )

                if should_roll:
                    self.logger.info(
                        f"ML Roll recommendation: {recommendation.action} "
                        f"(confidence: {recommendation.confidence:.0%}, "
                        f"expected improvement: ${recommendation.expected_pnl_improvement:.2f})"
                    )

                    # Store roll parameters for execution
                    position['_roll_recommendation'] = recommendation

                    # Map roll actions to close reasons
                    action_map = {
                        "ROLL_FORWARD": "ML_ROLL_FORWARD",
                        "ROLL_OUT": "ML_ROLL_OUT",
                        "CLOSE_EARLY": "ML_CLOSE_EARLY",
                    }
                    return True, action_map.get(recommendation.action, "ML_ROLL")

                self.logger.debug(
                    f"ML Roll: {recommendation.action} - {recommendation.reasoning}"
                )

            except Exception as e:
                self.logger.warning(f"ML roll optimization failed: {e}")
                # Fall through to rule-based logic

        # Rule-based roll logic (fallback when ML disabled or failed)
        # For Wheel strategy, we only close early in specific cases

        # Expiry check
        expiry = position.get("expiry")
        if expiry:
            if isinstance(expiry, str):
                try:
                    expiry_date = datetime.strptime(expiry, '%Y%m%d')
                except (ValueError, TypeError):
                    expiry_date = current_dt
            else:
                expiry_date = expiry

            dte = (expiry_date - current_dt).days if isinstance(current_dt, datetime) else 0

            # Roll Forward: High premium capture with time remaining
            if premium_captured_pct >= 80 and dte > 7:
                return True, "ROLL_FORWARD"

            # Near expiry - let it expire (Wheel strategy: time is our friend)
            if dte <= 0:
                return True, "EXPIRY"

        # Traditional profit target (user may have disabled it for pure Wheel)
        if not self._profit_target_disabled:
            profit_threshold = self.profit_target_pct / 100.0 * abs(entry_price)
            if abs(pnl) >= profit_threshold and pnl < 0:
                return True, "PROFIT_TARGET"

        # Traditional stop loss (only for extreme cases, user should consider disabling)
        if not self._stop_loss_disabled:
            loss_threshold = self.stop_loss_pct / 100.0 * abs(entry_price)
            if pnl >= loss_threshold:
                self.logger.warning(
                    f"Stop loss triggered for {position.get('symbol')}. "
                    f"Consider if this is appropriate for Wheel strategy."
                )
                return True, "STOP_LOSS"

        return False, ""

    def get_roll_parameters(self, position: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get roll parameters from ML recommendation.

        Args:
            position: Position with roll recommendation

        Returns:
            Dictionary with roll parameters or None
        """
        recommendation = position.get('_roll_recommendation')
        if not recommendation:
            return None

        return {
            'action': recommendation.action,
            'optimal_dte': recommendation.optimal_dte,
            'optimal_delta': recommendation.optimal_delta,
            'expected_improvement': recommendation.expected_pnl_improvement,
            'confidence': recommendation.confidence,
        }

    def generate_roll_signal(
        self,
        closed_trade: Dict[str, Any],
        current_date: str,
        underlying_price: float,
        iv: float,
    ) -> Optional[Signal]:
        """Generate a roll signal after a position is closed.

        Called by the backtest engine when a position is closed with PROFIT_TARGET
        (indicating premium captured, good candidate for roll forward).

        Args:
            closed_trade: The trade that was just closed
            current_date: Current date (YYYY-MM-DD)
            underlying_price: Current underlying price
            iv: Current implied volatility

        Returns:
            Signal for new position, or None if no roll
        """
        # Only roll on profit target (premium captured successfully)
        exit_reason = closed_trade.get('exit_reason', '')
        if exit_reason not in ('PROFIT_TARGET', 'ROLL_FORWARD', 'ROLL_OUT'):
            return None

        # Don't roll if assigned (we now hold stock or sold stock)
        if exit_reason == 'ASSIGNMENT':
            return None

        symbol = closed_trade.get('symbol', '')
        right = closed_trade.get('right', '')  # 'P' or 'C'
        quantity = closed_trade.get('quantity', 0)

        # Determine DTE for new position
        if self.ml_dte_optimization and self.ml_dte_optimizer:
            # Use ML-optimized DTE
            dte_result = self.ml_dte_optimizer.predict_optimal_dte(
                symbol=symbol,
                current_date=current_date,
                iv=iv,
                delta_target=self.call_delta if right == 'C' else self.put_delta,
            )
            target_dte = dte_result.predicted_dte
        else:
            # Use configured DTE range
            target_dte = self.dte_max  # Default to longer DTE for rolls

        # Calculate expiry date
        try:
            curr_date = datetime.strptime(current_date, '%Y-%m-%d')
        except (ValueError, TypeError):
            curr_date = datetime.now()

        expiry_date = curr_date + timedelta(days=target_dte)
        expiry_str = expiry_date.strftime('%Y%m%d')

        # Determine strike based on phase and right
        # Use ML Delta Optimizer if available for optimal delta selection
        if self.phase == "SP" and right == "P":
            # Continue selling puts
            if self.ml_delta_optimization and self.ml_integration:
                try:
                    # Get ML-optimized delta
                    ml_result = self.ml_integration.get_optimal_delta(
                        symbol=symbol,
                        current_date=current_date,
                        underlying_price=underlying_price,
                        iv=iv,
                        strategy_phase="SP"
                    )
                    target_delta = -abs(ml_result.optimal_delta)  # Put delta is negative
                except Exception as e:
                    self.logger.warning(f"ML delta optimization failed in roll: {e}")
                    target_delta = -abs(self.put_delta)
            else:
                target_delta = -abs(self.put_delta)
            strike = OptionsPricer.strike_from_delta(
                underlying_price, target_dte / 365.0, iv, target_delta, 'P'
            )
            trade_type = "BINBIN_PUT"

        elif self.phase == "CC" and right == 'C':
            # Continue selling calls
            if self.ml_delta_optimization and self.ml_integration:
                try:
                    # Get ML-optimized delta
                    ml_result = self.ml_integration.get_optimal_delta(
                        symbol=symbol,
                        current_date=current_date,
                        underlying_price=underlying_price,
                        iv=iv,
                        strategy_phase="CC"
                    )
                    target_delta = abs(ml_result.optimal_delta)  # Call delta is positive
                except Exception as e:
                    self.logger.warning(f"ML delta optimization failed in roll: {e}")
                    target_delta = self.call_delta
            else:
                target_delta = self.call_delta
            strike = OptionsPricer.strike_from_delta(
                underlying_price, target_dte / 365.0, iv, target_delta, 'C'
            )
            trade_type = "BINBIN_CALL"

        else:
            # Phase mismatch, don't roll
            return None

        # Calculate premium
        T = target_dte / 365.0
        if right == 'P':
            premium = OptionsPricer.put_price(underlying_price, strike, T, iv)
        else:
            premium = OptionsPricer.call_price(underlying_price, strike, T, iv)

        # SAFETY CHECK: Skip roll if premium is too low
        if premium < 0.01:
            logger.warning(
                f"Skipping roll signal for {symbol} {right}: premium too low ({premium:.4f}), "
                f"strike={strike:.2f}, underlying={underlying_price:.2f}"
            )
            return None

        # Generate signal
        signal = Signal(
            symbol=symbol,
            trade_type=trade_type,
            strike=round(strike, 2),
            expiry=expiry_str,
            right=right,
            quantity=quantity,  # Same size
            premium=premium,
            delta=target_delta,
            iv=iv,
            margin_requirement=strike * 100 if right == 'P' else None,
        )

        logger.info(
            f"Generated roll signal: {symbol} {trade_type} strike={strike:.2f} "
            f"dte={target_dte} premium={premium:.2f}"
        )

        return signal

    def on_assignment(self, position: Dict[str, Any]):
        """Called when option is assigned/exercised."""
        right = position.get("right", "")
        quantity = abs(position.get("quantity", 0))
        strike = position.get("strike", 0)
        symbol = position.get("symbol", "")  # Get symbol from position
        
        # Defensive check: ensure symbol is present
        if not symbol:
            logger.warning(f"Assignment position missing symbol field: {position}")
            return
        
        if right == "P":
            # Put assignment: we bought shares
            shares_acquired = quantity * 100
            # Use new multi-stock tracking
            self.stock_holding.add_shares(symbol, shares_acquired, strike)
            self.phase = "CC"  # Switch to Covered Call phase
            logger.info(f"Put assigned: Bought {shares_acquired} shares of {symbol} @ ${strike}, switching to CC phase")
            logger.info(f"Current holdings: {self.stock_holding.holdings}")
        
        elif right == "C":
            # Call assignment: we sold shares
            shares_sold = quantity * 100
            
            # Get the cost basis for this specific stock
            stock_cost_basis_per_share = self.stock_holding.holdings.get(symbol, {}).get("cost_basis", 0)
            shares_held = self.stock_holding.get_shares(symbol)
            
            # CRITICAL: Defensive check - if no shares held, this is an error
            if shares_held <= 0 or stock_cost_basis_per_share <= 0:
                logger.warning(
                    f"Call assigned but no {symbol} shares held! This indicates a bug. "
                    f"shares={shares_held}, cost_basis={stock_cost_basis_per_share:.2f}"
                )
                return
            
            # Limit shares_sold to actual shares held
            actual_shares_sold = min(shares_sold, shares_held)
            if actual_shares_sold != shares_sold:
                logger.warning(
                    f"Call assignment for {shares_sold} shares but only {shares_held} held. "
                    f"Adjusting to {actual_shares_sold} shares."
                )
                shares_sold = actual_shares_sold
            
            # Calculate realized stock P&L
            stock_cost_basis = stock_cost_basis_per_share * shares_sold
            stock_proceeds = strike * shares_sold
            stock_pnl = stock_proceeds - stock_cost_basis
           
            # Log complete P&L breakdown
            option_pnl = position.get("pnl", 0)
            total_trade_pnl = option_pnl + stock_pnl
            logger.info(
                f"Call assigned: Option P&L=${option_pnl:+.2f}, Stock P&L=${stock_pnl:+.2f}, "
                f"Total=${total_trade_pnl:+.2f} (bought at ${stock_cost_basis_per_share:.2f}, "
                f"sold at ${strike:.2f}, {shares_sold} shares of {symbol})"
            )
           
            # Remove shares using new method
            self.stock_holding.remove_shares(symbol, shares_sold)
            
            # Check if we still hold any shares
            if self.stock_holding.shares == 0:
                self.phase = "SP"  # Switch back to Sell Put phase
                logger.info("All shares sold, returning to SP phase")
            else:
                logger.info(f"Remaining holdings: {self.stock_holding.holdings}")
    
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
            symbol = trade.get("symbol", "")  # Get symbol from trade
            
            # Defensive check: ensure symbol is present
            if not symbol:
                logger.warning(f"Assignment trade missing symbol field: {trade}")
                return 0.0
            
            if right == "P":
                # Put assignment: we bought shares
                shares_acquired = quantity * 100
                
                # Use new multi-stock tracking
                self.stock_holding.add_shares(symbol, shares_acquired, strike)
                self.phase = "CC"  # Switch to Covered Call phase
                
                logger.info(f"Put assigned: Bought {shares_acquired} shares @ ${strike} of {symbol}, switching to CC phase")
                logger.info(f"Current holdings: {self.stock_holding.holdings}")
            
            elif right == "C":
                # Call assignment: we sold shares
                shares_sold = quantity * 100
                
                # Get the cost basis for this specific stock
                stock_cost_basis_per_share = self.stock_holding.holdings.get(symbol, {}).get("cost_basis", 0)
                shares_held = self.stock_holding.get_shares(symbol)
                
                # CRITICAL: Defensive check - if no shares held for this stock, this is an error
                if shares_held <= 0 or stock_cost_basis_per_share <= 0:
                    logger.warning(
                        f"Call assigned but no {symbol} shares held! This indicates a bug. "
                        f"shares={shares_held}, cost_basis={stock_cost_basis_per_share:.2f}"
                    )
                    return 0.0
                
                # Ensure shares_sold is a multiple of 100 to maintain proper lot sizes
                # But also ensure we don't sell more shares than we hold
                actual_shares_sold = min(shares_sold, shares_held)
                
                # Make sure we're selling in lots of 100 shares
                if actual_shares_sold >= 100:
                    actual_shares_sold = (actual_shares_sold // 100) * 100
                
                if actual_shares_sold != shares_sold:
                    logger.info(
                        f"Adjusted call assignment from {shares_sold} to {actual_shares_sold} shares "
                        f"to maintain proper lot size (100-share multiples)."
                    )
                    shares_sold = actual_shares_sold
                
                # Calculate realized stock P&L
                stock_cost_basis = stock_cost_basis_per_share * shares_sold
                stock_proceeds = strike * shares_sold
                stock_pnl = stock_proceeds - stock_cost_basis
                
                # IMPORTANT: Record stock P&L to be added to cumulative_pnl
                additional_stock_pnl = stock_pnl
               
                # Log complete P&L breakdown
                option_pnl = trade.get("pnl", 0)
                total_trade_pnl = option_pnl + stock_pnl
                logger.info(
                    f"Call assigned: Option P&L=${option_pnl:+.2f}, Stock P&L=${stock_pnl:+.2f}, "
                    f"Total=${total_trade_pnl:+.2f} (bought at ${stock_cost_basis_per_share:.2f}, "
                    f"sold at ${strike:.2f}, {shares_sold} shares of {symbol})"
                )
               
                # Remove shares using new method
                self.stock_holding.remove_shares(symbol, shares_sold)
                
                # Check if we still hold any shares
                if self.stock_holding.shares == 0:
                    self.phase = "SP"  # Switch back to Sell Put phase
                    logger.info("All shares sold, returning to SP phase")
                else:
                    logger.info(f"Remaining holdings: {self.stock_holding.holdings}")
        
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
    
    def select_strike_with_delta(
        self,
        underlying_price: float,
        iv: float,
        T: float,
        right: str,
        target_delta: float,
    ) -> float:
        """Select strike price with a specific delta value."""
        from core.backtesting.pricing import OptionsPricer
        
        # Binary search for the strike that gives target delta
        if right == "P":
            target_delta = -abs(target_delta)
            # For put options: when S > K (OTM), delta approaches 0
            # When S < K (ITM), delta approaches -1
            # So to get delta near -0.3 (slightly OTM), we want K slightly less than S
            low = underlying_price * 0.8  # e.g. 120 for S=150
            high = underlying_price * 1.1  # e.g. 165 for S=150
        else:
            target_delta = abs(target_delta)
            # For call options: when S > K (ITM), delta approaches 1
            # When S < K (OTM), delta approaches 0
            # So to get delta near 0.3 (slightly OTM for covered calls), we want K > S
            low = underlying_price * 0.8  # Wider range for better convergence
            high = underlying_price * 1.2

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
        # Debug logging
        self.logger.info(f"BinbinGod Performance Report: phase={self.phase}, shares={self.stock_holding.shares}, holdings={self.stock_holding.holdings}")

        return {
            "strategy": "binbin_god",
            # current_state: for compatibility with Wheel's monitoring dashboard
            "current_state": {
                "phase": self.phase,
                "shares_held": self.stock_holding.shares,
                "cost_basis": round(self.stock_holding.cost_basis, 2),
                "total_premium_collected": round(self.stock_holding.total_premium_collected, 2),
            },
            # Multi-stock holdings details
            "stock_holdings": dict(self.stock_holding.holdings) if self.stock_holding.holdings else {},
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
            # ML optimization status
            "ml_optimization": {
                "delta_enabled": self.ml_delta_optimization,
                "dte_enabled": self.ml_dte_optimization,
                "roll_enabled": self.ml_roll_optimization,
                "roll_model_loaded": self.ml_roll_optimizer.model is not None if self.ml_roll_optimizer else False,
            },
            # For UI compatibility with Wheel strategy monitoring
            "trade_history": [],  # Would need to track in on_trade_closed
            "phase_history": [],  # Would need to track on phase transitions
        }

    def build_market_data_for_roll(
        self,
        symbol: str,
        current_price: float,
        iv: float,
        bars: List[Dict] = None,
        option_price: float = None,
        current_delta: float = None,
    ) -> Dict[str, Any]:
        """Build market data dictionary for ML roll optimization.

        Args:
            symbol: Stock symbol
            current_price: Current underlying price
            iv: Current implied volatility
            bars: Price history bars
            option_price: Current option price
            current_delta: Current option delta

        Returns:
            Market data dictionary for ML roll features
        """
        from core.ml.market_data import MarketDataCalculator

        # Calculate momentum indicators from bars if available
        momentum_data = {}
        if bars and len(bars) >= 20:
            prices = [bar.get('close', 0) for bar in bars[-50:]]
            if len(prices) >= 20:
                momentum = MarketDataCalculator.calculate_momentum_indicators(prices, len(prices) - 1)
                momentum_data = {
                    'momentum_5d': momentum.get('momentum_5d', 0),
                    'momentum_10d': momentum.get('momentum_10d', 0),
                    'vs_ma20': momentum.get('vs_ma20', 0),
                    'vs_ma50': momentum.get('vs_ma50', 0),
                    'ma20': current_price / (1 + momentum.get('vs_ma20', 0) / 100),
                    'ma50': current_price / (1 + momentum.get('vs_ma50', 0) / 100),
                    'price_history': prices,
                }

        # Calculate historical volatility
        hv = MarketDataCalculator.calculate_historical_volatility(
            [bar.get('close', 0) for bar in bars[-30:]] if bars else [current_price]
        )

        # Build market data snapshot
        market_data = {
            'price': current_price,
            'iv': iv,
            'historical_volatility': hv,
            'iv_rank': 50.0,  # Would need IV history to calculate
            'iv_percentile': 50.0,
            'option_price': option_price,
            'delta': current_delta,
            'gamma': 0.05,  # Estimate
            'vega': 0.2,  # Estimate
            **momentum_data,
            'vix': 20.0,  # Default VIX
            'vix_percentile': 50.0,
            'vix_rank': 50.0,
            'vix_change_pct': 0.0,
            'vix_5d_ma': 20.0,
            'vix_20d_ma': 20.0,
            'vix_term_structure': 0.0,
            'market_regime': 1,  # Normal
        }

        return market_data
