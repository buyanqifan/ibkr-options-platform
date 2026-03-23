"""
BinbinGod Strategy for QuantConnect
====================================

A dynamic stock selection Wheel strategy for MAG7 stocks with ML optimization.

Features:
- Intelligent stock selection based on IV Rank, Momentum, Technicals
- Full Wheel strategy logic (Sell Put -> Covered Call cycle)
- CC+SP simultaneous mode
- Multi-stock holdings support
- ML-optimized delta and DTE selection
"""

from AlgorithmImports import *
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np

# Import ML modules only - use QC's built-in Greeks and pricing
from ml_integration import (
    BinbinGodMLIntegration, 
    MLOptimizationConfig, 
    StrategySignal,
    AdaptiveDeltaStrategy
)


# MAG7 Universe
MAG7_STOCKS = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]


@dataclass
class StockScore:
    """Score for a single stock."""
    symbol: str
    pe_ratio: float = 0.0
    iv_rank: float = 50.0
    momentum: float = 50.0
    technical_score: float = 50.0
    liquidity_score: float = 70.0
    ml_score_adjustment: float = 0.0
    total_score: float = 0.0


@dataclass
class StockHolding:
    """Tracks stock position from put assignment."""
    shares: int = 0
    cost_basis: float = 0.0
    total_premium_collected: float = 0.0
    symbol: str = ""
    holdings: Dict = field(default_factory=dict)
    
    def add_shares(self, symbol: str, shares: int, cost_basis: float):
        """Add shares of a stock to holdings."""
        if symbol not in self.holdings:
            self.holdings[symbol] = {"shares": 0, "cost_basis": 0.0, "premium": 0.0}
        
        existing = self.holdings[symbol]
        total_shares = existing["shares"] + shares
        if total_shares > 0:
            total_cost = existing["shares"] * existing["cost_basis"] + shares * cost_basis
            existing["cost_basis"] = total_cost / total_shares
        existing["shares"] = total_shares
        self._update_legacy_fields()
    
    def remove_shares(self, symbol: str, shares: int) -> int:
        """Remove shares of a stock."""
        if symbol not in self.holdings:
            return 0
        existing = self.holdings[symbol]
        removed = min(shares, existing["shares"])
        existing["shares"] -= removed
        if existing["shares"] <= 0:
            del self.holdings[symbol]
        self._update_legacy_fields()
        return removed
    
    def get_shares(self, symbol: str) -> int:
        """Get shares for a specific symbol."""
        return self.holdings.get(symbol, {}).get("shares", 0)
    
    def get_symbols(self) -> List[str]:
        """Get list of symbols with holdings."""
        return list(self.holdings.keys())
    
    def add_premium(self, symbol: str, premium: float):
        """Add premium collected for a symbol."""
        if symbol not in self.holdings:
            self.holdings[symbol] = {"shares": 0, "cost_basis": 0.0, "premium": 0.0}
        self.holdings[symbol]["premium"] = self.holdings[symbol].get("premium", 0.0) + premium
        self.total_premium_collected += premium
    
    def _update_legacy_fields(self):
        """Update legacy fields for backward compatibility."""
        self.shares = sum(h["shares"] for h in self.holdings.values())
        if self.holdings:
            primary = max(self.holdings.items(), key=lambda x: x[1]["shares"])
            self.symbol = primary[0]
            self.cost_basis = primary[1]["cost_basis"]


class BinbinGodStrategy(QCAlgorithm):
    """
    BinbinGod Strategy - Intelligent stock selection + Full Wheel logic + ML optimization.
    
    Phase 1 (SP): Sell OTM puts, collect premium
      - If expires worthless: keep premium, continue selling puts
      - If assigned: buy shares at strike, switch to Phase 2
    
    Phase 2 (CC): Sell OTM calls against owned shares
      - If expires worthless: keep premium + shares, continue selling calls
      - If assigned: sell shares at strike, return to Phase 1
    
    Optimization: CC phase can also open SP positions (simultaneous mode)
    
    ML Enhancements:
      - Dynamic delta selection based on market regime
      - Optimal DTE range based on IV and momentum
      - Kelly-based position sizing
      - Intelligent roll decisions
      - Volatility regime detection
    """
    
    def Initialize(self):
        """Initialize the strategy."""
        # Get parameters with defaults - use local variables, not properties
        start_date_param = self.GetParameter("start_date")
        end_date_param = self.GetParameter("end_date")
        
        if start_date_param:
            self.SetStartDate(datetime.strptime(start_date_param, "%Y-%m-%d"))
        else:
            self.SetStartDate(2020, 1, 1)
        
        if end_date_param:
            self.SetEndDate(datetime.strptime(end_date_param, "%Y-%m-%d"))
        else:
            self.SetEndDate(datetime.now() - timedelta(days=1))
        
        # Strategy parameters
        self.initial_capital = float(self.GetParameter("initial_capital", 100000))
        self.SetCash(self.initial_capital)
        
        self.max_positions = int(self.GetParameter("max_positions", 10))
        self.profit_target_pct = float(self.GetParameter("profit_target_pct", 50))
        self.stop_loss_pct = float(self.GetParameter("stop_loss_pct", 999999))  # Disabled by default - Wheel strategy doesn't use traditional stop loss
        
        # Disable flags (set to >= 999999 to disable)
        self._profit_target_disabled = self.profit_target_pct >= 999999
        self._stop_loss_disabled = self.stop_loss_pct >= 999999
        
        # CC+SP simultaneous mode parameters
        self.allow_sp_in_cc_phase = bool(self.GetParameter("allow_sp_in_cc_phase", True))
        self.sp_in_cc_margin_threshold = float(self.GetParameter("sp_in_cc_margin_threshold", 0.5))
        self.sp_in_cc_max_positions = int(self.GetParameter("sp_in_cc_max_positions", 3))
        
        # ML configuration
        self.ml_enabled = bool(self.GetParameter("ml_enabled", True))
        self.ml_exploration_rate = float(self.GetParameter("ml_exploration_rate", 0.1))
        self.ml_learning_rate = float(self.GetParameter("ml_learning_rate", 0.01))
        self.ml_adoption_rate = float(self.GetParameter("ml_adoption_rate", 0.5))  # How much to trust ML vs traditional
        self.ml_min_confidence = float(self.GetParameter("ml_min_confidence", 0.4))  # Min confidence to use ML result
        
        # Stock pool
        stock_pool_str = self.GetParameter("stock_pool", ",".join(MAG7_STOCKS))
        self.stock_pool = stock_pool_str.split(",")
        
        # Scoring weights - aligned with original binbin_god.py
        self.weights = {
            "iv_rank": 0.35,       # Higher IV = better premium income
            "technical": 0.25,     # RSI + MA position (trend quality)
            "momentum": 0.20,      # Price trend strength
            "pe_score": 0.20,      # Value factor
        }
        
        # Initialize ML integration
        ml_config = MLOptimizationConfig(
            ml_delta_enabled=self.ml_enabled,
            ml_dte_enabled=self.ml_enabled,
            ml_roll_enabled=self.ml_enabled,
            ml_position_enabled=self.ml_enabled,
            exploration_rate=self.ml_exploration_rate,
            learning_rate=self.ml_learning_rate,
        )
        self.ml_integration = BinbinGodMLIntegration(ml_config)
        
        # Initialize Adaptive Delta Strategy for smooth transition between traditional and ML
        self.adaptive_strategy = AdaptiveDeltaStrategy(
            ml_integration=self.ml_integration,
            adoption_rate=self.ml_adoption_rate,
            min_confidence=self.ml_min_confidence
        )
        
        # Flag to track if ML pretraining has been done
        self._ml_pretrained = False
        
        # Add equities and options for all stocks in pool
        self.equities = {}
        self.options = {}
        self.price_history = {}  # Store price history for ML
        
        for symbol in self.stock_pool:
            equity = self.AddEquity(symbol, Resolution.Daily)
            equity.SetDataNormalizationMode(DataNormalizationMode.Raw)
            self.equities[symbol] = equity
            self.price_history[symbol] = []
            
            # Add option universe
            option = self.AddOption(symbol, Resolution.Daily)
            option.SetFilter(-10, 10, timedelta(days=21), timedelta(days=60))
            self.options[symbol] = option
        
        # Add VIX for market sentiment indicator
        self.vix = self.AddEquity("VIXY", Resolution.Daily)  # VIX ETF as proxy
        self._current_vix = 20.0  # Default VIX value
        self._vix_history = []
        
        # Strategy state
        self.phase = "SP"
        self.stock_holding = StockHolding()
        self._last_selected_stock = None
        self._selection_count = 0
        self._min_hold_cycles = 3
        
        # Track open positions with ML signals
        self.open_option_positions: Dict[str, Dict] = {}
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.trade_history = []
        
        # ML tracking
        self.ml_signals_history = []
        
        # Warm up for indicators (60 days)
        self.SetWarmUp(60)
        
        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Rebalance
        )
        
        # Schedule daily check for expired options
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketClose("SPY", -10),
            self.CheckExpiredOptions
        )
        
        # Schedule ML model update
        self.Schedule.On(
            self.DateRules.MonthEnd(),
            self.TimeRules.AfterMarketClose("SPY", -30),
            self.UpdateMLModels
        )
        
        self.Log(f"BinbinGod Strategy initialized with stock pool: {self.stock_pool}")
        self.Log(f"ML optimization enabled: {self.ml_enabled}")
    
    def OnData(self, data):
        """Handle incoming data - update price history for ML."""
        
        for symbol in self.stock_pool:
            if symbol in data.Bars:
                bar = data.Bars[symbol]
                self.price_history[symbol].append({
                    'date': self.Time.strftime('%Y-%m-%d'),
                    'open': float(bar.Open),
                    'high': float(bar.High),
                    'low': float(bar.Low),
                    'close': float(bar.Close),
                    'volume': float(bar.Volume)
                })
                
                # Keep limited history
                if len(self.price_history[symbol]) > 500:
                    self.price_history[symbol] = self.price_history[symbol][-500:]
        
        # Update VIX value for market sentiment
        if "VIXY" in data.Bars:
            vix_bar = data.Bars["VIXY"]
            # VIXY is roughly 0.5x of VIX, so multiply by 2 for approximation
            self._current_vix = float(vix_bar.Close) * 2
            self._vix_history.append(self._current_vix)
            if len(self._vix_history) > 252:  # Keep 1 year of history
                self._vix_history = self._vix_history[-252:]
    
    def OnWarmupFinished(self):
        """Called when warmup is finished - pretrain ML models."""
        if not self.ml_enabled or self._ml_pretrained:
            return
        
        self.Log("Starting ML model pretraining with warmup data...")
        
        # Pretrain ML models using collected price history
        pretrain_stats = self._pretrain_ml_models()
        
        self._ml_pretrained = True
        self.Log(f"ML pretraining completed: {pretrain_stats}")
    
    def _pretrain_ml_models(self) -> Dict:
        """
        Pretrain ML models with historical price data collected during warmup.
        
        This allows ML models to have initial learning before live trading/backtesting.
        """
        stats = {
            'symbols_trained': 0,
            'total_simulations': 0,
            'status': 'skipped'
        }
        
        if not self.ml_enabled:
            stats['status'] = 'ml_disabled'
            return stats
        
        try:
            for symbol in self.stock_pool:
                bars = self.price_history.get(symbol, [])
                
                if len(bars) < 60:
                    self.Log(f"Skipping {symbol}: insufficient data ({len(bars)} bars)")
                    continue
                
                # Estimate IV from historical volatility
                iv_estimate = self._calculate_historical_vol(symbol) or 0.25
                
                # Pretrain ML models for this symbol
                symbol_stats = self.ml_integration.pretrain_models(
                    symbol=symbol,
                    historical_bars=bars,
                    iv_estimate=iv_estimate
                )
                
                if symbol_stats.get('status') == 'success':
                    stats['symbols_trained'] += 1
                    stats['total_simulations'] += symbol_stats.get('put_simulations', 0)
                    stats['total_simulations'] += symbol_stats.get('call_simulations', 0)
                    
                    self.Log(f"Pretrained {symbol}: {symbol_stats.get('put_simulations', 0)} put sims, "
                            f"{symbol_stats.get('call_simulations', 0)} call sims")
            
            stats['status'] = 'success' if stats['symbols_trained'] > 0 else 'no_data'
            
        except Exception as e:
            self.Log(f"ML pretraining error: {e}")
            stats['status'] = f'error: {e}'
        
        return stats
    
    def Rebalance(self):
        """Main rebalancing logic - called daily."""
        if self.IsWarmingUp:
            return
        
        # Check for early exit opportunities (profit target, roll forward, etc.)
        self._check_position_management()
        
        # Check current open positions
        open_count = len(self.open_option_positions)
        
        if open_count >= self.max_positions:
            self.Log(f"Max positions reached: {open_count} >= {self.max_positions}")
            return
        
        # Generate ML-optimized signals
        signals = self._generate_ml_signals()
        
        if not signals:
            self.Log("No signals generated")
            return
        
        # Select best signal
        best_signal = self._select_best_signal(signals)
        
        # Execute best signal with confidence filtering
        if best_signal and best_signal.confidence >= self.ml_min_confidence:
            self._execute_signal(best_signal)
        elif best_signal:
            self.Log(f"Signal filtered: {best_signal.symbol} confidence {best_signal.confidence:.2f} < threshold {self.ml_min_confidence}")
    
    def _check_position_management(self):
        """Check open positions for early exit opportunities."""
        positions_to_manage = list(self.open_option_positions.items())
        
        for position_id, pos_info in positions_to_manage:
            if position_id not in self.open_option_positions:
                continue
                
            option_symbol = pos_info['option_symbol']
            security = self.Securities.get(option_symbol)
            
            if not security:
                continue
            
            current_price = security.Price
            entry_price = pos_info['entry_price']
            
            if current_price <= 0 or entry_price <= 0:
                continue
            
            # Calculate P&L (short position: profit when price drops)
            pnl = (entry_price - current_price) * abs(pos_info['quantity']) * 100
            pnl_pct = pnl / (entry_price * abs(pos_info['quantity']) * 100) * 100
            
            # Get DTE
            expiry = pos_info['expiry']
            if hasattr(expiry, '__sub__'):
                dte = (expiry - self.Time).days
            else:
                dte = 30  # Default
            
            # Premium captured (for short options, profit when price drops)
            premium_captured_pct = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0
            
            # Build position and market data for ML Roll optimizer
            position_data = {**pos_info, 'current_price': current_price, 'pnl_pct': pnl_pct, 'dte': dte}
            equity = self.equities.get(pos_info['symbol'])
            underlying_price = self.Securities[equity.Symbol].Price if equity else 0
            market_data = {'price': underlying_price, 'iv': 0.25, 'vix': getattr(self, '_current_vix', 20)}
            
            # Use ML Roll optimizer if enabled
            if self.ml_enabled and hasattr(self.ml_integration, 'roll_optimizer'):
                try:
                    should_roll, roll_rec = self.ml_integration.roll_optimizer.should_roll(
                        position=position_data,
                        market_data=market_data,
                        current_date=self.Time.strftime('%Y-%m-%d'),
                        min_confidence=self.ml_min_confidence
                    )
                    
                    if should_roll and roll_rec.action in ["ROLL_FORWARD", "CLOSE_EARLY"]:
                        self.Log(f"ML Roll: {roll_rec.action} (confidence: {roll_rec.confidence:.0%})")
                        
                        if roll_rec.action == "ROLL_FORWARD":
                            signal = StrategySignal(
                                symbol=pos_info['symbol'],
                                action="ROLL",
                                delta=roll_rec.optimal_delta or abs(pos_info['delta_at_entry']),
                                dte_min=roll_rec.optimal_dte or 30,
                                dte_max=roll_rec.optimal_dte or 45,
                                num_contracts=abs(pos_info['quantity']),
                                confidence=roll_rec.confidence,
                                reasoning=roll_rec.reasoning
                            )
                            self._execute_roll(signal)
                            continue
                        elif roll_rec.action == "CLOSE_EARLY":
                            signal = StrategySignal(
                                symbol=pos_info['symbol'],
                                action="CLOSE",
                                delta=0, dte_min=0, dte_max=0, num_contracts=0,
                                confidence=roll_rec.confidence,
                                reasoning=roll_rec.reasoning
                            )
                            self._execute_close(signal)
                            continue
                except Exception as e:
                    self.Log(f"ML Roll optimizer error: {e}")
            
            # Rule-based roll logic (fallback)
            # Check roll forward: 80%+ premium captured with 7+ DTE remaining
            if premium_captured_pct >= 80 and dte > 7:
                self.Log(f"Roll forward opportunity: {pos_info['symbol']} "
                        f"premium_captured={premium_captured_pct:.1f}%, DTE={dte}")
                
                # Generate roll signal
                signal = StrategySignal(
                    symbol=pos_info['symbol'],
                    action="ROLL",
                    delta=abs(pos_info['delta_at_entry']),
                    dte_min=30,
                    dte_max=45,
                    num_contracts=abs(pos_info['quantity']),
                    confidence=0.8,
                    reasoning=f"Roll forward: {premium_captured_pct:.0f}% premium captured, {dte} DTE remaining"
                )
                self._execute_roll(signal)
                continue
            
            # Check profit target (Wheel strategy: optional, may be disabled)
            if not self._profit_target_disabled and pnl_pct >= self.profit_target_pct:
                self.Log(f"Profit target reached: {pos_info['symbol']} P&L={pnl_pct:.1f}%")
                
                signal = StrategySignal(
                    symbol=pos_info['symbol'],
                    action="CLOSE",
                    delta=0,
                    dte_min=0,
                    dte_max=0,
                    num_contracts=0,
                    confidence=0.9,
                    reasoning=f"Profit target: {pnl_pct:.0f}% gain"
                )
                self._execute_close(signal)
                continue
            
            # Check stop loss (DISABLED by default for Wheel strategy - time is our friend)
            # Can be enabled by setting stop_loss_pct < 999999 as extreme safety net
            if not self._stop_loss_disabled and pnl_pct <= -self.stop_loss_pct:
                self.Log(f"Stop loss triggered: {pos_info['symbol']} P&L={pnl_pct:.1f}%")
                
                signal = StrategySignal(
                    symbol=pos_info['symbol'],
                    action="CLOSE",
                    delta=0,
                    dte_min=0,
                    dte_max=0,
                    num_contracts=0,
                    confidence=1.0,
                    reasoning=f"Stop loss: {pnl_pct:.0f}% loss"
                )
                self._execute_close(signal)
    
    def _generate_ml_signals(self) -> List[StrategySignal]:
        """Generate ML-optimized signals for all candidate stocks."""
        
        signals = []
        
        # Get portfolio state
        portfolio_state = self._get_portfolio_state()
        
        # Generate signals based on phase
        if self.phase == "SP":
            # Sell Put phase - score all stocks
            for symbol in self.stock_pool:
                signal = self._generate_signal_for_symbol(symbol, "SP", portfolio_state)
                if signal:
                    signals.append(signal)
        
        elif self.phase == "CC":
            # Covered Call phase - generate CC signals for held stocks
            held_symbols = self.stock_holding.get_symbols()
            
            for symbol in held_symbols:
                signal = self._generate_signal_for_symbol(symbol, "CC", portfolio_state)
                if signal:
                    signals.append(signal)
            
            # CC+SP mode: also generate SP signals
            if self.allow_sp_in_cc_phase:
                sp_signals = self._generate_cc_sp_signals(portfolio_state)
                signals.extend(sp_signals)
        
        return signals
    
    def _generate_signal_for_symbol(
        self,
        symbol: str,
        strategy_phase: str,
        portfolio_state: Dict
    ) -> Optional[StrategySignal]:
        """Generate ML-optimized signal for a single symbol.
        
        Uses AdaptiveDeltaStrategy to combine traditional and ML delta selection,
        with automatic fallback when ML confidence is low.
        """
        
        equity = self.equities.get(symbol)
        if not equity:
            return None
        
        underlying_price = self.Securities[equity.Symbol].Price
        if underlying_price <= 0:
            return None
        
        # Get price history
        bars = self.price_history.get(symbol, [])
        if len(bars) < 20:
            return None
        
        # Get cost basis
        cost_basis = self.stock_holding.holdings.get(symbol, {}).get("cost_basis", 0)
        
        # Get current position if any
        current_position = self._get_current_position(symbol)
        
        # Traditional/fallback delta values
        traditional_put_delta = 0.30
        traditional_call_delta = 0.30
        
        # Use ML integration to generate signal
        signal = self.ml_integration.generate_signal(
            symbol=symbol,
            current_price=underlying_price,
            cost_basis=cost_basis,
            bars=bars,
            strategy_phase=strategy_phase,
            portfolio_state=portfolio_state,
            current_position=current_position
        )
        
        # Apply AdaptiveDeltaStrategy to adjust delta based on confidence
        if signal and self.ml_enabled:
            right = "P" if strategy_phase in ("SP", "CC+SP") else "C"
            traditional_delta = traditional_put_delta if right == "P" else traditional_call_delta
            
            # Use adaptive strategy to get final delta
            if right == "P":
                adaptive_delta, explanation = self.adaptive_strategy.select_put_delta(
                    traditional_delta=traditional_delta,
                    ml_delta=signal.delta,
                    ml_confidence=signal.delta_confidence,
                    reasoning=signal.reasoning
                )
            else:
                adaptive_delta, explanation = self.adaptive_strategy.select_call_delta(
                    traditional_delta=traditional_delta,
                    ml_delta=signal.delta,
                    ml_confidence=signal.delta_confidence,
                    reasoning=signal.reasoning
                )
            
            # Log the adaptive decision if delta changed significantly
            if abs(adaptive_delta - signal.delta) > 0.02:
                self.Log(f"Adaptive delta adjustment for {symbol}: {explanation}")
            
            # Update signal with adaptive delta
            signal.delta = adaptive_delta
        
        # Add stock scoring adjustment
        score = self._score_single_stock(symbol, bars, underlying_price)
        signal.ml_score_adjustment = (score.total_score - 50) / 100  # Normalize to -0.5 to 0.5
        
        return signal
    
    def _generate_cc_sp_signals(self, portfolio_state: Dict) -> List[StrategySignal]:
        """Generate SP signals during CC phase (simultaneous mode)."""
        
        signals = []
        
        # Check margin utilization
        margin_used = self.Portfolio.TotalMarginUsed
        margin_utilization = margin_used / self.Portfolio.TotalPortfolioValue
        
        if margin_utilization > self.sp_in_cc_margin_threshold:
            self.Log(f"CC+SP: margin utilization {margin_utilization:.1%} > threshold")
            return signals
        
        # Count current SP positions
        sp_positions = sum(1 for pos in self.open_option_positions.values() if pos.get('right') == 'P')
        
        if sp_positions >= self.sp_in_cc_max_positions:
            self.Log(f"CC+SP: max SP positions reached ({sp_positions})")
            return signals
        
        # Select best stock not currently held
        held_symbols = self.stock_holding.get_symbols()
        available_stocks = [s for s in self.stock_pool if s not in held_symbols]
        
        if not available_stocks:
            available_stocks = self.stock_pool
        
        # Generate signals for available stocks
        for symbol in available_stocks[:3]:  # Top 3 candidates
            signal = self._generate_signal_for_symbol(symbol, "CC+SP", portfolio_state)
            if signal and signal.confidence > 0.6:
                signals.append(signal)
        
        return signals
    
    def _score_single_stock(self, symbol: str, bars: List[Dict], current_price: float) -> StockScore:
        """Score a single stock for selection."""
        
        if len(bars) < 20:
            return StockScore(symbol=symbol, total_score=50.0)
        
        closes = np.array([b['close'] for b in bars])
        
        # Momentum (20-day price change)
        prev_20_price = closes[-20]
        raw_momentum = ((current_price - prev_20_price) / prev_20_price) * 100
        momentum = max(0, min(100, (raw_momentum + 20) / 70 * 100))
        
        # IV Rank approximation - calculate volatility from returns
        lookback = min(30, len(closes))
        if lookback > 2:
            recent = closes[-lookback:]
            returns = np.diff(recent) / recent[:-1]
            vol = np.std(returns)
            iv_rank = min(100, max(0, vol * 200 * 252))
        else:
            iv_rank = 50.0
        
        # Technical Score
        rsi_score = self._calculate_rsi_score(closes)
        ma_score = self._calculate_ma_score(closes)
        technical_score = rsi_score * 0.6 + ma_score * 0.4
        
        # Liquidity Score
        volumes = [b.get('volume', 1) for b in bars[-10:]]
        if volumes:
            recent_vol = np.mean(volumes[-5:])
            prior_vol = np.mean(volumes[-10:-5])
            liquidity_score = min(100, (recent_vol / prior_vol) * 50) if prior_vol > 0 else 70.0
        else:
            liquidity_score = 70.0
        
        # PE Score
        pe_ratio = max(5, min(60, 35 + raw_momentum * 0.3))
        pe_score = max(0, min(100, 100 - pe_ratio))
        
        # Calculate weighted total score
        total_score = (
            iv_rank * self.weights["iv_rank"] +
            technical_score * self.weights["technical"] +
            momentum * self.weights["momentum"] +
            pe_score * self.weights["pe_score"]
        )
        
        return StockScore(
            symbol=symbol,
            pe_ratio=pe_ratio,
            iv_rank=iv_rank,
            momentum=momentum,
            technical_score=technical_score,
            liquidity_score=liquidity_score,
            total_score=total_score
        )
    
    def _calculate_rsi_score(self, closes: np.ndarray) -> float:
        """Calculate RSI score."""
        if len(closes) < 14:
            return 50.0
        
        deltas = np.diff(closes[-14:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_score = max(0, min(100, 100 - abs(rsi - 60) * 2))
        return rsi_score
    
    def _calculate_ma_score(self, closes: np.ndarray) -> float:
        """Calculate MA position score."""
        if len(closes) < 20:
            return 50.0
        
        current_price = closes[-1]
        sma_20 = np.mean(closes[-20:])
        ma_position = (current_price - sma_20) / sma_20 * 100
        
        ma_score = max(0, min(100, 50 + ma_position * 5))
        return ma_score
    
    def _get_portfolio_state(self) -> Dict:
        """Get current portfolio state for ML."""
        
        positions = []
        for holding in self.Portfolio.Values:
            if holding.Invested:
                positions.append({
                    'symbol': str(holding.Symbol),
                    'quantity': holding.Quantity,
                    'market_value': holding.HoldingsValue,
                })
        
        return {
            'total_capital': self.Portfolio.TotalPortfolioValue,
            'available_margin': self.Portfolio.Cash,
            'margin_used': self.Portfolio.TotalMarginUsed,
            'drawdown': self._calculate_drawdown(),
            'positions': positions,
            'cost_basis': self.stock_holding.cost_basis,
        }
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown."""
        peak = self.initial_capital
        current = self.Portfolio.TotalPortfolioValue
        
        # Simple drawdown calculation
        if current < peak:
            return (peak - current) / peak * 100
        return 0.0
    
    def _get_current_position(self, symbol: str) -> Optional[Dict]:
        """Get current position info for a symbol."""
        
        for pos_id, pos_info in self.open_option_positions.items():
            if pos_info.get('symbol') == symbol:
                return pos_info
        return None
    
    def _select_best_signal(self, signals: List[StrategySignal]) -> StrategySignal:
        """Select the best signal from candidates."""
        
        if not signals:
            return None
        
        # Score each signal
        for signal in signals:
            # Adjust confidence with stock score
            signal.confidence += signal.ml_score_adjustment * 0.5
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals[0]
    
    def _execute_signal(self, signal: StrategySignal):
        """Execute a trading signal using QC's built-in Greeks data."""
        
        if not signal:
            return
        
        if signal.action == "HOLD":
            self.Log(f"HOLD: {signal.reasoning}")
            return
        
        if signal.action == "ROLL":
            self._execute_roll(signal)
            return
        
        if signal.action == "CLOSE":
            self._execute_close(signal)
            return
        
        # Get equity
        equity = self.equities.get(signal.symbol)
        if not equity:
            self.Log(f"No equity found for {signal.symbol}")
            return
        
        underlying_price = self.Securities[equity.Symbol].Price
        
        # Determine option type and target delta
        target_right = OptionRight.Put if signal.action == "SELL_PUT" else OptionRight.Call
        target_delta = -signal.delta if target_right == OptionRight.Put else signal.delta
        
        # Find suitable option using QC's Greeks
        selected = self._find_option_by_greeks(
            symbol=signal.symbol,
            equity_symbol=equity.Symbol,
            target_right=target_right,
            target_delta=target_delta,
            dte_min=signal.dte_min,
            dte_max=signal.dte_max,
            delta_tolerance=0.05
        )
        
        if not selected:
            self.Log(f"No suitable options found for {signal.symbol} with delta ~{target_delta:.2f}")
            return
        
        option_symbol = selected['option_symbol']
        strike = selected['strike']
        expiry = selected['expiry']
        delta = selected['delta']
        iv = selected['iv']
        premium = selected['premium']
        
        # Position sizing from signal
        quantity = -signal.num_contracts
        
        # Execute order
        ticket = self.MarketOrder(option_symbol, quantity)
        
        if ticket.Status == OrderStatus.Filled:
            right_str = 'P' if target_right == OptionRight.Put else 'C'
            position_id = f"{signal.symbol}_{self.Time.strftime('%Y%m%d')}_{strike}_{right_str}"
            
            # Use actual fill price
            fill_price = ticket.AverageFillPrice if ticket.AverageFillPrice > 0 else premium
            
            self.open_option_positions[position_id] = {
                'symbol': signal.symbol,
                'option_symbol': option_symbol,
                'right': right_str,
                'strike': strike,
                'expiry': expiry,
                'entry_date': self.Time.strftime('%Y-%m-%d'),
                'entry_price': fill_price,
                'quantity': quantity,
                'delta_at_entry': delta,
                'iv_at_entry': iv,
                'strategy_phase': self.phase,
                'ml_signal': signal,
            }
            
            # Record signal
            self.ml_signals_history.append({
                'date': self.Time.strftime('%Y-%m-%d'),
                'signal': signal,
            })
            
            self.Log(f"Executed: {signal.action} {signal.num_contracts} {signal.symbol} "
                    f"@ ${fill_price:.2f}, strike={strike:.2f}, delta={delta:.3f}, IV={iv:.1%}")
            self.Log(f"ML Reasoning: {signal.reasoning}")
        else:
            self.Log(f"Order not filled for {signal.symbol}: {ticket.Status}")
    
    def _execute_roll(self, signal: StrategySignal):
        """Execute a roll action - close existing position and open new one."""
        self.Log(f"Roll action for {signal.symbol}: {signal.reasoning}")
        
        # Find existing position for this symbol
        existing_position_id = None
        existing_position = None
        
        for pos_id, pos_info in self.open_option_positions.items():
            if pos_info.get('symbol') == signal.symbol:
                existing_position_id = pos_id
                existing_position = pos_info
                break
        
        if not existing_position:
            self.Log(f"No existing position found to roll for {signal.symbol}")
            return
        
        # Close existing position
        option_symbol = existing_position['option_symbol']
        quantity_to_close = -existing_position['quantity']  # Opposite side to close
        
        close_ticket = self.MarketOrder(option_symbol, quantity_to_close)
        
        if close_ticket.Status != OrderStatus.Filled:
            self.Log(f"Failed to close position for roll: {close_ticket.Status}")
            return
        
        close_price = close_ticket.AverageFillPrice
        entry_price = existing_position['entry_price']
        pnl = (entry_price - close_price) * abs(existing_position['quantity']) * 100
        
        self.Log(f"Closed position for roll: P&L=${pnl:.2f}")
        
        # Update tracking
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        
        self.trade_history.append({
            "date": self.Time.strftime("%Y-%m-%d"),
            "symbol": signal.symbol,
            "type": existing_position['right'],
            "pnl": pnl,
            "exit_reason": "ROLL"
        })
        
        # Remove old position
        del self.open_option_positions[existing_position_id]
        
        # Open new position using signal parameters
        equity = self.equities.get(signal.symbol)
        if not equity:
            return
        
        target_right = OptionRight.Put if signal.action == "ROLL" and existing_position['right'] == 'P' else OptionRight.Call
        target_delta = -signal.delta if target_right == OptionRight.Put else signal.delta
        
        # Find new option
        selected = self._find_option_by_greeks(
            symbol=signal.symbol,
            equity_symbol=equity.Symbol,
            target_right=target_right,
            target_delta=target_delta,
            dte_min=signal.dte_min,
            dte_max=signal.dte_max,
            delta_tolerance=0.05
        )
        
        if not selected:
            self.Log(f"Failed to find new option for roll: {signal.symbol}")
            return
        
        # Open new position
        new_quantity = -signal.num_contracts
        new_ticket = self.MarketOrder(selected['option_symbol'], new_quantity)
        
        if new_ticket.Status == OrderStatus.Filled:
            right_str = 'P' if target_right == OptionRight.Put else 'C'
            position_id = f"{signal.symbol}_{self.Time.strftime('%Y%m%d')}_{selected['strike']:.0f}_{right_str}"
            
            fill_price = new_ticket.AverageFillPrice
            
            self.open_option_positions[position_id] = {
                'symbol': signal.symbol,
                'option_symbol': selected['option_symbol'],
                'right': right_str,
                'strike': selected['strike'],
                'expiry': selected['expiry'],
                'entry_date': self.Time.strftime('%Y-%m-%d'),
                'entry_price': fill_price,
                'quantity': new_quantity,
                'delta_at_entry': selected['delta'],
                'iv_at_entry': selected['iv'],
                'strategy_phase': self.phase,
                'ml_signal': signal,
            }
            
            self.Log(f"Rolled to new position: {signal.symbol} @ ${fill_price:.2f}, "
                    f"strike={selected['strike']:.2f}, delta={selected['delta']:.3f}")
    
    def _execute_close(self, signal: StrategySignal):
        """Execute a close action - close existing position."""
        self.Log(f"Close action for {signal.symbol}: {signal.reasoning}")
        
        # Find existing position for this symbol
        positions_to_close = []
        
        for pos_id, pos_info in self.open_option_positions.items():
            if pos_info.get('symbol') == signal.symbol:
                positions_to_close.append((pos_id, pos_info))
        
        if not positions_to_close:
            self.Log(f"No positions found to close for {signal.symbol}")
            return
        
        for pos_id, pos_info in positions_to_close:
            option_symbol = pos_info['option_symbol']
            quantity_to_close = -pos_info['quantity']
            
            close_ticket = self.MarketOrder(option_symbol, quantity_to_close)
            
            if close_ticket.Status == OrderStatus.Filled:
                close_price = close_ticket.AverageFillPrice
                entry_price = pos_info['entry_price']
                pnl = (entry_price - close_price) * abs(pos_info['quantity']) * 100
                
                self.total_trades += 1
                self.total_pnl += pnl
                if pnl > 0:
                    self.winning_trades += 1
                
                self.trade_history.append({
                    "date": self.Time.strftime("%Y-%m-%d"),
                    "symbol": signal.symbol,
                    "type": pos_info['right'],
                    "pnl": pnl,
                    "exit_reason": signal.reasoning or "SIGNAL_CLOSE"
                })
                
                self.Log(f"Closed position: {signal.symbol} P&L=${pnl:.2f}")
                
                del self.open_option_positions[pos_id]
    
    def CheckExpiredOptions(self):
        """Check for expired options and handle assignments."""
        positions_to_remove = []
        
        for position_id, pos_info in self.open_option_positions.items():
            security = self.Securities.get(pos_info['option_symbol'])
            if not security:
                continue
            
            # Check if expired
            if security.IsDelisted or security.Price == 0:
                symbol = pos_info['symbol']
                right = pos_info['right']
                
                # Calculate P&L
                holdings = security.Holdings
                pnl = holdings.UnrealizedProfit
                
                # Update ML models
                self.ml_integration.update_performance({
                    'symbol': symbol,
                    'delta': abs(pos_info['delta_at_entry']),
                    'dte': (pos_info['expiry'] - datetime.strptime(pos_info['entry_date'], '%Y-%m-%d')).days,
                    'num_contracts': abs(pos_info['quantity']),
                    'pnl': pnl / 100,  # Normalize
                    'assigned': False,  # Would need to check actual assignment
                    'bars': self.price_history.get(symbol, []),
                    'cost_basis': self.stock_holding.holdings.get(symbol, {}).get('cost_basis', 0),
                    'strategy_phase': pos_info['strategy_phase'],
                })
                
                # Record trade
                self.total_trades += 1
                self.total_pnl += pnl
                if pnl > 0:
                    self.winning_trades += 1
                
                self.trade_history.append({
                    "date": self.Time.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "type": right,
                    "pnl": pnl,
                    "exit_reason": "EXPIRY"
                })
                
                self.Log(f"Option expired: {position_id}, P&L: ${pnl:.2f}")
                positions_to_remove.append(position_id)
                
                # Handle assignment logic
                equity = self.equities.get(symbol)
                if equity:
                    last_price = self.Securities[equity.Symbol].Price
                    strike = pos_info['strike']
                    
                    if right == "P" and last_price < strike:
                        # Put assignment
                        shares_acquired = abs(pos_info['quantity']) * 100
                        self.stock_holding.add_shares(symbol, shares_acquired, strike)
                        self.stock_holding.add_premium(symbol, pos_info['entry_price'] * 100)
                        self.phase = "CC"
                        self.Log(f"Put assigned: Bought {shares_acquired} shares of {symbol} @ ${strike:.2f}")
                    
                    elif right == "C" and last_price > strike:
                        # Call assignment
                        shares_sold = abs(pos_info['quantity']) * 100
                        cost_basis = self.stock_holding.holdings.get(symbol, {}).get('cost_basis', strike)
                        stock_pnl = (strike - cost_basis) * shares_sold
                        self.total_pnl += stock_pnl
                        
                        self.stock_holding.remove_shares(symbol, shares_sold)
                        self.Log(f"Call assigned: Sold {shares_sold} shares of {symbol} @ ${strike:.2f}")
                        
                        if self.stock_holding.shares == 0:
                            self.phase = "SP"
        
        # Remove expired positions
        for pos_id in positions_to_remove:
            del self.open_option_positions[pos_id]
    
    def UpdateMLModels(self):
        """Update ML models with accumulated data."""
        if self.ml_integration.should_retrain():
            self.Log("Retraining ML models...")
            # Would trigger retraining here
    
    def _estimate_iv(self, symbol: str, S: float, K: float, T: float) -> float:
        """Estimate implied volatility - now uses QC's IV from option chain."""
        # This method is kept for backward compatibility but should not be used
        # Prefer _find_option_by_greeks which uses QC's built-in IV
        return self._get_atm_iv_from_chain(symbol, S) or 0.25
    
    def _get_atm_iv_from_chain(self, symbol: str, underlying_price: float) -> Optional[float]:
        """Get ATM implied volatility from QC option chain.
        
        Args:
            symbol: Stock symbol
            underlying_price: Current underlying price
            
        Returns:
            ATM IV or None if not available
        """
        equity = self.equities.get(symbol)
        if not equity:
            return None
        
        option_chain = self.OptionChainProvider.GetOptionContractList(
            equity.Symbol, self.Time
        )
        
        if not option_chain:
            return None
        
        # Find option closest to ATM
        closest_iv = None
        min_strike_diff = float('inf')
        
        for option_symbol in option_chain:
            strike = option_symbol.ID.StrikePrice
            strike_diff = abs(strike - underlying_price)
            
            if strike_diff < min_strike_diff:
                security = self.Securities.get(option_symbol)
                if security and hasattr(security, 'ImpliedVolatility') and security.ImpliedVolatility:
                    closest_iv = security.ImpliedVolatility
                    min_strike_diff = strike_diff
        
        return closest_iv
    
    def _find_option_by_greeks(
        self,
        symbol: str,
        equity_symbol: Symbol,
        target_right: OptionRight,
        target_delta: float,
        dte_min: int,
        dte_max: int,
        delta_tolerance: float = 0.05
    ) -> Optional[Dict]:
        """Find suitable option using QC's built-in Greeks data.
        
        This method uses QuantConnect's native Greeks calculations instead of
        custom Black-Scholes implementation for better accuracy and real market data.
        
        Args:
            symbol: Stock symbol string
            equity_symbol: QC equity Symbol object
            target_right: OptionRight.Put or OptionRight.Call
            target_delta: Target delta value (negative for puts, positive for calls)
            dte_min: Minimum days to expiry
            dte_max: Maximum days to expiry
            delta_tolerance: Allowed deviation from target delta
            
        Returns:
            Dict with option details or None if not found
        """
        underlying_price = self.Securities[equity_symbol].Price
        
        option_chain = self.OptionChainProvider.GetOptionContractList(
            equity_symbol, self.Time
        )
        
        if not option_chain:
            self.Log(f"No option chain available for {symbol}")
            return None
        
        suitable_options = []
        
        for option_symbol in option_chain:
            # Filter by option type
            if option_symbol.ID.OptionRight != target_right:
                continue
            
            # Filter by DTE
            expiry = option_symbol.ID.Date
            dte = (expiry - self.Time).days
            if not (dte_min <= dte <= dte_max):
                continue
            
            strike = option_symbol.ID.StrikePrice
            
            # Get QC's built-in Greeks and IV
            security = self.Securities.get(option_symbol)
            if not security:
                continue
            
            # QC provides these Greeks based on actual market data
            delta = getattr(security, 'Delta', None)
            iv = getattr(security, 'ImpliedVolatility', None)
            
            # Skip if Greeks not available (may need warmup or data)
            if delta is None or iv is None:
                # Fallback: estimate delta from moneyness if QC data not ready
                # For OTM put: delta ~ -0.30 means strike is below price
                # For OTM call: delta ~ 0.30 means strike is above price
                if target_right == OptionRight.Put:
                    # Rough estimate: OTM put delta based on moneyness
                    moneyness = strike / underlying_price
                    if moneyness < 0.98:  # OTM put
                        delta = -max(0.15, min(0.40, (1 - moneyness) * 2))
                    else:
                        continue  # Skip ITM/ATM
                else:
                    # OTM call
                    moneyness = strike / underlying_price
                    if moneyness > 1.02:  # OTM call
                        delta = max(0.15, min(0.40, (moneyness - 1) * 2))
                    else:
                        continue  # Skip ITM/ATM
                
                # Use historical vol as IV fallback
                iv = self._calculate_historical_vol(symbol) or 0.25
            
            # Check if delta is within tolerance
            delta_diff = abs(delta - target_delta)
            if delta_diff <= delta_tolerance:
                # Get premium from bid/ask or last price
                bid = getattr(security, 'BidPrice', 0) or 0
                ask = getattr(security, 'AskPrice', 0) or 0
                
                if bid > 0 and ask > 0:
                    premium = (bid + ask) / 2  # Mid price
                else:
                    premium = security.Price or 0
                
                if premium <= 0:
                    continue
                
                suitable_options.append({
                    'option_symbol': option_symbol,
                    'strike': strike,
                    'expiry': expiry,
                    'dte': dte,
                    'delta': delta,
                    'iv': iv,
                    'premium': premium,
                    'delta_diff': delta_diff,
                    'bid': bid,
                    'ask': ask
                })
        
        if not suitable_options:
            self.Log(f"No options found for {symbol} {target_right} with delta {target_delta:.2f}±{delta_tolerance}")
            return None
        
        # Sort by delta difference (closest to target first)
        suitable_options.sort(key=lambda x: x['delta_diff'])
        
        best = suitable_options[0]
        
        self.Log(f"Found option: {symbol} {target_right} strike={best['strike']:.2f} "
                f"delta={best['delta']:.3f} (target={target_delta:.3f}) "
                f"IV={best['iv']:.1%} premium=${best['premium']:.2f} DTE={best['dte']}")
        
        return best
    
    def _calculate_historical_vol(self, symbol: str, window: int = 20) -> Optional[float]:
        """Calculate historical volatility from price history.
        
        Args:
            symbol: Stock symbol
            window: Lookback window in days
            
        Returns:
            Annualized historical volatility or None
        """
        bars = self.price_history.get(symbol, [])
        if len(bars) < window + 1:
            return None
        
        closes = np.array([b['close'] for b in bars[-(window+1):]])
        returns = np.diff(closes) / closes[:-1]
        
        if len(returns) < 2:
            return None
        
        # Annualize the volatility
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        return annual_vol if annual_vol > 0 else None
    
    def _calculate_iv_rank(self, symbol: str) -> float:
        """Calculate IV rank based on historical volatility percentile.
        
        IV Rank shows where current IV stands relative to its history.
        100 = highest IV in history, 0 = lowest.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            IV rank (0-100)
        """
        bars = self.price_history.get(symbol, [])
        if len(bars) < 60:
            return 50.0  # Default mid-range
        
        # Calculate rolling volatility for the past year
        window = 20
        vol_series = []
        closes = np.array([b['close'] for b in bars])
        
        for i in range(window, len(closes)):
            window_closes = closes[i-window:i+1]
            returns = np.diff(window_closes) / window_closes[:-1]
            vol = np.std(returns) * np.sqrt(252)  # Annualized
            vol_series.append(vol)
        
        if len(vol_series) < 10:
            return 50.0
        
        current_vol = vol_series[-1]
        min_vol = min(vol_series)
        max_vol = max(vol_series)
        
        if max_vol == min_vol:
            return 50.0
        
        iv_rank = (current_vol - min_vol) / (max_vol - min_vol) * 100
        return max(0, min(100, iv_rank))
    
    def _get_vix_percentile(self) -> float:
        """Calculate VIX percentile relative to its history.
        
        Returns:
            VIX percentile (0-100)
        """
        if len(self._vix_history) < 10:
            return 50.0
        
        current_vix = self._current_vix
        min_vix = min(self._vix_history)
        max_vix = max(self._vix_history)
        
        if max_vix == min_vix:
            return 50.0
        
        percentile = (current_vix - min_vix) / (max_vix - min_vix) * 100
        return max(0, min(100, percentile))
    
    def OnOrderEvent(self, orderEvent):
        """Handle order events."""
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(f"Order filled: {orderEvent.Symbol} @ ${orderEvent.FillPrice:.2f}")
    
    def OnEndOfAlgorithm(self):
        """Called at the end of the algorithm."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        self.Log("=" * 60)
        self.Log("BINBINGOD STRATEGY RESULTS (ML Enhanced)")
        self.Log("=" * 60)
        self.Log(f"Total Trades: {self.total_trades}")
        self.Log(f"Winning Trades: {self.winning_trades}")
        self.Log(f"Win Rate: {win_rate:.2f}%")
        self.Log(f"Total P&L: ${self.total_pnl:.2f}")
        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:.2f}")
        self.Log(f"Final Phase: {self.phase}")
        self.Log(f"Shares Held: {self.stock_holding.shares}")
        self.Log("")
        self.Log("--- ML Model Insights ---")
        self.Log(self.ml_integration.get_status_report())
        self.Log("=" * 60)