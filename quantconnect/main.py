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

# Import option pricing module
from option_pricing import BlackScholes, OptionsPricer

# Import ML modules
from ml_integration import (
    BinbinGodMLIntegration, 
    MLOptimizationConfig, 
    StrategySignal
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
        self.stop_loss_pct = float(self.GetParameter("stop_loss_pct", 200))
        
        # CC+SP simultaneous mode parameters
        self.allow_sp_in_cc_phase = bool(self.GetParameter("allow_sp_in_cc_phase", True))
        self.sp_in_cc_margin_threshold = float(self.GetParameter("sp_in_cc_margin_threshold", 0.5))
        self.sp_in_cc_max_positions = int(self.GetParameter("sp_in_cc_max_positions", 3))
        
        # ML configuration
        self.ml_enabled = bool(self.GetParameter("ml_enabled", True))
        self.ml_exploration_rate = float(self.GetParameter("ml_exploration_rate", 0.1))
        self.ml_learning_rate = float(self.GetParameter("ml_learning_rate", 0.01))
        
        # Stock pool
        stock_pool_str = self.GetParameter("stock_pool", ",".join(MAG7_STOCKS))
        self.stock_pool = stock_pool_str.split(",")
        
        # Scoring weights
        self.weights = {
            "iv_rank": 0.30,
            "technical": 0.25,
            "momentum": 0.20,
            "pe_score": 0.15,
            "ml_adjustment": 0.10,
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
    
    def Rebalance(self):
        """Main rebalancing logic - called daily."""
        if self.IsWarmingUp:
            return
        
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
        
        # Execute best signal
        best_signal = self._select_best_signal(signals)
        self._execute_signal(best_signal)
    
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
        """Generate ML-optimized signal for a single symbol."""
        
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
        """Execute a trading signal."""
        
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
        
        # Get equity and option chain
        equity = self.equities.get(signal.symbol)
        if not equity:
            return
        
        underlying_price = self.Securities[equity.Symbol].Price
        
        option_chain = self.OptionChainProvider.GetOptionContractList(
            equity.Symbol, self.Time
        )
        
        if not option_chain:
            self.Log(f"No option chain for {signal.symbol}")
            return
        
        # Find suitable option
        target_right = OptionRight.Put if signal.action == "SELL_PUT" else OptionRight.Call
        
        suitable_options = []
        for option_symbol in option_chain:
            # Get option properties from Symbol ID
            option_right = option_symbol.ID.OptionRight
            if option_right != target_right:
                continue
            
            strike = option_symbol.ID.StrikePrice
            expiry = option_symbol.ID.Date
            
            dte = (expiry - self.Time).days
            if not (signal.dte_min <= dte <= signal.dte_max):
                continue
            
            # Calculate delta
            T = dte / 365.0
            iv = self._estimate_iv(signal.symbol, underlying_price, strike, T)
            delta = BlackScholes.delta(
                underlying_price, strike, T, 0.05, iv, 
                "P" if target_right == OptionRight.Put else "C"
            )
            
            if signal.action == "SELL_PUT":
                if -signal.delta * 1.2 <= delta <= -signal.delta * 0.8:
                    premium = BlackScholes.put_price(
                        underlying_price, strike, T, 0.05, iv
                    )
                    suitable_options.append((option_symbol, strike, expiry, delta, premium, iv))
            else:
                if signal.delta * 0.8 <= delta <= signal.delta * 1.2:
                    premium = BlackScholes.call_price(
                        underlying_price, strike, T, 0.05, iv
                    )
                    suitable_options.append((option_symbol, strike, expiry, delta, premium, iv))
        
        if not suitable_options:
            self.Log(f"No suitable options found for {signal.symbol}")
            return
        
        # Select best option (closest to target delta)
        target_delta = signal.delta if target_right == OptionRight.Call else -signal.delta
        suitable_options.sort(key=lambda x: abs(x[3] - target_delta))
        selected = suitable_options[0]
        option_symbol, strike, expiry, delta, premium, iv = selected
        
        # Position sizing from signal
        quantity = -signal.num_contracts
        
        # Execute order
        ticket = self.MarketOrder(option_symbol, quantity)
        
        if ticket.Status == OrderStatus.Filled:
            right_str = 'P' if target_right == OptionRight.Put else 'C'
            position_id = f"{signal.symbol}_{self.Time.strftime('%Y%m%d')}_{strike}_{right_str}"
            
            self.open_option_positions[position_id] = {
                'symbol': signal.symbol,
                'option_symbol': option_symbol,
                'right': right_str,
                'strike': strike,
                'expiry': expiry,
                'entry_date': self.Time.strftime('%Y-%m-%d'),
                'entry_price': premium,
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
                    f"@ ${premium:.2f}, strike={strike:.2f}, delta={delta:.3f}")
            self.Log(f"ML Reasoning: {signal.reasoning}")
    
    def _execute_roll(self, signal: StrategySignal):
        """Execute a roll action."""
        self.Log(f"Roll action for {signal.symbol}: {signal.reasoning}")
        # Roll logic would close existing position and open new one
    
    def _execute_close(self, signal: StrategySignal):
        """Execute a close action."""
        self.Log(f"Close action for {signal.symbol}: {signal.reasoning}")
        # Close logic would close existing position
    
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
        """Estimate implied volatility."""
        # Use ML volatility model
        bars = self.price_history.get(symbol, [])
        if bars:
            prediction = self.ml_integration.volatility_model.predict(bars)
            return prediction.iv_estimate
        return 0.25
    
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