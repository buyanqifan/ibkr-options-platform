"""
BinbinGod Strategy for QuantConnect
A dynamic stock selection Wheel strategy for MAG7 stocks with ML optimization.
"""

from AlgorithmImports import *
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

# Import helper classes
from helpers import StockScore, StockHolding

# Import scoring functions
from scoring import score_single_stock, calculate_historical_vol, calculate_iv_rank

# Import ML modules
from ml_integration import (
    BinbinGodMLIntegration, MLOptimizationConfig, StrategySignal, AdaptiveDeltaStrategy
)

# Import option utilities
from option_utils import (
    calculate_historical_vol, estimate_premium_approx, filter_option_by_itm_protection,
    estimate_delta_from_moneyness, build_option_result, get_premium_from_security,
    should_roll_position, calculate_dte
)

# Import signal utilities
from signals import (
    select_best_signal_with_memory, calculate_position_risk,
    get_cc_optimization_params, build_position_data, calculate_pnl_metrics
)

# MAG7 Universe
MAG7_STOCKS = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]


class BinbinGodStrategy(QCAlgorithm):
    """BinbinGod Strategy - Intelligent stock selection + Full Wheel logic + ML optimization."""
    
    def Initialize(self):
        """Initialize the strategy."""
        self._init_dates()
        self._init_parameters()
        self._init_ml()
        self._init_securities()
        self._init_state()
        self._schedule_events()
        
        self.Log(f"BinbinGod Strategy initialized with stock pool: {self.stock_pool}")
        self.Log(f"ML optimization enabled: {self.ml_enabled}")
    
    def _init_dates(self):
        """Initialize start and end dates."""
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
    
    def _init_parameters(self):
        """Initialize strategy parameters."""
        self.initial_capital = float(self.GetParameter("initial_capital", 100000))
        self.SetCash(self.initial_capital)
        
        self.max_positions = int(self.GetParameter("max_positions", 10))
        self.profit_target_pct = float(self.GetParameter("profit_target_pct", 50))
        self.stop_loss_pct = float(self.GetParameter("stop_loss_pct", 999999))
        
        # Risk management
        self.max_risk_per_trade = float(self.GetParameter("max_risk_per_trade", 0.02))
        self.max_leverage = float(self.GetParameter("max_leverage", 1.0))
        
        # Disable flags
        self._profit_target_disabled = self.profit_target_pct >= 999999
        self._stop_loss_disabled = self.stop_loss_pct >= 999999
        
        # CC+SP simultaneous mode
        self.allow_sp_in_cc_phase = bool(self.GetParameter("allow_sp_in_cc_phase", True))
        self.sp_in_cc_margin_threshold = float(self.GetParameter("sp_in_cc_margin_threshold", 0.5))
        self.sp_in_cc_max_positions = int(self.GetParameter("sp_in_cc_max_positions", 3))
        
        # CC optimization
        self.cc_optimization_enabled = bool(self.GetParameter("cc_optimization_enabled", True))
        self.cc_min_delta_cost = float(self.GetParameter("cc_min_delta_cost", 0.15))
        self.cc_cost_basis_threshold = float(self.GetParameter("cc_cost_basis_threshold", 0.05))
        self.cc_min_strike_premium = float(self.GetParameter("cc_min_strike_premium", 0.02))
        
        # Stock selection memory
        self._last_selected_stock = None
        self._selection_count = 0
        self._min_hold_cycles = 3
        self._last_stock_scores = {}
        
        # ML config
        self.ml_enabled = bool(self.GetParameter("ml_enabled", True))
        self.ml_exploration_rate = float(self.GetParameter("ml_exploration_rate", 0.1))
        self.ml_learning_rate = float(self.GetParameter("ml_learning_rate", 0.01))
        self.ml_adoption_rate = float(self.GetParameter("ml_adoption_rate", 0.5))
        self.ml_min_confidence = float(self.GetParameter("ml_min_confidence", 0.4))
        
        # Stock pool
        stock_pool_str = self.GetParameter("stock_pool", ",".join(MAG7_STOCKS))
        self.stock_pool = stock_pool_str.split(",")
        
        # Scoring weights
        self.weights = {"iv_rank": 0.35, "technical": 0.25, "momentum": 0.20, "pe_score": 0.20}
    
    def _init_ml(self):
        """Initialize ML integration."""
        ml_config = MLOptimizationConfig(
            ml_delta_enabled=self.ml_enabled,
            ml_dte_enabled=self.ml_enabled,
            ml_roll_enabled=self.ml_enabled,
            ml_position_enabled=self.ml_enabled,
            exploration_rate=self.ml_exploration_rate,
            learning_rate=self.ml_learning_rate,
        )
        self.ml_integration = BinbinGodMLIntegration(ml_config)
        self.adaptive_strategy = AdaptiveDeltaStrategy(
            ml_integration=self.ml_integration,
            adoption_rate=self.ml_adoption_rate,
            min_confidence=self.ml_min_confidence
        )
        self._ml_pretrained = False
    
    def _init_securities(self):
        """Initialize equity and option securities."""
        self.equities = {}
        self.options = {}
        self.price_history = {}
        
        for symbol in self.stock_pool:
            equity = self.AddEquity(symbol, Resolution.Daily)
            equity.SetDataNormalizationMode(DataNormalizationMode.Raw)
            self.equities[symbol] = equity
            self.price_history[symbol] = []
            
            option = self.AddOption(symbol, Resolution.Daily)
            option.SetFilter(-10, 10, timedelta(days=25), timedelta(days=50))
            self.options[symbol] = option
        
        # VIX proxy
        self.vix = self.AddEquity("VIXY", Resolution.Daily)
        self._current_vix = 20.0
        self._vix_history = []
    
    def _init_state(self):
        """Initialize strategy state."""
        self.phase = "SP"
        self.stock_holding = StockHolding()
        self.open_option_positions: Dict[str, Dict] = {}
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.trade_history = []
        self.ml_signals_history = []
        self.SetWarmUp(60)
    
    def _schedule_events(self):
        """Schedule strategy events."""
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.AfterMarketOpen("SPY", 30), self.Rebalance)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.AfterMarketClose("SPY", -10), self.CheckExpiredOptions)
        self.Schedule.On(self.DateRules.MonthEnd(), self.TimeRules.AfterMarketClose("SPY", -30), self.UpdateMLModels)
    
    def OnData(self, data):
        """Handle incoming data."""
        for symbol in self.stock_pool:
            if symbol in data.Bars:
                bar = data.Bars[symbol]
                self.price_history[symbol].append({
                    'date': self.Time.strftime('%Y-%m-%d'),
                    'open': float(bar.Open), 'high': float(bar.High),
                    'low': float(bar.Low), 'close': float(bar.Close),
                    'volume': float(bar.Volume)
                })
                if len(self.price_history[symbol]) > 500:
                    self.price_history[symbol] = self.price_history[symbol][-500:]
        
        if "VIXY" in data.Bars:
            self._current_vix = float(data.Bars["VIXY"].Close) * 2
            self._vix_history.append(self._current_vix)
            if len(self._vix_history) > 252:
                self._vix_history = self._vix_history[-252:]
    
    def OnWarmupFinished(self):
        """Pretrain ML models with warmup data."""
        if not self.ml_enabled or self._ml_pretrained:
            return
        self.Log("Starting ML model pretraining...")
        stats = self._pretrain_ml_models()
        self._ml_pretrained = True
        self.Log(f"ML pretraining completed: {stats}")
    
    def _pretrain_ml_models(self) -> Dict:
        """Pretrain ML models with historical data."""
        stats = {'symbols_trained': 0, 'total_simulations': 0, 'status': 'skipped'}
        if not self.ml_enabled:
            return stats
        
        for symbol in self.stock_pool:
            bars = self.price_history.get(symbol, [])
            if len(bars) < 60:
                continue
            
            iv_estimate = calculate_historical_vol(bars)
            symbol_stats = self.ml_integration.pretrain_models(symbol=symbol, historical_bars=bars, iv_estimate=iv_estimate)
            
            if symbol_stats.get('status') == 'success':
                stats['symbols_trained'] += 1
                stats['total_simulations'] += symbol_stats.get('put_simulations', 0) + symbol_stats.get('call_simulations', 0)
        
        stats['status'] = 'success' if stats['symbols_trained'] > 0 else 'no_data'
        return stats
    
    def Rebalance(self):
        """Main rebalancing logic."""
        if self.IsWarmingUp:
            return
        
        self._check_position_management()
        
        if len(self.open_option_positions) >= self.max_positions:
            return
        
        signals = self._generate_ml_signals()
        if not signals:
            return
        
        best_signal, self._last_selected_stock, self._selection_count, self._last_stock_scores = \
            select_best_signal_with_memory(
                signals, self._last_selected_stock, self._selection_count,
                self._min_hold_cycles, self._last_stock_scores
            )
        
        if best_signal and best_signal.confidence >= self.ml_min_confidence:
            self._execute_signal(best_signal)
    
    def _check_position_management(self):
        """Check open positions for roll/close opportunities."""
        for position_id, pos_info in list(self.open_option_positions.items()):
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
            
            pnl, pnl_pct = calculate_pnl_metrics(entry_price, current_price, pos_info['quantity'])
            premium_captured_pct = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0
            dte = calculate_dte(pos_info['expiry'], self.Time)
            
            # ML Roll check
            if self.ml_enabled and hasattr(self.ml_integration, 'roll_optimizer'):
                position_data = build_position_data(pos_info, current_price, pnl_pct, dte)
                equity = self.equities.get(pos_info['symbol'])
                underlying_price = self.Securities[equity.Symbol].Price if equity else 0
                market_data = {'price': underlying_price, 'iv': 0.25, 'vix': self._current_vix}
                
                try:
                    should_roll, roll_rec = self.ml_integration.roll_optimizer.should_roll(
                        position=position_data, market_data=market_data,
                        current_date=self.Time.strftime('%Y-%m-%d'),
                        min_confidence=self.ml_min_confidence
                    )
                    if should_roll and roll_rec.action in ["ROLL_FORWARD", "CLOSE_EARLY"]:
                        self._handle_roll_action(roll_rec, pos_info, position_id)
                        continue
                except Exception as e:
                    self.Log(f"ML Roll error: {e}")
            
            # Rule-based roll
            action, reasoning = should_roll_position(
                premium_captured_pct, dte, pnl_pct,
                self.profit_target_pct, self._profit_target_disabled,
                self.stop_loss_pct, self._stop_loss_disabled
            )
            
            if action == "ROLL":
                signal = StrategySignal(symbol=pos_info['symbol'], action="ROLL",
                    delta=abs(pos_info['delta_at_entry']), dte_min=30, dte_max=45,
                    num_contracts=abs(pos_info['quantity']), confidence=0.8, reasoning=reasoning)
                self._execute_roll(signal)
            elif action.startswith("CLOSE"):
                signal = StrategySignal(symbol=pos_info['symbol'], action="CLOSE",
                    delta=0, dte_min=0, dte_max=0, num_contracts=0, confidence=0.9, reasoning=reasoning)
                self._execute_close(signal)
    
    def _handle_roll_action(self, roll_rec, pos_info, position_id):
        """Handle ML roll recommendation."""
        if roll_rec.action == "ROLL_FORWARD":
            signal = StrategySignal(symbol=pos_info['symbol'], action="ROLL",
                delta=roll_rec.optimal_delta or abs(pos_info['delta_at_entry']),
                dte_min=roll_rec.optimal_dte or 30, dte_max=roll_rec.optimal_dte or 45,
                num_contracts=abs(pos_info['quantity']), confidence=roll_rec.confidence,
                reasoning=roll_rec.reasoning)
            self._execute_roll(signal)
        elif roll_rec.action == "CLOSE_EARLY":
            signal = StrategySignal(symbol=pos_info['symbol'], action="CLOSE",
                delta=0, dte_min=0, dte_max=0, num_contracts=0,
                confidence=roll_rec.confidence, reasoning=roll_rec.reasoning)
            self._execute_close(signal)
    
    def _generate_ml_signals(self) -> List[StrategySignal]:
        """Generate ML-optimized signals."""
        signals = []
        portfolio_state = self._get_portfolio_state()
        
        if self.phase == "SP":
            for symbol in self.stock_pool:
                signal = self._generate_signal_for_symbol(symbol, "SP", portfolio_state)
                if signal:
                    signals.append(signal)
        elif self.phase == "CC":
            for symbol in self.stock_holding.get_symbols():
                signal = self._generate_signal_for_symbol(symbol, "CC", portfolio_state)
                if signal:
                    signals.append(signal)
            
            if self.allow_sp_in_cc_phase:
                signals.extend(self._generate_cc_sp_signals(portfolio_state))
        
        return signals
    
    def _generate_signal_for_symbol(self, symbol: str, strategy_phase: str, portfolio_state: Dict) -> Optional[StrategySignal]:
        """Generate signal for a single symbol."""
        equity = self.equities.get(symbol)
        if not equity:
            return None
        
        underlying_price = self.Securities[equity.Symbol].Price
        if underlying_price <= 0:
            return None
        
        bars = self.price_history.get(symbol, [])
        if len(bars) < 20:
            return None
        
        cost_basis = self.stock_holding.holdings.get(symbol, {}).get("cost_basis", 0)
        current_position = self._get_current_position(symbol)
        
        traditional_delta = 0.30
        cc_min_strike = None
        
        # CC Optimization
        if strategy_phase == "CC" and self.cc_optimization_enabled and cost_basis > 0:
            adj_delta, cc_min_strike, log_msg = get_cc_optimization_params(
                cost_basis, underlying_price, self.cc_optimization_enabled,
                self.cc_min_delta_cost, self.cc_cost_basis_threshold, self.cc_min_strike_premium
            )
            if log_msg:
                self.Log(log_msg)
                traditional_delta = adj_delta
        
        signal = self.ml_integration.generate_signal(
            symbol=symbol, current_price=underlying_price, cost_basis=cost_basis,
            bars=bars, strategy_phase=strategy_phase,
            portfolio_state=portfolio_state, current_position=current_position
        )
        
        if signal and self.ml_enabled:
            right = "P" if strategy_phase in ("SP", "CC+SP") else "C"
            trad_delta = traditional_delta
            
            if right == "P":
                adaptive_delta, _ = self.adaptive_strategy.select_put_delta(
                    trad_delta, signal.delta, signal.delta_confidence, signal.reasoning)
            else:
                adaptive_delta, _ = self.adaptive_strategy.select_call_delta(
                    trad_delta, signal.delta, signal.delta_confidence, signal.reasoning)
            
            signal.delta = adaptive_delta
        
        if signal:
            score = score_single_stock(symbol, bars, underlying_price, self.weights)
            signal.ml_score_adjustment = (score.total_score - 50) / 100
            if cc_min_strike is not None:
                signal.min_strike = cc_min_strike
        
        return signal
    
    def _generate_cc_sp_signals(self, portfolio_state: Dict) -> List[StrategySignal]:
        """Generate SP signals during CC phase."""
        signals = []
        
        margin_util = self.Portfolio.TotalMarginUsed / self.Portfolio.TotalPortfolioValue
        if margin_util > self.sp_in_cc_margin_threshold:
            return signals
        
        sp_positions = sum(1 for p in self.open_option_positions.values() if p.get('right') == 'P')
        if sp_positions >= self.sp_in_cc_max_positions:
            return signals
        
        held = self.stock_holding.get_symbols()
        available = [s for s in self.stock_pool if s not in held] or self.stock_pool
        
        for symbol in available[:3]:
            signal = self._generate_signal_for_symbol(symbol, "CC+SP", portfolio_state)
            if signal and signal.confidence > 0.6:
                signals.append(signal)
        
        return signals
    
    def _get_portfolio_state(self) -> Dict:
        """Get current portfolio state."""
        positions = [{'symbol': str(h.Symbol), 'quantity': h.Quantity, 'market_value': h.HoldingsValue}
                     for h in self.Portfolio.Values if h.Invested]
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
        current = self.Portfolio.TotalPortfolioValue
        if current < self.initial_capital:
            return (self.initial_capital - current) / self.initial_capital * 100
        return 0.0
    
    def _get_current_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol."""
        for pos_info in self.open_option_positions.values():
            if pos_info.get('symbol') == symbol:
                return pos_info
        return None
    
    def _execute_signal(self, signal: StrategySignal):
        """Execute a trading signal."""
        if not signal or signal.action == "HOLD":
            return
        if signal.action == "ROLL":
            self._execute_roll(signal)
            return
        if signal.action == "CLOSE":
            self._execute_close(signal)
            return
        
        equity = self.equities.get(signal.symbol)
        if not equity:
            return
        
        underlying_price = self.Securities[equity.Symbol].Price
        target_right = OptionRight.Put if signal.action == "SELL_PUT" else OptionRight.Call
        target_delta = -signal.delta if target_right == OptionRight.Put else signal.delta
        min_strike = getattr(signal, 'min_strike', 0.0)
        
        selected = self._find_option_by_greeks(
            symbol=signal.symbol, equity_symbol=equity.Symbol,
            target_right=target_right, target_delta=target_delta,
            dte_min=signal.dte_min, dte_max=signal.dte_max,
            delta_tolerance=0.05, min_strike=min_strike if min_strike > 0 else None
        )
        
        if not selected:
            self.Log(f"No suitable options for {signal.symbol} delta ~{target_delta:.2f}")
            return
        
        quantity, risk_msg = calculate_position_risk(
            selected['premium'], signal.num_contracts, self.Portfolio.TotalPortfolioValue,
            self.max_risk_per_trade, self.max_leverage, self.Portfolio.TotalMarginUsed
        )
        
        if risk_msg.startswith("LEVERAGE"):
            self.Log(f"Max leverage exceeded: {risk_msg}")
            return
        if risk_msg.startswith("RISK"):
            self.Log(risk_msg)
        
        ticket = self.MarketOrder(selected['option_symbol'], quantity)
        
        if ticket.Status == OrderStatus.Filled:
            fill_price = ticket.AverageFillPrice or selected['premium']
            right_str = 'P' if target_right == OptionRight.Put else 'C'
            position_id = f"{signal.symbol}_{self.Time.strftime('%Y%m%d')}_{selected['strike']:.0f}_{right_str}"
            
            self.open_option_positions[position_id] = {
                'symbol': signal.symbol, 'option_symbol': selected['option_symbol'],
                'right': right_str, 'strike': selected['strike'], 'expiry': selected['expiry'],
                'entry_date': self.Time.strftime('%Y-%m-%d'), 'entry_price': fill_price,
                'quantity': quantity, 'delta_at_entry': selected['delta'],
                'iv_at_entry': selected['iv'], 'strategy_phase': self.phase, 'ml_signal': signal,
            }
            
            self.Log(f"Executed: {signal.action} {signal.num_contracts} {signal.symbol} @ ${fill_price:.2f}")
    
    def _execute_roll(self, signal: StrategySignal):
        """Execute a roll action."""
        existing = None
        for pos_id, pos_info in self.open_option_positions.items():
            if pos_info.get('symbol') == signal.symbol:
                existing = (pos_id, pos_info)
                break
        
        if not existing:
            return
        
        pos_id, pos_info = existing
        close_ticket = self.MarketOrder(pos_info['option_symbol'], -pos_info['quantity'])
        
        if close_ticket.Status != OrderStatus.Filled:
            return
        
        pnl, _ = calculate_pnl_metrics(pos_info['entry_price'], close_ticket.AverageFillPrice, pos_info['quantity'])
        self._record_trade(signal.symbol, pos_info['right'], pnl, "ROLL")
        del self.open_option_positions[pos_id]
        
        equity = self.equities.get(signal.symbol)
        if not equity:
            return
        
        target_right = OptionRight.Put if pos_info['right'] == 'P' else OptionRight.Call
        target_delta = -signal.delta if target_right == OptionRight.Put else signal.delta
        
        new_selected = self._find_option_by_greeks(
            symbol=signal.symbol, equity_symbol=equity.Symbol,
            target_right=target_right, target_delta=target_delta,
            dte_min=signal.dte_min, dte_max=signal.dte_max
        )
        
        if new_selected:
            new_qty = -signal.num_contracts
            new_ticket = self.MarketOrder(new_selected['option_symbol'], new_qty)
            
            if new_ticket.Status == OrderStatus.Filled:
                right_str = 'P' if target_right == OptionRight.Put else 'C'
                self.open_option_positions[f"{signal.symbol}_{self.Time.strftime('%Y%m%d')}_{new_selected['strike']:.0f}_{right_str}"] = {
                    'symbol': signal.symbol, 'option_symbol': new_selected['option_symbol'],
                    'right': right_str, 'strike': new_selected['strike'], 'expiry': new_selected['expiry'],
                    'entry_date': self.Time.strftime('%Y-%m-%d'), 'entry_price': new_ticket.AverageFillPrice,
                    'quantity': new_qty, 'delta_at_entry': new_selected['delta'],
                    'iv_at_entry': new_selected['iv'], 'strategy_phase': self.phase, 'ml_signal': signal,
                }
    
    def _execute_close(self, signal: StrategySignal):
        """Execute a close action."""
        for pos_id, pos_info in list(self.open_option_positions.items()):
            if pos_info.get('symbol') != signal.symbol:
                continue
            
            close_ticket = self.MarketOrder(pos_info['option_symbol'], -pos_info['quantity'])
            
            if close_ticket.Status == OrderStatus.Filled:
                pnl, _ = calculate_pnl_metrics(pos_info['entry_price'], close_ticket.AverageFillPrice, pos_info['quantity'])
                self._record_trade(signal.symbol, pos_info['right'], pnl, signal.reasoning or "SIGNAL_CLOSE")
                del self.open_option_positions[pos_id]
    
    def _record_trade(self, symbol: str, right: str, pnl: float, reason: str):
        """Record trade result."""
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        self.trade_history.append({
            "date": self.Time.strftime("%Y-%m-%d"), "symbol": symbol,
            "type": right, "pnl": pnl, "exit_reason": reason
        })
    
    def CheckExpiredOptions(self):
        """Check for expired options."""
        for pos_id, pos_info in list(self.open_option_positions.items()):
            security = self.Securities.get(pos_info['option_symbol'])
            if not security or not (security.IsDelisted or security.Price == 0):
                continue
            
            symbol = pos_info['symbol']
            pnl = security.Holdings.UnrealizedProfit if hasattr(security, 'Holdings') else 0
            
            self.ml_integration.update_performance({
                'symbol': symbol, 'delta': abs(pos_info['delta_at_entry']),
                'dte': calculate_dte(pos_info['expiry'], datetime.strptime(pos_info['entry_date'], '%Y-%m-%d')),
                'num_contracts': abs(pos_info['quantity']), 'pnl': pnl / 100,
                'assigned': False, 'bars': self.price_history.get(symbol, []),
                'cost_basis': self.stock_holding.holdings.get(symbol, {}).get('cost_basis', 0),
                'strategy_phase': pos_info['strategy_phase'],
            })
            
            self._record_trade(symbol, pos_info['right'], pnl, "EXPIRY")
            del self.open_option_positions[pos_id]
            
            # Handle assignment
            equity = self.equities.get(symbol)
            if equity:
                last_price = self.Securities[equity.Symbol].Price
                strike = pos_info['strike']
                
                if pos_info['right'] == "P" and last_price < strike:
                    shares = abs(pos_info['quantity']) * 100
                    self.stock_holding.add_shares(symbol, shares, strike)
                    self.stock_holding.add_premium(symbol, pos_info['entry_price'] * 100)
                    self.phase = "CC"
                    self.Log(f"Put assigned: {shares} {symbol} @ ${strike:.2f}")
                
                elif pos_info['right'] == "C" and last_price > strike:
                    shares = abs(pos_info['quantity']) * 100
                    cost = self.stock_holding.holdings.get(symbol, {}).get('cost_basis', strike)
                    self.total_pnl += (strike - cost) * shares
                    self.stock_holding.remove_shares(symbol, shares)
                    self.Log(f"Call assigned: {shares} {symbol} @ ${strike:.2f}")
                    if self.stock_holding.shares == 0:
                        self.phase = "SP"
    
    def UpdateMLModels(self):
        """Update ML models."""
        if self.ml_integration.should_retrain():
            self.Log("Retraining ML models...")
    
    def _find_option_by_greeks(self, symbol: str, equity_symbol: Symbol, target_right: OptionRight,
                                target_delta: float, dte_min: int, dte_max: int,
                                delta_tolerance: float = 0.05, min_strike: float = None) -> Optional[Dict]:
        """Find suitable option using QC's Greeks data."""
        underlying_price = self.Securities[equity_symbol].Price
        option_chain = self.OptionChainProvider.GetOptionContractList(equity_symbol, self.Time)
        
        if not option_chain:
            return None
        
        suitable = []
        
        for option_symbol in option_chain:
            if option_symbol.ID.OptionRight != target_right:
                continue
            
            dte = (option_symbol.ID.Date - self.Time).days
            if not (dte_min <= dte <= dte_max):
                continue
            
            strike = option_symbol.ID.StrikePrice
            
            if min_strike and strike < min_strike:
                continue
            
            if not filter_option_by_itm_protection(strike, underlying_price, target_right):
                continue
            
            security = self.Securities.get(option_symbol)
            if not security:
                continue
            
            delta = getattr(security, 'Delta', None)
            iv = getattr(security, 'ImpliedVolatility', None)
            
            if delta is None or iv is None:
                delta = estimate_delta_from_moneyness(strike, underlying_price, target_right)
                iv = calculate_historical_vol(self.price_history.get(symbol, []))
                if delta is None:
                    continue
            
            if abs(delta - target_delta) > delta_tolerance:
                continue
            
            premium = get_premium_from_security(security)
            if premium <= 0:
                continue
            
            suitable.append(build_option_result(
                option_symbol, strike, option_symbol.ID.Date, dte, delta, iv, premium,
                abs(delta - target_delta), getattr(security, 'BidPrice', 0) or 0,
                getattr(security, 'AskPrice', 0) or 0
            ))
        
        if not suitable:
            return None
        
        suitable.sort(key=lambda x: x['delta_diff'])
        return suitable[0]
    
    def OnOrderEvent(self, orderEvent):
        """Handle order events."""
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(f"Order filled: {orderEvent.Symbol} @ ${orderEvent.FillPrice:.2f}")
    
    def OnEndOfAlgorithm(self):
        """Final results."""
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
        self.Log(self.ml_integration.get_status_report())
        self.Log("=" * 60)