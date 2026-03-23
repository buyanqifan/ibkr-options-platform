"""BinbinGod Strategy for QuantConnect - Wheel strategy with ML optimization."""
from AlgorithmImports import *
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

from helpers import StockScore, StockHolding
from scoring import score_single_stock, calculate_historical_vol
from ml_integration import BinbinGodMLIntegration, MLOptimizationConfig, StrategySignal, AdaptiveDeltaStrategy
from option_utils import filter_option_by_itm_protection, estimate_delta_from_moneyness, get_premium_from_security, should_roll_position, calculate_dte
from signals import select_best_signal_with_memory, get_cc_optimization_params, build_position_data, calculate_pnl_metrics

MAG7_STOCKS = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]


class BinbinGodStrategy(QCAlgorithm):
    """BinbinGod Strategy - Intelligent stock selection + Full Wheel logic + ML optimization."""

    def _make_signal(self, symbol, action, delta=0, dte_min=30, dte_max=45, num_contracts=1, confidence=0.5, reasoning=""):
        """Helper to create StrategySignal with default values."""
        return StrategySignal(symbol=symbol, action=action, delta=delta, dte_min=dte_min, dte_max=dte_max,
            num_contracts=num_contracts, confidence=confidence, reasoning=reasoning,
            expected_premium=0.0, expected_return=0.0, expected_risk=0.0, assignment_probability=0.0)

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
            self.SetStartDate(2024, 1, 1)
        
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
        self.max_risk_per_trade = float(self.GetParameter("max_risk_per_trade", 0.02))
        self.max_leverage = float(self.GetParameter("max_leverage", 1.0))
        self._profit_target_disabled = self.profit_target_pct >= 999999
        self._stop_loss_disabled = self.stop_loss_pct >= 999999
        self.allow_sp_in_cc_phase = bool(self.GetParameter("allow_sp_in_cc_phase", True))
        self.sp_in_cc_margin_threshold = float(self.GetParameter("sp_in_cc_margin_threshold", 0.5))
        self.sp_in_cc_max_positions = int(self.GetParameter("sp_in_cc_max_positions", 3))
        self.cc_optimization_enabled = bool(self.GetParameter("cc_optimization_enabled", True))
        self.cc_min_delta_cost = float(self.GetParameter("cc_min_delta_cost", 0.15))
        self.cc_cost_basis_threshold = float(self.GetParameter("cc_cost_basis_threshold", 0.05))
        self.cc_min_strike_premium = float(self.GetParameter("cc_min_strike_premium", 0.02))
        self._last_selected_stock, self._selection_count, self._min_hold_cycles, self._last_stock_scores = None, 0, 3, {}
        self.ml_enabled = bool(self.GetParameter("ml_enabled", True))
        self.ml_exploration_rate = float(self.GetParameter("ml_exploration_rate", 0.1))
        self.ml_learning_rate = float(self.GetParameter("ml_learning_rate", 0.01))
        self.ml_adoption_rate = float(self.GetParameter("ml_adoption_rate", 0.5))
        self.ml_min_confidence = float(self.GetParameter("ml_min_confidence", 0.4))
        self.stock_pool = self.GetParameter("stock_pool", ",".join(MAG7_STOCKS)).split(",")
        self.weights = {"iv_rank": 0.35, "technical": 0.25, "momentum": 0.20, "pe_score": 0.20}
    
    def _init_ml(self):
        """Initialize ML integration."""
        self.ml_integration = BinbinGodMLIntegration(MLOptimizationConfig(
            ml_delta_enabled=self.ml_enabled,
            ml_dte_enabled=self.ml_enabled,
            ml_roll_enabled=self.ml_enabled,
            ml_position_enabled=self.ml_enabled,
            dte_min=30,
            dte_max=45,
            exploration_rate=self.ml_exploration_rate,
            learning_rate=self.ml_learning_rate
        ))
        self.adaptive_strategy = AdaptiveDeltaStrategy(self.ml_integration, self.ml_adoption_rate, self.ml_min_confidence)
        self._ml_pretrained = False

    def _init_securities(self):
        """Initialize equity and option securities."""
        self.equities, self.options, self.price_history = {}, {}, {}
        for symbol in self.stock_pool:
            e = self.AddEquity(symbol, Resolution.Daily)
            e.SetDataNormalizationMode(DataNormalizationMode.Raw)
            self.equities[symbol], self.price_history[symbol] = e, []
            o = self.AddOption(symbol, Resolution.Daily)
            o.SetFilter(-10, 10, timedelta(days=25), timedelta(days=50))
            self.options[symbol] = o
        self.vix = self.AddEquity("VIXY", Resolution.Daily)
        self._current_vix, self._vix_history = 20.0, []

    def _init_state(self):
        """Initialize strategy state."""
        self.phase, self.stock_holding = "SP", StockHolding()
        self.open_option_positions, self.trade_history, self.ml_signals_history = {}, [], []
        self.total_trades, self.winning_trades, self.total_pnl = 0, 0, 0.0
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
        if not self.ml_enabled or self._ml_pretrained:
            return
        for symbol in self.stock_pool:
            bars = self.price_history.get(symbol, [])
            if len(bars) >= 60:
                self.ml_integration.pretrain_models(symbol=symbol, historical_bars=bars, iv_estimate=calculate_historical_vol(bars))
        self._ml_pretrained = True
        self.Log("ML pretraining done")
        
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
                signal = self._make_signal(pos_info['symbol'], "ROLL",
                    delta=abs(pos_info['delta_at_entry']), dte_min=30, dte_max=45,
                    num_contracts=abs(pos_info['quantity']), confidence=0.8, reasoning=reasoning)
                self._execute_roll(signal)
            elif action.startswith("CLOSE"):
                signal = self._make_signal(pos_info['symbol'], "CLOSE", dte_min=0, dte_max=0, num_contracts=0, confidence=0.9, reasoning=reasoning)
                self._execute_close(signal)

    def _handle_roll_action(self, roll_rec, pos_info, position_id):
        """Handle ML roll recommendation."""
        if roll_rec.action == "ROLL_FORWARD":
            signal = self._make_signal(pos_info['symbol'], "ROLL",
                delta=roll_rec.optimal_delta or abs(pos_info['delta_at_entry']),
                dte_min=roll_rec.optimal_dte or 30, dte_max=roll_rec.optimal_dte or 45,
                num_contracts=abs(pos_info['quantity']), confidence=roll_rec.confidence, reasoning=roll_rec.reasoning)
            self._execute_roll(signal)
        elif roll_rec.action == "CLOSE_EARLY":
            signal = self._make_signal(pos_info['symbol'], "CLOSE", dte_min=0, dte_max=0, num_contracts=0, confidence=roll_rec.confidence, reasoning=roll_rec.reasoning)
            self._execute_close(signal)
    
    def _generate_ml_signals(self) -> List[StrategySignal]:
        """Generate ML-optimized signals."""
        signals = []
        portfolio_state = self._get_portfolio_state()

        if self.phase == "SP":
            # Skip symbols that already have Put positions
            symbols_with_put = {p.get('symbol') for p in self.open_option_positions.values() if p.get('right') == 'P'}
            for symbol in self.stock_pool:
                if symbol not in symbols_with_put:
                    sig = self._generate_signal_for_symbol(symbol, "SP", portfolio_state)
                    if sig:
                        signals.append(sig)
        elif self.phase == "CC":
            for symbol in self.stock_holding.get_symbols():
                sig = self._generate_signal_for_symbol(symbol, "CC", portfolio_state)
                if sig:
                    signals.append(sig)
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

        # Check margin utilization
        margin_util = self.Portfolio.TotalMarginUsed / self.Portfolio.TotalPortfolioValue
        if margin_util > self.sp_in_cc_margin_threshold:
            return signals

        # Check current SP positions count
        sp_positions = sum(1 for p in self.open_option_positions.values() if p.get('right') == 'P')
        if sp_positions >= self.sp_in_cc_max_positions:
            return signals

        # Check total positions
        if len(self.open_option_positions) >= self.max_positions:
            return signals

        # Get symbols that already have Put positions
        symbols_with_put = set()
        for pos_info in self.open_option_positions.values():
            if pos_info.get('right') == 'P':
                symbols_with_put.add(pos_info.get('symbol'))

        # Select stocks not currently held as shares (prefer diversification)
        held = self.stock_holding.get_symbols()
        available = [s for s in self.stock_pool if s not in held] or self.stock_pool

        # Also exclude stocks that already have Put positions
        available = [s for s in available if s not in symbols_with_put]

        # Select only ONE best stock for CC+SP (same as original)
        best_signal = None
        for symbol in available:
            signal = self._generate_signal_for_symbol(symbol, "CC+SP", portfolio_state)
            if signal and signal.confidence > 0.6:
                if best_signal is None or signal.confidence > best_signal.confidence:
                    best_signal = signal

        if best_signal:
            signals.append(best_signal)

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

        # Calculate position size differently for PUT vs CALL
        current_positions = len(self.open_option_positions)

        if target_right == OptionRight.Put:
            # PUT: Based on INITIAL capital (1 contract per $10k) - same as original
            max_by_capital = max(1, int(self.initial_capital / 10000))
            max_by_limit = self.max_positions - current_positions
            quantity = min(max_by_capital, max_by_limit, 10)
        else:
            # CALL: Based on shares held (1 contract per 100 shares)
            shares_held = self.stock_holding.get_shares(signal.symbol)

            # Check existing call coverage
            existing_call_contracts = 0
            for pos_info in self.open_option_positions.values():
                if pos_info.get('symbol') == signal.symbol and pos_info.get('right') == 'C':
                    existing_call_contracts += abs(pos_info.get('quantity', 0))

            shares_covered = existing_call_contracts * 100
            shares_available = shares_held - shares_covered
            quantity = min(max(0, shares_available // 100), self.max_positions)

            if quantity <= 0:
                self.Log(f"No available shares for {signal.symbol} call: held={shares_held}, covered={shares_covered}")
                return

        if quantity <= 0:
            return

        quantity = -quantity  # NEGATIVE for selling

        self.Log(f"Selling {abs(quantity)} {signal.symbol} {target_right} @ ${selected['premium']:.2f}")
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
            # Keep same position size as original
            new_qty = pos_info['quantity']  # Already negative
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
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(f"Filled: {orderEvent.Symbol} @ ${orderEvent.FillPrice:.2f}")

    def OnEndOfAlgorithm(self):
        wr = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        self.Log(f"Trades: {self.total_trades}, WinRate: {wr:.1f}%, PnL: ${self.total_pnl:.2f}")
        self.Log(f"Portfolio: ${self.Portfolio.TotalPortfolioValue:.2f}, Phase: {self.phase}")
        self.Log(self.ml_integration.get_status_report())