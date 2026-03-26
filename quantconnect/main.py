"""BinbinGod Strategy for QuantConnect - Wheel strategy with ML optimization."""
from AlgorithmImports import *
from strategy_init import init_dates, init_parameters, init_ml, init_securities, init_state, schedule_events
from strategy_mixin import rebalance, on_end_of_algorithm
from expiry import check_expired_options, update_ml_models

class BinbinGodStrategy(QCAlgorithm):
    """BinbinGod Strategy - Intelligent stock selection + Full Wheel logic + ML optimization."""

    def Initialize(self):
        """Initialize the strategy."""
        init_dates(self)
        init_parameters(self)
        init_ml(self)
        init_securities(self)
        init_state(self)
        schedule_events(self)
        self.Log(f"BinbinGod Strategy initialized with stock pool: {self.stock_pool}")
        self.Log(f"ML optimization enabled: {self.ml_enabled}")

    def OnData(self, data):
        """Handle incoming data."""
        for symbol in self.stock_pool:
            if symbol in data.Bars:
                bar = data.Bars[symbol]
                self.price_history[symbol].append({
                    'date': self.Time.strftime('%Y-%m-%d'),
                    'open': float(bar.Open), 'high': float(bar.High),
                    'low': float(bar.Low), 'close': float(bar.Close),
                    'volume': float(bar.Volume)})
                if len(self.price_history[symbol]) > 500:
                    self.price_history[symbol] = self.price_history[symbol][-500:]
        if "VIXY" in data.Bars:
            self._current_vix = float(data.Bars["VIXY"].Close) * 2
            self._vix_history.append(self._current_vix)
            if len(self._vix_history) > 252:
                self._vix_history = self._vix_history[-252:]

    def OnWarmupFinished(self):
        """Called when warmup finished."""
        from scoring import calculate_historical_vol
        if not self.ml_enabled or self._ml_pretrained:
            return
        for symbol in self.stock_pool:
            bars = self.price_history.get(symbol, [])
            if len(bars) >= 60:
                self.ml_integration.pretrain_models(symbol=symbol, historical_bars=bars, iv_estimate=calculate_historical_vol(bars))
        self._ml_pretrained = True
        self.Log("ML pretraining done")

    def OnOrderEvent(self, orderEvent):
        """Called on order event."""
        if orderEvent.Status == OrderStatus.Filled:
            symbol = orderEvent.Symbol
            qty = orderEvent.FillQuantity
            price = orderEvent.FillPrice
            # Log stock trades separately for debugging
            if symbol.SecurityType == SecurityType.Equity:
                action = "BUY" if qty > 0 else "SELL"
                self.Log(f"STOCK_{action}: {symbol} qty={qty} @ ${price:.2f}")
            else:
                self.Log(f"Filled: {symbol} @ ${price:.2f}")

    def OnMarginCallWarning(self):
        """Called when margin is getting low."""
        margin_remaining = self.Portfolio.MarginRemaining
        margin_used = self.Portfolio.TotalMarginUsed
        self.Log(f"MARGIN_WARNING: Remaining=${margin_remaining:.2f}, Used=${margin_used:.2f}")

    def OnMarginCall(self, requests):
        """Called when margin call occurs - QC wants to liquidate positions.
        
        We prioritize keeping stock and selling options instead.
        Returns list of orders to submit for margin call.
        """
        self.Log(f"MARGIN_CALL: {len(requests)} liquidation requests")
        
        # Log what QC wants to liquidate
        for req in requests:
            self.Log(f"  -> Liquidate: {req.Symbol} qty={req.Quantity}")
        
        # Separate stock vs option requests
        stock_requests = [r for r in requests if r.Symbol.SecurityType == SecurityType.Equity]
        option_requests = [r for r in requests if r.Symbol.SecurityType == SecurityType.Option]
        
        # If QC wants to sell stock, try to liquidate options first
        if stock_requests:
            self.Log(f"MARGIN_CALL: Preventing {len(stock_requests)} stock liquidations")
            # Only return option liquidations - try to keep stock
            return option_requests
        
        return requests

    def Rebalance(self):
        rebalance(self)

    def CheckExpiredOptions(self):
        check_expired_options(self)

    def UpdateMLModels(self):
        update_ml_models(self)

    def OnEndOfAlgorithm(self):
        on_end_of_algorithm(self)