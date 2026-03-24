"""BinbinGod Strategy for QuantConnect - Wheel strategy with ML optimization."""
from AlgorithmImports import *
from strategy_mixin import (
    init_dates, init_parameters, init_ml, init_securities, init_state, schedule_events,
    rebalance, check_position_management, check_expired_options, update_ml_models, on_end_of_algorithm
)

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
        # Cache option chain data for use in scheduled events
        if not hasattr(self, '_cached_option_chains'):
            self._cached_option_chains = {}
        for symbol in self.stock_pool:
            if symbol in data.OptionChains:
                self._cached_option_chains[symbol] = data.OptionChains[symbol]

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
            self.Log(f"Filled: {orderEvent.Symbol} @ ${orderEvent.FillPrice:.2f}")

    def Rebalance(self):
        rebalance(self)

    def CheckExpiredOptions(self):
        check_expired_options(self)

    def UpdateMLModels(self):
        update_ml_models(self)

    def OnEndOfAlgorithm(self):
        on_end_of_algorithm(self)