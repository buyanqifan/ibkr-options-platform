"""Unified IBKR data client: wraps ib_insync calls with sync interface for Dash."""

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
from ib_insync import Stock, Option, Contract
from core.ibkr.connection import IBKRConnectionManager
from core.ibkr.event_bridge import AsyncEventBridge
from core.market_data.cache import DataCache
from utils.rate_limiter import RateLimiter
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("ibkr_data")


class IBKRDataClient:
    """High-level synchronous data client backed by ib_insync.

    All methods are safe to call from Dash callbacks (any thread).
    """

    def __init__(
        self,
        conn_mgr: IBKRConnectionManager,
        bridge: AsyncEventBridge,
        cache: DataCache,
    ):
        self._conn = conn_mgr
        self._bridge = bridge
        self._cache = cache
        self._limiter = RateLimiter(rate=settings.API_REQUESTS_PER_SECOND, burst=10)

    # ------------------------------------------------------------------
    # Contract helpers
    # ------------------------------------------------------------------

    def _stock(self, symbol: str, exchange: str = "SMART", currency: str = "USD") -> Stock:
        return Stock(symbol, exchange, currency)

    def _option(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Option:
        return Option(symbol, expiry, strike, right, exchange, currency=currency)

    def _qualify(self, contract: Contract) -> Contract:
        """Qualify a contract (resolve conId)."""
        results = self._bridge.run_coroutine(
            self._conn.ib.qualifyContractsAsync(contract), timeout=10
        )
        if results:
            return results[0]
        raise ValueError(f"Could not qualify contract: {contract}")

    # ------------------------------------------------------------------
    # Real-time Quotes
    # ------------------------------------------------------------------

    def get_realtime_quote(self, symbol: str) -> dict:
        """Get latest quote for a stock. Returns cached value if fresh."""
        cached = self._cache.get_quote(symbol)
        if cached:
            return cached

        self._limiter.acquire()
        contract = self._stock(symbol)
        ticker = self._bridge.run_coroutine(
            self._async_snapshot(contract), timeout=15
        )
        quote = {
            "symbol": symbol,
            "bid": ticker.bid if ticker.bid == ticker.bid else None,
            "ask": ticker.ask if ticker.ask == ticker.ask else None,
            "last": ticker.last if ticker.last == ticker.last else None,
            "volume": ticker.volume if ticker.volume == ticker.volume else 0,
            "high": ticker.high if ticker.high == ticker.high else None,
            "low": ticker.low if ticker.low == ticker.low else None,
            "close": ticker.close if ticker.close == ticker.close else None,
            "time": datetime.now().isoformat(),
        }
        self._cache.set_quote(symbol, quote)
        return quote

    async def _async_snapshot(self, contract):
        self._conn.ib.reqMktData(contract, "", True, False)
        await self._conn.ib.sleep(2)
        ticker = self._conn.ib.ticker(contract)
        self._conn.ib.cancelMktData(contract)
        return ticker

    # ------------------------------------------------------------------
    # Historical Data
    # ------------------------------------------------------------------

    def get_historical_bars(
        self,
        symbol: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
        what_to_show: str = "ADJUSTED_LAST",  # Changed from "TRADES" to use split-adjusted prices
        use_rth: bool = True,
    ) -> list[dict]:
        """Fetch historical OHLCV bars with split/dividend adjustments.
        
        Args:
            symbol: Stock symbol (e.g., 'NVDA')
            duration: Lookback period (e.g., '1 Y', '2 Y')
            bar_size: Bar interval (e.g., '1 day', '1 hour')
            what_to_show: Data type - uses ADJUSTED_LAST for split-adjusted prices
                         Other options: TRADES (raw), BID, ASK, etc.
            use_rth: Only regular trading hours
            
        Returns:
            List of OHLCV bars with adjusted prices
        """
        cached = self._cache.get_bars(symbol, bar_size)
        if cached:
            return cached

        self._limiter.acquire()
        contract = self._stock(symbol)
        bars = self._bridge.run_coroutine(
            self._async_historical(contract, duration, bar_size, what_to_show, use_rth),
            timeout=30,
        )
        result = [
            {
                "date": b.date.isoformat() if hasattr(b.date, "isoformat") else str(b.date),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "average": b.average,
                "barCount": b.barCount,
            }
            for b in bars
        ]
        self._cache.set_bars(symbol, bar_size, result)
        return result

    async def _async_historical(self, contract, duration, bar_size, what_to_show, use_rth):
        bars = await self._conn.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
        )
        return bars or []

    # ------------------------------------------------------------------
    # Options Chain
    # ------------------------------------------------------------------

    def get_option_chain_params(self, symbol: str) -> list[dict]:
        """Get available expirations and strikes for a symbol."""
        cache_key = f"opt_params:{symbol}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        self._limiter.acquire()
        contract = self._stock(symbol)
        contract = self._qualify(contract)
        chains = self._bridge.run_coroutine(
            self._async_option_params(contract), timeout=15
        )
        result = []
        for c in chains:
            if c.exchange == "SMART":
                result.append({
                    "exchange": c.exchange,
                    "underlyingConId": c.underlyingConId,
                    "tradingClass": c.tradingClass,
                    "multiplier": c.multiplier,
                    "expirations": sorted(c.expirations),
                    "strikes": sorted(c.strikes),
                })
        self._cache.set(cache_key, result, ttl=300)
        return result

    async def _async_option_params(self, contract):
        return await self._conn.ib.reqSecDefOptParamsAsync(
            contract.symbol, "", contract.secType, contract.conId
        )

    def get_option_chain(
        self,
        symbol: str,
        expiry: str,
        strikes: list[float] | None = None,
        right: str = "",
    ) -> list[dict]:
        """Fetch option chain data with Greeks for a given expiry."""
        cached = self._cache.get_options(symbol, expiry)
        if cached:
            return cached

        params = self.get_option_chain_params(symbol)
        if not params:
            return []

        available_strikes = params[0]["strikes"] if params else []
        if strikes is None:
            # Get quote to determine ATM, then pick nearby strikes
            quote = self.get_realtime_quote(symbol)
            price = quote.get("last") or quote.get("close") or 0
            if price > 0:
                strikes = [s for s in available_strikes if abs(s - price) / price < 0.15]
            else:
                strikes = available_strikes[:20]

        contracts = []
        for strike in strikes:
            rights = ["P", "C"] if not right else [right]
            for r in rights:
                contracts.append(self._option(symbol, expiry, strike, r))

        if not contracts:
            return []

        self._limiter.acquire()
        qualified = self._bridge.run_coroutine(
            self._conn.ib.qualifyContractsAsync(*contracts), timeout=30
        )
        qualified = [c for c in qualified if c.conId]

        # Request market data for all qualified options
        tickers = self._bridge.run_coroutine(
            self._async_option_tickers(qualified), timeout=30
        )

        result = []
        for ticker in tickers:
            c = ticker.contract
            result.append({
                "symbol": symbol,
                "expiry": c.lastTradeDateOrContractMonth,
                "strike": c.strike,
                "right": c.right,
                "bid": _safe_val(ticker.bid),
                "ask": _safe_val(ticker.ask),
                "last": _safe_val(ticker.last),
                "volume": _safe_val(ticker.volume, 0),
                "openInterest": _safe_val(getattr(ticker, "callOpenInterest" if c.right == "C" else "putOpenInterest", None), 0),
                "impliedVol": _safe_val(ticker.modelGreeks.impliedVol if ticker.modelGreeks else None),
                "delta": _safe_val(ticker.modelGreeks.delta if ticker.modelGreeks else None),
                "gamma": _safe_val(ticker.modelGreeks.gamma if ticker.modelGreeks else None),
                "theta": _safe_val(ticker.modelGreeks.theta if ticker.modelGreeks else None),
                "vega": _safe_val(ticker.modelGreeks.vega if ticker.modelGreeks else None),
            })
        self._cache.set_options(symbol, expiry, result)
        return result

    async def _async_option_tickers(self, contracts):
        tickers = []
        for c in contracts:
            self._conn.ib.reqMktData(c, "100,101,104,106", True, False)
        await self._conn.ib.sleep(3)
        for c in contracts:
            t = self._conn.ib.ticker(c)
            tickers.append(t)
            self._conn.ib.cancelMktData(c)
        return tickers

    # ------------------------------------------------------------------
    # Fundamental Data
    # ------------------------------------------------------------------

    def get_fundamentals(self, symbol: str) -> dict:
        """Fetch fundamental data (PE, market cap, revenue, etc.)."""
        cached = self._cache.get_fundamentals(symbol)
        if cached:
            return cached

        self._limiter.acquire()
        contract = self._stock(symbol)
        raw_xml = self._bridge.run_coroutine(
            self._async_fundamentals(contract), timeout=15
        )
        data = self._parse_fundamentals_xml(symbol, raw_xml)
        self._cache.set_fundamentals(symbol, data)
        return data

    async def _async_fundamentals(self, contract):
        try:
            xml_data = await self._conn.ib.reqFundamentalDataAsync(
                contract, "ReportsFinSummary"
            )
            return xml_data or ""
        except Exception as e:
            logger.warning("Fundamentals request failed: %s", e)
            return ""

    def _parse_fundamentals_xml(self, symbol: str, xml_str: str) -> dict:
        """Parse IBKR fundamental data XML into a dict."""
        data = {"symbol": symbol}
        if not xml_str:
            return data
        try:
            root = ET.fromstring(xml_str)
            # Try to extract common fields from FinancialSummary report
            for ratio in root.iter("Ratio"):
                field_name = ratio.get("FieldName", "")
                val = ratio.text
                if val is None:
                    continue
                try:
                    fval = float(val)
                except ValueError:
                    continue
                if field_name == "MKTCAP":
                    data["market_cap"] = fval * 1_000_000  # usually in millions
                elif field_name == "APENORM":
                    data["pe_ratio"] = fval
                elif field_name == "TTMEPSXCLX":
                    data["eps"] = fval
                elif field_name == "TTMREV":
                    data["revenue"] = fval * 1_000_000
                elif field_name == "TTMREVCHG":
                    data["revenue_growth"] = fval
                elif field_name == "TTMNPMGN":
                    data["profit_margin"] = fval
                elif field_name == "YIELD":
                    data["dividend_yield"] = fval
                elif field_name == "BETA":
                    data["beta"] = fval
                elif field_name == "NHIG":
                    data["week52_high"] = fval
                elif field_name == "NLOW":
                    data["week52_low"] = fval
                elif field_name == "SHARESOUT":
                    data["shares_outstanding"] = fval * 1_000_000
        except ET.ParseError:
            logger.warning("Failed to parse fundamentals XML for %s", symbol)
        return data

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_summary(self) -> dict:
        """Get account summary (NAV, cash, unrealized P&L)."""
        if not self._conn.is_connected:
            return {}
        try:
            values = self._bridge.run_coroutine(
                self._async_account_summary(), timeout=10
            )
            return values
        except Exception as e:
            logger.warning("Failed to get account summary: %s", e)
            return {}

    async def _async_account_summary(self):
        summary = self._conn.ib.accountSummary()
        result = {}
        for item in summary:
            if item.tag in ("NetLiquidation", "TotalCashValue", "UnrealizedPnL",
                            "RealizedPnL", "BuyingPower", "GrossPositionValue"):
                result[item.tag] = float(item.value) if item.value else 0.0
        if not result:
            # Trigger refresh
            self._conn.ib.reqAccountSummary()
            await self._conn.ib.sleep(2)
            summary = self._conn.ib.accountSummary()
            for item in summary:
                if item.tag in ("NetLiquidation", "TotalCashValue", "UnrealizedPnL",
                                "RealizedPnL", "BuyingPower", "GrossPositionValue"):
                    result[item.tag] = float(item.value) if item.value else 0.0
        return result

    def get_positions(self) -> list[dict]:
        """Get current portfolio positions."""
        if not self._conn.is_connected:
            return []
        try:
            positions = self._bridge.run_coroutine(
                self._async_positions(), timeout=10
            )
            return positions
        except Exception as e:
            logger.warning("Failed to get positions: %s", e)
            return []

    async def _async_positions(self):
        portfolio = self._conn.ib.portfolio()
        result = []
        for item in portfolio:
            c = item.contract
            result.append({
                "symbol": c.symbol,
                "secType": c.secType,
                "expiry": getattr(c, "lastTradeDateOrContractMonth", ""),
                "strike": getattr(c, "strike", 0),
                "right": getattr(c, "right", ""),
                "position": item.position,
                "avgCost": item.averageCost,
                "marketValue": item.marketValue,
                "unrealizedPNL": item.unrealizedPNL,
                "realizedPNL": item.realizedPNL,
            })
        return result


def _safe_val(v, default=None):
    """Return value if it's a valid number (not NaN), else default."""
    if v is None:
        return default
    try:
        if v != v:  # NaN check
            return default
    except (TypeError, ValueError):
        return default
    return v
