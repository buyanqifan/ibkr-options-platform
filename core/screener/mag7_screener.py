"""MAG7 Stock Screener for selecting optimal Wheel strategy candidates.

MAG7 Stocks:
- MSFT (Microsoft)
- AAPL (Apple)
- NVDA (NVIDIA)
- GOOGL (Alphabet/Google)
- AMZN (Amazon)
- META (Meta/Facebook)
- TSLA (Tesla)

Selection criteria:
1. P/E Ratio - Lower is better (value stocks preferred)
2. Option IV - Higher is better (more premium income)
3. Price momentum - Stable or upward trend preferred
4. Market cap - All are large cap, but relative comparison
5. Dividend yield - Bonus factor (some pay dividends)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class StockMetrics:
    """Fundamental and technical metrics for a stock."""
    symbol: str
    pe_ratio: float  # Price-to-Earnings ratio
    iv_rank: float   # Implied Volatility Rank (0-100%)
    iv_percentile: float  # IV Percentile (0-100%)
    price: float     # Current stock price
    market_cap: float  # Market capitalization
    dividend_yield: float  # Annual dividend yield (%)
    price_change_1m: float  # 1-month price change (%)
    price_change_3m: float  # 3-month price change (%)
    beta: float      # Beta (volatility vs market)
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "pe_ratio": round(self.pe_ratio, 2),
            "iv_rank": round(self.iv_rank, 2),
            "iv_percentile": round(self.iv_percentile, 2),
            "price": round(self.price, 2),
            "market_cap": f"${self.market_cap/1e9:.1f}B",
            "dividend_yield": round(self.dividend_yield, 2),
            "price_change_1m": round(self.price_change_1m, 2),
            "price_change_3m": round(self.price_change_3m, 2),
            "beta": round(self.beta, 2),
        }


@dataclass
class StockScore:
    """Calculated score for stock selection."""
    symbol: str
    total_score: float
    pe_score: float
    iv_score: float
    momentum_score: float
    stability_score: float
    metrics: StockMetrics
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "total_score": round(self.total_score, 2),
            "pe_score": round(self.pe_score, 2),
            "iv_score": round(self.iv_score, 2),
            "momentum_score": round(self.momentum_score, 2),
            "stability_score": round(self.stability_score, 2),
            "metrics": self.metrics.to_dict(),
        }


class MAG7Screener:
    """Screen and rank MAG7 stocks for Wheel strategy suitability."""
    
    # MAG7 universe
    MAG7_SYMBOLS = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
    
    # Default metrics (will be updated with real data when available)
    DEFAULT_METRICS = {
        "MSFT": {"pe_ratio": 35.0, "iv_rank": 30.0, "iv_percentile": 35.0, "price": 500.0, 
                 "market_cap": 3.7e12, "dividend_yield": 0.75, "price_change_1m": 2.0, 
                 "price_change_3m": 5.0, "beta": 0.9},
        "AAPL": {"pe_ratio": 30.0, "iv_rank": 25.0, "iv_percentile": 30.0, "price": 220.0,
                 "market_cap": 3.4e12, "dividend_yield": 0.50, "price_change_1m": 1.5,
                 "price_change_3m": 3.0, "beta": 1.0},
        "NVDA": {"pe_ratio": 65.0, "iv_rank": 60.0, "iv_percentile": 65.0, "price": 800.0,
                 "market_cap": 2.0e12, "dividend_yield": 0.03, "price_change_1m": 8.0,
                 "price_change_3m": 15.0, "beta": 1.7},
        "GOOGL": {"pe_ratio": 25.0, "iv_rank": 28.0, "iv_percentile": 32.0, "price": 200.0,
                  "market_cap": 2.5e12, "dividend_yield": 0.0, "price_change_1m": 3.0,
                  "price_change_3m": 6.0, "beta": 1.1},
        "AMZN": {"pe_ratio": 55.0, "iv_rank": 35.0, "iv_percentile": 40.0, "price": 220.0,
                 "market_cap": 2.3e12, "dividend_yield": 0.0, "price_change_1m": 4.0,
                 "price_change_3m": 8.0, "beta": 1.3},
        "META": {"pe_ratio": 28.0, "iv_rank": 40.0, "iv_percentile": 45.0, "price": 600.0,
                 "market_cap": 1.5e12, "dividend_yield": 0.0, "price_change_1m": 5.0,
                 "price_change_3m": 10.0, "beta": 1.2},
        "TSLA": {"pe_ratio": 70.0, "iv_rank": 75.0, "iv_percentile": 80.0, "price": 350.0,
                 "market_cap": 1.1e12, "dividend_yield": 0.0, "price_change_1m": -5.0,
                 "price_change_3m": 10.0, "beta": 2.0},
    }
    
    def __init__(self, data_client=None):
        """Initialize screener.
        
        Args:
            data_client: IBKR data client for fetching real-time metrics
        """
        self._client = data_client
    
    def get_stock_metrics(self, symbol: str) -> StockMetrics:
        """Get metrics for a single stock.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            StockMetrics object
        """
        # TODO: Fetch real data from IBKR when available
        # For now, use default values with some randomization for demo
        import numpy as np
        
        base_metrics = self.DEFAULT_METRICS.get(symbol.upper(), self.DEFAULT_METRICS["MSFT"])
        
        # Add small random variation for realism
        metrics = base_metrics.copy()
        for key in metrics:
            if isinstance(metrics[key], (int, float)):
                noise = np.random.normal(0, 0.02)  # 2% noise
                metrics[key] *= (1 + noise)
        
        return StockMetrics(symbol=symbol.upper(), **metrics)
    
    def calculate_pe_score(self, pe_ratio: float) -> float:
        """Calculate P/E score (lower PE = higher score).
        
        Score range: 0-100
        PE < 20: 100 points
        PE > 70: 20 points
        Linear interpolation in between
        """
        if pe_ratio <= 0:
            return 50.0  # No PE data
        
        # Clamp to reasonable range
        pe_clamped = max(10, min(pe_ratio, 80))
        
        # Inverse relationship: lower PE = higher score
        score = 100 - (pe_clamped - 10) * (80 / 70)
        return max(20, min(100, score))
    
    def calculate_iv_score(self, iv_rank: float, iv_percentile: float) -> float:
        """Calculate IV score (higher IV = more premium income).
        
        Score range: 0-100
        Weight: 60% IV Rank, 40% IV Percentile
        """
        iv_score = 0.6 * iv_rank + 0.4 * iv_percentile
        return max(0, min(100, iv_score))
    
    def calculate_momentum_score(self, price_change_1m: float, price_change_3m: float) -> float:
        """Calculate momentum score (positive momentum preferred).
        
        Score range: 0-100
        Ideal: Moderate positive momentum (not too hot, not too cold)
        """
        # Weight recent performance more
        momentum = 0.6 * price_change_1m + 0.4 * price_change_3m
        
        # Score based on momentum
        # -10% or worse: 20 points
        # +10% or better: 80 points
        # In between: linear
        if momentum <= -10:
            return 20.0
        elif momentum >= 10:
            return 80.0
        else:
            return 50 + momentum * 3
    
    def calculate_stability_score(self, beta: float, dividend_yield: float) -> float:
        """Calculate stability score (lower beta, higher dividend = more stable).
        
        Score range: 0-100
        Beta: 60% weight, Dividend: 40% weight
        """
        # Beta score (lower is better, ideal around 0.8-1.2)
        if beta <= 0:
            beta_score = 50.0
        elif beta < 1.0:
            beta_score = 80 + (1.0 - beta) * 20  # 80-100 for low beta
        elif beta < 1.5:
            beta_score = 70 - (beta - 1.0) * 40  # 70-50 for moderate beta
        else:
            beta_score = max(30, 70 - beta * 20)  # 30-50 for high beta
        
        # Dividend score (higher is better)
        div_score = min(100, dividend_yield * 20)  # 5% yield = 100 points
        
        # Combined score
        stability = 0.6 * beta_score + 0.4 * div_score
        return max(0, min(100, stability))
    
    def calculate_total_score(self, metrics: StockMetrics, 
                             weights: Dict[str, float] = None) -> StockScore:
        """Calculate total weighted score for a stock.
        
        Args:
            metrics: StockMetrics object
            weights: Custom weights for scoring categories
                    Keys: 'pe', 'iv', 'momentum', 'stability'
                    
        Returns:
            StockScore object with detailed breakdown
        """
        # Default weights (optimized for Wheel strategy)
        if weights is None:
            weights = {
                "pe": 0.20,          # 20% - Value matters
                "iv": 0.40,          # 40% - IV is crucial for premium
                "momentum": 0.20,    # 20% - Trend direction
                "stability": 0.20,   # 20% - Risk management
            }
        
        # Calculate individual scores
        pe_score = self.calculate_pe_score(metrics.pe_ratio)
        iv_score = self.calculate_iv_score(metrics.iv_rank, metrics.iv_percentile)
        momentum_score = self.calculate_momentum_score(
            metrics.price_change_1m, metrics.price_change_3m
        )
        stability_score = self.calculate_stability_score(
            metrics.beta, metrics.dividend_yield
        )
        
        # Total weighted score
        total_score = (
            weights["pe"] * pe_score +
            weights["iv"] * iv_score +
            weights["momentum"] * momentum_score +
            weights["stability"] * stability_score
        )
        
        return StockScore(
            symbol=metrics.symbol,
            total_score=total_score,
            pe_score=pe_score,
            iv_score=iv_score,
            momentum_score=momentum_score,
            stability_score=stability_score,
            metrics=metrics,
        )
    
    def screen_and_rank(self, weights: Dict[str, float] = None) -> List[StockScore]:
        """Screen all MAG7 stocks and rank by total score.
        
        Args:
            weights: Custom scoring weights (optional)
            
        Returns:
            List of StockScore objects, sorted by total_score descending
        """
        scores = []
        for symbol in self.MAG7_SYMBOLS:
            try:
                metrics = self.get_stock_metrics(symbol)
                score = self.calculate_total_score(metrics, weights)
                scores.append(score)
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by total score (descending)
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores
    
    def get_best_pick(self, weights: Dict[str, float] = None) -> Optional[StockScore]:
        """Get the highest-ranked stock for Wheel strategy.
        
        Args:
            weights: Custom scoring weights (optional)
            
        Returns:
            Best StockScore object or None if error
        """
        ranked = self.screen_and_rank(weights)
        return ranked[0] if ranked else None
    
    def get_analysis_report(self) -> dict:
        """Generate comprehensive analysis report for all MAG7 stocks.
        
        Returns:
            Dictionary with detailed analysis
        """
        ranked = self.screen_and_rank()
        
        report = {
            "analysis_date": np.datetime64('now').astype(str),
            "universe": self.MAG7_SYMBOLS,
            "ranked_stocks": [score.to_dict() for score in ranked],
            "best_pick": ranked[0].to_dict() if ranked else None,
            "worst_pick": ranked[-1].to_dict() if ranked else None,
            "summary": {
                "avg_pe": np.mean([s.metrics.pe_ratio for s in ranked]),
                "avg_iv_rank": np.mean([s.metrics.iv_rank for s in ranked]),
                "avg_total_score": np.mean([s.total_score for s in ranked]),
            }
        }
        
        return report
