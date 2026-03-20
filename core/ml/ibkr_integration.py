"""IBKR data integration for ML features.

Provides real-time market data from IBKR including:
- Live IV Rank from options chain
- Real-time Greeks
- Market data snapshots
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class IBKRDataIntegration:
    """Integration layer for fetching real-time data from IBKR."""
    
    def __init__(self, data_client=None):
        """Initialize IBKR data integration.
        
        Args:
            data_client: IBKR data client instance
        """
        self.data_client = data_client
    
    def get_iv_rank_from_chain(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str
    ) -> Optional[float]:
        """Fetch IV Rank from IBKR options chain.
        
        Args:
            symbol: Underlying symbol
            expiry: Option expiry (YYYYMMDD)
            strike: Strike price
            right: 'P' or 'C'
            
        Returns:
            IV Rank (0-100) or None if not available
        """
        if not self.data_client:
            return None
        
        try:
            # Fetch options chain data from IBKR
            # This would use the actual IBKR API call
            chain_data = self.data_client.get_options_chain(symbol)
            
            if not chain_data:
                return None
            
            # Extract IV data for this strike/expiry
            option_data = self._find_option_in_chain(
                chain_data, expiry, strike, right
            )
            
            if not option_data:
                return None
            
            current_iv = option_data.get('implied_volatility', 0)
            iv_52w_high = option_data.get('iv_52w_high', current_iv * 1.5)
            iv_52w_low = option_data.get('iv_52w_low', current_iv * 0.5)
            
            # Calculate IV Rank
            if iv_52w_high == iv_52w_low:
                return 50.0
            
            iv_rank = (current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100
            return max(0.0, min(100.0, iv_rank))
            
        except Exception as e:
            logger.error(f"Error fetching IV Rank from IBKR: {e}")
            return None
    
    def get_real_time_greeks(
        self,
        symbol: str,
        strike: float,
        expiry: str,
        right: str
    ) -> Optional[Dict[str, float]]:
        """Fetch real-time Greeks from IBKR.
        
        Args:
            symbol: Underlying symbol
            strike: Strike price
            expiry: Option expiry (YYYYMMDD)
            right: 'P' or 'C'
            
        Returns:
            Dictionary with delta, gamma, vega, theta or None
        """
        if not self.data_client:
            return None
        
        try:
            # Fetch option market data
            option_data = self.data_client.get_option_market_data(
                symbol, strike, expiry, right
            )
            
            if not option_data:
                return None
            
            return {
                'delta': option_data.get('delta', 0),
                'gamma': option_data.get('gamma', 0),
                'vega': option_data.get('vega', 0),
                'theta': option_data.get('theta', 0),
            }
            
        except Exception as e:
            logger.error(f"Error fetching Greeks from IBKR: {e}")
            return None
    
    def get_market_data_snapshot(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Get complete market data snapshot from IBKR.
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            Complete market data dictionary
        """
        if not self.data_client:
            return None
        
        try:
            # Fetch underlying data
            underlying_data = self.data_client.get_stock_data(symbol)
            
            if not underlying_data:
                return None
            
            # Fetch options chain for IV data
            chain_data = self.data_client.get_options_chain(symbol)
            
            # Calculate IV Rank from chain
            iv_data = self._calculate_iv_from_chain(chain_data)
            
            # Build market snapshot
            snapshot = {
                'price': underlying_data.get('close', 0),
                'historical_volatility': self._calculate_hv(underlying_data),
                'iv_rank': iv_data.get('iv_rank', 50),
                'iv_percentile': iv_data.get('iv_percentile', 50),
                'current_iv': iv_data.get('current_iv', 0.3),
                'momentum_5d': self._calculate_momentum(underlying_data, 5),
                'momentum_10d': self._calculate_momentum(underlying_data, 10),
                'vs_ma20': self._calculate_vs_ma(underlying_data, 20),
                'vs_ma50': self._calculate_vs_ma(underlying_data, 50),
                'market_regime': self._determine_regime(iv_data.get('iv_rank', 50)),
                'risk_free_rate': 0.05,
            }
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error fetching market snapshot from IBKR: {e}")
            return None
    
    def _find_option_in_chain(
        self,
        chain_data: Dict,
        expiry: str,
        strike: float,
        right: str
    ) -> Optional[Dict]:
        """Find specific option in options chain."""
        if not chain_data:
            return None
        
        for option in chain_data.get('options', []):
            if (option.get('expiry') == expiry and
                option.get('strike') == strike and
                option.get('right') == right):
                return option
        
        return None
    
    def _calculate_iv_from_chain(
        self,
        chain_data: Dict
    ) -> Dict[str, float]:
        """Calculate IV metrics from options chain."""
        if not chain_data:
            return {
                'current_iv': 0.3,
                'iv_rank': 50,
                'iv_percentile': 50,
            }
        
        # Extract all IVs from chain
        all_ivs = [
            opt.get('implied_volatility', 0)
            for opt in chain_data.get('options', [])
            if opt.get('implied_volatility', 0) > 0
        ]
        
        if not all_ivs:
            return {
                'current_iv': 0.3,
                'iv_rank': 50,
                'iv_percentile': 50,
            }
        
        current_iv = sum(all_ivs) / len(all_ivs)
        iv_52w_high = max(all_ivs) * 1.2  # Estimate
        iv_52w_low = min(all_ivs) * 0.8   # Estimate
        
        # Calculate IV Rank
        if iv_52w_high == iv_52w_low:
            iv_rank = 50.0
        else:
            iv_rank = (current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100
        
        # Calculate IV Percentile
        days_lower = sum(1 for iv in all_ivs if iv < current_iv)
        iv_percentile = days_lower / len(all_ivs) * 100
        
        return {
            'current_iv': current_iv,
            'iv_rank': max(0, min(100, iv_rank)),
            'iv_percentile': iv_percentile,
            'iv_52w_high': iv_52w_high,
            'iv_52w_low': iv_52w_low,
        }
    
    def _calculate_hv(self, underlying_data: Dict) -> float:
        """Calculate historical volatility from price data."""
        prices = underlying_data.get('price_history', [])
        if len(prices) < 21:
            return 30.0  # Default
        
        from core.ml.market_data import MarketDataCalculator
        return MarketDataCalculator.calculate_historical_volatility(prices)
    
    def _calculate_momentum(
        self,
        underlying_data: Dict,
        days: int
    ) -> float:
        """Calculate price momentum."""
        prices = underlying_data.get('price_history', [])
        if len(prices) < days + 1:
            return 0.0
        
        current = prices[-1]
        past = prices[-days - 1]
        
        return (current - past) / past * 100
    
    def _calculate_vs_ma(
        self,
        underlying_data: Dict,
        window: int
    ) -> float:
        """Calculate price vs moving average."""
        import numpy as np
        
        prices = underlying_data.get('price_history', [])
        if len(prices) < window:
            return 0.0
        
        ma = np.mean(prices[-window:])
        current = prices[-1]
        
        return (current - ma) / ma * 100
    
    def _determine_regime(self, iv_rank: float) -> int:
        """Determine market regime based on IV Rank."""
        if iv_rank < 20:
            return 0  # Low vol
        elif iv_rank > 50:
            return 2  # High vol
        else:
            return 1  # Normal
