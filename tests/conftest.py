"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(scope='session')
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp('test_data')


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    import numpy as np
    
    np.random.seed(42)
    n = 100
    prices = 150 * np.cumprod(1 + np.random.normal(0, 0.02, n))
    
    return [
        {
            'date': f'2024-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}',
            'close': price,
            'high': price * 1.01,
            'low': price * 0.99,
        }
        for i, price in enumerate(prices)
    ]


@pytest.fixture
def sample_options_data():
    """Generate sample options data for testing."""
    import numpy as np
    
    strikes = np.arange(140, 161, 2.5)
    
    options = []
    for strike in strikes:
        options.append({
            'strike': strike,
            'right': 'P',
            'bid': max(0.01, 150 - strike) * 0.5,
            'ask': max(0.01, 150 - strike) * 0.55,
            'delta': -max(0.05, min(0.45, (150 - strike) / 50)),
            'iv': 0.25,
            'volume': 1000,
        })
        options.append({
            'strike': strike,
            'right': 'C',
            'bid': max(0.01, strike - 150) * 0.5,
            'ask': max(0.01, strike - 150) * 0.55,
            'delta': max(0.05, min(0.45, (strike - 150) / 50)),
            'iv': 0.25,
            'volume': 1000,
        })
    
    return options