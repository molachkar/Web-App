"""
market/__init__.py
Expose market data public API.
"""

from market.ranges import fetch_ranges

__all__ = ["fetch_ranges"]
