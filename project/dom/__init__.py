"""
dom/__init__.py
Expose DOM handler public API.
"""

from dom.handler import broadcast, tcp_server, get_dom_stats, clients, stats

__all__ = ["broadcast", "tcp_server", "get_dom_stats", "clients", "stats"]
