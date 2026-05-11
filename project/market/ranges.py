"""
market/ranges.py
Compute intraday and weekly price ranges for XAUUSD.

Public API:
    fetch_ranges() → dict
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

log = logging.getLogger("sentinel.market")


def fetch_ranges() -> dict:
    """
    Fetch current XAUUSD price and compute:
      - Intraday range (today's high - low)
      - Weekly range (week's high - low)
    
    Returns dict with:
        current_price: float
        intraday_high: float
        intraday_low: float
        intraday_range: float
        intraday_range_pct: float
        weekly_high: float
        weekly_low: float
        weekly_range: float
        weekly_range_pct: float
    """
    try:
        # Fetch 5-day data to ensure we have this week + some context
        end = datetime.utcnow()
        start = end - timedelta(days=7)
        
        raw = yf.download("GC=F", start=start, end=end,
                          interval="1m", auto_adjust=False, progress=False)
        
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        
        if raw.empty:
            log.warning("Ranges: No data available from yfinance")
            return {
                "current_price": None,
                "intraday_high": None,
                "intraday_low": None,
                "intraday_range": None,
                "intraday_range_pct": None,
                "weekly_high": None,
                "weekly_low": None,
                "weekly_range": None,
                "weekly_range_pct": None,
                "error": "No data available",
            }
        
        current_price = float(raw["Close"].iloc[-1])
        
        # Intraday: today's candles
        today = end.date()
        today_mask = raw.index.date == today
        today_data = raw[today_mask]
        
        if len(today_data) > 0:
            intraday_high = float(today_data["High"].max())
            intraday_low = float(today_data["Low"].min())
            intraday_range = intraday_high - intraday_low
            intraday_range_pct = (intraday_range / intraday_low) * 100 if intraday_low else 0
        else:
            # Use last available day if today has no data
            intraday_high = float(raw["High"].iloc[-1])
            intraday_low = float(raw["Low"].iloc[-1])
            intraday_range = intraday_high - intraday_low
            intraday_range_pct = (intraday_range / intraday_low) * 100 if intraday_low else 0
        
        # Weekly: Monday to now
        monday = end - timedelta(days=end.weekday())
        monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        week_mask = raw.index >= monday
        week_data = raw[week_mask]
        
        if len(week_data) > 0:
            weekly_high = float(week_data["High"].max())
            weekly_low = float(week_data["Low"].min())
            weekly_range = weekly_high - weekly_low
            weekly_range_pct = (weekly_range / weekly_low) * 100 if weekly_low else 0
        else:
            weekly_high = intraday_high
            weekly_low = intraday_low
            weekly_range = weekly_high - weekly_low
            weekly_range_pct = (weekly_range / weekly_low) * 100 if weekly_low else 0
        
        result = {
            "current_price": round(current_price, 2),
            "intraday_high": round(intraday_high, 2),
            "intraday_low": round(intraday_low, 2),
            "intraday_range": round(intraday_range, 2),
            "intraday_range_pct": round(intraday_range_pct, 3),
            "weekly_high": round(weekly_high, 2),
            "weekly_low": round(weekly_low, 2),
            "weekly_range": round(weekly_range, 2),
            "weekly_range_pct": round(weekly_range_pct, 3),
        }
        
        log.info(f"Ranges: intraday={intraday_range:.2f} ({intraday_range_pct:.2f}%), "
                 f"weekly={weekly_range:.2f} ({weekly_range_pct:.2f}%)")
        
        return result
    
    except Exception as exc:
        log.error(f"Ranges: failed to compute: {exc}")
        return {
            "current_price": None,
            "intraday_high": None,
            "intraday_low": None,
            "intraday_range": None,
            "intraday_range_pct": None,
            "weekly_high": None,
            "weekly_low": None,
            "weekly_range": None,
            "weekly_range_pct": None,
            "error": str(exc),
        }
