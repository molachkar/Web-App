"""
market/ranges.py
Intraday and weekly XAUUSD price ranges.

Uses 1H bars (not 1m) — avoids the timeout issue seen with 1m data over 7 days.

Public API:
    fetch_ranges() → dict
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

log = logging.getLogger("sentinel.market")

_YF_TIMEOUT = 12


def fetch_ranges() -> dict:
    """
    Fetch 1H XAUUSD bars for the past 7 days.
    Computes:
      - Intraday range (today's session high - low)
      - Weekly range (Mon–now high - low)

    Returns dict. All float fields are None on failure.
    """
    try:
        end   = datetime.utcnow()
        start = end - timedelta(days=7)

        raw = yf.download(
            "GC=F", start=start, end=end,
            interval="1h",
            auto_adjust=False, progress=False,
            timeout=_YF_TIMEOUT,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)

        if raw.empty:
            log.warning("Ranges: no 1H data from yfinance")
            return _empty("no data")

        current_price = float(raw["Close"].iloc[-1])

        # ── Intraday ──────────────────────────────────────────────────────────
        today       = end.date()
        today_data  = raw[raw.index.date == today]

        if today_data.empty:
            # Market not yet opened today — use most recent session
            last_date  = raw.index[-1].date()
            today_data = raw[raw.index.date == last_date]

        i_high = float(today_data["High"].max())
        i_low  = float(today_data["Low"].min())
        i_rng  = i_high - i_low
        i_pct  = (i_rng / i_low * 100) if i_low else 0.0

        # ── Weekly (Mon 00:00 UTC → now) ──────────────────────────────────────
        monday    = end - timedelta(days=end.weekday())
        monday    = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        week_data = raw[raw.index >= monday]

        if week_data.empty:
            week_data = today_data   # fallback: just today

        w_high = float(week_data["High"].max())
        w_low  = float(week_data["Low"].min())
        w_rng  = w_high - w_low
        w_pct  = (w_rng / w_low * 100) if w_low else 0.0

        log.info(
            f"Ranges: intraday={i_rng:.2f} ({i_pct:.2f}%), "
            f"weekly={w_rng:.2f} ({w_pct:.2f}%)"
        )

        return {
            "current_price":     round(current_price, 2),
            "intraday_high":     round(i_high, 2),
            "intraday_low":      round(i_low,  2),
            "intraday_range":    round(i_rng,  2),
            "intraday_range_pct":round(i_pct,  3),
            "weekly_high":       round(w_high, 2),
            "weekly_low":        round(w_low,  2),
            "weekly_range":      round(w_rng,  2),
            "weekly_range_pct":  round(w_pct,  3),
        }

    except Exception as exc:
        log.error(f"Ranges: {exc}")
        return _empty(str(exc))


def _empty(reason: str) -> dict:
    return {
        "current_price": None, "intraday_high": None,
        "intraday_low": None,  "intraday_range": None,
        "intraday_range_pct": None, "weekly_high": None,
        "weekly_low": None,    "weekly_range": None,
        "weekly_range_pct": None, "error": reason,
    }