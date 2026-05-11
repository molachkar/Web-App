"""
data/candle_validator.py
Determines whether today's daily candle is "settled" (safe to trade the signal)
and computes the current safe-run window status.

Logic:
  - Daily gold candle settles at 13:30 NY time (CME close)
  - CME maintenance window: 21:15–22:00 NY (avoid trading)
  - We classify the current moment into one of three states:
      SAFE    → candle settled, not in maintenance, regular session
      CAUTION → near settlement boundary or approaching maintenance
      DANGER  → inside maintenance window, or candle not yet settled

Public API:
    candle_status()  → dict  (used by /status endpoint)
"""

from datetime import datetime, timedelta
from core.config import (
    NY_TZ, MOROCCO_TZ, UTC_TZ,
    SETTLEMENT_NY_HOUR, MAINTENANCE_START_NY, MAINTENANCE_END_NY,
)


def _ny_decimal_hour(dt: datetime) -> float:
    """Convert a datetime to decimal NY hour (0.0 – 24.0)."""
    ny = dt.astimezone(NY_TZ)
    return ny.hour + ny.minute / 60 + ny.second / 3600


def candle_status() -> dict:
    """
    Returns a dict describing the current candle / session state.

    Keys:
        settled      bool   – today's candle is closed and safe to use
        window       str    – "SAFE" | "CAUTION" | "DANGER"
        window_color str    – "#10d988" | "#f5c842" | "#ff4d6a"
        reason       str    – human-readable explanation
        ny_time      str    – current NY time HH:MM
        morocco_time str    – current Morocco time HH:MM
        utc_time     str    – current UTC HH:MM
        candle_date  str    – date of most recently settled candle (YYYY-MM-DD)
    """
    now = datetime.now(UTC_TZ)
    ny_h = _ny_decimal_hour(now)

    ny_str      = now.astimezone(NY_TZ).strftime("%H:%M")
    morocco_str = now.astimezone(MOROCCO_TZ).strftime("%H:%M")
    utc_str     = now.strftime("%H:%M")

    # ── is today's candle settled? ────────────────────────────────────────────
    # Settlement is at 13:30 NY.  Before that, use yesterday's candle.
    settled = ny_h >= SETTLEMENT_NY_HOUR

    if settled:
        candle_date = now.astimezone(NY_TZ).date()
    else:
        candle_date = (now.astimezone(NY_TZ) - timedelta(days=1)).date()

    # ── classify window ───────────────────────────────────────────────────────
    # DANGER: maintenance window 21:15–22:00 NY
    if MAINTENANCE_START_NY <= ny_h < MAINTENANCE_END_NY:
        window       = "DANGER"
        window_color = "#ff4d6a"
        reason       = "CME maintenance window — avoid trading"

    # CAUTION: 30 min before maintenance
    elif ny_h >= MAINTENANCE_START_NY - 0.5:
        window       = "CAUTION"
        window_color = "#f5c842"
        reason       = "Approaching CME maintenance — be cautious"

    # CAUTION: 30 min before settlement (candle not yet official)
    elif not settled and ny_h >= SETTLEMENT_NY_HOUR - 0.5:
        window       = "CAUTION"
        window_color = "#f5c842"
        reason       = "Candle settling soon — wait for official close"

    # CAUTION: pre-settlement
    elif not settled:
        window       = "CAUTION"
        window_color = "#f5c842"
        reason       = "Candle not yet settled (before 13:30 NY) — using yesterday's signal"

    # SAFE: settled, not in or near maintenance
    else:
        window       = "SAFE"
        window_color = "#10d988"
        reason       = "Candle settled · Session clear · Signal valid"

    return {
        "settled":      settled,
        "window":       window,
        "window_color": window_color,
        "reason":       reason,
        "ny_time":      ny_str,
        "morocco_time": morocco_str,
        "utc_time":     utc_str,
        "candle_date":  str(candle_date),
    }