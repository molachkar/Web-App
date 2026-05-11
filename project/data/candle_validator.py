"""
data/candle_validator.py
Determines whether today's daily candle is settled and computes the safe-run window.

Logic:
  - Daily gold candle settles at 13:30 NY (CME close)
  - CME maintenance: 21:15-22:00 NY time (avoid trading)
  - Three states: SAFE | CAUTION | DANGER

Public API:
    candle_status() -> dict
"""

from datetime import datetime, timedelta

from core.config import (
    UTC_TZ, NY_TZ, MOROCCO_TZ,
    SETTLEMENT_NY_HOUR,
    MAINTENANCE_START_NY, MAINTENANCE_END_NY,
)


def _decimal_hour(dt: datetime, tz) -> float:
    local = dt.astimezone(tz)
    return local.hour + local.minute / 60 + local.second / 3600


def candle_status() -> dict:
    """
    Returns a dict describing the current candle / session state.

    Keys:
        settled      bool   - today's candle is closed and safe to use
        window       str    - SAFE | CAUTION | DANGER
        window_color str    - hex color for frontend badge
        reason       str    - human-readable one-liner
        ny_time      str    - HH:MM
        morocco_time str    - HH:MM
        utc_time     str    - HH:MM
        candle_date  str    - YYYY-MM-DD of most recently settled candle
    """
    now  = datetime.now(UTC_TZ)
    ny_h = _decimal_hour(now, NY_TZ)

    ny_str      = now.astimezone(NY_TZ).strftime("%H:%M")
    morocco_str = now.astimezone(MOROCCO_TZ).strftime("%H:%M")
    utc_str     = now.strftime("%H:%M")

    # ── Settlement ─────────────────────────────────────────────────────────
    settled = ny_h >= SETTLEMENT_NY_HOUR

    if settled:
        candle_date = now.astimezone(NY_TZ).date()
    else:
        candle_date = (now.astimezone(NY_TZ) - timedelta(days=1)).date()

    # ── Window classification ───────────────────────────────────────────────
    if MAINTENANCE_START_NY <= ny_h < MAINTENANCE_END_NY:
        window       = "DANGER"
        window_color = "#ff4d6a"
        reason       = "CME maintenance window — avoid trading"

    elif ny_h >= MAINTENANCE_START_NY - 0.5:
        window       = "CAUTION"
        window_color = "#f5c842"
        reason       = "Approaching CME maintenance — be cautious"

    elif not settled and ny_h >= SETTLEMENT_NY_HOUR - 0.5:
        window       = "CAUTION"
        window_color = "#f5c842"
        reason       = "Candle settling soon — wait for official close"

    elif not settled:
        window       = "CAUTION"
        window_color = "#f5c842"
        reason       = "Candle not yet settled (before 13:30 NY) — using yesterday's signal"

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