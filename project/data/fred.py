"""
data/fred.py
Fetches FRED macro series (DFII10, DFII5, DGS2, FEDFUNDS).
Falls back to local CSV files in fred_cache/ if the API is unreachable.
Returns a tidy DataFrame aligned to business-day index.

Public API:
    fetch_fred(start, end)  → pd.DataFrame  (columns = MACRO_SERIES)
    fred_series_ages()      → dict[str, float]  (days since last observation)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from core.config import FRED_API_KEY, MACRO_SERIES, FRED_CACHE_DIR, FRED_STALE_DAYS

log = logging.getLogger("sentinel.fred")


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_local_csv(series_id: str,
                    start: datetime,
                    end: datetime) -> Optional[pd.Series]:
    """
    Load a FRED series from the local CSV fallback.
    CSV format: date index (col 0), value (col 1), '.' for missing.
    """
    path = os.path.join(FRED_CACHE_DIR, f"{series_id}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        col = df.columns[0]
        s = df[col].replace(".", np.nan).astype(float)
        s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
        log.info(f"FRED [{series_id}] loaded from local CSV ({len(s)} rows)")
        return s
    except Exception as exc:
        log.error(f"FRED [{series_id}] CSV read failed: {exc}")
        return None


def _fetch_via_api(series_id: str,
                   start: datetime,
                   end: datetime) -> Optional[pd.Series]:
    """Attempt live fetch from FRED API using fredapi."""
    try:
        from fredapi import Fred
        fred_obj = Fred(api_key=FRED_API_KEY)
        data = fred_obj.get_series(series_id, start, end)
        data.index = pd.to_datetime(data.index).tz_localize(None)
        log.info(f"FRED [{series_id}] fetched from API ({len(data)} rows)")
        return data
    except Exception as exc:
        log.warning(f"FRED [{series_id}] API error: {exc}")
        return None


# ── public ────────────────────────────────────────────────────────────────────

def fetch_fred(start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch all MACRO_SERIES and return as a single DataFrame
    aligned to a business-day index between start and end.

    Tries live API first; falls back to local CSV.
    Raises RuntimeError if both fail for any series.
    """
    series: dict[str, pd.Series] = {}

    for sid in MACRO_SERIES:
        data = _fetch_via_api(sid, start, end)
        if data is None or data.empty:
            data = _read_local_csv(sid, start, end)
        if data is None or data.empty:
            raise RuntimeError(
                f"FRED series '{sid}': API failed and no local CSV found at "
                f"{os.path.join(FRED_CACHE_DIR, sid + '.csv')}"
            )
        series[sid] = data

    macro = pd.DataFrame(series)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)

    # Align to business-day index, forward-fill gaps (FRED is weekly/monthly)
    bdays = pd.date_range(start=start, end=end, freq="B")
    macro = macro.reindex(bdays).ffill().bfill()

    return macro


def fred_series_ages() -> dict[str, float]:
    """
    Return approximate age (in days) of the most recent observation
    for each MACRO_SERIES, based on local CSV files.

    Useful for the status bar stale-data warning.
    Returns 999.0 if a file is missing or unreadable.
    """
    ages: dict[str, float] = {}
    today = datetime.utcnow()

    for sid in MACRO_SERIES:
        path = os.path.join(FRED_CACHE_DIR, f"{sid}.csv")
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            last_date = df.index.max()
            age_days = (today - last_date).days
            ages[sid] = age_days
        except Exception:
            ages[sid] = 999.0

    return ages