"""
data/prices.py
Fetches price strip data (XAU, SPX, NAS, BTC, SILVER, EUR, JPY, OIL)
and returns JSON-ready dicts for the header ticker tape.

Also used by ml pipeline for XAUUSD daily close + EURUSD/USDJPY pairs.

Public API:
    fetch_price_strip()              → list[dict]   (for /prices endpoint)
    fetch_ohlcv(ticker, start, end)  → pd.DataFrame (for ML pipeline)
    fetch_ml_prices(start, end)      → pd.DataFrame (gold + fx combined)
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import PRICE_STRIP

log = logging.getLogger("sentinel.prices")


# ── low-level download with retry ─────────────────────────────────────────────

def fetch_ohlcv(ticker: str,
                start: datetime,
                end: datetime,
                retries: int = 3) -> pd.DataFrame:
    """
    Download OHLCV from yfinance with retry logic.
    Returns empty DataFrame on persistent failure (caller decides how to handle).
    """
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if not df.empty:
                return df
        except Exception as exc:
            log.warning(f"yfinance [{ticker}] attempt {attempt+1}/{retries}: {exc}")
            if attempt < retries - 1:
                time.sleep(1 + attempt)

    log.error(f"yfinance [{ticker}] all {retries} attempts failed")
    return pd.DataFrame()


# ── price strip ───────────────────────────────────────────────────────────────

def fetch_price_strip() -> list[dict]:
    """
    Fetch latest prices for all PRICE_STRIP instruments.
    Returns a list of dicts sorted by config order:
        [{ "symbol": "XAU", "price": 3250.40, "change_pct": 0.42, "status": "ok" }, ...]
    """
    end   = datetime.utcnow()
    start = end - timedelta(days=5)   # fetch 5 days so we always get at least 2 closes

    results = []

    for symbol, ticker in PRICE_STRIP.items():
        try:
            df = fetch_ohlcv(ticker, start, end)
            if df.empty or len(df) < 1:
                results.append(_strip_error(symbol))
                continue

            close = float(df["Close"].iloc[-1])
            prev  = float(df["Close"].iloc[-2]) if len(df) >= 2 else close
            chg   = (close - prev) / prev * 100 if prev else 0.0

            results.append({
                "symbol":     symbol,
                "ticker":     ticker,
                "price":      round(close, _decimals(symbol)),
                "change_pct": round(chg, 3),
                "direction":  "up" if chg > 0 else "down" if chg < 0 else "flat",
                "status":     "ok",
            })
        except Exception as exc:
            log.warning(f"Price strip [{symbol}]: {exc}")
            results.append(_strip_error(symbol))

    return results


def _strip_error(symbol: str) -> dict:
    return {"symbol": symbol, "price": None, "change_pct": None,
            "direction": "flat", "status": "error"}


def _decimals(symbol: str) -> int:
    """Display precision per instrument."""
    return {"EUR": 5, "JPY": 3, "BTC": 2}.get(symbol, 2)


# ── ML input data ─────────────────────────────────────────────────────────────

def fetch_ml_prices(start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch XAUUSD, EURUSD, USDJPY daily OHLCV and combine into one DataFrame
    used by the feature engineering pipeline.

    Returns columns: Close_XAUUSD, Volume_XAUUSD, Close_EURUSD, Close_USDJPY
    aligned to a business-day index.
    Raises RuntimeError if gold data is unavailable.
    """
    gold = fetch_ohlcv("GC=F",     start, end)
    eur  = fetch_ohlcv("EURUSD=X", start, end)
    jpy  = fetch_ohlcv("JPY=X",    start, end)

    if gold.empty:
        raise RuntimeError("XAUUSD (GC=F) data unavailable — cannot build ML features")

    prices = pd.DataFrame({
        "Close_XAUUSD":  gold["Close"],
        "Volume_XAUUSD": gold.get("Volume", pd.Series(dtype=float)),
        "Close_EURUSD":  eur["Close"] if not eur.empty else np.nan,
        "Close_USDJPY":  jpy["Close"] if not jpy.empty else np.nan,
    })

    bdays  = pd.date_range(start=prices.index.min(), end=prices.index.max(), freq="B")
    prices = prices.reindex(bdays).ffill().bfill()
    prices.index.name = "Date"

    return prices