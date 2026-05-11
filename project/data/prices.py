"""
data/prices.py
Price strip fetch (header ticker tape) and ML input prices.

Public API:
    fetch_price_strip()              -> list[dict]   for /prices endpoint
    fetch_ohlcv(ticker, start, end)  -> pd.DataFrame
    fetch_ml_prices(start, end)      -> pd.DataFrame (gold + fx for ML pipeline)
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import PRICE_STRIP, STOOQ_MAP

log = logging.getLogger("sentinel.prices")

_YF_RETRIES = 3
_YF_TIMEOUT = 12
_YF_SLEEP   = 1.5


# ── Low-level download ────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, start: datetime, end: datetime,
                interval: str = "1d") -> pd.DataFrame:
    """yfinance download with retry. Returns empty DataFrame on persistent failure."""
    for attempt in range(_YF_RETRIES):
        try:
            df = yf.download(ticker, start=start, end=end,
                             interval=interval,
                             auto_adjust=False, progress=False,
                             timeout=_YF_TIMEOUT)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if not df.empty:
                return df
        except Exception as exc:
            log.warning(f"yfinance [{ticker}] attempt {attempt+1}/{_YF_RETRIES}: {exc}")
            if attempt < _YF_RETRIES - 1:
                time.sleep(_YF_SLEEP * (attempt + 1))

    # stooq fallback (daily only)
    if interval == "1d":
        df = _fetch_stooq(ticker, start, end)
        if not df.empty:
            log.info(f"{ticker}: fell back to stooq")
            return df

    log.error(f"{ticker}: all sources failed")
    return pd.DataFrame()


def _fetch_stooq(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        import pandas_datareader as pdr
        stooq_ticker = STOOQ_MAP.get(ticker, ticker)
        df = pdr.DataReader(stooq_ticker, "stooq", start, end)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.sort_index()
    except Exception as e:
        log.warning(f"stooq [{ticker}]: {e}")
        return pd.DataFrame()


# ── Price strip ───────────────────────────────────────────────────────────────

def fetch_price_strip() -> list:
    """
    Latest price + % change for all PRICE_STRIP instruments.
    Fetches 5 days so we always have at least 2 closes for the delta.
    """
    end   = datetime.utcnow()
    start = end - timedelta(days=5)
    results = []

    for symbol, ticker in PRICE_STRIP.items():
        try:
            df = fetch_ohlcv(ticker, start, end)
            if df.empty or len(df) < 1:
                results.append(_error_row(symbol, ticker))
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
            results.append(_error_row(symbol, ticker))

    return results


def _error_row(symbol: str, ticker: str) -> dict:
    return {"symbol": symbol, "ticker": ticker,
            "price": None, "change_pct": None, "direction": "flat", "status": "error"}


def _decimals(symbol: str) -> int:
    return {"EUR": 5, "JPY": 3, "BTC": 2}.get(symbol, 2)


# ── ML input prices ───────────────────────────────────────────────────────────

def fetch_ml_prices(start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch XAU/USD, EURUSD, USDJPY daily OHLCV.
    Returns business-day aligned DataFrame:
        Close_XAUUSD, Volume_XAUUSD, Close_EURUSD, Close_USDJPY
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
        "Close_EURUSD":  eur["Close"]  if not eur.empty  else np.nan,
        "Close_USDJPY":  jpy["Close"]  if not jpy.empty  else np.nan,
    })

    bdays  = pd.date_range(start=prices.index.min(), end=prices.index.max(), freq="B")
    prices = prices.reindex(bdays).ffill().bfill()
    prices.index.name = "Date"
    return prices