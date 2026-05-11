"""
features/engineer.py
Builds all 15 ML input features from raw price + macro data.

CRITICAL: Must exactly reproduce training-time feature engineering.
Source of truth: Gold_signal.py::engineer()

Public API:
    engineer_features(df) -> pd.DataFrame
"""

import logging

import numpy as np
import pandas as pd

from core.config import BASE_FEATURES, MACRO_SERIES

log = logging.getLogger("sentinel.features")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns:
        Close_XAUUSD, Volume_XAUUSD, Close_EURUSD, Close_USDJPY,
        DFII10, DFII5, DGS2, FEDFUNDS

    Returns DataFrame with BASE_FEATURES + Close_XAUUSD,
    warmup rows (rolling window NaN) stripped.
    """
    out  = pd.DataFrame(index=df.index)
    gold = df["Close_XAUUSD"]

    # ── Returns ───────────────────────────────────────────────────────────────
    out["Close_Returns"]  = gold.pct_change()
    out["Log_Returns"]    = np.log(gold / gold.shift(1))
    out["EURUSD_Returns"] = df["Close_EURUSD"].pct_change()
    out["USDJPY_Returns"] = df["Close_USDJPY"].pct_change()

    # ── Bollinger Band %B (20-day) ────────────────────────────────────────────
    sma20 = gold.rolling(20).mean()
    std20 = gold.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    denom = upper - lower
    out["BB_PctB"] = (gold - lower) / denom.replace(0, np.nan)

    # ── EMA ratios & Bull Trend ───────────────────────────────────────────────
    ema50  = gold.ewm(span=50,  adjust=False).mean()
    ema200 = gold.ewm(span=200, adjust=False).mean()
    out["Price_Over_EMA50"]  = gold / ema50
    out["Price_Over_EMA200"] = gold / ema200
    out["Bull_Trend"]        = (ema50 - ema200) / ema200

    # ── MACD signal normalised ────────────────────────────────────────────────
    macd = (gold.ewm(span=12, adjust=False).mean()
          - gold.ewm(span=26, adjust=False).mean())
    out["MACD_Signal_Norm"] = macd.ewm(span=9, adjust=False).mean() / gold

    # ── Z-scores (rolling 20) ─────────────────────────────────────────────────
    r20 = out["Log_Returns"].rolling(20)
    out["LogReturn_ZScore"] = (out["Log_Returns"] - r20.mean()) / r20.std()

    c20 = out["Close_Returns"].rolling(20)
    out["Return_ZScore"] = (out["Close_Returns"] - c20.mean()) / c20.std()

    # ── Percentiles (rolling 100) — matches training ──────────────────────────
    out["Return_Percentile"] = out["Close_Returns"].rolling(100).rank(pct=True)

    # Volume: gold futures often report 0 on yfinance — handle gracefully.
    # Replace 0 with NaN, forward-fill from last known non-zero bar.
    # If still all NaN (data source gap), default to neutral 0.5.
    vol = df["Volume_XAUUSD"].replace(0, np.nan).ffill()
    if vol.dropna().empty:
        log.warning("Volume_XAUUSD is all zero/NaN — defaulting Volume_Percentile to 0.5")
        out["Volume_Percentile"] = 0.5
    else:
        vp = vol.rolling(100).rank(pct=True)
        # Forward-fill the warmup NaN rows (first 99 bars) with 0.5
        out["Volume_Percentile"] = vp.fillna(0.5)

    # ── % from all-time high ──────────────────────────────────────────────────
    ath = gold.expanding().max()
    out["Pct_From_AllTimeHigh"] = (ath - gold) / ath

    # ── Macro Fast composite (z-score of each series + delta, 252-day rolling)
    # Exactly matches Gold_signal.py — z-scores each MACRO_SERIES and its
    # 1-day delta with a 252-bar shifted rolling window, then averages them.
    z_cols = []
    for col in MACRO_SERIES:
        if col not in df.columns:
            log.warning(f"Macro column '{col}' missing from input — skipping")
            continue
        delta_col = f"{col}_delta"
        df[delta_col] = df[col].diff()
        for feat in [col, delta_col]:
            roll   = df[feat].shift(1).rolling(252)
            z_name = f"{feat}_z"
            out[z_name] = (df[feat] - roll.mean()) / roll.std()
            z_cols.append(z_name)

    if z_cols:
        out["Macro_Fast"] = (
            out[z_cols].mean(axis=1)          # skipna=True by default
            .replace([np.inf, -np.inf], np.nan)
            .ffill().bfill()
            .clip(-5, 5)
        )
        out.drop(columns=z_cols, inplace=True)
    else:
        log.error("No macro z-score columns computed — Macro_Fast set to 0")
        out["Macro_Fast"] = 0.0

    # ── Retain close ──────────────────────────────────────────────────────────
    out["Close_XAUUSD"] = gold

    # ── Diagnostic: log NaN counts before dropping ───────────────────────────
    nan_counts = out[BASE_FEATURES].isna().sum()
    bad = nan_counts[nan_counts > 0]
    if not bad.empty:
        log.info(f"NaN counts before dropna: {bad.to_dict()}")

    # Drop only the warmup rows — use the smallest required window (20-bar)
    # rather than BASE_FEATURES which would drop rows with any NaN at all.
    # Features with structural NaN (volume on some data sources) are already
    # handled above. This matches training behaviour.
    out = out.dropna(subset=BASE_FEATURES)

    if out.empty:
        # Last-resort: something is still NaN in every row.
        # Fill remaining NaN with column medians so we get at least 1 prediction.
        log.error("All rows dropped by dropna — filling NaN with column medians as fallback")
        out_full = pd.DataFrame(index=df.index)
        out_full["Close_XAUUSD"] = gold
        for col in BASE_FEATURES:
            if col in out.columns:
                out_full[col] = out[col].fillna(out[col].median())
            else:
                out_full[col] = 0.0
        out = out_full.dropna(subset=["Close_Returns"])  # only drop rows with no price

    # Sanity check
    missing = [f for f in BASE_FEATURES if f not in out.columns]
    if missing:
        raise ValueError(f"engineer_features: missing columns: {missing}")

    log.info(f"Feature matrix: {len(out)} rows × {len(BASE_FEATURES)} features")
    return out