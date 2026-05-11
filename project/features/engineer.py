"""
features/engineer.py
Builds all 15 ML input features + auxiliary columns from raw price/macro data.

Input:  DataFrame with columns:
        Close_XAUUSD, Volume_XAUUSD, Close_EURUSD, Close_USDJPY,
        DFII10, DFII5, DGS2, FEDFUNDS

Output: DataFrame with all BASE_FEATURES + Close_XAUUSD column retained

Public API:
    engineer_features(df)  → pd.DataFrame
"""

import numpy as np
import pandas as pd

from core.config import BASE_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all 15 model features from raw daily price + macro DataFrame.

    The input df is expected to come from data/prices.py + data/fred.py,
    already forward-filled and aligned to a business-day index.
    """
    out  = pd.DataFrame(index=df.index)
    gold = df["Close_XAUUSD"]

    # ── Price returns ─────────────────────────────────────────────────────────
    out["Close_Returns"]  = gold.pct_change()
    out["Log_Returns"]    = np.log(gold / gold.shift(1))
    out["EURUSD_Returns"] = df["Close_EURUSD"].pct_change()
    out["USDJPY_Returns"] = df["Close_USDJPY"].pct_change()

    # ── Bollinger Band %B ─────────────────────────────────────────────────────
    sma20 = gold.rolling(20).mean()
    std20 = gold.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["BB_PctB"] = (gold - lower) / (upper - lower)

    # ── EMA ratios & Bull Trend ───────────────────────────────────────────────
    ema50  = gold.ewm(span=50,  adjust=False).mean()
    ema200 = gold.ewm(span=200, adjust=False).mean()
    out["Price_Over_EMA50"]  = gold / ema50
    out["Price_Over_EMA200"] = gold / ema200
    out["Bull_Trend"]        = (ema50 - ema200) / ema200

    # ── MACD signal (normalised) ──────────────────────────────────────────────
    macd_line = (gold.ewm(span=12, adjust=False).mean()
               - gold.ewm(span=26, adjust=False).mean())
    out["MACD_Signal_Norm"] = macd_line.ewm(span=9, adjust=False).mean() / gold

    # ── Z-scores ─────────────────────────────────────────────────────────────
    r20 = out["Log_Returns"].rolling(20)
    out["LogReturn_ZScore"] = (out["Log_Returns"] - r20.mean()) / r20.std()

    c20 = out["Close_Returns"].rolling(20)
    out["Return_ZScore"] = (out["Close_Returns"] - c20.mean()) / c20.std()

    # ── Percentiles (rolling 252-day) ─────────────────────────────────────────
    out["Return_Percentile"] = (
        out["Close_Returns"]
        .rolling(252)
        .apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else 0.5,
               raw=True)
    )
    out["Volume_Percentile"] = (
        df["Volume_XAUUSD"]
        .rolling(252)
        .apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else 0.5,
               raw=True)
    )

    # ── % from all-time high ──────────────────────────────────────────────────
    out["Pct_From_AllTimeHigh"] = (gold / gold.expanding().max()) - 1

    # ── Macro Fast composite ──────────────────────────────────────────────────
    # Weighted blend: real yield (DFII10) drives gold inversely
    # Higher real yield → tighter financial conditions → bearish for gold
    dfii10   = df.get("DFII10", pd.Series(0, index=df.index)).ffill()
    dfii5    = df.get("DFII5",  pd.Series(0, index=df.index)).ffill()
    dgs2     = df.get("DGS2",   pd.Series(0, index=df.index)).ffill()
    fedfunds = df.get("FEDFUNDS", pd.Series(0, index=df.index)).ffill()

    out["Macro_Fast"] = (
        0.40 * dfii10
      + 0.25 * dfii5
      + 0.20 * dgs2
      + 0.15 * fedfunds
    )

    # ── carry original close for downstream use ───────────────────────────────
    out["Close_XAUUSD"] = gold

    out.dropna(subset=["Close_Returns", "Log_Returns"], inplace=True)

    # Verify all required features are present
    missing = [f for f in BASE_FEATURES if f not in out.columns]
    if missing:
        raise ValueError(f"Feature engineering missing columns: {missing}")

    return out