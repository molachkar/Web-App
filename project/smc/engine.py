"""
smc/engine.py
Smart Money Concepts detector — 4H XAUUSD candles.
Detects: swing highs/lows, BOS, CHoCH, Order Blocks, S/R clusters.

Public API:
    fetch_smc_levels(current_price) → dict
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import (
    SMC_SWING_LENGTH, SMC_SR_TOLERANCE, SMC_SR_MIN_HITS, SMC_LOOKBACK_DAYS,
)

log = logging.getLogger("sentinel.smc")

_YF_TIMEOUT = 12


def fetch_smc_levels(current_price: float) -> dict:
    empty = {
        "bos_bull": [], "bos_bear": [],
        "choch_bull": [], "choch_bear": [],
        "ob_bull": [], "ob_bear": [],
        "sr": [],
    }
    try:
        ohlc = _fetch_4h()
        if ohlc is None or ohlc.empty:
            return empty

        sh_list, sl_list, hl, lvl = _detect_swings(ohlc)
        result = _detect_bos_choch_ob(ohlc, sh_list, sl_list)
        result["sr"] = _detect_sr(hl, lvl)

        log.info(
            f"SMC: {len(result['sr'])} S/R | "
            f"{len(result['ob_bull'])} bull OB | {len(result['ob_bear'])} bear OB"
        )
        return result

    except Exception as exc:
        log.error(f"SMC engine error: {exc}")
        return empty


# ── Internal ──────────────────────────────────────────────────────────────────

def _fetch_4h() -> pd.DataFrame | None:
    try:
        end   = datetime.utcnow()
        start = end - timedelta(days=SMC_LOOKBACK_DAYS)
        raw   = yf.download("GC=F", start=start, end=end,
                            interval="1h", auto_adjust=False,
                            progress=False, timeout=_YF_TIMEOUT)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        if raw.empty:
            return None

        ohlc           = raw["Close"].resample("4h").ohlc()
        ohlc.columns   = ["Open", "High", "Low", "Close"]
        ohlc["Volume"] = raw["Volume"].resample("4h").sum()
        ohlc           = ohlc.dropna().iloc[:-1]   # drop incomplete last bar
        log.info(f"SMC: {len(ohlc)} 4H candles")
        return ohlc
    except Exception as exc:
        log.error(f"SMC fetch failed: {exc}")
        return None


def _detect_swings(ohlc: pd.DataFrame) -> tuple:
    highs = ohlc["High"].values
    lows  = ohlc["Low"].values
    n     = len(ohlc)
    L     = SMC_SWING_LENGTH

    hl  = np.zeros(n, dtype=int)
    lvl = np.full(n, np.nan)

    for i in range(L, n - L):
        wh = np.concatenate([highs[i-L:i], highs[i+1:i+L+1]])
        wl = np.concatenate([lows[i-L:i],  lows[i+1:i+L+1]])
        if highs[i] > wh.max():
            hl[i] = 1;  lvl[i] = highs[i]
        elif lows[i] < wl.min():
            hl[i] = -1; lvl[i] = lows[i]

    sh_list = [(i, lvl[i]) for i in range(n) if hl[i] ==  1]
    sl_list = [(i, lvl[i]) for i in range(n) if hl[i] == -1]
    return sh_list, sl_list, hl, lvl


def _detect_bos_choch_ob(ohlc, sh_list, sl_list) -> dict:
    highs  = ohlc["High"].values
    lows   = ohlc["Low"].values
    opens  = ohlc["Open"].values
    closes = ohlc["Close"].values
    n      = len(ohlc)

    res = {k: [] for k in ["bos_bull", "bos_bear", "choch_bull", "choch_bear",
                            "ob_bull", "ob_bear"]}
    trend   = 0
    last_sh = last_sl = None

    for i in range(1, n):
        for si, sv in sh_list:
            if si < i and (last_sh is None or si > last_sh[0]):
                last_sh = (si, sv)
        for si, sv in sl_list:
            if si < i and (last_sl is None or si > last_sl[0]):
                last_sl = (si, sv)

        if last_sh and closes[i] > last_sh[1]:
            key = "bos_bull" if trend == 1 else "choch_bull"
            res[key].append({"price": round(last_sh[1], 1),
                             "when": ohlc.index[i].strftime("%m-%d %H:%M")})
            for j in range(last_sh[0]-1, max(0, last_sh[0]-30), -1):
                if closes[j] < opens[j]:
                    if not any(lows[k] < lows[j] for k in range(j+1, min(j+40, n))):
                        res["ob_bull"].append({
                            "top":    round(highs[j], 1),
                            "bottom": round(lows[j],  1),
                            "mid":    round((highs[j]+lows[j])/2, 1),
                            "when":   ohlc.index[j].strftime("%m-%d %H:%M"),
                        })
                    break
            trend = 1; last_sh = None

        elif last_sl and closes[i] < last_sl[1]:
            key = "bos_bear" if trend == -1 else "choch_bear"
            res[key].append({"price": round(last_sl[1], 1),
                             "when": ohlc.index[i].strftime("%m-%d %H:%M")})
            for j in range(last_sl[0]-1, max(0, last_sl[0]-30), -1):
                if closes[j] > opens[j]:
                    if not any(highs[k] > highs[j] for k in range(j+1, min(j+40, n))):
                        res["ob_bear"].append({
                            "top":    round(highs[j], 1),
                            "bottom": round(lows[j],  1),
                            "mid":    round((highs[j]+lows[j])/2, 1),
                            "when":   ohlc.index[j].strftime("%m-%d %H:%M"),
                        })
                    break
            trend = -1; last_sl = None

    return res


def _detect_sr(hl: np.ndarray, lvl: np.ndarray) -> list:
    prices = [lvl[i] for i in range(len(lvl)) if hl[i] != 0 and not np.isnan(lvl[i])]
    if not prices:
        return []

    used = [False] * len(prices)
    sr   = []

    for i, p in enumerate(prices):
        if used[i]:
            continue
        nearby = [p]
        for j in range(i+1, len(prices)):
            if not used[j] and abs(prices[j] - p) / p < SMC_SR_TOLERANCE:
                nearby.append(prices[j]); used[j] = True
        used[i] = True
        if len(nearby) >= SMC_SR_MIN_HITS:
            mid = round(np.mean(nearby), 1)
            if not any(abs(mid - s["price"]) / mid < SMC_SR_TOLERANCE * 2 for s in sr):
                sr.append({"price": mid, "hits": len(nearby)})

    return sorted(sr, key=lambda x: x["price"])