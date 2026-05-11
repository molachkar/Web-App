"""
smc/engine.py
Smart Money Concepts detector for 4H gold candles.
Detects: swing highs/lows, BOS, CHoCH, Order Blocks, Support/Resistance.

Public API:
    fetch_smc_levels(current_price: float) → dict
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import SMC_SWING_LENGTH, SMC_SR_TOLERANCE, SMC_SR_MIN_HITS

log = logging.getLogger("sentinel.smc")


def _fetch_4h_candles() -> Optional[pd.DataFrame]:
    """
    Fetch ~200 bars of 4H gold candles from yfinance.
    Returns OHLCV DataFrame indexed by datetime.
    """
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=58)  # ~200 4H candles
        
        raw = yf.download("GC=F", start=start, end=end,
                          interval="1h", auto_adjust=False, progress=False)
        
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        
        if raw.empty:
            log.warning("SMC: No 4H data available from yfinance")
            return None
        
        # Resample to 4H
        ohlc = raw["Close"].resample("4h").ohlc()
        ohlc.columns = ["Open", "High", "Low", "Close"]
        ohlc["Volume"] = raw["Volume"].resample("4h").sum()
        ohlc = ohlc.dropna().iloc[:-1]  # drop incomplete last candle
        
        log.info(f"SMC: fetched {len(ohlc)} 4H candles")
        return ohlc
    
    except Exception as exc:
        log.error(f"SMC: failed to fetch 4H candles: {exc}")
        return None


def _detect_swings(ohlc: pd.DataFrame) -> tuple:
    """
    Detect swing highs and lows using a simple pivot approach.
    Returns arrays of swing indices and levels.
    """
    highs = ohlc["High"].values
    lows = ohlc["Low"].values
    n = len(ohlc)
    L = SMC_SWING_LENGTH
    
    # hl[i] = 1 for swing high, -1 for swing low, 0 otherwise
    hl = np.zeros(n, dtype=int)
    lvl = np.full(n, np.nan)
    
    for i in range(L, n - L):
        wh = np.concatenate([highs[i-L:i], highs[i+1:i+L+1]])
        wl = np.concatenate([lows[i-L:i], lows[i+1:i+L+1]])
        
        if highs[i] > wh.max():
            hl[i] = 1
            lvl[i] = highs[i]
        elif lows[i] < wl.min():
            hl[i] = -1
            lvl[i] = lows[i]
    
    sh_list = [(i, lvl[i]) for i in range(n) if hl[i] == 1]
    sl_list = [(i, lvl[i]) for i in range(n) if hl[i] == -1]
    
    return sh_list, sl_list, hl, lvl


def _detect_bos_choch_ob(ohlc: pd.DataFrame, 
                          sh_list: list, 
                          sl_list: list) -> dict:
    """
    Detect Break of Structure (BOS), Change of Character (CHoCH),
    and Order Blocks (OB) based on swing points.
    """
    highs = ohlc["High"].values
    lows = ohlc["Low"].values
    opens = ohlc["Open"].values
    closes = ohlc["Close"].values
    n = len(ohlc)
    
    res = {
        "bos_bull": [],
        "bos_bear": [],
        "choch_bull": [],
        "choch_bear": [],
        "ob_bull": [],
        "ob_bear": [],
    }
    
    trend = 0  # 0 = neutral, 1 = bullish, -1 = bearish
    last_sh = None
    last_sl = None
    
    for i in range(1, n):
        # Update most recent swing high/low before bar i
        for si, sv in sh_list:
            if si < i and (last_sh is None or si > last_sh[0]):
                last_sh = (si, sv)
        
        for si, sv in sl_list:
            if si < i and (last_sl is None or si > last_sl[0]):
                last_sl = (si, sv)
        
        # Bullish BOS/CHoCH
        if last_sh and closes[i] > last_sh[1]:
            key = "bos_bull" if trend == 1 else "choch_bull"
            res[key].append({
                "price": round(last_sh[1], 1),
                "when": ohlc.index[i].strftime("%m-%d %H:%M"),
            })
            
            # Find bullish order block (last down candle before breakout)
            for j in range(last_sh[0] - 1, max(0, last_sh[0] - 30), -1):
                if closes[j] < opens[j]:  # down candle
                    # Check it's not taken out within next 40 bars
                    if not any(lows[k] < lows[j] for k in range(j + 1, min(j + 40, n))):
                        res["ob_bull"].append({
                            "top": round(highs[j], 1),
                            "bottom": round(lows[j], 1),
                            "mid": round((highs[j] + lows[j]) / 2, 1),
                            "when": ohlc.index[j].strftime("%m-%d %H:%M"),
                        })
                    break
            
            trend = 1
            last_sh = None
        
        # Bearish BOS/CHoCH
        elif last_sl and closes[i] < last_sl[1]:
            key = "bos_bear" if trend == -1 else "choch_bear"
            res[key].append({
                "price": round(last_sl[1], 1),
                "when": ohlc.index[i].strftime("%m-%d %H:%M"),
            })
            
            # Find bearish order block (last up candle before breakout)
            for j in range(last_sl[0] - 1, max(0, last_sl[0] - 30), -1):
                if closes[j] > opens[j]:  # up candle
                    # Check it's not taken out within next 40 bars
                    if not any(highs[k] > highs[j] for k in range(j + 1, min(j + 40, n))):
                        res["ob_bear"].append({
                            "top": round(highs[j], 1),
                            "bottom": round(lows[j], 1),
                            "mid": round((highs[j] + lows[j]) / 2, 1),
                            "when": ohlc.index[j].strftime("%m-%d %H:%M"),
                        })
                    break
            
            trend = -1
            last_sl = None
    
    return res


def _detect_sr(hl: np.ndarray, lvl: np.ndarray) -> list:
    """
    Cluster swing points into Support/Resistance levels.
    Groups nearby swings within tolerance and requires minimum hits.
    """
    all_prices = [lvl[i] for i in range(len(lvl)) 
                  if hl[i] != 0 and not np.isnan(lvl[i])]
    
    if not all_prices:
        return []
    
    used = [False] * len(all_prices)
    sr = []
    
    for i, p in enumerate(all_prices):
        if used[i]:
            continue
        
        nearby = [p]
        for j in range(i + 1, len(all_prices)):
            if not used[j] and abs(all_prices[j] - p) / p < SMC_SR_TOLERANCE:
                nearby.append(all_prices[j])
                used[j] = True
        
        used[i] = True
        
        if len(nearby) >= SMC_SR_MIN_HITS:
            mid = round(np.mean(nearby), 1)
            # Avoid duplicate levels
            if not any(abs(mid - s["price"]) / mid < SMC_SR_TOLERANCE * 2 
                       for s in sr):
                sr.append({"price": mid, "hits": len(nearby)})
    
    return sorted(sr, key=lambda x: x["price"])


def fetch_smc_levels(current_price: float) -> dict:
    """
    Main entry point: fetch 4H candles and compute all SMC levels.
    
    Args:
        current_price: Current XAUUSD price for reference
        
    Returns:
        dict with keys:
            - bos_bull, bos_bear: list of {price, when}
            - choch_bull, choch_bear: list of {price, when}
            - ob_bull: list of {top, bottom, mid, when}
            - ob_bear: list of {top, bottom, mid, when}
            - sr: list of {price, hits}
    """
    empty = {
        "bos_bull": [], "bos_bear": [],
        "choch_bull": [], "choch_bear": [],
        "ob_bull": [], "ob_bear": [],
        "sr": [],
    }
    
    ohlc = _fetch_4h_candles()
    if ohlc is None or ohlc.empty:
        return empty
    
    sh_list, sl_list, hl, lvl = _detect_swings(ohlc)
    bos_choch_ob = _detect_bos_choch_ob(ohlc, sh_list, sl_list)
    sr = _detect_sr(hl, lvl)
    
    result = {**bos_choch_ob, "sr": sr}
    
    log.info(f"SMC: detected {len(result['sr'])} S/R levels, "
             f"{len(result['ob_bull'])} bull OB, {len(result['ob_bear'])} bear OB")
    
    return result
