"""
XAUUSD Gold Intelligence — Streamlit Inference App
===================================================
Stage 1 : LightGBM base model  (cv_best_fold_model.pkl)
Stage 2 : Logistic calibrator  (calibrator.pkl)

NEW:
  - SMC 4H chart (BOS, CHoCH, Order Blocks, Key S/R)
  - Fixed news: RSS-based headlines for 6 assets
    XAU/USD · S&P500 · Oil · Nasdaq · Bitcoin · Silver
"""

import os, pickle, warnings, time, re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────
DAYS_BACK       = 520
PROB_THRESHOLD  = 0.45
Z_THRESHOLD     = 0.6
PRED_Z_LOOKBACK = 252
FRED_API_KEY    = "219d0c44b2e3b4a8b690c3f69b91a5bb"
MACRO_SERIES    = ["DFII10", "DFII5", "DGS2", "FEDFUNDS"]
ARTEFACT_DIR    = os.path.dirname(os.path.abspath(__file__))

BASE_FEATURES = [
    "Close_Returns", "Log_Returns", "EURUSD_Returns", "USDJPY_Returns",
    "BB_PctB", "Price_Over_EMA50", "Price_Over_EMA200", "MACD_Signal_Norm",
    "LogReturn_ZScore", "Return_ZScore", "Return_Percentile", "Volume_Percentile",
    "Pct_From_AllTimeHigh", "Bull_Trend", "Macro_Fast",
]

CALIB_FEATURES = [
    "prediction_value", "abs_prediction",
    "Bull_Trend", "Macro_Fast", "BB_PctB", "Price_Over_EMA200",
]

NEWS_ASSETS = [
    {"label": "XAU/USD",  "ticker": "GC=F",    "color": "#f5c842",
     "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC%3DF&region=US&lang=en-US"},
    {"label": "S&P 500",  "ticker": "^GSPC",   "color": "#378ADD",
     "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US"},
    {"label": "Oil",      "ticker": "CL=F",    "color": "#D85A30",
     "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=CL%3DF&region=US&lang=en-US"},
    {"label": "Nasdaq",   "ticker": "^IXIC",   "color": "#7F77DD",
     "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EIXIC&region=US&lang=en-US"},
    {"label": "Bitcoin",  "ticker": "BTC-USD", "color": "#EF9F27",
     "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US"},
    {"label": "Silver",   "ticker": "SI=F",    "color": "#B4B2A9",
     "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SI%3DF&region=US&lang=en-US"},
]

GOLD_EFFECT_CONTEXT = {
    "XAU/USD":  "Direct asset. Analyse its own technicals, macro drivers, and momentum.",
    "S&P 500":  "Inverse risk-off proxy. When equities fall hard, gold typically benefits as safe haven.",
    "Oil":      "Inflation driver. Rising oil raises CPI expectations, lowering real yields, which is bullish for gold.",
    "Nasdaq":   "Risk appetite barometer. Nasdaq weakness often precedes gold inflows. AI capex affects real yields.",
    "Bitcoin":  "Competing safe haven. Short-term stress: BTC sells first, gold benefits. Medium-term: both rise on dollar weakness.",
    "Silver":   "High-beta gold proxy. Amplifies gold moves. Supply deficit adds independent bullish pressure.",
}

# ── SMC constants ──────────────────────────────────────────────────────────────
SMC_SWING_LENGTH  = 5
SMC_SR_TOLERANCE  = 0.0015
SMC_SR_MIN_HITS   = 2
SMC_OB_EXTEND     = 40
C_BG              = "#05070a"
C_SURF            = "#0c0f14"
C_BORDER          = "#1c2030"
C_TEXT            = "#e2e8f0"
C_MUTED           = "#5a6a80"
C_GOLD            = "#f5c842"
C_BUY_C           = "#10d988"
C_SELL_C          = "#ff4d6a"


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING — ML PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def _download(ticker, start, end, retries=3):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if not df.empty:
                return df
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return pd.DataFrame()


def fetch_fred_local(start, end):
    series = {}
    fred_obj = Fred(api_key=FRED_API_KEY)
    for s in MACRO_SERIES:
        try:
            data = fred_obj.get_series(s, start, end)
            series[s] = data
        except Exception:
            local = os.path.join(ARTEFACT_DIR, f"{s}.csv")
            if os.path.exists(local):
                df = pd.read_csv(local, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                col = df.columns[0]
                data = df[col].replace(".", np.nan).astype(float)
                data = data[(data.index >= str(start)) & (data.index <= str(end))]
                series[s] = data
            else:
                raise FileNotFoundError(
                    f"FRED API failed and no local {s}.csv found."
                )
    macro = pd.DataFrame(series)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    return macro


def fetch_data(start, end):
    gold = _download("GC=F",     start, end)
    eur  = _download("EURUSD=X", start, end)
    jpy  = _download("JPY=X",    start, end)

    prices = pd.DataFrame({
        "Close_XAUUSD":  gold["Close"],
        "Volume_XAUUSD": gold["Volume"],
        "Close_EURUSD":  eur["Close"],
        "Close_USDJPY":  jpy["Close"],
    })

    macro    = fetch_fred_local(start, end)
    full_idx = pd.date_range(start=prices.index.min(),
                             end=prices.index.max(), freq="B")
    prices   = prices.reindex(full_idx)
    macro    = macro.reindex(full_idx)
    df       = prices.join(macro, how="left").ffill().bfill()
    df.dropna(subset=["Close_XAUUSD"], inplace=True)
    df.index.name = "Date"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df):
    out  = pd.DataFrame(index=df.index)
    gold = df["Close_XAUUSD"]

    out["Close_Returns"]  = gold.pct_change()
    out["Log_Returns"]    = np.log(gold / gold.shift(1))
    out["EURUSD_Returns"] = df["Close_EURUSD"].pct_change()
    out["USDJPY_Returns"] = df["Close_USDJPY"].pct_change()

    sma20 = gold.rolling(20).mean()
    std20 = gold.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["BB_PctB"] = (gold - lower) / (upper - lower)

    ema50  = gold.ewm(span=50,  adjust=False).mean()
    ema200 = gold.ewm(span=200, adjust=False).mean()
    out["Price_Over_EMA50"]  = gold / ema50
    out["Price_Over_EMA200"] = gold / ema200
    out["Bull_Trend"]        = (ema50 - ema200) / ema200

    macd = (gold.ewm(span=12, adjust=False).mean()
           - gold.ewm(span=26, adjust=False).mean())
    out["MACD_Signal_Norm"] = macd.ewm(span=9, adjust=False).mean() / gold

    r20 = out["Log_Returns"].rolling(20)
    out["LogReturn_ZScore"] = (out["Log_Returns"] - r20.mean()) / r20.std()
    c20 = out["Close_Returns"].rolling(20)
    out["Return_ZScore"] = (out["Close_Returns"] - c20.mean()) / c20.std()

    out["Return_Percentile"] = out["Close_Returns"].rolling(100).rank(pct=True)
    vol = df["Volume_XAUUSD"].replace(0, np.nan).ffill()
    out["Volume_Percentile"] = vol.rolling(100).rank(pct=True)

    ath = gold.expanding().max()
    out["Pct_From_AllTimeHigh"] = (ath - gold) / ath

    z_cols = []
    for col in MACRO_SERIES:
        df[f"{col}_delta"] = df[col].diff()
        for feat in [col, f"{col}_delta"]:
            roll = df[feat].shift(1).rolling(252)
            out[f"{feat}_z"] = (df[feat] - roll.mean()) / roll.std()
            z_cols.append(f"{feat}_z")
    out["Macro_Fast"] = (out[z_cols].mean(axis=1)
                         .replace([np.inf, -np.inf], np.nan)
                         .ffill().bfill().clip(-5, 5))
    out.drop(columns=z_cols, inplace=True)
    out["Close_XAUUSD"] = gold
    return out.dropna(subset=BASE_FEATURES)


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def load_artefacts():
    def _load(name):
        path = os.path.join(ARTEFACT_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found in {ARTEFACT_DIR}")
        with open(path, "rb") as f:
            return pickle.load(f)

    base_model  = _load("cv_best_fold_model.pkl")
    calibrator  = _load("calibrator.pkl")
    oof_history = pd.read_csv(
        os.path.join(ARTEFACT_DIR, "cv_predictions_oof.csv"),
        index_col=0, parse_dates=True
    )
    return base_model, calibrator, oof_history


def run_inference(feat_df, base_model, calibrator, oof_history):
    today      = feat_df.iloc[[-1]].copy()
    today_date = today.index[0]
    pred_val   = float(base_model.predict(today[BASE_FEATURES].values)[0])
    abs_pred   = abs(pred_val)

    hist   = oof_history["oof_prediction"].dropna().tail(PRED_Z_LOOKBACK)
    h_std  = hist.std()
    pred_z = float((pred_val - hist.mean()) / h_std) if h_std > 0 else 0.0

    bull_trend = float(today["Bull_Trend"].iloc[0])
    macro_fast = float(today["Macro_Fast"].iloc[0])
    bb_pctb    = float(today["BB_PctB"].iloc[0])
    ema200     = float(today["Price_Over_EMA200"].iloc[0])
    close      = float(today["Close_XAUUSD"].iloc[0])

    calib_input = pd.DataFrame(
        [[pred_val, abs_pred, bull_trend, macro_fast, bb_pctb, ema200]],
        columns=CALIB_FEATURES,
    )
    prob = float(calibrator.predict_proba(calib_input)[0][1])

    signal = "NO SIGNAL"
    if prob >= PROB_THRESHOLD and abs(pred_z) >= Z_THRESHOLD:
        signal = "BUY" if pred_val > 0 else "SELL"

    recent = feat_df["Close_XAUUSD"].tail(20)
    return {
        "date": today_date, "signal": signal,
        "pred_val": pred_val, "abs_pred": abs_pred,
        "pred_z": pred_z, "abs_pred_z": abs(pred_z),
        "prob": prob, "bull_trend": bull_trend,
        "macro_fast": macro_fast, "bb_pctb": bb_pctb,
        "ema200": ema200, "close": close,
        "recent": recent, "feat_df": feat_df,
    }


def _fetch_rss_headlines(url: str, max_items: int = 8) -> list:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=7) as resp:
            raw = resp.read()
        root    = ET.fromstring(raw)
        channel = root.find("channel")
        if channel is None:
            return []
        items  = []
        cutoff = datetime.utcnow() - timedelta(days=5)
        for item in channel.findall("item")[:max_items * 2]:
            title   = (item.findtext("title") or "").strip()
            link    = (item.findtext("link")  or "#").strip()
            pub_raw = (item.findtext("pubDate") or "").strip()
            desc    = (item.findtext("description") or "").strip()
            if not title:
                continue
            age_str = ""
            try:
                pub_dt = datetime.strptime(pub_raw, "%a, %d %b %Y %H:%M:%S %z").replace(tzinfo=None)
                if pub_dt < cutoff:
                    continue
                delta   = datetime.utcnow() - pub_dt
                hrs     = int(delta.total_seconds() // 3600)
                age_str = f"{hrs}h ago" if hrs < 24 else f"{delta.days}d ago"
            except Exception:
                pass
            items.append({"title": title, "url": link, "age": age_str, "desc": desc})
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


def _fetch_yf_fallback(ticker: str, max_items: int = 8) -> list:
    try:
        t        = yf.Ticker(ticker)
        raw_news = t.news or []
        cutoff   = datetime.utcnow() - timedelta(days=5)
        items    = []
        for n in raw_news[:max_items * 2]:
            ts  = n.get("providerPublishTime", 0)
            pub = datetime.utcfromtimestamp(ts) if ts else None
            if pub and pub < cutoff:
                continue
            hrs     = int((datetime.utcnow() - pub).total_seconds() // 3600) if pub else 0
            age_str = f"{hrs}h ago" if hrs < 24 else f"{int((datetime.utcnow()-pub).days)}d ago" if pub else ""
            items.append({"title": n.get("title", ""), "url": n.get("link", "#"), "age": age_str, "desc": ""})
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


@st.cache_data(ttl=1800)
def fetch_all_headlines() -> dict:
    result = {}
    for asset in NEWS_ASSETS:
        headlines = _fetch_rss_headlines(asset["rss"])
        if not headlines:
            headlines = _fetch_yf_fallback(asset["ticker"])
        result[asset["label"]] = headlines
    return result


def _ai_write_article(label: str, headlines: list, gold_price: float) -> dict:
    import json, urllib.request
    if not headlines:
        return {
            "headline": f"No recent headlines available for {label}",
            "body": "No news data was returned from the feed in the last 5 days. Check back later or verify the RSS source is accessible from your server.",
            "effect": "Unknown — no data available.",
            "effect_direction": "neutral",
        }

    headlines_text = "\n".join(
        f"- {h['title']} ({h['age']})" + (f"\n  {h['desc'][:200]}" if h.get("desc") else "")
        for h in headlines[:8]
    )
    gold_context = GOLD_EFFECT_CONTEXT.get(label, "")

    prompt = f"""You are a professional financial analyst writing a market brief for a gold trader.

Asset: {label}
Current gold price: ${gold_price:,.0f}
Gold relationship context: {gold_context}

Recent headlines (last 5 days):
{headlines_text}

Write a structured market brief in this exact JSON format. No markdown, no backticks, only valid JSON:
{{
  "headline": "One sharp sentence summarising the most important development (max 120 chars)",
  "body": "Three paragraphs separated by \\n\\n. Paragraph 1: what is happening and why (fundamentals, data, events). Paragraph 2: technical picture and price levels if relevant. Paragraph 3: forward outlook and key risks. Each paragraph 3-5 sentences. Be specific with numbers and dates from the headlines. Do not be vague.",
  "effect": "One sentence starting with BULLISH FOR GOLD / BEARISH FOR GOLD / NEUTRAL FOR GOLD / MIXED — then explain why in 1-2 sentences.",
  "effect_direction": "bullish or bearish or neutral or mixed"
}}"""

    try:
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"content-type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data     = json.loads(resp.read())
            raw_text = data["content"][0]["text"].strip()
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            return json.loads(raw_text)
    except Exception as e:
        return {
            "headline": headlines[0]["title"] if headlines else f"{label} market update",
            "body": "\n\n".join([
                f"Recent {label} news covers the following key developments: " + ". ".join(h["title"] for h in headlines[:3]) + ".",
                "Detailed AI analysis is temporarily unavailable. Please review the source headlines above for the latest information.",
                f"Monitor {label} closely given current market conditions and its relationship to gold positioning.",
            ]),
            "effect": gold_context,
            "effect_direction": "neutral",
        }


# ══════════════════════════════════════════════════════════════════════════════
# SMC ENGINE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=900)
def fetch_4h() -> pd.DataFrame:
    end   = datetime.utcnow()
    start = end - timedelta(days=58)
    raw   = yf.download("GC=F", start=start, end=end,
                         interval="1h", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    if raw.empty:
        return pd.DataFrame()
    ohlc            = raw["Close"].resample("4h").ohlc()
    ohlc.columns    = ["Open", "High", "Low", "Close"]
    ohlc["Volume"]  = raw["Volume"].resample("4h").sum()
    ohlc.dropna(inplace=True)
    return ohlc.iloc[:-1]          # drop incomplete candle


def _find_swings(df: pd.DataFrame, length: int = SMC_SWING_LENGTH) -> pd.DataFrame:
    highs = df["High"].values
    lows  = df["Low"].values
    n     = len(df)
    hl    = np.zeros(n, dtype=int)
    lvl   = np.full(n, np.nan)
    for i in range(length, n - length):
        wh = np.concatenate([highs[i-length:i], highs[i+1:i+length+1]])
        wl = np.concatenate([lows[i-length:i],  lows[i+1:i+length+1]])
        if highs[i] > wh.max():
            hl[i]  = 1;  lvl[i] = highs[i]
        elif lows[i] < wl.min():
            hl[i]  = -1; lvl[i] = lows[i]
    return pd.DataFrame({"HighLow": hl, "Level": lvl}, index=df.index)


def _find_bos_choch(df: pd.DataFrame, swings: pd.DataFrame) -> list:
    events = []
    closes = df["Close"].values
    idx    = df.index
    n      = len(df)
    trend  = 0
    last_sh = last_sl = None

    swing_highs = [(i, swings["Level"].iloc[i]) for i in range(len(swings)) if swings["HighLow"].iloc[i] == 1]
    swing_lows  = [(i, swings["Level"].iloc[i]) for i in range(len(swings)) if swings["HighLow"].iloc[i] == -1]

    for i in range(1, n):
        for si, sv in swing_highs:
            if si < i and (last_sh is None or si > last_sh[0]):
                last_sh = (si, sv)
        for si, sv in swing_lows:
            if si < i and (last_sl is None or si > last_sl[0]):
                last_sl = (si, sv)

        if last_sh is not None and closes[i] > last_sh[1]:
            ev_type = "BOS" if trend == 1 else "CHoCH"
            events.append({"type": ev_type, "direction": 1,
                           "level": last_sh[1], "bar_index": last_sh[0],
                           "broken_index": i, "broken_time": idx[i],
                           "level_time": idx[last_sh[0]]})
            trend = 1; last_sh = None
        elif last_sl is not None and closes[i] < last_sl[1]:
            ev_type = "BOS" if trend == -1 else "CHoCH"
            events.append({"type": ev_type, "direction": -1,
                           "level": last_sl[1], "bar_index": last_sl[0],
                           "broken_index": i, "broken_time": idx[i],
                           "level_time": idx[last_sl[0]]})
            trend = -1; last_sl = None
    return events


def _find_order_blocks(df: pd.DataFrame, bos_events: list) -> list:
    opens  = df["Open"].values
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    idx    = df.index
    n      = len(df)
    obs    = []

    for ev in bos_events:
        bi = ev["broken_index"]
        if bi < 2:
            continue
        if ev["direction"] == 1:          # bullish — find last bearish candle
            ob_idx = None
            for j in range(bi - 1, max(0, bi - 30), -1):
                if closes[j] < opens[j]:
                    ob_idx = j; break
            color = "bull"
        else:                             # bearish — find last bullish candle
            ob_idx = None
            for j in range(bi - 1, max(0, bi - 30), -1):
                if closes[j] > opens[j]:
                    ob_idx = j; break
            color = "bear"

        if ob_idx is None:
            continue

        top     = highs[ob_idx]
        bottom  = lows[ob_idx]
        end_idx = min(ob_idx + SMC_OB_EXTEND, n - 1)
        mitigated = False
        for k in range(ob_idx + 1, end_idx + 1):
            if color == "bull" and lows[k] < bottom:
                mitigated = True; end_idx = k; break
            if color == "bear" and highs[k] > top:
                mitigated = True; end_idx = k; break

        obs.append({"direction": color, "top": top, "bottom": bottom,
                    "start_time": idx[ob_idx], "end_time": idx[end_idx],
                    "mitigated": mitigated, "ob_index": ob_idx})

    # deduplicate
    seen, deduped = [], []
    for ob in sorted(obs, key=lambda x: x["ob_index"], reverse=True):
        mid = (ob["top"] + ob["bottom"]) / 2
        if not any(abs(mid - s) / mid < SMC_SR_TOLERANCE for s in seen):
            seen.append(mid); deduped.append(ob)
    return deduped


def _find_key_levels(df: pd.DataFrame, swings: pd.DataFrame) -> list:
    prices = swings.loc[swings["HighLow"] != 0, "Level"].dropna().values
    if len(prices) < SMC_SR_MIN_HITS:
        return []
    clusters, used = [], np.zeros(len(prices), dtype=bool)
    for i, p in enumerate(prices):
        if used[i]: continue
        nearby = [p]
        for j in range(i + 1, len(prices)):
            if not used[j] and abs(prices[j] - p) / p < SMC_SR_TOLERANCE:
                nearby.append(prices[j]); used[j] = True
        used[i] = True
        if len(nearby) >= SMC_SR_MIN_HITS:
            mid = np.mean(nearby)
            clusters.append({"price": mid, "hits": len(nearby),
                             "top": mid * (1 + SMC_SR_TOLERANCE),
                             "bottom": mid * (1 - SMC_SR_TOLERANCE)})
    return clusters






# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');
:root {
    --bg:#05070a; --surf:#0c0f14; --surf2:#111520; --border:#1c2030;
    --border2:#252a38; --text:#e2e8f0; --muted:#3d4a5c; --muted2:#5a6a80;
    --accent:#f5c842; --accent2:#e6a800; --buy:#10d988; --buy2:#0aab68;
    --sell:#ff4d6a; --sell2:#cc2a45;
    --mono:'JetBrains Mono',monospace; --sans:'Space Grotesk',sans-serif;
    --glow-buy:0 0 24px rgba(16,217,136,0.18);
    --glow-sell:0 0 24px rgba(255,77,106,0.18);
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"],
section.main,.main .block-container{background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;}
body{background:var(--bg)!important;}
#MainMenu,footer,header{visibility:hidden!important;}
.block-container{padding:2.4rem 1.6rem 6rem!important;max-width:960px!important;}
*{box-sizing:border-box;}
.app-header{display:flex;align-items:center;justify-content:space-between;
    padding:0 0 1.4rem;border-bottom:1px solid var(--border);margin-bottom:2.2rem;}
.app-logo{display:flex;align-items:center;gap:0.75rem;}
.logo-hex{width:34px;height:34px;background:var(--accent);
    clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
    display:flex;align-items:center;justify-content:center;
    font-family:var(--mono);font-size:0.6rem;font-weight:700;color:#000;flex-shrink:0;}
.logo-text{font-family:var(--mono);font-size:0.78rem;font-weight:600;letter-spacing:0.2em;color:var(--accent);text-transform:uppercase;}
.logo-sub{font-family:var(--mono);font-size:0.56rem;color:var(--muted2);letter-spacing:0.12em;margin-top:2px;}
.header-right{text-align:right;}
.header-ts{font-family:var(--mono);font-size:0.58rem;color:var(--muted2);letter-spacing:0.08em;}
.header-status{font-family:var(--mono);font-size:0.56rem;color:var(--buy);letter-spacing:0.12em;margin-top:3px;}
div.stButton>button{background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%)!important;
    color:#000!important;border:none!important;border-radius:2px!important;
    font-family:var(--mono)!important;font-size:0.72rem!important;font-weight:700!important;
    letter-spacing:0.2em!important;text-transform:uppercase!important;
    padding:0.85rem 2rem!important;width:100%!important;transition:all 0.15s!important;
    box-shadow:0 0 20px rgba(245,200,66,0.25)!important;}
div.stButton>button:hover{opacity:0.88!important;box-shadow:0 0 32px rgba(245,200,66,0.4)!important;}
.sig-banner{position:relative;border-radius:3px;overflow:hidden;margin:1.8rem 0;
    padding:2rem 2rem 2rem 2.4rem;display:flex;align-items:center;
    justify-content:space-between;background:var(--surf);}
.sig-banner::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;}
.sb-buy{border:1px solid rgba(16,217,136,0.3);box-shadow:var(--glow-buy);}
.sb-buy::before{background:var(--buy);}
.sb-sell{border:1px solid rgba(255,77,106,0.3);box-shadow:var(--glow-sell);}
.sb-sell::before{background:var(--sell);}
.sb-none{border:1px solid var(--border2);}
.sb-none::before{background:var(--muted);}
.sig-label{font-family:var(--mono);font-size:2.6rem;font-weight:700;letter-spacing:0.04em;line-height:1;}
.sb-buy .sig-label{color:var(--buy);text-shadow:0 0 30px rgba(16,217,136,0.5);}
.sb-sell .sig-label{color:var(--sell);text-shadow:0 0 30px rgba(255,77,106,0.5);}
.sb-none .sig-label{color:var(--muted2);}
.sig-sub{font-family:var(--mono);font-size:0.6rem;color:var(--muted2);letter-spacing:0.12em;text-transform:uppercase;margin-top:0.5rem;}
.sig-right{text-align:right;}
.sig-prob{font-family:var(--mono);font-size:1.4rem;font-weight:600;}
.sb-buy .sig-prob{color:var(--buy);}
.sb-sell .sig-prob{color:var(--sell);}
.sb-none .sig-prob{color:var(--muted2);}
.sig-prob-lbl{font-family:var(--mono);font-size:0.56rem;color:var(--muted2);letter-spacing:0.12em;text-transform:uppercase;margin-top:0.3rem;}
.sig-reason{font-family:var(--mono);font-size:0.6rem;color:var(--muted2);line-height:1.9;margin-top:0.5rem;}
.section-label{font-family:var(--mono);font-size:0.56rem;letter-spacing:0.22em;
    text-transform:uppercase;color:var(--muted2);padding:0 0 0.6rem;
    border-bottom:1px solid var(--border);margin:1.8rem 0 1rem;}
.kpi-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
    background:var(--border);border:1px solid var(--border);border-radius:2px;overflow:hidden;margin-bottom:1px;}
.kpi-cell{background:var(--surf);padding:1.1rem 1rem;}
.kpi-lbl{font-family:var(--mono);font-size:0.52rem;color:var(--muted2);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.4rem;}
.kpi-val{font-family:var(--mono);font-size:1.1rem;font-weight:600;color:var(--text);}
.kpi-val.gold{color:var(--accent);}
.kpi-val.bull{color:var(--buy);}
.kpi-val.bear{color:var(--sell);}
.kpi-val.muted{color:var(--muted2);}
.kpi-sub{font-family:var(--mono);font-size:0.52rem;color:var(--muted);margin-top:0.25rem;}
.spark-wrap{background:var(--surf);border:1px solid var(--border);border-radius:2px;padding:1.1rem 1.2rem;margin-bottom:1px;}
.spark-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;}
.spark-title{font-family:var(--mono);font-size:0.56rem;letter-spacing:0.16em;color:var(--muted2);text-transform:uppercase;}
.spark-price{font-family:var(--mono);font-size:0.78rem;font-weight:600;color:var(--accent);}
svg.spark{width:100%;height:56px;display:block;}
.dtbl{background:var(--surf);border:1px solid var(--border);border-radius:2px;overflow:hidden;}
.dtbl-row{display:flex;justify-content:space-between;align-items:center;
    padding:0.55rem 1.2rem;border-bottom:1px solid var(--border);}
.dtbl-row:last-child{border-bottom:none;}
.dtbl-row:hover{background:var(--surf2);}
.dtbl-k{font-family:var(--mono);font-size:0.62rem;color:var(--muted2);letter-spacing:0.08em;}
.dtbl-v{font-family:var(--mono);font-size:0.68rem;font-weight:500;color:var(--text);}
.dtbl-v.buy{color:var(--buy);}
.dtbl-v.sell{color:var(--sell);}
.dtbl-v.gold{color:var(--accent);}
.dtbl-v.muted{color:var(--muted2);}
/* news grid */
.news-tabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:1rem;}
.news-tab{font-family:var(--mono);font-size:0.58rem;letter-spacing:0.1em;
    padding:4px 10px;border-radius:2px;border:1px solid var(--border2);
    color:var(--muted2);cursor:pointer;background:var(--surf);transition:all 0.15s;}
.news-tab.active{border-color:var(--accent);color:var(--accent);}
.news-list{display:flex;flex-direction:column;gap:8px;}
.news-item{background:var(--surf);border:1px solid var(--border);border-radius:2px;
    padding:0.7rem 1rem;transition:border-color 0.15s;}
.news-item:hover{border-color:var(--border2);}
.news-item a{text-decoration:none;}
.news-src{font-family:var(--mono);font-size:0.5rem;letter-spacing:0.14em;
    text-transform:uppercase;margin-bottom:0.3rem;}
.news-title{font-family:var(--sans);font-size:0.78rem;font-weight:400;
    color:var(--text);line-height:1.5;margin-bottom:0.25rem;}
.news-age{font-family:var(--mono);font-size:0.5rem;color:var(--muted2);}
/* smc legend */
.smc-legend{display:flex;flex-wrap:wrap;gap:14px;font-family:var(--mono);
    font-size:0.58rem;color:var(--muted2);margin-top:0.5rem;padding:0 0.2rem;}
.model-row{display:flex;gap:1px;background:var(--border);margin-bottom:1px;}
.model-cell{background:var(--surf);padding:0.9rem 1.1rem;flex:1;}
.model-lbl{font-family:var(--mono);font-size:0.5rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--muted2);margin-bottom:0.3rem;}
.model-val{font-family:var(--mono);font-size:0.78rem;font-weight:500;color:var(--text);}
.idle{font-family:var(--mono);font-size:0.68rem;color:var(--muted);letter-spacing:0.1em;
    text-align:center;padding:4rem 0;border:1px dashed var(--border2);
    border-radius:2px;margin-top:1.6rem;background:var(--surf);}
.idle-sub{font-size:0.56rem;color:var(--muted);margin-top:0.5rem;letter-spacing:0.08em;}
.stSpinner>div{border-top-color:var(--accent)!important;}
[data-testid="stStatusWidget"]{display:none!important;}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# SPARKLINE
# ══════════════════════════════════════════════════════════════════════════════
def make_sparkline(prices: pd.Series, color: str) -> str:
    vals = prices.dropna().values.astype(float)
    if len(vals) < 2:
        return ""
    mn, mx = vals.min(), vals.max()
    rng = mx - mn if mx != mn else 1
    W, H, PAD = 400, 56, 4
    pts = []
    for i, v in enumerate(vals):
        x = PAD + (i / (len(vals) - 1)) * (W - 2 * PAD)
        y = PAD + (1 - (v - mn) / rng) * (H - 2 * PAD)
        pts.append(f"{x:.1f},{y:.1f}")
    poly     = " ".join(pts)
    fill_pts = f"{PAD},{H-PAD} " + poly + f" {W-PAD},{H-PAD}"
    return (
        f'<svg class="spark" viewBox="0 0 {W} {H}" preserveAspectRatio="none">'
        f'<defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.25"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
        f'</linearGradient></defs>'
        f'<polygon points="{fill_pts}" fill="url(#sg)"/>'
        f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round"/>'
        f'</svg>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _tbl_row(k, v, cls=""):
    return (f'<div class="dtbl-row">'
            f'<span class="dtbl-k">{k}</span>'
            f'<span class="dtbl-v {cls}">{v}</span>'
            f'</div>')

def _kpi(lbl, val, cls="", sub=""):
    s = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi-cell">'
            f'<div class="kpi-lbl">{lbl}</div>'
            f'<div class="kpi-val {cls}">{val}</div>{s}</div>')

def _section(label):
    return f'<div class="section-label">{label}</div>'


# ══════════════════════════════════════════════════════════════════════════════
# NEWS RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def _render_news(gold_price: float = 0.0):
    st.markdown(_section("MARKET INTELLIGENCE — 6 ASSET BRIEFS"), unsafe_allow_html=True)

    all_headlines = fetch_all_headlines()

    effect_colors = {
        "bullish": "#10d988",
        "bearish": "#ff4d6a",
        "mixed":   "#f5c842",
        "neutral": "#5a6a80",
    }

    news_css = """
    <style>
    .brief-wrap{display:flex;flex-direction:column;gap:24px;margin-top:4px}
    .brief-card{background:#0c0f14;border:1px solid #1c2030;border-radius:6px;overflow:hidden}
    .brief-top{padding:14px 16px 10px;border-bottom:1px solid #1c2030}
    .brief-asset{font-family:'JetBrains Mono',monospace;font-size:0.62rem;font-weight:600;letter-spacing:0.18em;margin-bottom:6px}
    .brief-headline{font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:500;color:#e2e8f0;line-height:1.45}
    .brief-body{padding:14px 16px;font-family:'Space Grotesk',sans-serif;font-size:0.82rem;color:#9ca8b8;line-height:1.8}
    .brief-body p{margin-bottom:10px}
    .brief-body p:last-child{margin-bottom:0}
    .brief-effect{padding:10px 16px;border-top:1px solid #1c2030;font-family:'JetBrains Mono',monospace;font-size:0.6rem;line-height:1.6}
    .brief-sources{padding:8px 16px;border-top:1px solid #111520;display:flex;flex-wrap:wrap;gap:8px}
    .brief-src-link{font-family:'JetBrains Mono',monospace;font-size:0.5rem;color:#3d4a5c;text-decoration:none;transition:color 0.15s}
    .brief-src-link:hover{color:#5a6a80}
    .brief-src-age{color:#252a38;margin-left:2px}
    </style>
    """

    cards_html = '<div class="brief-wrap">'

    for asset in NEWS_ASSETS:
        label     = asset["label"]
        color     = asset["color"]
        headlines = all_headlines.get(label, [])

        with st.spinner(f"Writing {label} brief..."):
            article = _ai_write_article(label, headlines, gold_price)

        edir  = article.get("effect_direction", "neutral")
        ecol  = effect_colors.get(edir, "#5a6a80")
        body_paragraphs = "".join(
            f"<p>{p.strip()}</p>"
            for p in article.get("body", "").split("\n\n")
            if p.strip()
        )

        sources_html = ""
        for h in headlines[:6]:
            title_short = h["title"][:60] + ("..." if len(h["title"]) > 60 else "")
            sources_html += (
                f'<a class="brief-src-link" href="{h["url"]}" target="_blank">'
                f'{title_short}'
                f'<span class="brief-src-age">&nbsp;{h["age"]}</span>'
                f'</a>'
            )

        if not sources_html:
            sources_html = '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.5rem;color:#252a38">No source links available</span>'

        cards_html += f"""
        <div class="brief-card">
          <div class="brief-top">
            <div class="brief-asset" style="color:{color}">{label}</div>
            <div class="brief-headline">{article.get("headline","")}</div>
          </div>
          <div class="brief-body">{body_paragraphs}</div>
          <div class="brief-effect" style="color:{ecol}">
            GOLD IMPACT &nbsp;·&nbsp; {article.get("effect","")}
          </div>
          <div class="brief-sources">{sources_html}</div>
        </div>"""

    cards_html += "</div>"
    st.markdown(news_css + cards_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SMC RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def _render_smc(ml_signal: str = "NO SIGNAL"):
    st.markdown(_section("XAU/USD · 4H · SMART MONEY LEVELS"), unsafe_allow_html=True)

    with st.spinner("Computing SMC levels from 4H data..."):
        df4 = fetch_4h()

    if df4.empty or len(df4) < SMC_SWING_LENGTH * 2 + 5:
        st.warning("Not enough 4H data — try again shortly.")
        return

    swings     = _find_swings(df4)
    bos_events = _find_bos_choch(df4, swings)
    obs        = _find_order_blocks(df4, bos_events)
    key_levels = _find_key_levels(df4, swings)

    current    = float(df4["Close"].iloc[-1])
    active_obs = [o for o in obs if not o["mitigated"]]
    last_ev    = bos_events[-1] if bos_events else None
    last_str   = f"{last_ev['type']} ({'Bull' if last_ev['direction']==1 else 'Bear'})" if last_ev else "—"
    trend_col  = "#10d988" if last_ev and last_ev["direction"] == 1 else "#ff4d6a" if last_ev else "#5a6a80"
    sig_col    = "#10d988" if ml_signal == "BUY" else "#ff4d6a" if ml_signal == "SELL" else "#5a6a80"

    all_levels = []
    for lvl in key_levels:
        all_levels.append({"price": lvl["price"], "type": "S/R", "dir": "Neutral",
                           "top": lvl["top"], "bot": lvl["bottom"], "hits": lvl["hits"],
                           "col": "#f5c842"})
    for ob in active_obs:
        mid = (ob["top"] + ob["bottom"]) / 2
        d   = "Bull" if ob["direction"] == "bull" else "Bear"
        c   = "#10d988" if ob["direction"] == "bull" else "#ff4d6a"
        all_levels.append({"price": mid, "type": "Order Block", "dir": d,
                           "top": ob["top"], "bot": ob["bottom"], "hits": 1, "col": c})
    for ev in bos_events[-6:]:
        d = "Bull" if ev["direction"] == 1 else "Bear"
        c = "#10d988" if ev["direction"] == 1 else "#ff4d6a"
        t = ev["type"]
        if t == "CHoCH":
            c = "#5DCAA5" if ev["direction"] == 1 else "#F0997B"
        all_levels.append({"price": ev["level"], "type": t, "dir": d,
                           "top": ev["level"] * 1.001, "bot": ev["level"] * 0.999,
                           "hits": 1, "col": c})

    seen, deduped = [], []
    for lv in sorted(all_levels, key=lambda x: x["price"], reverse=True):
        if not any(abs(lv["price"] - s) / lv["price"] < SMC_SR_TOLERANCE * 2 for s in seen):
            seen.append(lv["price"])
            deduped.append(lv)

    resistance = sorted([l for l in deduped if l["price"] > current], key=lambda x: x["price"])
    support    = sorted([l for l in deduped if l["price"] < current], key=lambda x: x["price"], reverse=True)

    def row(lv, is_res):
        dist    = (lv["price"] - current) / current * 100
        dc      = "#ff4d6a" if is_res else "#10d988"
        dist_s  = f"+{dist:.2f}%" if dist > 0 else f"{dist:.2f}%"
        hits_s  = f"{lv['hits']}x" if lv["hits"] > 1 else ""
        zone_s  = f"${lv['bot']:,.1f} – ${lv['top']:,.1f}"
        return (
            f'<div class="smc-row">'
            f'<span class="smc-price">${lv["price"]:,.1f}</span>'
            f'<span class="smc-tag" style="color:{lv["col"]};border-color:{lv["col"]}40">{lv["type"]}</span>'
            f'<span class="smc-dir" style="color:{lv["col"]}">{lv["dir"]}</span>'
            f'<span class="smc-zone">{zone_s}</span>'
            f'<span class="smc-hits">{hits_s}</span>'
            f'<span class="smc-dist" style="color:{dc}">{dist_s}</span>'
            f'</div>'
        )

    res_rows = "".join(row(l, True)  for l in resistance[:8])
    sup_rows = "".join(row(l, False) for l in support[:8])

    html = f"""
    <style>
    .smc-wrap{{font-family:'JetBrains Mono',monospace;margin-top:4px}}
    .smc-col-hdr{{display:grid;grid-template-columns:90px 100px 60px 1fr 30px 60px;
        gap:0;padding:4px 10px 6px;font-size:0.5rem;letter-spacing:0.12em;
        color:#3d4a5c;border-bottom:1px solid #1c2030;margin-bottom:4px}}
    .smc-row{{display:grid;grid-template-columns:90px 100px 60px 1fr 30px 60px;
        gap:0;padding:6px 10px;border-radius:4px;margin-bottom:2px;
        background:#0c0f1408;transition:background 0.15s}}
    .smc-row:hover{{background:#0c0f14}}
    .smc-price{{font-size:0.72rem;font-weight:500;color:#e2e8f0}}
    .smc-tag{{font-size:0.55rem;padding:2px 6px;border-radius:3px;
        border:1px solid;display:inline-block;align-self:center;letter-spacing:0.05em}}
    .smc-dir{{font-size:0.58rem;color:#5a6a80;align-self:center}}
    .smc-zone{{font-size:0.55rem;color:#3d4a5c;align-self:center}}
    .smc-hits{{font-size:0.55rem;color:#5a6a80;align-self:center;text-align:center}}
    .smc-dist{{font-size:0.65rem;font-weight:500;text-align:right;align-self:center}}
    .smc-divider{{display:flex;align-items:center;gap:10px;margin:10px 0;padding:0 10px}}
    .smc-cur-price{{font-size:1.1rem;font-weight:500;color:#e2e8f0}}
    .smc-cur-line{{flex:1;height:1px;background:#252a38}}
    .smc-zone-label{{font-size:0.5rem;letter-spacing:0.2em;color:#3d4a5c;
        padding:4px 10px 3px;margin-bottom:2px}}
    </style>
    <div class="smc-wrap">
      <div class="smc-col-hdr">
        <span>PRICE</span><span>TYPE</span><span>DIR</span>
        <span>ZONE</span><span>HITS</span><span style="text-align:right">DIST</span>
      </div>
      <div class="smc-zone-label">RESISTANCE</div>
      {res_rows}
      <div class="smc-divider">
        <div class="smc-cur-line"></div>
        <div>
          <div class="smc-cur-price">${current:,.1f}</div>
          <div style="font-size:0.5rem;color:#5a6a80;margin-top:1px">
            CURRENT &nbsp;·&nbsp;
            <span style="color:{trend_col}">{last_str}</span>
            &nbsp;·&nbsp; ML: <span style="color:{sig_col}">{ml_signal}</span>
          </div>
        </div>
        <div class="smc-cur-line"></div>
      </div>
      <div class="smc-zone-label">SUPPORT</div>
      {sup_rows}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="XAUUSD Intelligence",
        page_icon="⬡",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    now = datetime.now()
    st.markdown(f"""
    <div class="app-header">
      <div class="app-logo">
        <div class="logo-hex">XAU</div>
        <div>
          <div class="logo-text">XAUUSD &nbsp; Intelligence</div>
          <div class="logo-sub">ML Signal System &nbsp;·&nbsp; Stage 1 + Stage 2 + SMC</div>
        </div>
      </div>
      <div class="header-right">
        <div class="header-ts">{now.strftime('%Y-%m-%d &nbsp; %H:%M:%S UTC')}</div>
        <div class="header-status">● LIVE</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    run = st.button("▶  GENERATE TODAY'S SIGNAL", use_container_width=True)

    if not run:
        st.markdown("""
        <div class="idle">
            Press ▶ to generate today's ML signal
            <div class="idle-sub">Fetches live market data · Runs LightGBM + Calibrator · Outputs probability-weighted signal</div>
        </div>
        """, unsafe_allow_html=True)
        _render_smc("NO SIGNAL")
        _render_news(gold_price=0.0)
        return

    # ── run inference ──────────────────────────────────────────────────────────
    try:
        end   = now
        start = end - timedelta(days=DAYS_BACK)
        with st.spinner("Fetching market data  (GC=F · EURUSD · USDJPY · FRED) ..."):
            raw = fetch_data(start, end)
        with st.spinner("Engineering 15 features ..."):
            feat_df = engineer_features(raw)
        with st.spinner("Loading model artefacts ..."):
            base_model, calibrator, oof_history = load_artefacts()
        with st.spinner("Running Stage 1 + Stage 2 inference ..."):
            r = run_inference(feat_df, base_model, calibrator, oof_history)
    except FileNotFoundError as e:
        st.error(f"Missing artefact: {e}")
        return
    except Exception as e:
        st.exception(e)
        return

    sig  = r["signal"]
    sc   = {"BUY": "sb-buy", "SELL": "sb-sell", "NO SIGNAL": "sb-none"}[sig]
    vc   = {"BUY": "buy",    "SELL": "sell",     "NO SIGNAL": "muted"}[sig]
    bt   = r["bull_trend"]
    regime_label = "BULL" if bt > 0.02 else "BEAR" if bt < -0.02 else "NEUTRAL"
    regime_cls   = "bull" if bt > 0.02 else "bear" if bt < -0.02 else "muted"

    reason_html = ""
    if sig == "NO SIGNAL":
        reasons = []
        if abs(r["pred_z"]) < Z_THRESHOLD:
            reasons.append(f"|z| {r['pred_z']:.2f} &lt; {Z_THRESHOLD} (low conviction)")
        if r["prob"] < PROB_THRESHOLD:
            reasons.append(f"prob {r['prob']:.4f} &lt; {PROB_THRESHOLD} (below threshold)")
        reason_html = "<br>".join(reasons)

    # signal banner
    st.markdown(f"""
    <div class="sig-banner {sc}">
      <div>
        <div class="sig-label">{sig}</div>
        <div class="sig-sub">XAUUSD · {r['date'].strftime('%Y-%m-%d')} · Daily Close</div>
        <div class="sig-reason">{reason_html}</div>
      </div>
      <div class="sig-right">
        <div class="sig-prob">{r['prob']:.1%}</div>
        <div class="sig-prob-lbl">Win Probability</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # sparkline
    spark_color = "#10d988" if r["recent"].iloc[-1] >= r["recent"].iloc[0] else "#ff4d6a"
    chg_pct     = (r["recent"].iloc[-1] / r["recent"].iloc[0] - 1) * 100
    spark_svg   = make_sparkline(r["recent"], spark_color)
    st.markdown(f"""
    <div class="spark-wrap">
      <div class="spark-hdr">
        <span class="spark-title">XAU/USD · 20-Day Price</span>
        <span class="spark-price">${r['close']:,.2f}
          <span style="font-size:0.62rem;color:var(--{'buy' if chg_pct>=0 else 'sell'})">
            &nbsp;{'+' if chg_pct>=0 else ''}{chg_pct:.2f}%
          </span>
        </span>
      </div>
      {spark_svg}
    </div>
    """, unsafe_allow_html=True)

    # KPI strips
    macro_lbl = "TIGHT" if r["macro_fast"] > 0 else "EASY"
    macro_cls = "sell" if r["macro_fast"] > 0.5 else "bull" if r["macro_fast"] < -0.5 else ""
    pct_from_ath = r["feat_df"]["Pct_From_AllTimeHigh"].iloc[-1]
    st.markdown(f"""
    <div class="kpi-strip">
      {_kpi("XAU/USD Close",   f"${r['close']:,.2f}",                   "gold")}
      {_kpi("Market Regime",   f"{regime_label} {bt:+.3f}",             regime_cls)}
      {_kpi("Win Probability", f"{r['prob']:.1%}",                      vc)}
      {_kpi("Macro Pressure",  f"{macro_lbl} ({r['macro_fast']:+.2f})", macro_cls)}
    </div>
    <div class="kpi-strip">
      {_kpi("BB %B",           f"{r['bb_pctb']:.3f}",    "gold" if r['bb_pctb']>0.8 else "")}
      {_kpi("EMA200 Ratio",    f"{r['ema200']:.4f}",     "bull" if r['ema200']>1 else "bear")}
      {_kpi("Pred Z-Score",    f"{r['pred_z']:+.3f}",    "bull" if r['pred_z']>0 else "bear")}
      {_kpi("% From ATH",      f"{pct_from_ath:.2%}",   "muted")}
    </div>
    """, unsafe_allow_html=True)

    # full output table
    st.markdown(_section("MODEL OUTPUT — FULL DETAIL"), unsafe_allow_html=True)
    rows = [
        ("DATE",            r["date"].strftime("%Y-%m-%d"),           ""),
        ("CLOSE PRICE",     f"${r['close']:,.2f}",                    "gold"),
        ("SIGNAL",          sig,                                       vc),
        ("WIN PROBABILITY", f"{r['prob']:.4f}  ({r['prob']:.1%})",    vc),
        ("BASE PREDICTION", f"{r['pred_val']:+.8f}",                  ""),
        ("ABS PREDICTION",  f"{r['abs_pred']:.8f}",                   ""),
        ("PRED Z-SCORE",    f"{r['pred_z']:+.4f}",                    ""),
        ("|Z| CONVICTION",  f"{r['abs_pred_z']:.4f}",
            "bull" if r['abs_pred_z'] >= Z_THRESHOLD else "muted"),
        ("BULL TREND",      f"{r['bull_trend']:+.4f}  ({regime_label})",  regime_cls),
        ("MACRO FAST",      f"{r['macro_fast']:+.4f}  ({macro_lbl})",     macro_cls),
        ("BB PCTB",         f"{r['bb_pctb']:.4f}",                    ""),
        ("PRICE / EMA200",  f"{r['ema200']:.4f}",
            "bull" if r['ema200'] > 1 else "bear"),
        ("PROB THRESHOLD",  f"{PROB_THRESHOLD}",                      "muted"),
        ("Z THRESHOLD",     f"{Z_THRESHOLD}",                         "muted"),
    ]
    rows_html = "".join(_tbl_row(k, v, c) for k, v, c in rows)
    st.markdown(f'<div class="dtbl">{rows_html}</div>', unsafe_allow_html=True)

    # feature snapshot
    st.markdown(_section("LIVE FEATURE SNAPSHOT — LAST 5 TRADING DAYS"), unsafe_allow_html=True)
    snap_cols = ["Close_XAUUSD", "Bull_Trend", "Macro_Fast", "BB_PctB",
                 "Price_Over_EMA200", "Return_ZScore", "Volume_Percentile"]
    snap = r["feat_df"][snap_cols].tail(5).copy()
    snap["Close_XAUUSD"] = snap["Close_XAUUSD"].map("${:,.2f}".format)
    snap.index = snap.index.strftime("%Y-%m-%d")
    snap_html = snap.to_html(classes="", border=0)
    st.markdown(f"""
    <div class="dtbl" style="overflow-x:auto;">
      <style>
        .dtbl table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:0.6rem;}}
        .dtbl th{{padding:0.55rem 0.8rem;border-bottom:1px solid var(--border);
            color:var(--muted2);letter-spacing:0.1em;text-align:left;
            background:var(--bg);font-weight:400;}}
        .dtbl td{{padding:0.5rem 0.8rem;border-bottom:1px solid var(--border);color:var(--text);}}
        .dtbl tr:last-child td{{border-bottom:none;}}
        .dtbl tr:hover td{{background:var(--surf2);}}
      </style>
      {snap_html}
    </div>
    """, unsafe_allow_html=True)

    # model info
    st.markdown(_section("MODEL ARTEFACTS"), unsafe_allow_html=True)
    st.markdown(f"""
    <div class="model-row">
      <div class="model-cell"><div class="model-lbl">Stage 1</div><div class="model-val">LightGBM Regressor</div></div>
      <div class="model-cell"><div class="model-lbl">Features</div><div class="model-val">{len(BASE_FEATURES)} inputs</div></div>
      <div class="model-cell"><div class="model-lbl">Target</div><div class="model-val">Log Return (t+1)</div></div>
    </div>
    <div class="model-row">
      <div class="model-cell"><div class="model-lbl">Stage 2</div><div class="model-val">Logistic Regression</div></div>
      <div class="model-cell"><div class="model-lbl">Features</div><div class="model-val">{len(CALIB_FEATURES)} inputs</div></div>
      <div class="model-cell"><div class="model-lbl">Target</div><div class="model-val">P(direction correct)</div></div>
    </div>
    <div class="model-row">
      <div class="model-cell"><div class="model-lbl">Entry Gate</div><div class="model-val">Prob ≥ {PROB_THRESHOLD} AND |Z| ≥ {Z_THRESHOLD}</div></div>
      <div class="model-cell"><div class="model-lbl">OOF Window</div><div class="model-val">Last {PRED_Z_LOOKBACK} days</div></div>
      <div class="model-cell"><div class="model-lbl">Data Fetch</div><div class="model-val">{DAYS_BACK}-day window</div></div>
    </div>
    """, unsafe_allow_html=True)

    _render_smc(sig)

    _render_news(gold_price=r["close"])


if __name__ == "__main__":
    main()