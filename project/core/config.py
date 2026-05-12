"""
core/config.py
Single source of truth for all constants, thresholds, paths, and API keys.
Zero dependencies — no imports from within the project.
"""

import os
from pathlib import Path
from datetime import timezone, timedelta

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
BASE_DIR       = PROJECT_ROOT
ARTEFACT_DIR   = PROJECT_ROOT / "ml" / "artefacts"
FRED_CACHE_DIR = PROJECT_ROOT / "fred_cache"
FRONTEND_DIR   = PROJECT_ROOT / "frontend"
CACHE_FILE     = PROJECT_ROOT / "daily_cache.json"

# ── API keys (set FRED_API_KEY env var in production) ────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "219d0c44b2e3b4a8b690c3f69b91a5bb")

# ── ML pipeline thresholds ────────────────────────────────────────────────────
DAYS_BACK       = 520
PROB_THRESHOLD  = 0.45
Z_THRESHOLD     = 0.6
PRED_Z_LOOKBACK = 252

# Must exactly match what the trained model/calibrator was built on
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

# ── FRED macro series ─────────────────────────────────────────────────────────
MACRO_SERIES    = ["DFII10", "DFII5", "DGS2", "FEDFUNDS"]
FRED_STALE_DAYS = 3

# ── SMC engine parameters ─────────────────────────────────────────────────────
SMC_SWING_LENGTH  = 5
SMC_SR_TOLERANCE  = 0.0015
SMC_SR_MIN_HITS   = 2
SMC_OB_EXTEND     = 40
SMC_LOOKBACK_DAYS = 58

# ── Timezones ─────────────────────────────────────────────────────────────────
UTC_TZ     = timezone.utc
NY_TZ      = timezone(timedelta(hours=-5))
MOROCCO_TZ = timezone(timedelta(hours=1))

# ── Session thresholds (decimal hours, NY time) ───────────────────────────────
SETTLEMENT_NY        = 13.5    # 1:30 PM NY — CME gold settle
SETTLEMENT_NY_HOUR   = 13.5    # alias used by candle_validator
MAINTENANCE_START    = 21.25   # 9:15 PM NY — CME maintenance begins
MAINTENANCE_END      = 22.0    # 10:00 PM NY — CME maintenance ends
MAINTENANCE_START_NY = 21.25   # same, referenced explicitly as NY decimal hour
MAINTENANCE_END_NY   = 22.0

# ── Stooq fallback ticker map ─────────────────────────────────────────────────
STOOQ_MAP = {
    "GC=F":     "GC.F",
    "EURUSD=X": "EURUSD",
    "JPY=X":    "JPYUSD",
}

# ── Price strip (symbol → yfinance ticker, ordered for frontend display) ──────
PRICE_STRIP = {
    "XAU":    "GC=F",
    "SPX":    "^GSPC",
    "NAS":    "^IXIC",
    "BTC":    "BTC-USD",
    "SILVER": "SI=F",
    "EUR":    "EURUSD=X",
    "JPY":    "JPY=X",
    "OIL":    "CL=F",
}

# ── News assets registry ──────────────────────────────────────────────────────
NEWS_ASSETS = [
    {"label": "XAU/USD", "ticker": "GC=F",   "color": "#f5c842",
     "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC%3DF&region=US&lang=en-US"},
    {"label": "S&P 500", "ticker": "^GSPC",  "color": "#378ADD",
     "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US"},
    {"label": "Oil",     "ticker": "CL=F",   "color": "#D85A30",
     "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=CL%3DF&region=US&lang=en-US"},
    {"label": "Nasdaq",  "ticker": "^IXIC",  "color": "#7F77DD",
     "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EIXIC&region=US&lang=en-US"},
    {"label": "Bitcoin", "ticker": "BTC-USD","color": "#EF9F27",
     "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US"},
    {"label": "Silver",  "ticker": "SI=F",   "color": "#B4B2A9",
     "rss": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SI%3DF&region=US&lang=en-US"},
]

GOLD_EFFECT_CONTEXT = {
    "XAU/USD":  "Direct asset. Analyse its own technicals, macro drivers, and momentum.",
    "S&P 500":  "Inverse risk-off proxy. Equity selloffs channel capital into gold.",
    "Oil":      "Inflation driver. Rising oil lifts CPI expectations, compresses real yields → bullish gold.",
    "Nasdaq":   "Risk appetite barometer. Nasdaq weakness precedes gold inflows.",
    "Bitcoin":  "Competing safe haven. Stress: BTC sells first, gold benefits. Medium-term: both rise on USD weakness.",
    "Silver":   "High-beta gold proxy. Amplifies gold moves. Supply deficit adds independent pressure.",
}

# ── UI color palette ──────────────────────────────────────────────────────────
C_BG     = "#05070a"
C_SURF   = "#0c0f14"
C_SURF2  = "#111520"
C_BORDER = "#1c2030"
C_TEXT   = "#e2e8f0"
C_MUTED  = "#5a6a80"
C_GOLD   = "#f5c842"
C_BUY    = "#10d988"
C_SELL   = "#ff4d6a"
C_PURPLE = "#a78bfa"
C_BLUE   = "#378ADD"

# ── Network ───────────────────────────────────────────────────────────────────
TCP_HOST = "0.0.0.0"
TCP_PORT = 5555
WS_HOST  = "0.0.0.0"
WS_PORT  = 8000

# ── Cache TTLs  (BUG 1 FIX — were missing, crashed scheduler.py on import) ───
CACHE_TTL_SECONDS  = 3600   # 1 hour  — signal + SMC route cache
PRICES_TTL_SECONDS = 30     # 30 secs — price strip route cache