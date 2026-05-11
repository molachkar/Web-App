"""
core/config.py
All constants, thresholds, paths, and API keys for the Sentinel trading terminal.
No dependencies on other project modules.
"""

import os
from datetime import timezone, timedelta

# ── ROOT ──────────────────────────────────────────────────────────────────────
# Get the directory where this config file is located
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is one level up from core/config.py (since core/ is inside project/)
PROJECT_ROOT = CONFIG_DIR  # Already at project level since we're in core/
if os.path.basename(CONFIG_DIR) == "core":
    PROJECT_ROOT = os.path.dirname(CONFIG_DIR)

ARTEFACT_DIR  = os.path.join(PROJECT_ROOT, "ml", "artefacts")
FRED_CACHE_DIR = os.path.join(PROJECT_ROOT, "fred_cache")
FRONTEND_DIR  = os.path.join(PROJECT_ROOT, "frontend")

# ── TIMEZONES ─────────────────────────────────────────────────────────────────
NY_TZ      = timezone(timedelta(hours=-5))   # EST (no DST handling; adjust if needed)
MOROCCO_TZ = timezone(timedelta(hours=1))    # WAT (Morocco)
UTC_TZ     = timezone.utc

# ── TRADING SESSION TIMES (NY hours, 24h float) ──────────────────────────────
SETTLEMENT_NY_HOUR   = 13.5   # 13:30 NY = gold daily close
MAINTENANCE_START_NY = 21.25  # 21:15 NY = CME maintenance window start
MAINTENANCE_END_NY   = 22.0   # 22:00 NY = CME maintenance window end

# ── ML MODEL ─────────────────────────────────────────────────────────────────
DAYS_BACK       = 520          # how far back to fetch for feature engineering
PROB_THRESHOLD  = 0.45         # minimum calibrated probability to emit BUY/SELL
Z_THRESHOLD     = 0.6          # minimum |pred_z| to emit BUY/SELL
PRED_Z_LOOKBACK = 252          # rolling window (trading days) for z-score normalisation

BASE_FEATURES = [
    "Close_Returns",
    "Log_Returns",
    "EURUSD_Returns",
    "USDJPY_Returns",
    "BB_PctB",
    "Price_Over_EMA50",
    "Price_Over_EMA200",
    "MACD_Signal_Norm",
    "LogReturn_ZScore",
    "Return_ZScore",
    "Return_Percentile",
    "Volume_Percentile",
    "Pct_From_AllTimeHigh",
    "Bull_Trend",
    "Macro_Fast",
]

CALIB_FEATURES = [
    "prediction_value",
    "abs_prediction",
    "Bull_Trend",
    "Macro_Fast",
    "BB_PctB",
    "Price_Over_EMA200",
]

# ── FRED MACRO SERIES ─────────────────────────────────────────────────────────
FRED_API_KEY  = "219d0c44b2e3b4a8b690c3f69b91a5bb"
MACRO_SERIES  = ["DFII10", "DFII5", "DGS2", "FEDFUNDS"]

# ── PRICE STRIP TICKERS ───────────────────────────────────────────────────────
# label: (yfinance_ticker, display_symbol)
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

# ── SMC ENGINE ────────────────────────────────────────────────────────────────
SMC_SWING_LENGTH = 5        # bars each side to qualify a swing high/low
SMC_SR_TOLERANCE = 0.0015   # 0.15% price tolerance for S/R clustering
SMC_SR_MIN_HITS  = 2        # minimum touches to confirm an S/R level
SMC_OB_EXTEND    = 40       # max bars to extend an order block rectangle
SMC_4H_BARS      = 200      # how many 4H candles to fetch for SMC analysis

# ── DOM / TCP ─────────────────────────────────────────────────────────────────
TCP_HOST = "0.0.0.0"
TCP_PORT = 5555

# ── FASTAPI / WS ─────────────────────────────────────────────────────────────
WS_HOST = "0.0.0.0"
WS_PORT = 8000

# ── CACHE ─────────────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 3600        # 1 hour: re-fetch signal / SMC after this
PRICES_TTL_SECONDS = 30         # price strip refresh interval
FRED_STALE_DAYS   = 2           # warn if FRED series older than this many days

# ── UI COLOURS (mirrored in frontend for reference) ───────────────────────────
C_BG    = "#05070a"
C_SURF  = "#0c0f14"
C_BORDER = "#1c2030"
C_TEXT  = "#e2e8f0"
C_MUTED = "#5a6a80"
C_GOLD  = "#f5c842"
C_BUY   = "#10d988"
C_SELL  = "#ff4d6a"
C_BID   = "#4a8fd4"
C_ASK   = "#c94040"