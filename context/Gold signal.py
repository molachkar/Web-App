import os, json, pickle, warnings, time, requests
import numpy as np
import pandas as pd
import streamlit as st
from fredapi import Fred
from datetime import datetime, timedelta, timezone
import yfinance as yf

try:
    from qwen_briefing import render_qwen_section
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

warnings.filterwarnings("ignore")

ARTEFACT_DIR      = os.path.dirname(os.path.abspath(__file__))
FRED_API_KEY      = "219d0c44b2e3b4a8b690c3f69b91a5bb"
MACRO_SERIES      = ["DFII10", "DFII5", "DGS2", "FEDFUNDS"]
DAYS_BACK         = 520
PRED_Z_LOOKBACK   = 252
PROB_THRESHOLD    = 0.45
Z_THRESHOLD       = 0.6
SMC_SWING         = 5
SR_TOL            = 0.0015
NY_TZ             = timezone(timedelta(hours=-5))
MOROCCO_TZ        = timezone(timedelta(hours=1))
SETTLEMENT_NY     = 13.5
MAINTENANCE_START = 21.25
MAINTENANCE_END   = 22.0

BASE_FEATURES = [
    "Close_Returns","Log_Returns","EURUSD_Returns","USDJPY_Returns",
    "BB_PctB","Price_Over_EMA50","Price_Over_EMA200","MACD_Signal_Norm",
    "LogReturn_ZScore","Return_ZScore","Return_Percentile","Volume_Percentile",
    "Pct_From_AllTimeHigh","Bull_Trend","Macro_Fast",
]
CALIB_FEATURES = [
    "prediction_value","abs_prediction",
    "Bull_Trend","Macro_Fast","BB_PctB","Price_Over_EMA200",
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');
:root{
    --bg:#05070a;--surf:#0c0f14;--surf2:#111520;--border:#1c2030;
    --text:#e2e8f0;--sub:#e2e8f0;
    --accent:#f5c842;--buy:#10d988;--sell:#ff4d6a;--purple:#a78bfa;
    --mono:'JetBrains Mono',monospace;--sans:'Space Grotesk',sans-serif;
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
.logo-sub{font-family:var(--mono);font-size:0.56rem;color:#e2e8f0;letter-spacing:0.12em;margin-top:2px;}
.header-right{text-align:right;}
.header-ts{font-family:var(--mono);font-size:0.58rem;color:#e2e8f0;letter-spacing:0.08em;}
.header-status{font-family:var(--mono);font-size:0.56rem;letter-spacing:0.12em;margin-top:3px;}
.section-label{font-family:var(--mono);font-size:0.54rem;letter-spacing:0.24em;
    text-transform:uppercase;color:#e2e8f0;padding:0 0 0.5rem;
    border-bottom:1px solid var(--border);margin:1.6rem 0 0.9rem;}
.sig-banner{position:relative;border-radius:6px;overflow:hidden;margin:1.2rem 0;
    padding:1.6rem 2rem 1.6rem 2.4rem;display:flex;align-items:center;
    justify-content:space-between;background:var(--surf);}
.sig-banner::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;}
.sb-buy{border:1px solid rgba(16,217,136,0.35);box-shadow:var(--glow-buy);}
.sb-buy::before{background:var(--buy);}
.sb-sell{border:1px solid rgba(255,77,106,0.35);box-shadow:var(--glow-sell);}
.sb-sell::before{background:var(--sell);}
.sb-none{border:1px solid var(--border);}
.sb-none::before{background:#e2e8f0;}
.sig-label{font-family:var(--mono);font-size:2.6rem;font-weight:700;letter-spacing:0.04em;line-height:1;}
.sb-buy .sig-label{color:var(--buy);text-shadow:0 0 30px rgba(16,217,136,0.5);}
.sb-sell .sig-label{color:var(--sell);text-shadow:0 0 30px rgba(255,77,106,0.5);}
.sb-none .sig-label{color:#e2e8f0;}
.sig-sub{font-family:var(--mono);font-size:0.58rem;color:#e2e8f0;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.4rem;}
.sig-right{text-align:right;}
.sig-prob{font-family:var(--mono);font-size:1.4rem;font-weight:600;}
.sb-buy .sig-prob{color:var(--buy);}
.sb-sell .sig-prob{color:var(--sell);}
.sb-none .sig-prob{color:#e2e8f0;}
.sig-prob-lbl{font-family:var(--mono);font-size:0.52rem;color:#e2e8f0;letter-spacing:0.12em;text-transform:uppercase;margin-top:0.3rem;}
.sig-reason{font-family:var(--mono);font-size:0.6rem;color:#e2e8f0;line-height:1.9;margin-top:0.4rem;}
.dtbl{background:var(--surf);border:1px solid var(--border);border-radius:6px;overflow:hidden;}
.dtbl-row{display:flex;justify-content:space-between;align-items:center;
    padding:0.5rem 1.2rem;border-bottom:1px solid var(--border);}
.dtbl-row:last-child{border-bottom:none;}
.dtbl-row:hover{background:var(--surf2);}
.dtbl-k{font-family:var(--mono);font-size:0.6rem;color:#e2e8f0;letter-spacing:0.06em;}
.dtbl-v{font-family:var(--mono);font-size:0.68rem;font-weight:500;color:#e2e8f0;}
.dtbl-v.buy{color:var(--buy);}
.dtbl-v.sell{color:var(--sell);}
.dtbl-v.gold{color:var(--accent);}
.dtbl-v.muted{color:#e2e8f0;}
.candle-status{font-family:var(--mono);font-size:0.62rem;padding:9px 14px;border-radius:4px;margin:0.5rem 0;}
.cs-ok{background:rgba(16,217,136,0.07);border:1px solid rgba(16,217,136,0.25);color:var(--buy);}
.cs-warn{background:rgba(245,200,66,0.07);border:1px solid rgba(245,200,66,0.25);color:var(--accent);}
.cs-danger{background:rgba(255,77,106,0.07);border:1px solid rgba(255,77,106,0.25);color:var(--sell);}
.feat-table-wrap{overflow-x:auto;overflow-y:auto;max-height:420px;}
.feat-table{border-collapse:collapse;font-family:var(--mono);font-size:0.58rem;width:100%;}
.feat-table th{background:var(--bg);color:#e2e8f0;padding:5px 8px;border-bottom:1px solid var(--border);letter-spacing:0.08em;white-space:nowrap;position:sticky;top:0;}
.feat-table td{padding:4px 8px;border-bottom:1px solid var(--border);color:var(--text);white-space:nowrap;}
.feat-table tr:hover td{background:var(--surf2);}
.feat-table tr.stats-row td{color:var(--accent);background:var(--surf);font-weight:500;}
.ml-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;
    background:var(--border);border:1px solid var(--border);border-radius:6px;overflow:hidden;margin-top:6px;}
.ml-card{background:var(--surf);padding:11px 12px 11px;}
.ml-card:hover{background:var(--surf2);}
.ml-card-title{font-family:var(--mono);font-size:0.5rem;font-weight:700;letter-spacing:0.14em;
    text-transform:uppercase;color:#e2e8f0;margin-bottom:7px;}
.ml-card-val{font-family:var(--mono);font-size:0.82rem;font-weight:600;line-height:1.4;}
.stSpinner>div{border-top-color:var(--accent)!important;}
[data-testid="stStatusWidget"]{display:none!important;}
</style>
"""

def _kpi(lbl, val, cls="", sub=""):
    s = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f'<div class="kpi-cell"><div class="kpi-lbl">{lbl}</div><div class="kpi-val {cls}">{val}</div>{s}</div>'

def _section(lbl):
    return f'<div class="section-label">{lbl}</div>'

def _row(k, v, cls=""):
    return f'<div class="dtbl-row"><span class="dtbl-k">{k}</span><span class="dtbl-v {cls}">{v}</span></div>'


def _now_ny():
    return datetime.now(NY_TZ)

def _now_morocco():
    return datetime.now(MOROCCO_TZ)

def _is_candle_settled():
    ny  = _now_ny()
    h   = ny.hour + ny.minute / 60.0
    dow = ny.weekday()
    if dow == 5:
        return True, "Saturday — market closed, Friday candle confirmed"
    if dow == 6:
        if h < 17.0:
            return True, "Sunday before 5pm NY — Friday candle confirmed"
        return False, "Sunday after 5pm NY — new session open, still Friday candle"
    if h >= SETTLEMENT_NY:
        return True, "Past 1:30 PM NY — today's candle is closed and confirmed"
    return False, f"Before settlement ({ny.strftime('%H:%M')} NY) — will use yesterday's candle"

def dist(price, current):
    pct  = (price - current) / current * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"



STOOQ_MAP = {"GC=F": "GC.F", "EURUSD=X": "EURUSD", "JPY=X": "JPYUSD"}

def _fetch_stooq(ticker, start, end):
    try:
        import pandas_datareader as pdr
        stooq_ticker = STOOQ_MAP.get(ticker, ticker)
        df = pdr.DataReader(stooq_ticker, "stooq", start, end)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()

def _fetch_yf(ticker, start, end, retries=2):
    for i in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=False, progress=False,
                             timeout=15)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if not df.empty:
                return df
        except Exception:
            if i < retries - 1:
                time.sleep(2)
    # fallback to Stooq
    df = _fetch_stooq(ticker, start, end)
    if not df.empty:
        return df
    return pd.DataFrame()


def fetch_fred_data(start, end):
    fred_obj = Fred(api_key=FRED_API_KEY)
    series, ages, warnings_list = {}, {}, []
    for s in MACRO_SERIES:
        data = None
        try:
            data = fred_obj.get_series(s, start, end)
            series[s] = data
        except Exception:
            local = os.path.join(ARTEFACT_DIR, f"{s}.csv")
            if os.path.exists(local):
                df   = pd.read_csv(local, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                data = df[df.columns[0]].replace(".", np.nan).astype(float)
                data = data[(data.index >= str(start)) & (data.index <= str(end))]
                series[s] = data
                warnings_list.append(f"{s}: loaded from local CSV")
            else:
                warnings_list.append(f"{s}: unavailable — will ffill from last known value")
                series[s] = pd.Series(dtype=float)
        if data is not None and not data.dropna().empty:
            last     = data.dropna().index[-1]
            bday_age = len(pd.bdate_range(start=last, end=pd.Timestamp.today().normalize())) - 1
            ages[s]  = (last.strftime("%Y-%m-%d"), bday_age)
        else:
            ages[s] = ("unavailable", 99)
    macro = pd.DataFrame(series)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    return macro, ages, warnings_list


CACHE_FILE = os.path.join(ARTEFACT_DIR, "daily_cache.json")

def _save_cache(df, fred_ages, fill_report, fetch_log, candle_note, fred_warnings):
    today = datetime.today().strftime("%Y-%m-%d")
    payload = {
        "date":          today,
        "df":            df.to_json(),
        "fred_ages":     fred_ages,
        "fill_report":   fill_report,
        "fetch_log":     fetch_log,
        "candle_note":   candle_note,
        "fred_warnings": fred_warnings,
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(payload, f)

def _load_cache():
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            payload = json.load(f)
        today = datetime.today().strftime("%Y-%m-%d")
        if payload.get("date") != today:
            os.remove(CACHE_FILE)
            return None
        df = pd.read_json(payload["df"])
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df, payload["fred_ages"], payload["fill_report"], payload["fetch_log"], payload["candle_note"], payload["fred_warnings"]
    except Exception:
        return None


def fetch_all_daily():
    cached = _load_cache()
    if cached is not None:
        df, fred_ages, fill_report, fetch_log, candle_note, fred_warnings = cached
        fetch_log = {k: (v[0], "cached") for k, v in fetch_log.items()}
        return df, fred_ages, fill_report, fetch_log, candle_note, fred_warnings

    end   = datetime.today()
    start = end - timedelta(days=DAYS_BACK)
    fetch_log = {}

    gold = _fetch_yf("GC=F", start, end)
    fetch_log["XAU/USD"] = ("yfinance GC=F", "ok")

    eur = _fetch_yf("EURUSD=X", start, end)
    fetch_log["EURUSD"] = ("yfinance EURUSD=X", "ok")
    jpy = _fetch_yf("JPY=X", start, end)
    fetch_log["USDJPY"] = ("yfinance JPY=X", "ok")
    macro, fred_ages, fred_warnings = fetch_fred_data(start, end)
    fetch_log["FRED"] = ("FRED API", "ok")

    prices = pd.DataFrame({
        "Close_XAUUSD":  gold["Close"],
        "Volume_XAUUSD": gold.get("Volume", pd.Series(dtype=float)),
        "Close_EURUSD":  eur["Close"],
        "Close_USDJPY":  jpy["Close"],
    })
    full_idx = pd.date_range(start=prices.index.min(), end=prices.index.max(), freq="B")
    prices   = prices.reindex(full_idx)
    macro    = macro.reindex(full_idx)
    df       = prices.join(macro, how="left")

    fill_report = {}
    for col in df.columns:
        nans = int(df[col].isna().sum())
        if nans > 0:
            gap_sizes, in_gap, g = [], False, 0
            for v in df[col]:
                if pd.isna(v):
                    in_gap = True; g += 1
                else:
                    if in_gap: gap_sizes.append(g)
                    in_gap = False; g = 0
            fill_report[col] = {"nan_filled": nans, "max_gap_days": max(gap_sizes) if gap_sizes else 0}

    df = df.ffill().bfill()

    settled, _ = _is_candle_settled()
    today_n    = pd.Timestamp.today().normalize()
    candle_note = ""
    if df.index[-1] >= today_n:
        if settled:
            candle_note = f"Today's candle ({df.index[-1].date()}) is confirmed closed — kept"
        else:
            candle_note = f"Dropped today's unsettled candle ({df.index[-1].date()}) — using yesterday"
            df = df.iloc[:-1]

    df.dropna(subset=["Close_XAUUSD"], inplace=True)
    df.index.name = "Date"
    _save_cache(df, fred_ages, fill_report, fetch_log, candle_note, fred_warnings)
    return df, fred_ages, fill_report, fetch_log, candle_note, fred_warnings


def engineer(df):
    out  = pd.DataFrame(index=df.index)
    gold = df["Close_XAUUSD"]

    out["Close_Returns"]  = gold.pct_change()
    out["Log_Returns"]    = np.log(gold / gold.shift(1))
    out["EURUSD_Returns"] = df["Close_EURUSD"].pct_change()
    out["USDJPY_Returns"] = df["Close_USDJPY"].pct_change()

    sma20 = gold.rolling(20).mean()
    std20 = gold.rolling(20).std()
    upper = sma20 + 2*std20; lower = sma20 - 2*std20
    out["BB_PctB"] = (gold - lower) / (upper - lower)

    ema50  = gold.ewm(span=50,  adjust=False).mean()
    ema200 = gold.ewm(span=200, adjust=False).mean()
    out["Price_Over_EMA50"]  = gold / ema50
    out["Price_Over_EMA200"] = gold / ema200
    out["Bull_Trend"]        = (ema50 - ema200) / ema200

    macd = gold.ewm(span=12,adjust=False).mean() - gold.ewm(span=26,adjust=False).mean()
    out["MACD_Signal_Norm"] = macd.ewm(span=9,adjust=False).mean() / gold

    r20 = out["Log_Returns"].rolling(20)
    out["LogReturn_ZScore"] = (out["Log_Returns"] - r20.mean()) / r20.std()
    c20 = out["Close_Returns"].rolling(20)
    out["Return_ZScore"]    = (out["Close_Returns"] - c20.mean()) / c20.std()

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
                         .replace([np.inf,-np.inf], np.nan)
                         .ffill().bfill().clip(-5, 5))
    out.drop(columns=z_cols, inplace=True)
    out["Close_XAUUSD"] = gold
    return out.dropna(subset=BASE_FEATURES)


def run_ml(feat_df):
    def _load(name):
        path = os.path.join(ARTEFACT_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found")
        with open(path, "rb") as f:
            return pickle.load(f)

    model      = _load("cv_best_fold_model.pkl")
    calibrator = _load("calibrator.pkl")
    oof        = pd.read_csv(os.path.join(ARTEFACT_DIR, "cv_predictions_oof.csv"),
                             index_col=0, parse_dates=True)
    today    = feat_df.iloc[[-1]].copy()
    pred_val = float(model.predict(today[BASE_FEATURES].values)[0])
    abs_pred = abs(pred_val)
    hist     = oof["oof_prediction"].dropna().tail(PRED_Z_LOOKBACK)
    h_std    = hist.std()
    pred_z   = float((pred_val - hist.mean()) / h_std) if h_std > 0 else 0.0

    calib_in = pd.DataFrame([[pred_val, abs_pred,
        float(today["Bull_Trend"].iloc[0]), float(today["Macro_Fast"].iloc[0]),
        float(today["BB_PctB"].iloc[0]),    float(today["Price_Over_EMA200"].iloc[0]),
    ]], columns=CALIB_FEATURES)
    prob = float(calibrator.predict_proba(calib_in)[0][1])

    signal = "NO SIGNAL"
    if prob >= PROB_THRESHOLD and abs(pred_z) >= Z_THRESHOLD:
        signal = "BUY" if pred_val > 0 else "SELL"

    return dict(signal=signal, prob=prob, pred_val=pred_val, pred_z=pred_z,
                abs_pred_z=abs(pred_z),
                bull_trend=float(today["Bull_Trend"].iloc[0]),
                macro_fast=float(today["Macro_Fast"].iloc[0]),
                bb_pctb=float(today["BB_PctB"].iloc[0]),
                ema200=float(today["Price_Over_EMA200"].iloc[0]),
                close=float(today["Close_XAUUSD"].iloc[0]))


def weekly_range(feat_df):
    try:
        raw = _fetch_yf("GC=F", datetime.today()-timedelta(days=14), datetime.today())
        if not raw.empty:
            raw = raw[raw.index < pd.Timestamp.today().normalize()].tail(7)
            return float(raw["High"].max()), float(raw["Low"].min()), \
                   f"{raw.index[0].date()} to {raw.index[-1].date()}"
    except Exception:
        pass
    c = feat_df["Close_XAUUSD"].tail(7)
    return float(c.max()), float(c.min()), "approx from close prices"


def intraday_range():
    try:
        now   = datetime.utcnow()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if start >= now:
            start = start - timedelta(days=1)
        raw = yf.download("GC=F", start=start, end=now,
                          interval="1h", auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty:
            return None, None, 0
        return float(raw["High"].max()), float(raw["Low"].min()), len(raw)
    except Exception:
        return None, None, 0


def smc_4h(current):
    empty = {k: [] for k in ["bos_bull","bos_bear","choch_bull","choch_bear","ob_bull","ob_bear","sr"]}
    try:
        end   = datetime.utcnow()
        start = end - timedelta(days=58)
        raw   = yf.download("GC=F", start=start, end=end,
                             interval="1h", auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        if raw.empty:
            return empty

        ohlc           = raw["Close"].resample("4h").ohlc()
        ohlc.columns   = ["Open","High","Low","Close"]
        ohlc["Volume"] = raw["Volume"].resample("4h").sum()
        ohlc           = ohlc.dropna().iloc[:-1]

        highs  = ohlc["High"].values
        lows   = ohlc["Low"].values
        opens  = ohlc["Open"].values
        closes = ohlc["Close"].values
        n, L   = len(ohlc), SMC_SWING

        hl  = np.zeros(n, dtype=int)
        lvl = np.full(n, np.nan)
        for i in range(L, n-L):
            wh = np.concatenate([highs[i-L:i], highs[i+1:i+L+1]])
            wl = np.concatenate([lows[i-L:i],  lows[i+1:i+L+1]])
            if highs[i] > wh.max():   hl[i]=1;  lvl[i]=highs[i]
            elif lows[i] < wl.min():  hl[i]=-1; lvl[i]=lows[i]

        sh_list = [(i, lvl[i]) for i in range(n) if hl[i]==1]
        sl_list = [(i, lvl[i]) for i in range(n) if hl[i]==-1]
        trend = 0; last_sh = last_sl = None
        res = {k: [] for k in ["bos_bull","bos_bear","choch_bull","choch_bear","ob_bull","ob_bear"]}

        for i in range(1, n):
            for si, sv in sh_list:
                if si < i and (last_sh is None or si > last_sh[0]): last_sh = (si, sv)
            for si, sv in sl_list:
                if si < i and (last_sl is None or si > last_sl[0]): last_sl = (si, sv)

            if last_sh and closes[i] > last_sh[1]:
                res["bos_bull" if trend==1 else "choch_bull"].append(
                    {"price": round(last_sh[1],1), "when": ohlc.index[i].strftime("%m-%d %H:%M")})
                for j in range(last_sh[0]-1, max(0,last_sh[0]-30), -1):
                    if closes[j] < opens[j]:
                        if not any(lows[k]<lows[j] for k in range(j+1,min(j+40,n))):
                            res["ob_bull"].append({
                                "top": round(highs[j],1), "bottom": round(lows[j],1),
                                "mid": round((highs[j]+lows[j])/2,1),
                                "when": ohlc.index[j].strftime("%m-%d %H:%M")})
                        break
                trend=1; last_sh=None

            elif last_sl and closes[i] < last_sl[1]:
                res["bos_bear" if trend==-1 else "choch_bear"].append(
                    {"price": round(last_sl[1],1), "when": ohlc.index[i].strftime("%m-%d %H:%M")})
                for j in range(last_sl[0]-1, max(0,last_sl[0]-30), -1):
                    if closes[j] > opens[j]:
                        if not any(highs[k]>highs[j] for k in range(j+1,min(j+40,n))):
                            res["ob_bear"].append({
                                "top": round(highs[j],1), "bottom": round(lows[j],1),
                                "mid": round((highs[j]+lows[j])/2,1),
                                "when": ohlc.index[j].strftime("%m-%d %H:%M")})
                        break
                trend=-1; last_sl=None

        all_prices = [lvl[i] for i in range(n) if hl[i]!=0 and not np.isnan(lvl[i])]
        used, sr   = [False]*len(all_prices), []
        for i, p in enumerate(all_prices):
            if used[i]: continue
            nearby = [p]
            for j in range(i+1, len(all_prices)):
                if not used[j] and abs(all_prices[j]-p)/p < SR_TOL:
                    nearby.append(all_prices[j]); used[j]=True
            used[i] = True
            if len(nearby) >= 2:
                mid = round(np.mean(nearby), 1)
                if not any(abs(mid-s["price"])/mid < SR_TOL*2 for s in sr):
                    sr.append({"price": mid, "hits": len(nearby)})

        res["sr"] = sorted(sr, key=lambda x: x["price"])
        return res
    except Exception as e:
        return empty


def _render_qwen(sig, r, current, w_high, w_low, i_high, i_low, deduped, raw_df, feat_df=None):
    if not QWEN_AVAILABLE:
        return
    render_qwen_section(sig, r, current, w_high, w_low, i_high, i_low, deduped, raw_df, feat_df=feat_df)


def main():
    st.set_page_config(page_title="Gold Signal", layout="centered",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)

    now = datetime.now()
    mor = _now_morocco()
    ny  = _now_ny()
    mor_h = mor.hour + mor.minute / 60.0
    settled, reason = _is_candle_settled()

    status_col = "var(--buy)" if settled and not (MAINTENANCE_START <= mor_h < MAINTENANCE_END) else "var(--accent)" if not settled else "var(--sell)"
    st.markdown(f"""
    <div class="app-header">
      <div class="app-logo">
        <div class="logo-hex">XAU</div>
        <div>
          <div class="logo-text">XAUUSD &nbsp; Signal Validator</div>
          <div class="logo-sub">Data integrity · SMC levels · ML inference</div>
        </div>
      </div>
      <div class="header-right">
        <div class="header-ts">{now.strftime('%Y-%m-%d &nbsp; %H:%M:%S')}</div>
        <div class="header-status" style="color:{status_col}">
          Morocco {mor.strftime('%H:%M')} &nbsp;·&nbsp; NY {ny.strftime('%H:%M')}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    cs_cls = "cs-ok" if settled and not (MAINTENANCE_START <= mor_h < MAINTENANCE_END) else "cs-danger" if (MAINTENANCE_START <= mor_h < MAINTENANCE_END) else "cs-warn"
    st.markdown(f'<div class="candle-status {cs_cls}">{reason}</div>', unsafe_allow_html=True)

    with st.expander("Safe run windows"):
        st.markdown("""<div style="font-family:var(--mono);font-size:0.62rem;color:var(--muted2);line-height:2.2">
        <span style="color:var(--buy)">BEST</span> &nbsp; 6:30 PM – 9:15 PM Morocco &nbsp;·&nbsp; settled + FRED fresh<br>
        <span style="color:var(--accent)">OK</span> &nbsp;&nbsp; 10:00 PM – 5:00 AM Morocco &nbsp;·&nbsp; yesterday candle confirmed<br>
        <span style="color:var(--sell)">AVOID</span> 9:15 PM – 10:00 PM Morocco &nbsp;·&nbsp; CME maintenance<br>
        <span style="color:var(--muted2)">FRED</span> &nbsp; DFII10/DFII5/DGS2 ready ~8:30 PM Morocco &nbsp;·&nbsp; FEDFUNDS changes on FOMC days only
        </div>""", unsafe_allow_html=True)

    # ── fetch all data silently ──────────────────────────────────────────
    with st.spinner("Loading..."):
        raw_df, fred_ages, fill_report, fetch_log, candle_note, fred_warnings = fetch_all_daily()
        feat_df = engineer(raw_df.copy())

    if fred_warnings:
        for w in fred_warnings:
            st.markdown(f'<div class="candle-status cs-warn">FRED: {w}</div>', unsafe_allow_html=True)

    is_cached  = all(s == "cached" for _, s in fetch_log.values())
    cache_note = "Cached — data fetched today" if is_cached else "Fetched fresh"
    cache_col  = "#10d988" if is_cached else "#e2e8f0"
    candle_col = "#10d988" if "confirmed" in (candle_note or "") else "#f5c842"
    candle_txt = candle_note or reason

    st.markdown(f"""
    <div style="background:#0c0f14;border:1px solid #1c2030;border-radius:6px;
                overflow:hidden;margin:0.5rem 0;font-family:'JetBrains Mono',monospace;">
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:#1c2030;">
        <div style="background:#0c0f14;padding:8px 14px;">
          <div style="font-size:0.44rem;letter-spacing:0.18em;color:#e2e8f0;text-transform:uppercase;margin-bottom:3px">CANDLE</div>
          <div style="font-size:0.68rem;font-weight:600;color:{candle_col}">{candle_txt}</div>
        </div>
        <div style="background:#0c0f14;padding:8px 14px;">
          <div style="font-size:0.44rem;letter-spacing:0.18em;color:#e2e8f0;text-transform:uppercase;margin-bottom:3px">DATA</div>
          <div style="font-size:0.68rem;font-weight:600;color:{cache_col}">{cache_note}</div>
        </div>
        <div style="background:#0c0f14;padding:8px 14px;">
          <div style="font-size:0.44rem;letter-spacing:0.18em;color:#e2e8f0;text-transform:uppercase;margin-bottom:3px">RUN TIME</div>
          <div style="font-size:0.68rem;font-weight:600;color:#e2e8f0">{mor.strftime('%H:%M')} Morocco</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── ML inference ────────────────────────────────────────────────────
    st.markdown(_section("ML INFERENCE"), unsafe_allow_html=True)
    try:
        r       = run_ml(feat_df)
        sig     = r["signal"]
        current = r["close"]
        bt_lbl  = "BULL" if r["bull_trend"]>0.02 else "BEAR" if r["bull_trend"]<-0.02 else "NEUTRAL"
        mf_lbl  = "TIGHT" if r["macro_fast"]>0.5 else "EASY" if r["macro_fast"]<-0.5 else "NEUTRAL"
        sc      = {"BUY":"sb-buy","SELL":"sb-sell","NO SIGNAL":"sb-none"}[sig]
        vc      = {"BUY":"buy","SELL":"sell","NO SIGNAL":"muted"}[sig]

        reason_html = ""
        if sig == "NO SIGNAL":
            parts = []
            if abs(r["pred_z"]) < Z_THRESHOLD:
                parts.append(f"|z|={abs(r['pred_z']):.2f} < {Z_THRESHOLD}")
            if r["prob"] < PROB_THRESHOLD:
                parts.append(f"prob={r['prob']:.3f} < {PROB_THRESHOLD}")
            reason_html = f'<div class="sig-reason">{" &nbsp;|&nbsp; ".join(parts)}</div>'

        st.markdown(f"""
        <div class="sig-banner {sc}">
          <div>
            <div class="sig-label">{sig}</div>
            <div class="sig-sub">XAU/USD &nbsp;·&nbsp; {raw_df.index[-1].date()} &nbsp;·&nbsp; Daily close</div>
            {reason_html}
          </div>
          <div class="sig-right">
            <div class="sig-prob">{r['prob']:.1%}</div>
            <div class="sig-prob-lbl">Win Probability</div>
          </div>
        </div>""", unsafe_allow_html=True)

        rc_col = "#5a6a80" if bt_lbl=="NEUTRAL" else ("#10d988" if bt_lbl=="BULL" else "#ff4d6a")
        mc_col = "#5a6a80" if mf_lbl=="NEUTRAL" else ("#ff4d6a" if mf_lbl=="TIGHT" else "#10d988")
        vc_col = "#10d988" if sig=="BUY" else "#ff4d6a" if sig=="SELL" else "#5a6a80"
        pz_col = "#10d988" if r["pred_z"]>0 else "#ff4d6a"
        e2_col = "#10d988" if r["ema200"]>1 else "#ff4d6a"

        cards = [
            ("Close",        f"${r['close']:,.2f}",                     "#f5c842"),
            ("Win prob",     f"{r['prob']:.4f}",                         vc_col),
            ("Pred Z",       f"{r['pred_z']:+.4f}",                      pz_col),
            ("Pred value",   f"{r['pred_val']:+.6f}",                    "#e2e8f0"),
            ("Bull Trend",   f"{r['bull_trend']:+.4f}<br><small style='font-size:0.55rem'>{bt_lbl}</small>", rc_col),
            ("Macro Fast",   f"{r['macro_fast']:+.4f}<br><small style='font-size:0.55rem'>{mf_lbl}</small>", mc_col),
            ("BB PctB",      f"{r['bb_pctb']:.4f}",                     "#e2e8f0"),
            ("EMA200",       f"{r['ema200']:.4f}",                       e2_col),
            ("Prob gate",    f"≥ {PROB_THRESHOLD}",                     "#378ADD"),
            ("Z gate",       f"≥ {Z_THRESHOLD}",                        "#378ADD"),
        ]

        cards_html = '<div class="ml-grid">'
        for title, value, col in cards:
            cards_html += (f'<div class="ml-card">'
                           f'<div class="ml-card-title">{title}</div>'
                           f'<div class="ml-card-val" style="color:{col}">{value}</div>'
                           f'</div>')
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

    except FileNotFoundError as e:
        st.markdown(f'<div class="candle-status cs-warn">Model files not found: {e} — showing data only</div>', unsafe_allow_html=True)
        current = float(feat_df["Close_XAUUSD"].iloc[-1])
        sig     = "NO SIGNAL"
        r       = None

    # ── market overview table (always fresh) ────────────────────────────
    st.markdown(_section("MARKET OVERVIEW"), unsafe_allow_html=True)
    with st.spinner("Computing levels..."):
        smc    = smc_4h(current)
        i_high, i_low, _ = intraday_range()
        w_high, w_low, _ = weekly_range(feat_df)

    all_levels = []
    for ev in smc["bos_bull"]:
        all_levels.append({"price": ev["price"], "type": "BOS",   "dir": "Bull", "col": "#10d988"})
    for ev in smc["choch_bull"]:
        all_levels.append({"price": ev["price"], "type": "CHoCH", "dir": "Bull", "col": "#5DCAA5"})
    for ob in smc["ob_bull"]:
        all_levels.append({"price": ob["mid"],   "type": "OB",    "dir": "Bull", "col": "#10d988"})
    for ev in smc["bos_bear"]:
        all_levels.append({"price": ev["price"], "type": "BOS",   "dir": "Bear", "col": "#ff4d6a"})
    for ev in smc["choch_bear"]:
        all_levels.append({"price": ev["price"], "type": "CHoCH", "dir": "Bear", "col": "#F0997B"})
    for ob in smc["ob_bear"]:
        all_levels.append({"price": ob["mid"],   "type": "OB",    "dir": "Bear", "col": "#ff4d6a"})
    for sr in smc["sr"]:
        all_levels.append({"price": sr["price"], "type": "S/R",   "dir": "—",    "col": "#f5c842"})

    seen, deduped = [], []
    for lv in sorted(all_levels, key=lambda x: x["price"], reverse=True):
        if not any(abs(lv["price"] - s) / lv["price"] < SR_TOL * 2 for s in seen):
            seen.append(lv["price"]); deduped.append(lv)

    # add weekly/daily range levels into the pool
    range_levels = []
    range_levels.append({"price": w_high, "type": "W.Res", "dir": "Weekly", "col": "#e2e8f0"})
    range_levels.append({"price": w_low,  "type": "W.Sup", "dir": "Weekly", "col": "#e2e8f0"})
    if i_high and i_low:
        range_levels.append({"price": i_high, "type": "D.Res", "dir": "Daily", "col": "#a78bfa"})
        range_levels.append({"price": i_low,  "type": "D.Sup", "dir": "Daily", "col": "#a78bfa"})

    # merge all levels, sort by price descending
    all_merged = sorted(deduped + range_levels, key=lambda x: x["price"], reverse=True)

    # data status
    xau_status  = fetch_log.get("XAU/USD", ("", ""))[1]
    fred_status = "OK" if all(age <= 1 or s == "FEDFUNDS" for s, (d, age) in fred_ages.items()) else "check"
    xau_col     = "#10d988" if xau_status in ("ok","cached") else "#f5c842"
    fred_col    = "#10d988" if fred_status == "OK" else "#f5c842"

    sig_col = "#10d988" if sig=="BUY" else "#ff4d6a" if sig=="SELL" else "#e2e8f0"

    # build all rows as one string then render once
    all_dists = [abs(lv["price"] - current) for lv in all_merged]
    max_dist  = max(all_dists) if all_dists else 1

    above = sorted([l for l in all_merged if l["price"] > current], key=lambda x: x["price"])
    below = sorted([l for l in all_merged if l["price"] < current], key=lambda x: x["price"], reverse=True)

    def _ov_row(lv):
        price = lv["price"]
        col   = lv["col"]
        d_pct = (price - current) / current * 100
        d_col = "#ff4d6a" if price > current else "#10d988"
        bar_w = min(int(abs(price - current) / max_dist * 100), 100)
        mono  = "'JetBrains Mono',monospace"
        return (
            f'<div style="display:flex;align-items:center;padding:6px 16px;'
            f'border-bottom:1px solid #1c2030;gap:0">'
            f'<span style="font-family:{mono};font-size:0.74rem;font-weight:600;'
            f'color:{col};width:96px;flex-shrink:0">${price:,.1f}</span>'
            f'<span style="font-family:{mono};font-size:0.48rem;width:96px;flex-shrink:0;'
            f'padding:2px 6px;border-radius:3px;border:1px solid {col}44;color:{col};'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
            f'{lv["type"]} {lv["dir"]}</span>'
            f'<div style="flex:1;height:3px;background:#1a2030;border-radius:2px;overflow:hidden;margin:0 12px">'
            f'<div style="width:{bar_w}%;height:100%;background:{col};border-radius:2px;opacity:0.85"></div></div>'
            f'<span style="font-family:{mono};font-size:0.58rem;'
            f'color:{d_col};width:56px;text-align:right;flex-shrink:0">{d_pct:+.2f}%</span>'
            f'</div>'
        )

    cur_divider = (
        f'<div style="background:#05070a;border-top:1px solid #1c2030;'
        f'border-bottom:1px solid #1c2030;padding:9px 16px;'
        f'display:flex;align-items:baseline;gap:10px">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:1rem;'
        f'font-weight:700;color:#e2e8f0">${current:,.1f}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.44rem;'
        f'letter-spacing:0.18em;color:#e2e8f0">CURRENT</span>'
        f'</div>'
    )

    ov_html = (
        f'<div style="background:#0c0f14;border:1px solid #1c2030;border-radius:6px;overflow:hidden;margin-top:4px">'
        # status bar
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#1c2030">'
        f'<div style="background:#0c0f14;padding:9px 14px">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.44rem;letter-spacing:0.18em;color:#e2e8f0;text-transform:uppercase;margin-bottom:3px">XAU/USD</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;font-weight:600;color:{xau_col}">{xau_status}</div></div>'
        f'<div style="background:#0c0f14;padding:9px 14px">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.44rem;letter-spacing:0.18em;color:#e2e8f0;text-transform:uppercase;margin-bottom:3px">FRED</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;font-weight:600;color:{fred_col}">{fred_status}</div></div>'
        f'<div style="background:#0c0f14;padding:9px 14px">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.44rem;letter-spacing:0.18em;color:#e2e8f0;text-transform:uppercase;margin-bottom:3px">SIGNAL</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;font-weight:600;color:{sig_col}">{sig}</div></div>'
        f'<div style="background:#0c0f14;padding:9px 14px">'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.44rem;letter-spacing:0.18em;color:#e2e8f0;text-transform:uppercase;margin-bottom:3px">CANDLE</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;font-weight:600;color:#e2e8f0">{str(raw_df.index[-1].date())}</div></div>'
        f'</div>'
        # levels
        + "".join(_ov_row(l) for l in above)
        + cur_divider
        + "".join(_ov_row(l) for l in below)
        + f'</div>'
    )
    st.markdown(ov_html, unsafe_allow_html=True)

    st.markdown(_section("FEATURE TABLE — LAST 252 ROWS"), unsafe_allow_html=True)
    tail  = feat_df[["Close_XAUUSD"] + BASE_FEATURES].tail(PRED_Z_LOOKBACK).copy()
    means = tail.mean()
    stds  = tail.std()

    th = "".join(f"<th>{c}</th>" for c in ["Date","Close"] + BASE_FEATURES)
    mean_tds = f"<td>MEAN</td><td>{means['Close_XAUUSD']:.2f}</td>" + "".join(f"<td>{means[f]:.4f}</td>" for f in BASE_FEATURES)
    std_tds  = f"<td>STD</td><td>{stds['Close_XAUUSD']:.2f}</td>"  + "".join(f"<td>{stds[f]:.4f}</td>"  for f in BASE_FEATURES)
    data_rows = ""
    for date, row in tail.iterrows():
        tds = f"<td>{date.strftime('%Y-%m-%d')}</td><td>{row['Close_XAUUSD']:,.1f}</td>"
        tds += "".join(f"<td>{row[f]:.4f}</td>" if not np.isnan(row[f]) else "<td>—</td>" for f in BASE_FEATURES)
        data_rows += f"<tr>{tds}</tr>"

    table_html = (f'<div class="feat-table-wrap"><table class="feat-table">'
                  f'<thead><tr>{th}</tr></thead>'
                  f'<tbody>'
                  f'<tr class="stats-row">{mean_tds}</tr>'
                  f'<tr class="stats-row">{std_tds}</tr>'
                  f'{data_rows}'
                  f'</tbody></table></div>')
    st.markdown(table_html, unsafe_allow_html=True)

    _render_qwen(sig, r if "r" in dir() else None,
                 current, w_high, w_low, i_high, i_low,
                 deduped, raw_df, feat_df=feat_df)


if __name__ == "__main__":
    main()