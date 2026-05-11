"""
ml/inference.py
Two-stage inference pipeline:
  Stage 1: LightGBM regressor  → raw prediction value (log-return direction)
  Stage 2: Logistic calibrator → calibrated win probability

Artefacts live in ml/artefacts/:
  cv_best_fold_model.pkl
  calibrator.pkl
  cv_predictions_oof.csv    ← used for pred_z normalisation

Public API:
    load_artefacts()                             → (model, calibrator, oof)
    run_inference(feat_df, model, calib, oof)    → dict
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd

from core.config import (
    ARTEFACT_DIR, BASE_FEATURES, CALIB_FEATURES,
    PROB_THRESHOLD, Z_THRESHOLD, PRED_Z_LOOKBACK,
)

log = logging.getLogger("sentinel.ml")


# ── artefact loading ──────────────────────────────────────────────────────────

def load_artefacts() -> tuple:
    """
    Load and return (base_model, calibrator, oof_history).
    oof_history is a pd.Series of out-of-fold predictions used for z-scoring.
    Raises FileNotFoundError if any artefact is missing.
    """
    model_path = os.path.join(ARTEFACT_DIR, "cv_best_fold_model.pkl")
    calib_path = os.path.join(ARTEFACT_DIR, "calibrator.pkl")
    oof_path   = os.path.join(ARTEFACT_DIR, "cv_predictions_oof.csv")

    for p in (model_path, calib_path, oof_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Artefact not found: {p}\n"
                f"Place cv_best_fold_model.pkl, calibrator.pkl, and "
                f"cv_predictions_oof.csv in {ARTEFACT_DIR}"
            )

    with open(model_path, "rb") as f:
        base_model = pickle.load(f)

    with open(calib_path, "rb") as f:
        calibrator = pickle.load(f)

    oof_df      = pd.read_csv(oof_path)
    oof_col     = "prediction_value" if "prediction_value" in oof_df.columns else oof_df.columns[0]
    oof_history = oof_df[oof_col].dropna()

    log.info(f"Artefacts loaded | OOF rows: {len(oof_history)}")
    return base_model, calibrator, oof_history


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(feat_df: pd.DataFrame,
                  base_model,
                  calibrator,
                  oof_history: pd.Series) -> dict:
    """
    Run both stages of inference on the most recent row of feat_df.

    Returns a rich result dict consumed by the /signal endpoint and
    the frontend ML Signal widget.
    """
    latest = feat_df[BASE_FEATURES].iloc[-1:].copy()

    # ── Stage 1: LightGBM ─────────────────────────────────────────────────────
    pred_val = float(base_model.predict(latest)[0])
    abs_pred = abs(pred_val)

    # Z-score normalisation against OOF history (rolling window)
    window  = oof_history.tail(PRED_Z_LOOKBACK)
    oof_mu  = float(window.mean())
    oof_std = float(window.std())
    pred_z  = (pred_val - oof_mu) / oof_std if oof_std > 0 else 0.0
    abs_z   = abs(pred_z)

    # ── Stage 2: Calibrator ───────────────────────────────────────────────────
    calib_input = pd.DataFrame([{
        "prediction_value": pred_val,
        "abs_prediction":   abs_pred,
        "Bull_Trend":       float(feat_df["Bull_Trend"].iloc[-1]),
        "Macro_Fast":       float(feat_df["Macro_Fast"].iloc[-1]),
        "BB_PctB":          float(feat_df["BB_PctB"].iloc[-1]),
        "Price_Over_EMA200":float(feat_df["Price_Over_EMA200"].iloc[-1]),
    }])

    # calibrator may return 1-D (binary class) or 2-D array
    raw_prob = calibrator.predict_proba(calib_input[CALIB_FEATURES])
    prob = float(raw_prob[0, 1]) if raw_prob.ndim == 2 else float(raw_prob[0])

    # ── Signal decision ───────────────────────────────────────────────────────
    conviction = prob >= PROB_THRESHOLD and abs_z >= Z_THRESHOLD

    if not conviction:
        signal = "NO SIGNAL"
    elif pred_val > 0:
        signal = "BUY"
    else:
        signal = "SELL"

    # ── recent 20-day close for sparkline ─────────────────────────────────────
    recent = feat_df["Close_XAUUSD"].tail(20)

    log.info(
        f"Inference | signal={signal} prob={prob:.4f} pred_z={pred_z:+.3f} "
        f"abs_z={abs_z:.3f} close={float(feat_df['Close_XAUUSD'].iloc[-1]):,.2f}"
    )

    return {
        # core signal
        "signal":     signal,
        "prob":       prob,
        "pred_val":   pred_val,
        "abs_pred":   abs_pred,
        "pred_z":     pred_z,
        "abs_pred_z": abs_z,

        # last known market state
        "close":      float(feat_df["Close_XAUUSD"].iloc[-1]),
        "date":       feat_df.index[-1],

        # key features (for frontend display)
        "bull_trend": float(feat_df["Bull_Trend"].iloc[-1]),
        "macro_fast": float(feat_df["Macro_Fast"].iloc[-1]),
        "bb_pctb":    float(feat_df["BB_PctB"].iloc[-1]),
        "ema200":     float(feat_df["Price_Over_EMA200"].iloc[-1]),

        # for sparkline
        "recent":     recent,

        # full feature frame (for status bar / feature table)
        "feat_df":    feat_df,
    }