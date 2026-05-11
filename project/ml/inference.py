"""
ml/inference.py
Two-stage inference:
  Stage 1: LightGBM regressor  → raw prediction (log-return direction proxy)
  Stage 2: Logistic calibrator → calibrated win probability

Artefacts in ml/artefacts/:
    cv_best_fold_model.pkl
    calibrator.pkl
    cv_predictions_oof.csv    ← OOF predictions for pred_z normalisation

Public API:
    load_artefacts()                          → (model, calibrator, oof_series)
    run_inference(feat_df, model, calib, oof) → dict
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd

from core.config import (
    ARTEFACT_DIR, BASE_FEATURES, CALIB_FEATURES,
    PROB_THRESHOLD, Z_THRESHOLD, PRED_Z_LOOKBACK,
)

log = logging.getLogger("sentinel.ml")


def load_artefacts() -> tuple:
    """
    Load (base_model, calibrator, oof_series).
    oof_series is pd.Series of out-of-fold predictions for z-score normalisation.
    Raises FileNotFoundError if any artefact is missing.
    """
    model_path = os.path.join(ARTEFACT_DIR, "cv_best_fold_model.pkl")
    calib_path = os.path.join(ARTEFACT_DIR, "calibrator.pkl")
    oof_path   = os.path.join(ARTEFACT_DIR, "cv_predictions_oof.csv")

    for p in (model_path, calib_path, oof_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Artefact not found: {p}\n"
                f"Expected in: {ARTEFACT_DIR}"
            )

    with open(model_path, "rb") as f:
        base_model = pickle.load(f)
    with open(calib_path, "rb") as f:
        calibrator = pickle.load(f)

    oof_df = pd.read_csv(oof_path, index_col=0, parse_dates=True)

    # Column name used by the training pipeline is "oof_prediction"
    # Accept either name for forward compatibility
    if "oof_prediction" in oof_df.columns:
        oof_series = oof_df["oof_prediction"].dropna()
    elif "prediction_value" in oof_df.columns:
        oof_series = oof_df["prediction_value"].dropna()
    else:
        oof_series = oof_df.iloc[:, 0].dropna()

    log.info(f"Artefacts loaded | OOF rows: {len(oof_series)}")
    return base_model, calibrator, oof_series


def run_inference(feat_df: pd.DataFrame,
                  base_model,
                  calibrator,
                  oof_series: pd.Series) -> dict:
    """
    Run both inference stages on the most recent row of feat_df.
    Returns a rich result dict for /signal endpoint and frontend.
    """
    latest = feat_df[BASE_FEATURES].iloc[-1:].copy()

    # ── Stage 1: LightGBM ─────────────────────────────────────────────────────
    pred_val = float(base_model.predict(latest.values)[0])
    abs_pred = abs(pred_val)

    window  = oof_series.tail(PRED_Z_LOOKBACK)
    oof_std = float(window.std())
    pred_z  = float((pred_val - window.mean()) / oof_std) if oof_std > 0 else 0.0
    abs_z   = abs(pred_z)

    # ── Stage 2: Calibrator ───────────────────────────────────────────────────
    calib_input = pd.DataFrame([{
        "prediction_value":  pred_val,
        "abs_prediction":    abs_pred,
        "Bull_Trend":        float(feat_df["Bull_Trend"].iloc[-1]),
        "Macro_Fast":        float(feat_df["Macro_Fast"].iloc[-1]),
        "BB_PctB":           float(feat_df["BB_PctB"].iloc[-1]),
        "Price_Over_EMA200": float(feat_df["Price_Over_EMA200"].iloc[-1]),
    }])[CALIB_FEATURES]

    raw_prob = calibrator.predict_proba(calib_input)
    prob = float(raw_prob[0, 1]) if raw_prob.ndim == 2 else float(raw_prob[0])

    # ── Signal decision ───────────────────────────────────────────────────────
    if prob >= PROB_THRESHOLD and abs_z >= Z_THRESHOLD:
        signal = "BUY" if pred_val > 0 else "SELL"
    else:
        signal = "NO SIGNAL"

    log.info(
        f"Inference | signal={signal} prob={prob:.4f} "
        f"pred_z={pred_z:+.3f} abs_z={abs_z:.3f} "
        f"close={float(feat_df['Close_XAUUSD'].iloc[-1]):,.2f}"
    )

    return {
        "signal":     signal,
        "prob":       prob,
        "pred_val":   pred_val,
        "abs_pred":   abs_pred,
        "pred_z":     pred_z,
        "abs_pred_z": abs_z,
        "close":      float(feat_df["Close_XAUUSD"].iloc[-1]),
        "date":       str(feat_df.index[-1].date()),
        "bull_trend": float(feat_df["Bull_Trend"].iloc[-1]),
        "macro_fast": float(feat_df["Macro_Fast"].iloc[-1]),
        "bb_pctb":    float(feat_df["BB_PctB"].iloc[-1]),
        "ema200":     float(feat_df["Price_Over_EMA200"].iloc[-1]),
    }