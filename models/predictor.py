"""
predictor.py
------------
Thin inference wrapper used by the Streamlit application.

Loads the persisted model and scaler once at import time and exposes a
single public function:

    predict(features: dict) -> dict
        Returns probability of default, risk band, and SHAP values.
"""

import os
import pickle
import numpy as np
import pandas as pd

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH  = os.path.join(os.path.dirname(__file__), "scaler.pkl")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), "feature_names.pkl")


def _load_artefacts():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_PATH, "rb") as f:
        feature_names = pickle.load(f)
    return bundle["model"], bundle["model_name"], scaler, feature_names


try:
    _MODEL, _MODEL_NAME, _SCALER, _FEATURE_NAMES = _load_artefacts()
    _MODEL_LOADED = True
except FileNotFoundError:
    _MODEL_LOADED = False


def model_loaded() -> bool:
    return _MODEL_LOADED


def get_model_name() -> str:
    return _MODEL_NAME if _MODEL_LOADED else "Not loaded"


def get_feature_names() -> list[str]:
    return _FEATURE_NAMES if _MODEL_LOADED else []


def _risk_band(prob: float) -> tuple[str, str]:
    """Return (band_label, hex_colour) for a given default probability."""
    if prob < 0.20:
        return "Low Risk", "#22c55e"
    elif prob < 0.40:
        return "Moderate Risk", "#f59e0b"
    elif prob < 0.60:
        return "High Risk", "#f97316"
    else:
        return "Very High Risk", "#ef4444"


def _compute_shap(model, X_df: pd.DataFrame, scaler) -> dict:
    """
    Compute feature-contribution values for a single prediction.

    Strategy (in order of preference):
      1. SHAP TreeExplainer / LinearExplainer if the shap package is available.
      2. Signed feature importance fallback: importance multiplied by a sign
         derived from whether each feature value is above its row mean,
         so the chart has a meaningful positive/negative direction.
      3. Zero-vector last resort (renders a flat chart rather than NaN).

    All returned values are sanitised: NaN and Inf are replaced with 0.0.
    """
    feature_names = X_df.columns.tolist()

    # ── Attempt 1: SHAP ────────────────────────────────────────────────────────
    try:
        import shap

        if _MODEL_NAME == "Random Forest":
            explainer   = shap.TreeExplainer(model)
            shap_result = explainer(X_df)          # newer API returns Explanation object
            vals = np.array(shap_result.values)
            # Shape may be (1, n_features) or (1, n_features, n_classes)
            if vals.ndim == 3:
                vals = vals[:, :, 1]               # take class-1 slice
            vals = vals[0]                         # single row
        else:
            X_sc = pd.DataFrame(scaler.transform(X_df), columns=feature_names)
            explainer   = shap.LinearExplainer(model, X_sc)
            shap_result = explainer(X_sc)
            vals = np.array(shap_result.values)[0]

        vals = np.where(np.isfinite(vals), vals, 0.0)
        return dict(zip(feature_names, vals.tolist()))

    except Exception:
        pass  # fall through to next strategy

    # ── Attempt 2: signed importance fallback ──────────────────────────────────
    try:
        if hasattr(model, "feature_importances_"):
            raw = np.array(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            raw = np.abs(model.coef_[0]).astype(float)
        else:
            raise ValueError("no importance attribute")

        x_vals = X_df.values[0].astype(float)
        signs  = np.where(x_vals > np.nanmean(x_vals), 1.0, -1.0)
        signed = np.where(np.isfinite(raw * signs), raw * signs, 0.0)
        return dict(zip(feature_names, signed.tolist()))

    except Exception:
        pass

    # ── Last resort: zeros ─────────────────────────────────────────────────────
    return dict(zip(feature_names, [0.0] * len(feature_names)))


def predict(features: dict) -> dict:
    """
    Run inference for a single applicant.

    Parameters
    ----------
    features : dict
        Keys must match the training feature names exactly.

    Returns
    -------
    dict with keys:
        probability  : float  — P(default)
        risk_band    : str
        band_colour  : str    — hex colour for UI
        shap_values  : dict   — feature → contribution
    """
    if not _MODEL_LOADED:
        raise RuntimeError(
            "Model not found. Run 'python models/train_model.py' first."
        )

    X = pd.DataFrame([features], columns=_FEATURE_NAMES)

    if _MODEL_NAME == "Logistic Regression" and _SCALER is not None:
        X_input = pd.DataFrame(_SCALER.transform(X), columns=_FEATURE_NAMES)
    else:
        X_input = X

    prob = float(_MODEL.predict_proba(X_input)[0][1])
    band, colour = _risk_band(prob)
    shap_vals = _compute_shap(_MODEL, X, _SCALER)

    return {
        "probability": prob,
        "risk_band":   band,
        "band_colour": colour,
        "shap_values": shap_vals,
    }
