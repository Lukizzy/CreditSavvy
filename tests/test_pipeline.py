"""
test_pipeline.py
----------------
Unit and integration tests for the CreditGuard pipeline.

Run with:
    pytest tests/ -v
"""

import os, sys, json, pickle
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.explainer import get_shap_values, global_feature_importance

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
BEST_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def meta():
    if not os.path.exists(META_PATH):
        pytest.skip("Model not trained yet — run `python models/train.py` first.")
    with open(META_PATH) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def model(meta):
    with open(BEST_PATH, "rb") as f:
        return pickle.load(f)

@pytest.fixture
def sample_input(meta):
    """Returns a valid single-row DataFrame with median-ish values."""
    data = {col: 0 for col in meta["feature_cols"]}
    data.update({
        "LIMIT_BAL": 200_000, "SEX": 2, "EDUCATION": 2,
        "MARRIAGE": 1, "AGE": 35,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0,
        "PAY_4": 0, "PAY_5": -1, "PAY_6": -1,
        "BILL_AMT1": 50_000, "BILL_AMT2": 48_000, "BILL_AMT3": 46_000,
        "BILL_AMT4": 44_000, "BILL_AMT5": 42_000, "BILL_AMT6": 40_000,
        "PAY_AMT1": 5_000, "PAY_AMT2": 5_000, "PAY_AMT3": 5_000,
        "PAY_AMT4": 5_000, "PAY_AMT5": 5_000, "PAY_AMT6": 5_000,
    })
    return pd.DataFrame([data])


# ── Data tests ─────────────────────────────────────────────────────────────────

def test_meta_keys(meta):
    """Metadata JSON contains all required keys."""
    required = {"feature_cols", "feature_labels", "best_model",
                "lr_metrics", "rf_metrics", "train_size", "test_size", "default_rate"}
    assert required.issubset(meta.keys())

def test_meta_auc_above_baseline(meta):
    """Both models must outperform random (AUC > 0.5)."""
    assert meta["lr_metrics"]["roc_auc"] > 0.5, "LR AUC below baseline"
    assert meta["rf_metrics"]["roc_auc"] > 0.5, "RF AUC below baseline"

def test_default_rate_plausible(meta):
    """Default rate should be between 1% and 50%."""
    dr = meta["default_rate"]
    assert 0.01 < dr < 0.50, f"Unexpected default rate: {dr}"


# ── Model prediction tests ─────────────────────────────────────────────────────

def test_predict_proba_shape(model, sample_input):
    """predict_proba returns an (n, 2) array."""
    proba = model.predict_proba(sample_input)
    assert proba.shape == (1, 2)

def test_predict_proba_sums_to_one(model, sample_input):
    """Probabilities must sum to 1.0."""
    proba = model.predict_proba(sample_input)
    assert abs(proba.sum(axis=1)[0] - 1.0) < 1e-6

def test_probability_in_range(model, sample_input):
    """Default probability must be in [0, 1]."""
    p = model.predict_proba(sample_input)[0, 1]
    assert 0.0 <= p <= 1.0

def test_high_risk_applicant(model, meta):
    """Applicant with multiple severe late payments should score higher risk."""
    low_risk = {col: 0 for col in meta["feature_cols"]}
    low_risk.update({"LIMIT_BAL": 500_000, "AGE": 40, "SEX": 1, "EDUCATION": 1,
                     "MARRIAGE": 1, "PAY_0": -1, "PAY_2": -1, "PAY_3": -1,
                     "BILL_AMT1": 20_000, "PAY_AMT1": 20_000})

    high_risk = {col: 0 for col in meta["feature_cols"]}
    high_risk.update({"LIMIT_BAL": 20_000, "AGE": 30, "SEX": 1, "EDUCATION": 3,
                      "MARRIAGE": 2, "PAY_0": 4, "PAY_2": 3, "PAY_3": 3,
                      "BILL_AMT1": 19_000, "PAY_AMT1": 500})

    X_low  = pd.DataFrame([low_risk])
    X_high = pd.DataFrame([high_risk])
    p_low  = model.predict_proba(X_low)[0, 1]
    p_high = model.predict_proba(X_high)[0, 1]
    assert p_high > p_low, "High-risk applicant should have higher default probability"


# ── Explainability tests ───────────────────────────────────────────────────────

def test_shap_output_shape(model, sample_input, meta):
    """SHAP DataFrame has correct columns and row count."""
    shap_df = get_shap_values(model, sample_input, meta["feature_labels"])
    assert set(["feature", "label", "value", "shap_value"]).issubset(shap_df.columns)
    assert len(shap_df) == len(meta["feature_cols"])

def test_shap_sorted_by_importance(model, sample_input, meta):
    """SHAP DataFrame is sorted by absolute value descending."""
    shap_df = get_shap_values(model, sample_input, meta["feature_labels"])
    abs_vals = shap_df["shap_value"].abs().values
    assert all(abs_vals[i] >= abs_vals[i+1] for i in range(len(abs_vals)-1))

def test_global_importance_sums_near_one(model, meta):
    """Global feature importances should sum to approximately 1.0."""
    fi_df = global_feature_importance(model, meta["feature_cols"], meta["feature_labels"])
    total = fi_df["importance"].sum()
    assert abs(total - 1.0) < 0.01, f"Importances sum to {total}"

def test_global_importance_positive(model, meta):
    """All feature importances should be non-negative."""
    fi_df = global_feature_importance(model, meta["feature_cols"], meta["feature_labels"])
    assert (fi_df["importance"] >= 0).all()
