"""
train.py
--------
Trains Logistic Regression and Random Forest classifiers on the credit
default dataset, evaluates both, and persists the best model plus the
scaler and feature list for use by the dashboard app.

Usage:
    python models/train.py
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, brier_score_loss
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "credit_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")


FEATURE_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET_COL = "default"

# Human-readable labels for the dashboard
FEATURE_LABELS = {
    "LIMIT_BAL": "Credit Limit (NT$)",
    "SEX": "Gender",
    "EDUCATION": "Education Level",
    "MARRIAGE": "Marital Status",
    "AGE": "Age",
    "PAY_0": "Repayment Status (Sep)",
    "PAY_2": "Repayment Status (Aug)",
    "PAY_3": "Repayment Status (Jul)",
    "PAY_4": "Repayment Status (Jun)",
    "PAY_5": "Repayment Status (May)",
    "PAY_6": "Repayment Status (Apr)",
    "BILL_AMT1": "Bill Amount (Sep, NT$)",
    "BILL_AMT2": "Bill Amount (Aug, NT$)",
    "BILL_AMT3": "Bill Amount (Jul, NT$)",
    "BILL_AMT4": "Bill Amount (Jun, NT$)",
    "BILL_AMT5": "Bill Amount (May, NT$)",
    "BILL_AMT6": "Bill Amount (Apr, NT$)",
    "PAY_AMT1": "Payment Amount (Sep, NT$)",
    "PAY_AMT2": "Payment Amount (Aug, NT$)",
    "PAY_AMT3": "Payment Amount (Jul, NT$)",
    "PAY_AMT4": "Payment Amount (Jun, NT$)",
    "PAY_AMT5": "Payment Amount (May, NT$)",
    "PAY_AMT6": "Payment Amount (Apr, NT$)",
}


def load_data():
    """Load dataset, validate columns, return X and y."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run `python data/prepare_data.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df[TARGET_COL]
    return X, y


def evaluate(model, X_test, y_test, name: str) -> dict:
    """Return a dict of evaluation metrics for a fitted model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc    = roc_auc_score(y_test, y_prob)
    brier  = brier_score_loss(y_test, y_prob)
    cm     = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n── {name} ──────────────────────────────")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Brier   : {brier:.4f}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(classification_report(y_test, y_pred))

    return {"name": name, "roc_auc": auc, "brier": brier,
            "accuracy": report["accuracy"], "confusion_matrix": cm}


def train():
    print("Loading data...")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"Default rate (train): {y_train.mean():.2%}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Model 1: Logistic Regression ──────────────────────────────────────────
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced",
                             random_state=42, solver="lbfgs")
    lr.fit(X_train_s, y_train)

    # Wrap in a pipeline so predict_proba works with raw input
    lr_pipe = Pipeline([("scaler", scaler), ("clf", lr)])

    # ── Model 2: Random Forest ────────────────────────────────────────────────
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    rf_pipe = Pipeline([("scaler", scaler), ("clf", rf)])

    # ── Evaluation ────────────────────────────────────────────────────────────
    lr_metrics = evaluate(lr_pipe, X_test, y_test, "Logistic Regression")
    rf_metrics = evaluate(rf_pipe, X_test, y_test, "Random Forest")

    # Cross-validation AUC (5-fold)
    cv_lr = cross_val_score(lr_pipe, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    cv_rf = cross_val_score(rf_pipe, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"\nCV AUC — LR: {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")
    print(f"CV AUC — RF: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

    # Select best model by ROC-AUC
    best_pipe = rf_pipe if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"] else lr_pipe
    best_name = "Random Forest" if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "Logistic Regression"
    print(f"\nBest model: {best_name}")

    # ── Persist artefacts ─────────────────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_pipe, f)
    with open(os.path.join(MODEL_DIR, "lr_model.pkl"), "wb") as f:
        pickle.dump(lr_pipe, f)
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "wb") as f:
        pickle.dump(rf_pipe, f)

    # Save metrics and metadata for the dashboard
    meta = {
        "feature_cols": FEATURE_COLS,
        "feature_labels": FEATURE_LABELS,
        "best_model": best_name,
        "lr_metrics": lr_metrics,
        "rf_metrics": rf_metrics,
        "cv_lr": {"mean": float(cv_lr.mean()), "std": float(cv_lr.std())},
        "cv_rf": {"mean": float(cv_rf.mean()), "std": float(cv_rf.std())},
        "train_size": len(X_train),
        "test_size": len(X_test),
        "default_rate": float(y_train.mean()),
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nAll artefacts saved to models/")
    return meta


if __name__ == "__main__":
    train()
