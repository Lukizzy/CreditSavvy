"""
train_model.py
--------------
Trains a Logistic Regression and a Random Forest classifier on the UCI
Credit Card Default dataset, evaluates both, and persists the better-
performing model (by AUC-ROC) as 'model.pkl'.

A StandardScaler is saved as 'scaler.pkl' for use at inference time.

Run from the project root:
    python models/train_model.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "credit_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), "feature_names.pkl")

TARGET = "default"
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train both models and return the better one with its metadata."""

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    lr.fit(X_train_sc, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_sc)[:, 1])

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)  # RF does not need scaling
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    print(f"\n{'='*50}")
    print(f"  Logistic Regression  AUC-ROC: {lr_auc:.4f}")
    print(f"  Random Forest        AUC-ROC: {rf_auc:.4f}")
    print(f"{'='*50}")

    # Choose better model by AUC
    if rf_auc >= lr_auc:
        print("\n→ Random Forest selected as production model.")
        print("\nClassification Report (Random Forest):")
        print(classification_report(y_test, rf.predict(X_test)))
        return rf, None, rf_auc, "Random Forest"
    else:
        print("\n→ Logistic Regression selected as production model.")
        print("\nClassification Report (Logistic Regression):")
        print(classification_report(y_test, lr.predict(X_test_sc)))
        return lr, scaler, lr_auc, "Logistic Regression"


def cross_validate(model, X, y, scaler=None):
    """5-fold stratified cross-validation AUC."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if scaler:
        X_sc = scaler.fit_transform(X)
        scores = cross_val_score(model, X_sc, y, cv=cv, scoring="roc_auc")
    else:
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"\n5-Fold CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")


def main():
    print("Loading data …")
    X, y = load_data()
    print(f"  {len(X):,} samples | {X.shape[1]} features | default rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model, scaler, auc, model_name = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    cross_validate(model, X, y, scaler)

    # Persist artefacts
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "model_name": model_name, "auc": auc}, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    with open(FEATURE_PATH, "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"\nModel saved  → {MODEL_PATH}")
    print(f"Scaler saved → {SCALER_PATH}")
    print(f"Features saved → {FEATURE_PATH}")


if __name__ == "__main__":
    main()
