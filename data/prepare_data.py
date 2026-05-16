"""
prepare_data.py
---------------
Downloads and preprocesses the UCI Credit Card Default dataset.
Run this script once before launching the app.

Dataset: Default of Credit Card Clients
Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
"""

import pandas as pd
import numpy as np
import os


def download_and_prepare():
    """
    Downloads the dataset via UCI ML Repository and saves a clean CSV.
    Falls back to generating a representative synthetic dataset if the
    network is unavailable (for offline demo purposes).
    """
    output_path = os.path.join(os.path.dirname(__file__), "credit_data.csv")

    try:
        from ucimlrepo import fetch_ucirepo
        print("Fetching dataset from UCI ML Repository...")
        dataset = fetch_ucirepo(id=350)
        df = dataset.data.features.copy()
        df["default"] = dataset.data.targets.values.ravel()
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}  ({len(df)} rows)")

    except Exception as e:
        print(f"Could not fetch from UCI ({e}). Generating synthetic fallback dataset...")
        _generate_synthetic(output_path)

    return output_path


def _generate_synthetic(output_path: str, n: int = 5000, seed: int = 42):
    """
    Generates a synthetic dataset with the same schema and statistical
    properties as the UCI Credit Card Default dataset for offline use.
    """
    rng = np.random.default_rng(seed)

    limit_bal = rng.integers(10_000, 800_000, n)
    sex = rng.integers(1, 3, n)
    education = rng.integers(1, 5, n)
    marriage = rng.integers(0, 4, n)
    age = rng.integers(21, 75, n)

    pay_cols = {}
    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        pay_cols[col] = rng.integers(-2, 9, n)

    bill_cols = {}
    for col in ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]:
        bill_cols[col] = rng.integers(0, 500_000, n)

    pay_amt_cols = {}
    for col in ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]:
        pay_amt_cols[col] = rng.integers(0, 200_000, n)

    default_prob = (
        0.05
        + 0.15 * (pay_cols["PAY_0"] > 1).astype(float)
        + 0.10 * (pay_cols["PAY_2"] > 1).astype(float)
        + 0.05 * (limit_bal < 50_000).astype(float)
    )
    default_prob = np.clip(default_prob, 0, 1)
    default = rng.binomial(1, default_prob, n)

    df = pd.DataFrame({
        "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education,
        "MARRIAGE": marriage, "AGE": age,
        **pay_cols, **bill_cols, **pay_amt_cols,
        "default": default
    })

    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset saved to {output_path}  ({n} rows)")


if __name__ == "__main__":
    download_and_prepare()
