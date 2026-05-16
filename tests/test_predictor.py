"""
test_predictor.py
-----------------
Unit and integration tests for the CreditLens predictor module.

Run with:
    pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

VALID_FEATURES = {
    "credit_limit": 100000,
    "sex":          1,
    "education":    2,
    "marriage":     1,
    "age":          35,
    "pay_sep":      0,
    "pay_aug":      0,
    "pay_jul":     -1,
    "pay_jun":     -1,
    "pay_may":     -1,
    "pay_apr":     -2,
    "bill_sep":    50000,
    "bill_aug":    48000,
    "bill_jul":    45000,
    "bill_jun":    43000,
    "bill_may":    40000,
    "bill_apr":    38000,
    "paid_sep":     5000,
    "paid_aug":     5000,
    "paid_jul":     5000,
    "paid_jun":     5000,
    "paid_may":     5000,
    "paid_apr":     5000,
}

HIGH_RISK_FEATURES = {
    **VALID_FEATURES,
    "pay_sep":  8,
    "pay_aug":  8,
    "pay_jul":  7,
    "pay_jun":  6,
    "credit_limit": 10000,
    "paid_sep": 0,
    "paid_aug": 0,
}

LOW_RISK_FEATURES = {
    **VALID_FEATURES,
    "pay_sep":  -2,
    "pay_aug":  -2,
    "pay_jul":  -2,
    "pay_jun":  -2,
    "credit_limit": 500000,
    "paid_sep": 50000,
    "paid_aug": 50000,
}


# ─────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────

class TestPredictorOutput:
    """Tests for the predict() function contract."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "model.pkl")),
        reason="Model not trained yet — run python models/train_model.py"
    )
    def test_predict_returns_expected_keys(self):
        from models.predictor import predict
        result = predict(VALID_FEATURES)
        assert "probability"  in result
        assert "risk_band"    in result
        assert "band_colour"  in result
        assert "shap_values"  in result

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "model.pkl")),
        reason="Model not trained yet"
    )
    def test_probability_in_unit_interval(self):
        from models.predictor import predict
        result = predict(VALID_FEATURES)
        assert 0.0 <= result["probability"] <= 1.0

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "model.pkl")),
        reason="Model not trained yet"
    )
    def test_high_risk_higher_than_low_risk(self):
        from models.predictor import predict
        high = predict(HIGH_RISK_FEATURES)["probability"]
        low  = predict(LOW_RISK_FEATURES)["probability"]
        assert high > low, (
            f"Expected high-risk prob ({high:.3f}) > low-risk prob ({low:.3f})"
        )

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "model.pkl")),
        reason="Model not trained yet"
    )
    def test_shap_values_cover_all_features(self):
        from models.predictor import predict, get_feature_names
        result   = predict(VALID_FEATURES)
        expected = set(get_feature_names())
        actual   = set(result["shap_values"].keys())
        assert expected == actual

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "model.pkl")),
        reason="Model not trained yet"
    )
    def test_risk_band_values(self):
        from models.predictor import predict
        result = predict(VALID_FEATURES)
        valid_bands = {"Low Risk", "Moderate Risk", "High Risk", "Very High Risk"}
        assert result["risk_band"] in valid_bands

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "model.pkl")),
        reason="Model not trained yet"
    )
    def test_band_colour_is_hex(self):
        from models.predictor import predict
        result = predict(VALID_FEATURES)
        colour = result["band_colour"]
        assert colour.startswith("#") and len(colour) == 7


class TestRiskBandLogic:
    """White-box tests for the _risk_band helper via public interface."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "model.pkl")),
        reason="Model not trained yet"
    )
    def test_consistent_predictions(self):
        """Two calls with same features must return identical result."""
        from models.predictor import predict
        r1 = predict(VALID_FEATURES)
        r2 = predict(VALID_FEATURES)
        assert r1["probability"] == r2["probability"]
        assert r1["risk_band"]   == r2["risk_band"]


class TestModelLoader:
    """Tests for model loading and metadata."""

    def test_model_loaded_returns_bool(self):
        from models.predictor import model_loaded
        assert isinstance(model_loaded(), bool)

    def test_get_feature_names_returns_list(self):
        from models.predictor import get_feature_names
        names = get_feature_names()
        assert isinstance(names, list)
        if names:  # only if model is loaded
            assert len(names) == 23

    def test_get_model_name_returns_string(self):
        from models.predictor import get_model_name
        assert isinstance(get_model_name(), str)


class TestDataIntegrity:
    """Tests that validate the cleaned dataset."""

    DATA_PATH = os.path.join("data", "credit_data.csv")

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("data", "credit_data.csv")),
        reason="Dataset not yet prepared — run python data/prepare_data.py"
    )
    def test_dataset_has_expected_columns(self):
        import pandas as pd
        df = pd.read_csv(self.DATA_PATH)
        required = {"credit_limit", "sex", "education", "marriage", "age", "default"}
        assert required.issubset(set(df.columns))

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("data", "credit_data.csv")),
        reason="Dataset not prepared"
    )
    def test_no_null_values(self):
        import pandas as pd
        df = pd.read_csv(self.DATA_PATH)
        assert df.isnull().sum().sum() == 0, "Dataset contains null values"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("data", "credit_data.csv")),
        reason="Dataset not prepared"
    )
    def test_target_is_binary(self):
        import pandas as pd
        df = pd.read_csv(self.DATA_PATH)
        assert set(df["default"].unique()).issubset({0, 1})

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("data", "credit_data.csv")),
        reason="Dataset not prepared"
    )
    def test_minimum_row_count(self):
        import pandas as pd
        df = pd.read_csv(self.DATA_PATH)
        assert len(df) >= 25000, f"Expected ≥25,000 rows, got {len(df)}"
