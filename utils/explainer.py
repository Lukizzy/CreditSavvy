"""
explainer.py
------------
SHAP-based model explainability utilities.
Provides feature importance values for individual predictions and
global feature importance for the model performance page.
"""

import numpy as np
import pandas as pd


def get_shap_values(model_pipeline, X_input: pd.DataFrame, feature_labels: dict):
    """
    Compute SHAP values for a single applicant using the fitted pipeline.
    Falls back to coefficient-based approximation if shap is unavailable.

    Parameters
    ----------
    model_pipeline : sklearn Pipeline with 'scaler' and 'clf' steps.
    X_input : pd.DataFrame  Single-row DataFrame with raw feature values.
    feature_labels : dict   Mapping from column name to human-readable label.

    Returns
    -------
    pd.DataFrame with columns [feature, label, value, shap_value]
    sorted by absolute SHAP value descending.
    """
    try:
        import shap
        clf = model_pipeline.named_steps["clf"]
        scaler = model_pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(X_input)

        clf_type = type(clf).__name__
        if "Forest" in clf_type or "Tree" in clf_type:
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_scaled)
            sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        else:
            explainer = shap.LinearExplainer(clf, X_scaled,
                                             feature_perturbation="correlation_dependent")
            shap_vals = explainer.shap_values(X_scaled)
            sv = shap_vals[0]
    except Exception:
        sv = _coefficient_proxy(model_pipeline, X_input)

    features = X_input.columns.tolist()
    result = pd.DataFrame({
        "feature": features,
        "label": [feature_labels.get(f, f) for f in features],
        "value": X_input.iloc[0].values,
        "shap_value": sv,
    })
    result["abs_shap"] = result["shap_value"].abs()
    return result.sort_values("abs_shap", ascending=False).reset_index(drop=True)


def _coefficient_proxy(pipeline, X_input: pd.DataFrame) -> np.ndarray:
    clf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    X_s = scaler.transform(X_input)[0]
    if hasattr(clf, "coef_"):
        return clf.coef_[0] * X_s
    elif hasattr(clf, "feature_importances_"):
        return clf.feature_importances_ * np.sign(X_s)
    return np.zeros(X_input.shape[1])


def global_feature_importance(model_pipeline, feature_cols: list, feature_labels: dict) -> pd.DataFrame:
    """Global feature importance from model coefficients or tree importances."""
    clf = model_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_[0])
        importance = importance / importance.sum()
    else:
        importance = np.ones(len(feature_cols)) / len(feature_cols)

    return pd.DataFrame({
        "feature": feature_cols,
        "label": [feature_labels.get(f, f) for f in feature_cols],
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
