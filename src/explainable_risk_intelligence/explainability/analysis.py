"""Permutation importance and SHAP-based attribution utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def permutation_importance_table(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def _positive_class_values(values: np.ndarray) -> np.ndarray:
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def shap_attributions(
    base_pipeline: Any,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Explain the selected uncalibrated base estimator in transformed space."""

    preprocessor = base_pipeline.named_steps["preprocessor"]
    estimator = base_pipeline.named_steps["model"]
    background = preprocessor.transform(X_background)
    explained = preprocessor.transform(X_explain)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.Explainer(estimator, background, feature_names=feature_names)
    try:
        explanation = explainer(explained, check_additivity=False)
    except TypeError:
        explanation = explainer(explained)
    values = _positive_class_values(np.asarray(explanation.values))
    global_table = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    return global_table, values


def summarize_local_attributions(
    feature_names: list[str],
    transformed_feature_values: np.ndarray,
    original_feature_values: pd.DataFrame,
    shap_values: np.ndarray,
    probabilities: np.ndarray,
    record_ids: np.ndarray,
    case_labels: list[str],
    probability_source: str,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Create labelled attribution summaries in original and transformed units."""

    if len(case_labels) != len(probabilities):
        raise ValueError("One case label is required per local explanation")

    cases: list[dict[str, Any]] = []
    for row_index in range(len(probabilities)):
        attributions = np.asarray(shap_values[row_index])
        positive = np.argsort(attributions)[::-1][:top_n]
        negative = np.argsort(attributions)[:top_n]

        positive_details = [
            {
                "feature": feature_names[index],
                "original_feature_value": float(
                    original_feature_values.iloc[row_index][feature_names[index]]
                ),
                "transformed_feature_value": float(transformed_feature_values[row_index, index]),
                "attribution": float(attributions[index]),
            }
            for index in positive
            if abs(attributions[index]) > 1e-12
        ]
        negative_details = [
            {
                "feature": feature_names[index],
                "original_feature_value": float(
                    original_feature_values.iloc[row_index][feature_names[index]]
                ),
                "transformed_feature_value": float(transformed_feature_values[row_index, index]),
                "attribution": float(attributions[index]),
            }
            for index in negative
            if abs(attributions[index]) > 1e-12
        ]

        cases.append(
            {
                "case_label": case_labels[row_index],
                "record_id": int(record_ids[row_index]),
                "predicted_default_probability": float(probabilities[row_index]),
                "probability_source": probability_source,
                "attribution_scope": "uncalibrated selected base estimator in transformed feature space",
                "risk_increasing_attributions": positive_details,
                "risk_reducing_attributions": negative_details,
                "interpretation_warning": "Feature attribution is not causality.",
            }
        )
    return cases
