"""Descriptive group metrics; these do not establish that a model is fair."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score


def group_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: pd.Series,
    threshold: float,
    attribute: str,
) -> pd.DataFrame:
    """Calculate operational error rates for each observed subgroup."""

    rows: list[dict[str, float | int | str]] = []
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    group_values = groups.fillna("unknown").astype(str).to_numpy()
    for group in sorted(np.unique(group_values)):
        mask = group_values == group
        group_y = y[mask]
        predictions = (probability[mask] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(group_y, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "attribute": attribute,
                "group": group,
                "count": int(mask.sum()),
                "observed_default_rate": float(group_y.mean()),
                "selection_rate": float(predictions.mean()),
                "true_positive_rate": float(tp / (tp + fn)) if tp + fn else np.nan,
                "false_positive_rate": float(fp / (fp + tn)) if fp + tn else np.nan,
                "precision": float(precision_score(group_y, predictions, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def audit_fairness(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    attributes: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Return group metrics and simple between-group gap indicators."""

    tables = [
        group_metrics(
            frame["default_next_month"].to_numpy(),
            probabilities,
            frame[attribute],
            threshold,
            attribute,
        )
        for attribute in attributes
    ]
    metrics = pd.concat(tables, ignore_index=True)
    gaps: dict[str, dict[str, float]] = {}
    for attribute, subset in metrics.groupby("attribute"):
        selection_gap = subset["selection_rate"].max() - subset["selection_rate"].min()
        tpr_gap = subset["true_positive_rate"].max() - subset["true_positive_rate"].min()
        fpr_gap = subset["false_positive_rate"].max() - subset["false_positive_rate"].min()
        gaps[str(attribute)] = {
            "demographic_parity_difference": float(selection_gap),
            "equal_opportunity_difference": float(tpr_gap),
            "equalized_odds_difference": float(max(tpr_gap, fpr_gap)),
        }
    return metrics, gaps
