"""Validation-only threshold policy analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from explainable_risk_intelligence.evaluation.metrics import calculate_metrics


def threshold_analysis(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
) -> pd.DataFrame:
    """Evaluate an illustrative review policy across candidate thresholds."""

    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        metrics = calculate_metrics(y_true, probabilities, float(threshold))
        total_cost = (
            metrics["false_positive"] * false_positive_cost
            + metrics["false_negative"] * false_negative_cost
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "review_rate": float((np.asarray(probabilities) >= threshold).mean()),
                "illustrative_cost": float(total_cost),
                "cost_per_case": float(total_cost / len(y_true)),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(analysis: pd.DataFrame) -> float:
    """Choose the lowest-cost threshold, resolving ties toward higher recall."""

    ordered = analysis.sort_values(
        ["illustrative_cost", "recall", "threshold"], ascending=[True, False, True]
    )
    return float(ordered.iloc[0]["threshold"])
