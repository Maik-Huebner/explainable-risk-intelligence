"""Bootstrap confidence intervals for central test metrics."""

from __future__ import annotations

import numpy as np

from explainable_risk_intelligence.evaluation.metrics import calculate_metrics

BOOTSTRAP_METRICS = ["roc_auc", "average_precision", "brier_score", "recall", "precision"]


def bootstrap_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Return percentile 95% intervals from paired iid row resamples.

    Resamples that contain only one target class are rejected because ranking
    metrics are undefined for them. Class proportions are not held fixed.
    """

    y = np.asarray(y_true)
    probability = np.asarray(probabilities)
    generator = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(iterations):
        indices = generator.integers(0, len(y), size=len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        values = calculate_metrics(y[indices], probability[indices], threshold)
        for metric in BOOTSTRAP_METRICS:
            collected[metric].append(float(values[metric]))

    return {
        metric: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for metric, values in collected.items()
    }
