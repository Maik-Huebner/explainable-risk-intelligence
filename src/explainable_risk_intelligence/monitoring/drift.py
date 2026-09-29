"""Small PSI and KS drift demonstrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def population_stability_index(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    bins: int = 10,
) -> float:
    """Compute PSI using reference quantile bins and numerical safeguards."""

    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("PSI requires finite reference and current observations")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.allclose(ref.mean(), cur.mean()) else float("inf")
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_counts = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_counts = np.clip(ref_counts, 1e-6, None)
    cur_counts = np.clip(cur_counts, 1e-6, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def numeric_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for feature in features:
        reference_values = reference[feature].dropna().to_numpy(dtype=float)
        current_values = current[feature].dropna().to_numpy(dtype=float)
        ks = ks_2samp(reference_values, current_values)
        rows.append(
            {
                "feature": feature,
                "reference_mean": float(reference_values.mean()),
                "current_mean": float(current_values.mean()),
                "psi": population_stability_index(reference_values, current_values),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
