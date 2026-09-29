"""Read-only helpers for the curated portfolio notebook and reviewers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from explainable_risk_intelligence.config import ProjectPaths
from explainable_risk_intelligence.utils.io import read_json


def load_portfolio_evidence(root: Path | None = None) -> dict[str, Any]:
    """Load only publishable report evidence through one documented API."""

    paths = ProjectPaths(root=root)
    json_files = {
        "manifest": paths.metrics / "data_manifest.json",
        "metadata": paths.metrics / "model_metadata.json",
        "test": paths.metrics / "test_metrics.json",
        "fairness_summary": paths.metrics / "fairness_summary.json",
        "error_analysis": paths.metrics / "error_analysis.json",
        "robustness": paths.metrics / "robustness.json",
        "drift_summary": paths.metrics / "drift_monitoring_summary.json",
        "local_explanations": paths.metrics / "local_explanations_test.json",
    }
    csv_files = {
        "model_comparison": paths.metrics / "model_comparison_validation.csv",
        "calibration": paths.metrics / "calibration_comparison_validation.csv",
        "fairness_groups": paths.metrics / "fairness_metrics_test.csv",
        "error_segments": paths.metrics / "error_segments_test.csv",
        "shap_global": paths.metrics / "shap_global_test.csv",
        "permutation_importance": paths.metrics / "permutation_importance_test.csv",
        "drift_features": paths.metrics / "drift_monitoring_demo.csv",
    }
    missing = [
        str(path) for path in (*json_files.values(), *csv_files.values()) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Portfolio evidence is incomplete; run `make reproduce` first: " + ", ".join(missing)
        )
    return {
        **{name: read_json(path) for name, path in json_files.items()},
        **{name: pd.read_csv(path) for name, path in csv_files.items()},
    }


def final_metrics_table(evidence: dict[str, Any]) -> pd.DataFrame:
    """Return final test metrics and available bootstrap intervals."""

    test = evidence["test"]
    intervals = test["confidence_intervals"]
    rows = []
    for metric, value in test["metrics"].items():
        if metric == "confusion_matrix":
            continue
        interval = intervals.get(metric, {})
        rows.append(
            {
                "metric": metric,
                "value": value,
                "lower_95": interval.get("lower_95"),
                "upper_95": interval.get("upper_95"),
            }
        )
    return pd.DataFrame(rows).set_index("metric")
