from __future__ import annotations

import numpy as np
import pandas as pd

from explainable_risk_intelligence.explainability.analysis import summarize_local_attributions
from explainable_risk_intelligence.fairness.metrics import audit_fairness, group_metrics
from explainable_risk_intelligence.monitoring.drift import (
    numeric_drift_report,
    population_stability_index,
)


def test_group_metrics_counts_every_row() -> None:
    y = np.array([0, 1, 0, 1, 1, 0])
    probabilities = np.array([0.1, 0.9, 0.6, 0.7, 0.4, 0.2])
    groups = pd.Series(["a", "a", "a", "b", "b", "b"])
    result = group_metrics(y, probabilities, groups, 0.5, "group")
    assert result["count"].sum() == len(y)
    assert set(result["group"]) == {"a", "b"}


def test_fairness_gap_definitions() -> None:
    frame = pd.DataFrame(
        {
            "default_next_month": [0, 1, 0, 1, 0, 1, 0, 1],
            "audit_sex": ["a"] * 4 + ["b"] * 4,
        }
    )
    probabilities = np.array([0.1, 0.9, 0.7, 0.8, 0.1, 0.4, 0.2, 0.3])
    _, gaps = audit_fairness(frame, probabilities, 0.5, ["audit_sex"])
    assert gaps["audit_sex"]["demographic_parity_difference"] == 0.75
    assert gaps["audit_sex"]["equal_opportunity_difference"] == 1.0
    assert gaps["audit_sex"]["equalized_odds_difference"] == 1.0


def test_psi_detects_identical_and_shifted_distributions() -> None:
    reference = np.arange(1, 101)
    assert population_stability_index(reference, reference) == 0
    assert population_stability_index(reference, reference + 100) > 0.1


def test_numeric_drift_report_is_sorted() -> None:
    reference = pd.DataFrame({"stable": np.arange(100), "shifted": np.arange(100)})
    current = pd.DataFrame({"stable": np.arange(100), "shifted": np.arange(100) + 100})
    report = numeric_drift_report(reference, current, ["stable", "shifted"])
    assert report.iloc[0]["feature"] == "shifted"


def test_local_explanation_summary_separates_directions() -> None:
    cases = summarize_local_attributions(
        ["a", "b", "c"],
        np.array([[1.0, 2.0, 3.0]]),
        pd.DataFrame([[10.0, 20.0, 30.0]], columns=["a", "b", "c"]),
        np.array([[0.5, -0.2, 0.1]]),
        np.array([0.7]),
        np.array([99]),
        ["high_risk"],
        "sigmoid-calibrated final model",
        top_n=2,
    )
    assert cases[0]["record_id"] == 99
    assert cases[0]["case_label"] == "high_risk"
    assert cases[0]["risk_increasing_attributions"][0]["feature"] == "a"
    assert cases[0]["risk_increasing_attributions"][0]["original_feature_value"] == 10.0
    assert cases[0]["risk_increasing_attributions"][0]["transformed_feature_value"] == 1.0
    assert cases[0]["risk_reducing_attributions"][0]["feature"] == "b"
    assert cases[0]["interpretation_warning"] == "Feature attribution is not causality."
