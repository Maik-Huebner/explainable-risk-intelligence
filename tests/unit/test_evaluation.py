from __future__ import annotations

import numpy as np
import pandas as pd

from explainable_risk_intelligence.config import ProjectPaths
from explainable_risk_intelligence.evaluation.bootstrap import bootstrap_intervals
from explainable_risk_intelligence.evaluation.metrics import calculate_metrics
from explainable_risk_intelligence.evaluation.run import _error_analysis
from explainable_risk_intelligence.evaluation.threshold import choose_threshold, threshold_analysis
from explainable_risk_intelligence.features.build import AUDIT_COLUMNS, RECORD_ID, TARGET


def test_metric_calculation_known_confusion_matrix() -> None:
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.7, 0.2])
    result = calculate_metrics(y, probabilities, threshold=0.5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert 0 <= result["brier_score"] <= 1


def test_threshold_policy_uses_explicit_costs() -> None:
    y = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.6, 0.4, 0.9])
    analysis = threshold_analysis(y, probabilities, np.array([0.3, 0.5, 0.7]), 1, 5)
    selected = choose_threshold(analysis)
    expected = analysis.sort_values(
        ["illustrative_cost", "recall", "threshold"], ascending=[True, False, True]
    ).iloc[0]["threshold"]
    assert selected == expected


def test_bootstrap_intervals_are_bounded() -> None:
    y = np.array([0, 1] * 50)
    probabilities = np.linspace(0.01, 0.99, 100)
    intervals = bootstrap_intervals(y, probabilities, threshold=0.5, iterations=30, seed=42)
    for interval in intervals.values():
        assert 0 <= interval["lower_95"] <= interval["upper_95"] <= 1


def test_choose_threshold_breaks_cost_tie_toward_recall() -> None:
    analysis = pd.DataFrame(
        {
            "threshold": [0.3, 0.5],
            "illustrative_cost": [10.0, 10.0],
            "recall": [0.9, 0.7],
        }
    )
    assert choose_threshold(analysis) == 0.3


def test_error_analysis_reports_normalized_segment_and_threshold_rates(
    tmp_path: object,
) -> None:
    frame = pd.DataFrame(
        {
            RECORD_ID: range(1, 9),
            TARGET: [0, 0, 1, 1, 0, 0, 1, 1],
            "audit_sex": ["female"] * 4 + ["male"] * 4,
            "audit_age": [25, 26, 27, 28, 45, 46, 47, 48],
            "audit_age_band": ["under_30"] * 4 + ["40_to_49"] * 4,
            "audit_education": ["1"] * 8,
            "audit_marriage": ["2"] * 8,
        }
    )
    assert set(AUDIT_COLUMNS).issubset(frame.columns)
    probabilities = np.array([0.1, 0.6, 0.4, 0.9, 0.2, 0.3, 0.49, 0.8])
    paths = ProjectPaths(root=tmp_path)  # type: ignore[arg-type]
    paths.ensure_generated_directories()
    analysis = _error_analysis(frame, probabilities, threshold=0.5, paths=paths)
    segments = pd.read_csv(paths.metrics / "error_segments_test.csv")
    assert analysis["sample_size"] == 8
    assert analysis["rates"]["overall_error_rate"] == 3 / 8
    assert analysis["rates"]["false_positive_rate"] == 1 / 4
    assert analysis["rates"]["false_negative_rate"] == 2 / 4
    assert set(
        ["sample_size", "error_rate", "false_positive_rate", "false_negative_rate"]
    ).issubset(segments.columns)
    assert segments["sample_size"].sum() == 8
    assert len(analysis["observations"]) == 4
    assert "; n=4)." in analysis["observations"][0]
    assert "; 1 / 2 non-default cases)." in analysis["observations"][1]
    assert "; 1 / 2 default cases)." in analysis["observations"][2]
