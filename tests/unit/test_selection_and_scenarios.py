from __future__ import annotations

import numpy as np
import pandas as pd

from explainable_risk_intelligence.config import ProjectPaths
from explainable_risk_intelligence.evaluation.scenarios import (
    apply_robustness_perturbation,
    apply_synthetic_drift_shift,
)
from explainable_risk_intelligence.features.build import (
    BILL_COLUMNS,
    MODEL_FEATURES,
    PAYMENT_COLUMNS,
    build_features,
)
from explainable_risk_intelligence.models.train import (
    calibration_is_acceptable,
    persist_model_metadata,
    select_best_model_name,
    select_validation_threshold,
    stable_search_results,
)
from explainable_risk_intelligence.utils.io import read_json


def test_model_selection_uses_validation_average_precision_then_brier() -> None:
    comparison = pd.DataFrame(
        [
            {"model": "dummy_prior", "average_precision": 0.9, "brier_score": 0.1},
            {"model": "a", "average_precision": 0.6, "brier_score": 0.2},
            {"model": "b", "average_precision": 0.6, "brier_score": 0.15},
        ]
    )
    assert select_best_model_name(comparison) == "b"


def test_calibration_acceptance_rule_uses_brier_and_ap_tolerance() -> None:
    base = {"brier_score": 0.14, "average_precision": 0.56}
    assert calibration_is_acceptable(base, {"brier_score": 0.13, "average_precision": 0.551})
    assert not calibration_is_acceptable(base, {"brier_score": 0.141, "average_precision": 0.57})
    assert not calibration_is_acceptable(base, {"brier_score": 0.13, "average_precision": 0.549})


def test_threshold_selection_consumes_only_supplied_validation_arrays() -> None:
    validation_target = np.array([0, 0, 0, 1, 1])
    validation_probability = np.array([0.1, 0.2, 0.6, 0.4, 0.9])
    config = {
        "grid_start": 0.3,
        "grid_stop": 0.7,
        "grid_step": 0.2,
        "false_positive_cost": 1.0,
        "false_negative_cost": 5.0,
    }
    analysis, selected = select_validation_threshold(
        validation_target, validation_probability, config
    )
    expected = analysis.sort_values(
        ["illustrative_cost", "recall", "threshold"], ascending=[True, False, True]
    ).iloc[0]["threshold"]
    assert selected == expected


def test_search_evidence_excludes_machine_dependent_timings() -> None:
    results = stable_search_results(
        {
            "mean_fit_time": [0.12],
            "std_fit_time": [0.01],
            "mean_score_time": [0.03],
            "std_score_time": [0.002],
            "params": [{"model__max_iter": 100}],
            "mean_test_score": [0.55],
            "rank_test_score": [1],
        }
    )
    assert list(results.columns) == ["params", "mean_test_score", "rank_test_score"]


def test_model_metadata_is_identical_in_runtime_and_published_evidence(tmp_path: object) -> None:
    paths = ProjectPaths(root=tmp_path)  # type: ignore[arg-type]
    paths.ensure_generated_directories()
    metadata = {"selected_model": "fixture", "selected_threshold": 0.17}
    persist_model_metadata(paths, metadata)
    assert read_json(paths.artifacts / "model_metadata.json") == metadata
    assert read_json(paths.metrics / "model_metadata.json") == metadata


def test_robustness_perturbation_rebuilds_dependent_features(raw_frame: pd.DataFrame) -> None:
    features = build_features(raw_frame).head(5)[MODEL_FEATURES]
    perturbed = apply_robustness_perturbation(features)
    np.testing.assert_allclose(perturbed["LIMIT_BAL"], features["LIMIT_BAL"] * 1.10)
    np.testing.assert_allclose(perturbed[PAYMENT_COLUMNS], features[PAYMENT_COLUMNS] * 0.95)
    np.testing.assert_allclose(
        perturbed["avg_payment_amount"], perturbed[PAYMENT_COLUMNS].mean(axis=1)
    )
    np.testing.assert_allclose(
        perturbed["credit_utilization"],
        perturbed[BILL_COLUMNS].mean(axis=1) / perturbed["LIMIT_BAL"],
    )


def test_drift_shift_rebuilds_all_dependent_features(raw_frame: pd.DataFrame) -> None:
    features = build_features(raw_frame).head(5)[MODEL_FEATURES]
    shifted = apply_synthetic_drift_shift(features)
    np.testing.assert_allclose(shifted[BILL_COLUMNS], features[BILL_COLUMNS] * 1.15)
    np.testing.assert_allclose(shifted[PAYMENT_COLUMNS], features[PAYMENT_COLUMNS] * 0.85)
    np.testing.assert_allclose(shifted["avg_bill_amount"], shifted[BILL_COLUMNS].mean(axis=1))
    np.testing.assert_allclose(
        shifted["payment_to_bill_ratio"],
        shifted[PAYMENT_COLUMNS].mean(axis=1) / shifted[BILL_COLUMNS].mean(axis=1).abs(),
    )
