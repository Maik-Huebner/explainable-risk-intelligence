"""Model comparison, tuning, calibration, and validation-only policy selection."""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from explainable_risk_intelligence.config import ProjectPaths, load_config
from explainable_risk_intelligence.data.prepare import load_train_validation_splits
from explainable_risk_intelligence.evaluation.metrics import calculate_metrics
from explainable_risk_intelligence.evaluation.threshold import choose_threshold, threshold_analysis
from explainable_risk_intelligence.features.build import MODEL_FEATURES, RECORD_ID, TARGET
from explainable_risk_intelligence.models.pipeline import (
    candidate_models,
    hgb_search_space,
    make_model_pipeline,
)
from explainable_risk_intelligence.utils.io import write_json

SEARCH_TIMING_COLUMNS = [
    "mean_fit_time",
    "std_fit_time",
    "mean_score_time",
    "std_score_time",
]


def _metric_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "roc_auc",
        "average_precision",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "brier_score",
        "log_loss",
    ]
    return {"model": name, **{key: metrics[key] for key in keys}}


def select_best_model_name(comparison: pd.DataFrame) -> str:
    """Select the strongest non-dummy candidate from validation evidence."""

    eligible = comparison[comparison["model"] != "dummy_prior"].sort_values(
        ["average_precision", "brier_score"], ascending=[False, True]
    )
    if eligible.empty:
        raise ValueError("At least one non-dummy model is required")
    return str(eligible.iloc[0]["model"])


def calibration_is_acceptable(
    base_metrics: dict[str, Any],
    calibrated_metrics: dict[str, Any],
    average_precision_tolerance: float = 0.01,
) -> bool:
    """Apply the documented validation-only calibration acceptance rule."""

    return bool(
        calibrated_metrics["brier_score"] <= base_metrics["brier_score"]
        and calibrated_metrics["average_precision"]
        >= base_metrics["average_precision"] - average_precision_tolerance
    )


def select_validation_threshold(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    threshold_config: dict[str, float],
) -> tuple[pd.DataFrame, float]:
    """Build and select the policy using validation arrays supplied by the caller."""

    grid = np.arange(
        threshold_config["grid_start"],
        threshold_config["grid_stop"] + threshold_config["grid_step"] / 2,
        threshold_config["grid_step"],
    )
    analysis = threshold_analysis(
        y_validation,
        validation_probabilities,
        thresholds=grid,
        false_positive_cost=threshold_config["false_positive_cost"],
        false_negative_cost=threshold_config["false_negative_cost"],
    )
    return analysis, choose_threshold(analysis)


def stable_search_results(cv_results: dict[str, Any]) -> pd.DataFrame:
    """Return reproducible search evidence without machine-dependent timings."""

    return pd.DataFrame(cv_results).drop(columns=SEARCH_TIMING_COLUMNS, errors="ignore")


def persist_model_metadata(paths: ProjectPaths, metadata: dict[str, Any]) -> None:
    """Persist one canonical metadata object for runtime and published evidence."""

    for destination in (
        paths.artifacts / "model_metadata.json",
        paths.metrics / "model_metadata.json",
    ):
        write_json(destination, metadata)


def train_models(fast: bool = False) -> dict[str, Any]:
    """Train without accessing the final test set."""

    config = load_config()
    paths = ProjectPaths()
    paths.ensure_generated_directories()
    train, validation = load_train_validation_splits()
    seed = int(config["project"]["random_seed"])
    X_train = train[MODEL_FEATURES]
    y_train = train[TARGET].to_numpy()
    X_validation = validation[MODEL_FEATURES]
    y_validation = validation[TARGET].to_numpy()

    fitted: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for name, pipeline in candidate_models(seed, fast=fast).items():
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_validation)[:, 1]
        metrics = calculate_metrics(y_validation, probabilities)
        comparison_rows.append(_metric_row(name, metrics))
        fitted[name] = pipeline

    cv_folds = 2 if fast else int(config["training"]["cv_folds"])
    search_iterations = 1 if fast else int(config["training"]["search_iterations"])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    tuned_pipeline = make_model_pipeline(HistGradientBoostingClassifier(random_state=seed))
    search = RandomizedSearchCV(
        estimator=tuned_pipeline,
        param_distributions=hgb_search_space(fast=fast),
        n_iter=search_iterations,
        scoring=config["training"]["scoring"],
        cv=cv,
        refit=True,
        n_jobs=-1,
        random_state=seed,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    tuned_probabilities = search.best_estimator_.predict_proba(X_validation)[:, 1]
    tuned_metrics = calculate_metrics(y_validation, tuned_probabilities)
    comparison_rows.append(_metric_row("tuned_hist_gradient_boosting", tuned_metrics))
    fitted["tuned_hist_gradient_boosting"] = search.best_estimator_

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["average_precision", "brier_score"], ascending=[False, True]
    )
    comparison.to_csv(
        paths.metrics / "model_comparison_validation.csv", index=False, float_format="%.12g"
    )
    stable_search_results(search.cv_results_).sort_values("rank_test_score").to_csv(
        paths.metrics / "hyperparameter_search.csv", index=False, float_format="%.12g"
    )

    selected_name = select_best_model_name(comparison)
    base_model = fitted[selected_name]
    base_probabilities = base_model.predict_proba(X_validation)[:, 1]
    base_metrics = calculate_metrics(y_validation, base_probabilities)

    calibrated_model = CalibratedClassifierCV(
        estimator=clone(base_model),
        method=config["training"]["calibration_method"],
        cv=cv,
    )
    calibrated_model.fit(X_train, y_train)
    calibrated_probabilities = calibrated_model.predict_proba(X_validation)[:, 1]
    calibrated_metrics = calculate_metrics(y_validation, calibrated_probabilities)

    calibration_comparison = pd.DataFrame(
        [
            _metric_row("uncalibrated", base_metrics),
            _metric_row("sigmoid_calibrated", calibrated_metrics),
        ]
    )
    calibration_comparison.to_csv(
        paths.metrics / "calibration_comparison_validation.csv", index=False
    )
    use_calibrated_model = calibration_is_acceptable(base_metrics, calibrated_metrics)
    final_model = calibrated_model if use_calibrated_model else base_model
    final_variant = "sigmoid_calibrated" if use_calibrated_model else "uncalibrated"
    validation_probabilities = (
        calibrated_probabilities if use_calibrated_model else base_probabilities
    )

    threshold_config = config["threshold"]
    analysis, selected_threshold = select_validation_threshold(
        y_validation,
        validation_probabilities,
        threshold_config,
    )
    analysis.to_csv(paths.metrics / "threshold_analysis_validation.csv", index=False)
    pd.DataFrame(
        {
            RECORD_ID: validation[RECORD_ID],
            "y_true": y_validation,
            "probability": validation_probabilities,
            "prediction": (validation_probabilities >= selected_threshold).astype(int),
        }
    ).to_csv(paths.metrics / "validation_predictions.csv", index=False)

    joblib.dump(base_model, paths.artifacts / "base_model.joblib")
    joblib.dump(final_model, paths.artifacts / "model.joblib")
    metadata = {
        "selected_model": selected_name,
        "final_variant": final_variant,
        "selected_threshold": selected_threshold,
        "selection_metric": config["training"]["scoring"],
        "best_cv_score": float(search.best_score_),
        "best_hyperparameters": search.best_params_,
        "validation_metrics": calculate_metrics(
            y_validation, validation_probabilities, selected_threshold
        ),
        "model_features": MODEL_FEATURES,
        "sensitive_features_excluded": ["SEX", "AGE", "EDUCATION", "MARRIAGE"],
        "training_rows": len(train),
        "validation_rows": len(validation),
        "fast_mode": fast,
        "random_seed": seed,
        "python_hash_seed": os.getenv("PYTHONHASHSEED", "not-set"),
    }
    persist_model_metadata(paths, metadata)
    write_json(
        paths.metrics / "threshold_policy.json",
        {
            "selected_threshold": selected_threshold,
            "false_positive_cost": threshold_config["false_positive_cost"],
            "false_negative_cost": threshold_config["false_negative_cost"],
            "selection_data": "validation only",
            "purpose": "illustrative decision-support review policy; not a lending decision",
        },
    )
    return metadata
