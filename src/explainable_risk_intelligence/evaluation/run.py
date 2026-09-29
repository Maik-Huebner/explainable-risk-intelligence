"""One-time final test evaluation and portfolio artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from explainable_risk_intelligence.config import ProjectPaths, load_config
from explainable_risk_intelligence.data.prepare import load_split, load_test_split
from explainable_risk_intelligence.data.schema import validate_inference_features
from explainable_risk_intelligence.evaluation.bootstrap import bootstrap_intervals
from explainable_risk_intelligence.evaluation.metrics import calculate_metrics
from explainable_risk_intelligence.evaluation.scenarios import (
    apply_robustness_perturbation,
    apply_synthetic_drift_shift,
)
from explainable_risk_intelligence.explainability.analysis import (
    permutation_importance_table,
    shap_attributions,
    summarize_local_attributions,
)
from explainable_risk_intelligence.fairness.metrics import audit_fairness
from explainable_risk_intelligence.features.build import (
    AUDIT_COLUMNS,
    MODEL_FEATURES,
    RECORD_ID,
    TARGET,
)
from explainable_risk_intelligence.monitoring.drift import numeric_drift_report
from explainable_risk_intelligence.utils.io import read_json, write_json

sns.set_theme(style="whitegrid", context="talk")


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_core_curves(
    y: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    paths: ProjectPaths,
) -> None:
    RocCurveDisplay.from_predictions(y, probabilities, name="Final model")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.title("Final test ROC curve")
    _save_figure(paths.figures / "roc_curve_test.png")

    PrecisionRecallDisplay.from_predictions(y, probabilities, name="Final model")
    plt.axhline(y.mean(), linestyle="--", color="grey", label="Prevalence")
    plt.title("Final test precision-recall curve")
    _save_figure(paths.figures / "precision_recall_curve_test.png")

    ConfusionMatrixDisplay.from_predictions(
        y, predictions, display_labels=["No default", "Default"]
    )
    plt.title("Final test confusion matrix at validation-selected threshold")
    _save_figure(paths.figures / "confusion_matrix_test.png")

    observed, predicted = calibration_curve(y, probabilities, n_bins=10, strategy="quantile")
    plt.figure(figsize=(8, 6))
    plt.plot(predicted, observed, marker="o", label="Final model")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed default rate")
    plt.title("Calibration curve — final test")
    plt.legend()
    _save_figure(paths.figures / "calibration_curve_test.png")


def _plot_validation_artifacts(paths: ProjectPaths) -> None:
    threshold = pd.read_csv(paths.metrics / "threshold_analysis_validation.csv")
    selected = read_json(paths.metrics / "threshold_policy.json")["selected_threshold"]
    _figure, axis = plt.subplots(1, 2, figsize=(14, 5))
    axis[0].plot(threshold["threshold"], threshold["precision"], label="Precision")
    axis[0].plot(threshold["threshold"], threshold["recall"], label="Recall")
    axis[0].plot(threshold["threshold"], threshold["f1"], label="F1")
    axis[0].axvline(selected, linestyle="--", color="black", label=f"Selected {selected:.2f}")
    axis[0].set_title("Validation threshold trade-off")
    axis[0].set_xlabel("Threshold")
    axis[0].legend()
    axis[1].plot(threshold["threshold"], threshold["cost_per_case"], color="darkred")
    axis[1].axvline(selected, linestyle="--", color="black")
    axis[1].set_title("Illustrative validation cost per case")
    axis[1].set_xlabel("Threshold")
    _save_figure(paths.figures / "threshold_analysis_validation.png")

    comparison = pd.read_csv(paths.metrics / "model_comparison_validation.csv")
    order = comparison.sort_values("average_precision", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(order["model"], order["average_precision"], color="#315f72")
    plt.xlabel("Validation average precision")
    plt.title("Model comparison — selection data only")
    _save_figure(paths.figures / "model_comparison_validation.png")


def _error_analysis(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    paths: ProjectPaths,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    errors = test[[RECORD_ID, TARGET, *AUDIT_COLUMNS]].copy()
    errors["probability"] = probabilities
    errors["prediction"] = predictions
    errors["error_type"] = np.select(
        [
            (errors[TARGET] == 0) & (errors["prediction"] == 1),
            (errors[TARGET] == 1) & (errors["prediction"] == 0),
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )
    errors["distance_to_threshold"] = (errors["probability"] - threshold).abs()
    error_cases = errors[errors["error_type"] != "correct"].sort_values("distance_to_threshold")
    error_cases.to_csv(paths.metrics / "error_cases_test.csv", index=False)
    near_threshold_mask = errors["distance_to_threshold"] <= 0.05
    near_threshold_errors = errors[near_threshold_mask & (errors["error_type"] != "correct")]
    near_threshold_errors.to_csv(paths.metrics / "near_threshold_errors_test.csv", index=False)

    segment_rows: list[dict[str, Any]] = []
    for (sex, age_band), subset in errors.groupby(
        ["audit_sex", "audit_age_band"], dropna=False, observed=True
    ):
        positives = int((subset[TARGET] == 1).sum())
        negatives = int((subset[TARGET] == 0).sum())
        false_positives = int((subset["error_type"] == "false_positive").sum())
        false_negatives = int((subset["error_type"] == "false_negative").sum())
        correct = int((subset["error_type"] == "correct").sum())
        near = subset["distance_to_threshold"] <= 0.05
        near_errors = near & (subset["error_type"] != "correct")
        segment_rows.append(
            {
                "audit_sex": str(sex),
                "audit_age_band": str(age_band),
                "sample_size": len(subset),
                "default_cases": positives,
                "non_default_cases": negatives,
                "correct_count": correct,
                "false_positive_count": false_positives,
                "false_negative_count": false_negatives,
                "error_count": false_positives + false_negatives,
                "error_rate": (false_positives + false_negatives) / len(subset),
                "false_positive_rate": false_positives / negatives if negatives else np.nan,
                "false_negative_rate": false_negatives / positives if positives else np.nan,
                "near_threshold_count": int(near.sum()),
                "near_threshold_error_count": int(near_errors.sum()),
                "near_threshold_error_rate": (
                    float(near_errors.sum() / near.sum()) if near.sum() else np.nan
                ),
            }
        )
    segment = pd.DataFrame(segment_rows).sort_values("error_rate", ascending=False)
    segment.to_csv(paths.metrics / "error_segments_test.csv", index=False)
    uncertain = errors.nsmallest(25, "distance_to_threshold")
    uncertain.to_csv(paths.metrics / "most_uncertain_cases_test.csv", index=False)

    plot = segment.copy()
    plot["segment"] = plot["audit_sex"] + " / " + plot["audit_age_band"]
    plot = plot.melt(
        id_vars=["segment"],
        value_vars=["error_rate", "false_positive_rate", "false_negative_rate"],
        var_name="rate",
        value_name="value",
    )
    plt.figure(figsize=(13, 7))
    sns.barplot(data=plot, x="segment", y="value", hue="rate")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Observed rate")
    plt.title("Descriptive test errors by audit segment")
    _save_figure(paths.figures / "error_rates_by_segment_test.png")

    counts = errors["error_type"].value_counts().to_dict()
    defaults = int((errors[TARGET] == 1).sum())
    non_defaults = int((errors[TARGET] == 0).sum())
    highest_error = segment.iloc[0]
    highest_fpr = segment.loc[segment["false_positive_rate"].idxmax()]
    highest_fnr = segment.loc[segment["false_negative_rate"].idxmax()]
    near_count = int(near_threshold_mask.sum())
    near_error_count = len(near_threshold_errors)
    near_error_rate = float(near_error_count / near_count) if near_count else float("nan")

    def describe(row: pd.Series, metric: str) -> str:
        if metric == "false_positive_rate":
            denominator = (
                f"{int(row['false_positive_count'])} / {int(row['non_default_cases'])} "
                "non-default cases"
            )
        elif metric == "false_negative_rate":
            denominator = (
                f"{int(row['false_negative_count'])} / {int(row['default_cases'])} default cases"
            )
        else:
            denominator = f"n={int(row['sample_size'])}"
        return (
            f"{row['audit_sex']} / {row['audit_age_band']} has the highest observed "
            f"{metric.replace('_', ' ')} ({row[metric]:.3f}; {denominator})."
        )

    return {
        "sample_size": len(errors),
        "counts": {key: int(value) for key, value in counts.items()},
        "rates": {
            "overall_error_rate": float((errors["error_type"] != "correct").mean()),
            "false_positive_rate": float(counts.get("false_positive", 0) / non_defaults),
            "false_negative_rate": float(counts.get("false_negative", 0) / defaults),
        },
        "near_threshold_analysis": {
            "absolute_probability_window": 0.05,
            "sample_size": near_count,
            "error_count": near_error_count,
            "error_rate": near_error_rate,
            "false_positive_count": int(
                (near_threshold_errors["error_type"] == "false_positive").sum()
            ),
            "false_negative_count": int(
                (near_threshold_errors["error_type"] == "false_negative").sum()
            ),
        },
        "segment_analysis": {
            "segment_count": len(segment),
            "minimum_segment_size": int(segment["sample_size"].min()),
            "maximum_segment_size": int(segment["sample_size"].max()),
        },
        "observations": [
            describe(highest_error, "error_rate"),
            describe(highest_fpr, "false_positive_rate"),
            describe(highest_fnr, "false_negative_rate"),
            (
                f"Within ±0.05 of the threshold, {near_error_count} of {near_count} cases "
                f"are errors ({near_error_rate:.3f})."
            ),
        ],
        "interpretation": (
            "Rates are descriptive test-set observations under the illustrative review policy. "
            "They do not identify causes, prove demographic effects, or define lending decisions."
        ),
    }


def _robustness_analysis(
    model: Any,
    test: pd.DataFrame,
    baseline_metrics: dict[str, Any],
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    X = test[MODEL_FEATURES].copy()
    missing = X.copy()
    missing_mask = generator.random(missing.shape) < 0.05
    missing = missing.mask(missing_mask)
    missing_probabilities = model.predict_proba(missing)[:, 1]
    missing_metrics = calculate_metrics(test[TARGET].to_numpy(), missing_probabilities, threshold)

    perturbed = apply_robustness_perturbation(X)
    perturbed_probabilities = model.predict_proba(perturbed)[:, 1]
    valid_input_accepted = True
    try:
        validate_inference_features(X.head(1), MODEL_FEATURES)
    except ValueError:
        valid_input_accepted = False

    invalid_status_rejected = False
    invalid_status = X.head(1).copy()
    invalid_status.loc[:, "PAY_0"] = 10
    try:
        validate_inference_features(invalid_status, MODEL_FEATURES)
    except ValueError as error:
        invalid_status_rejected = "Repayment status" in str(error)

    negative_limit_rejected = False
    negative_limit = X.head(1).copy()
    negative_limit.loc[:, "LIMIT_BAL"] = -1
    try:
        validate_inference_features(negative_limit, MODEL_FEATURES)
    except ValueError as error:
        negative_limit_rejected = "LIMIT_BAL" in str(error)

    return {
        "missingness_stress_test": {
            "injected_cell_fraction": 0.05,
            "average_precision": missing_metrics["average_precision"],
            "average_precision_change": missing_metrics["average_precision"]
            - baseline_metrics["average_precision"],
            "brier_score": missing_metrics["brier_score"],
            "brier_score_change": missing_metrics["brier_score"] - baseline_metrics["brier_score"],
        },
        "plausible_perturbation": {
            "assumption": (
                "LIMIT_BAL +10%; payment amounts -5%; all dependent engineered features rebuilt"
            ),
            "mean_absolute_probability_change": float(
                np.mean(
                    np.abs(
                        perturbed_probabilities - model.predict_proba(test[MODEL_FEATURES])[:, 1]
                    )
                )
            ),
        },
        "invalid_input_rejected": bool(
            valid_input_accepted and invalid_status_rejected and negative_limit_rejected
        ),
        "input_validation_checks": {
            "valid_complete_row_accepted": valid_input_accepted,
            "repayment_status_above_9_rejected": invalid_status_rejected,
            "negative_limit_rejected": negative_limit_rejected,
        },
        "scope_warning": "Sensitivity checks are not adversarial-ML evidence.",
    }


def _explain(
    model: Any,
    base_model: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    paths: ProjectPaths,
    seed: int,
    fast: bool,
    final_variant: str,
) -> None:
    sample_size = min(300 if not fast else 40, len(test))
    importance_size = min(2_000 if not fast else 100, len(test))
    importance_sample = test.sample(importance_size, random_state=seed)
    permutation = permutation_importance_table(
        model,
        importance_sample[MODEL_FEATURES],
        importance_sample[TARGET].to_numpy(),
        repeats=5 if not fast else 1,
        seed=seed,
    )
    permutation.to_csv(paths.metrics / "permutation_importance_test.csv", index=False)
    top = permutation.head(15).sort_values("importance_mean")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Decrease in average precision")
    plt.title("Permutation importance — final test sample")
    _save_figure(paths.figures / "permutation_importance_test.png")

    case_indices = np.array(
        [
            int(np.argmax(probabilities)),
            int(np.argmin(probabilities)),
            int(np.argmin(np.abs(probabilities - threshold))),
        ]
    )
    case_labels = ["high_risk", "low_risk", "near_threshold"]
    explain_sample = test.sample(sample_size, random_state=seed)
    background = train.sample(min(200 if not fast else 30, len(train)), random_state=seed)
    global_shap, _shap_values = shap_attributions(
        base_model,
        background[MODEL_FEATURES],
        explain_sample[MODEL_FEATURES],
    )
    global_shap.to_csv(paths.metrics / "shap_global_test.csv", index=False)
    shap_top = global_shap.head(15).sort_values("mean_abs_shap")
    plt.figure(figsize=(9, 7))
    plt.barh(shap_top["feature"], shap_top["mean_abs_shap"], color="#8f5d3b")
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Global SHAP attribution — selected base estimator")
    _save_figure(paths.figures / "shap_summary_test.png")

    cases = test.iloc[case_indices]
    preprocessor = base_model.named_steps["preprocessor"]
    transformed_cases = preprocessor.transform(cases[MODEL_FEATURES])
    _, local_values = shap_attributions(
        base_model,
        background[MODEL_FEATURES],
        cases[MODEL_FEATURES],
    )
    local = summarize_local_attributions(
        list(preprocessor.get_feature_names_out()),
        np.asarray(transformed_cases),
        cases[MODEL_FEATURES].reset_index(drop=True),
        local_values,
        probabilities[case_indices],
        cases[RECORD_ID].to_numpy(),
        case_labels,
        "sigmoid-calibrated final model"
        if final_variant == "sigmoid_calibrated"
        else "uncalibrated final model",
    )
    write_json(paths.metrics / "local_explanations_test.json", local)


def evaluate_model(fast: bool = False) -> dict[str, Any]:
    """Open the final test set once and create the complete evidence package."""

    config = load_config()
    paths = ProjectPaths()
    paths.ensure_generated_directories()
    metadata = read_json(paths.artifacts / "model_metadata.json")
    model = joblib.load(paths.artifacts / "model.joblib")
    base_model = joblib.load(paths.artifacts / "base_model.joblib")
    train = load_split("train")
    test = load_test_split()
    X_test = test[MODEL_FEATURES]
    y_test = test[TARGET].to_numpy()
    threshold = float(metadata["selected_threshold"])
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = calculate_metrics(y_test, probabilities, threshold)
    iterations = 40 if fast else int(config["training"]["bootstrap_iterations"])
    intervals = bootstrap_intervals(
        y_test,
        probabilities,
        threshold,
        iterations=iterations,
        seed=int(config["project"]["random_seed"]),
    )
    result = {"metrics": metrics, "confidence_intervals": intervals, "test_rows": len(test)}
    write_json(paths.metrics / "test_metrics.json", result)

    prediction_frame = test[[RECORD_ID, TARGET, *AUDIT_COLUMNS]].copy()
    prediction_frame["probability"] = probabilities
    prediction_frame["prediction"] = predictions
    prediction_frame.to_csv(paths.metrics / "test_predictions.csv", index=False)
    _plot_core_curves(y_test, probabilities, predictions, paths)
    _plot_validation_artifacts(paths)

    fairness, gaps = audit_fairness(
        test,
        probabilities,
        threshold,
        attributes=["audit_sex", "audit_age_band"],
    )
    fairness.to_csv(paths.metrics / "fairness_metrics_test.csv", index=False)
    write_json(
        paths.metrics / "fairness_summary.json",
        {
            "gaps": gaps,
            "warning": (
                "Observed benchmark disparities are descriptive. They neither prove fairness nor "
                "establish legal or regulatory acceptability."
            ),
        },
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fairness, x="group", y="true_positive_rate", hue="attribute")
    plt.ylim(0, 1)
    plt.title("Subgroup true-positive rates — final test")
    _save_figure(paths.figures / "fairness_tpr_test.png")

    write_json(
        paths.metrics / "error_analysis.json",
        _error_analysis(test, probabilities, threshold, paths),
    )
    robustness = _robustness_analysis(
        model,
        test,
        metrics,
        threshold,
        seed=int(config["project"]["random_seed"]),
    )
    write_json(paths.metrics / "robustness.json", robustness)

    drift_current = apply_synthetic_drift_shift(test[MODEL_FEATURES])
    drift_features = [
        "LIMIT_BAL",
        "avg_bill_amount",
        "avg_payment_amount",
        "credit_utilization",
        "max_payment_delay",
    ]
    drift = numeric_drift_report(train[MODEL_FEATURES], drift_current, drift_features)
    drift.to_csv(paths.metrics / "drift_monitoring_demo.csv", index=False)
    reference_probabilities = model.predict_proba(train[MODEL_FEATURES])[:, 1]
    current_probabilities = model.predict_proba(drift_current)[:, 1]
    drift_summary = {
        "reference_mean_prediction": float(reference_probabilities.mean()),
        "current_mean_prediction": float(current_probabilities.mean()),
        "mean_prediction_shift": float(
            current_probabilities.mean() - reference_probabilities.mean()
        ),
        "warning": "Monitoring prototype using synthetic perturbations; not evidence of production operation.",
    }
    write_json(paths.metrics / "drift_monitoring_summary.json", drift_summary)
    plt.figure(figsize=(9, 6))
    sns.histplot(reference_probabilities, bins=30, stat="density", alpha=0.45, label="Reference")
    sns.histplot(current_probabilities, bins=30, stat="density", alpha=0.45, label="Perturbed")
    plt.xlabel("Predicted default probability")
    plt.title("Monitoring demonstrator — prediction distribution shift")
    plt.legend()
    _save_figure(paths.figures / "prediction_drift_demo.png")

    _explain(
        model,
        base_model,
        train,
        test,
        probabilities,
        threshold,
        paths,
        seed=int(config["project"]["random_seed"]),
        fast=fast,
        final_variant=str(metadata["final_variant"]),
    )
    return result
