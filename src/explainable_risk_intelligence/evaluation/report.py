"""Generate human-readable evidence from machine-readable artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from explainable_risk_intelligence.config import ProjectPaths, load_config
from explainable_risk_intelligence.utils.io import read_json


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


def _results_markdown(
    metadata: dict[str, Any],
    test: dict[str, Any],
    calibration: pd.DataFrame,
) -> str:
    metrics = test["metrics"]
    intervals = test["confidence_intervals"]
    rows = []
    for key, label in (
        ("roc_auc", "ROC-AUC"),
        ("average_precision", "Average precision / PR-AUC"),
        ("brier_score", "Brier score"),
        ("log_loss", "Log loss"),
        ("precision", "Precision at policy threshold"),
        ("recall", "Recall at policy threshold"),
        ("f1", "F1 at policy threshold"),
    ):
        interval = intervals.get(key)
        interval_text = (
            f"{_format_metric(interval['lower_95'])} to {_format_metric(interval['upper_95'])}"
            if interval
            else "—"
        )
        rows.append(f"| {label} | {_format_metric(metrics[key])} | {interval_text} |")
    calibrated_row = calibration.loc[calibration["model"] == metadata["final_variant"]].iloc[0]
    return "\n".join(
        [
            f"Selected estimator: **{metadata['selected_model']}** ({metadata['final_variant']}).",
            "",
            "| Final test metric | Value | Bootstrap 95% CI |",
            "|---|---:|---:|",
            *rows,
            "",
            f"The validation-selected illustrative review threshold is **{metrics['threshold']:.2f}** "
            f"(validation calibration Brier score: {calibrated_row['brier_score']:.3f}). "
            "The final test was not used for model, calibration, or threshold selection.",
        ]
    )


def _replace_generated_section(readme: Path, generated: str) -> None:
    start = "<!-- GENERATED_RESULTS_START -->"
    end = "<!-- GENERATED_RESULTS_END -->"
    content = readme.read_text(encoding="utf-8")
    if start not in content or end not in content:
        raise ValueError("README generated-result markers are missing")
    prefix, remainder = content.split(start, maxsplit=1)
    _, suffix = remainder.split(end, maxsplit=1)
    readme.write_text(
        f"{prefix}{start}\n{generated}\n{end}{suffix}",
        encoding="utf-8",
    )


def generate_reports() -> None:
    """Generate the executive summary, model card, and README results."""

    config = load_config()
    paths = ProjectPaths()
    metadata = read_json(paths.metrics / "model_metadata.json")
    test = read_json(paths.metrics / "test_metrics.json")
    fairness = read_json(paths.metrics / "fairness_summary.json")
    robustness = read_json(paths.metrics / "robustness.json")
    drift = read_json(paths.metrics / "drift_monitoring_summary.json")
    errors = read_json(paths.metrics / "error_analysis.json")
    calibration = pd.read_csv(paths.metrics / "calibration_comparison_validation.csv")
    generated_results = _results_markdown(metadata, test, calibration)
    _replace_generated_section(paths.root / "README.md", generated_results)

    metrics = test["metrics"]
    model_card = f"""# Model Card: Explainable Risk Intelligence

## Model details

- **Version:** 1.0.0
- **Model:** `{metadata["selected_model"]}`; final variant `{metadata["final_variant"]}`
- **Purpose:** Portfolio and research implementation of calibrated probability-of-default decision support.
- **Owner:** Maik Hübner

## Intended use

The model estimates next-month default probability for retrospective analysis, model-risk discussion,
and human decision-support demonstrations. It is intended to demonstrate leakage-safe applied machine
learning, probability calibration, explainability, subgroup auditing, and monitoring design.

## Out-of-scope use

It must not autonomously approve, decline, price, or otherwise determine credit. It is not validated
for a current population, another jurisdiction, protected-class decisioning, regulatory reporting, or
production operation. It is not financial or legal advice.

## Dataset

UCI *Default of Credit Card Clients* (30,000 Taiwanese credit-card clients; April-September 2005),
DOI [{config["dataset"]["doi"]}]({config["dataset"]["doi"]}), licensed CC BY 4.0. The official archive
is verified with SHA-256 `{config["dataset"]["archive_sha256"]}`. The target is next-month default.

## Features and preprocessing

ID, sex, age, education, and marital status are excluded from model inputs. Demographic fields are
retained only for descriptive audit slices. Payment history, bill amounts, payment amounts, credit
limit, and documented aggregate features are median-imputed and scaled inside the fitted pipeline.

## Training and selection

The data uses a reproducible stratified 60/20/20 train/validation/test split because no suitable event
timestamp is available. Cross-validation and hyperparameter tuning use training data only. Model,
calibration, and threshold selection use validation data. The training path cannot load the test split;
final evaluation opens it only after the development choices and artifacts have been persisted.

## Final evaluation

- ROC-AUC: {metrics["roc_auc"]:.4f}
- Average precision: {metrics["average_precision"]:.4f}
- Brier score: {metrics["brier_score"]:.4f}
- Log loss: {metrics["log_loss"]:.4f}
- Precision / recall / F1 at threshold {metrics["threshold"]:.2f}: {metrics["precision"]:.4f} / {metrics["recall"]:.4f} / {metrics["f1"]:.4f}

Bootstrap percentile intervals are stored in `reports/metrics/test_metrics.json`.

## Calibration and threshold policy

Sigmoid calibration is retained only when it improves validation Brier score without a material
average-precision loss. The observed calibration gain is positive but marginal and should not be
over-interpreted. The review threshold uses validation data and illustrative costs of 1 for a false
positive and 5 for a false negative. These assumptions demonstrate trade-offs; they do not encode a
real lender's economics or policy.

## Explainability

Global permutation importance and SHAP attributions are produced, plus local examples for high-, low-,
and near-threshold cases. Local records explicitly distinguish original from transformed feature values.
SHAP describes model attribution, not causality. When calibration is selected, SHAP explains the
underlying base estimator while displayed probabilities come from the calibrated model.

## Fairness and responsible AI

Sex and age bands are audited using selection rate, TPR, FPR, precision, demographic-parity difference,
equal-opportunity difference, and equalized-odds difference. Observed gaps are descriptive benchmark
evidence, not proof of fairness or legal acceptability. Summary: `{fairness["gaps"]}`.

## Error analysis and robustness

The test error rate is `{errors["rates"]["overall_error_rate"]:.4f}` across
`{errors["sample_size"]}` records; false-positive and false-negative rates are reported separately with
segment sample sizes in `reports/metrics/error_segments_test.csv`. These observations are descriptive,
not causal demographic findings. Missing-value injection, internally consistent feature perturbation,
input-range rejection, and bootstrap stability are demonstrated. Missingness AP change:
`{robustness["missingness_stress_test"]["average_precision_change"]:.4f}`.

## Monitoring

PSI, KS statistics, feature-distribution changes, and prediction-distribution changes are demonstrated
with deliberately shifted base fields whose dependent engineered features are rebuilt before scoring.
{drift["warning"]}

## Limitations and ethical considerations

The benchmark is historical, geographically specific, and lacks a timestamp appropriate for temporal
validation. Demographic labels are coarse and context-limited. Calibration and group metrics may drift.
Real deployment would require current representative data, legal review, adverse-action governance,
human oversight, security controls, ongoing validation, and institution-specific cost analysis.

## Reproducibility

Run `make install`, then `make reproduce`. Publishable machine-readable evidence is under
`reports/metrics/`; fitted models and their runtime metadata remain Git-ignored under `artifacts/`.
"""
    (paths.model_card / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")

    executive = f"""# Executive Evaluation Summary

The final `{metadata["selected_model"]}` ({metadata["final_variant"]}) achieved ROC-AUC
**{metrics["roc_auc"]:.3f}**, average precision **{metrics["average_precision"]:.3f}**, and Brier score
**{metrics["brier_score"]:.3f}** on {test["test_rows"]:,} untouched test records. At the validation-selected
threshold of **{metrics["threshold"]:.2f}**, recall was **{metrics["recall"]:.3f}** and precision was
**{metrics["precision"]:.3f}**.

This is a decision-support prototype, not an automated lending system. The strongest evidence is the
separation of train/validation/test responsibilities, calibrated probability evaluation, confidence
intervals, labelled local and global explanations, normalized subgroup error analysis, explicit error-cost
assumptions, and an internally consistent synthetic monitoring demonstration. The largest limitations are
the historical 2005 population, random rather than temporal validation, and absence of real deployment
outcomes.
"""
    (paths.root / "reports" / "EXECUTIVE_SUMMARY.md").write_text(executive, encoding="utf-8")

    observations = "\n".join(f"- {observation}" for observation in errors["observations"])
    near = errors["near_threshold_analysis"]
    error_analysis = f"""# Error Analysis

This analysis describes errors on the locked final test set at the validation-selected threshold. It
does not identify causes or establish demographic effects.

## Overall profile

- Sample size: {errors["sample_size"]:,}
- Overall error rate: {errors["rates"]["overall_error_rate"]:.3f}
- False-positive rate among non-defaults: {errors["rates"]["false_positive_rate"]:.3f}
- False-negative rate among defaults: {errors["rates"]["false_negative_rate"]:.3f}
- Counts: {errors["counts"]}

## Threshold-near cases

Within ±{near["absolute_probability_window"]:.2f} of the illustrative threshold, the sample contains
{near["sample_size"]:,} records and {near["error_count"]:,} errors
({near["error_rate"]:.3f}). These ambiguous cases are useful for manual review of policy trade-offs;
they do not define an automated credit decision.

## Reproducible observations

{observations}

Every segment rate is accompanied by its sample size in
`reports/metrics/error_segments_test.csv`. The chart `reports/figures/error_rates_by_segment_test.png`
shows overall, false-positive, and false-negative rates separately. Differences are descriptive and
must not be interpreted as causal, inherently demographic, fair, or legally acceptable.
"""
    (paths.root / "reports" / "ERROR_ANALYSIS.md").write_text(error_analysis, encoding="utf-8")
