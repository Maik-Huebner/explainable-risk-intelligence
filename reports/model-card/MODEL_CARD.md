# Model Card: Explainable Risk Intelligence

## Model details

- **Version:** 1.0.0
- **Model:** `tuned_hist_gradient_boosting`; final variant `sigmoid_calibrated`
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
DOI [https://doi.org/10.24432/C55S3H](https://doi.org/10.24432/C55S3H), licensed CC BY 4.0. The official archive
is verified with SHA-256 `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`. The target is next-month default.

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

- ROC-AUC: 0.7810
- Average precision: 0.5593
- Brier score: 0.1345
- Log loss: 0.4283
- Precision / recall / F1 at threshold 0.17: 0.3755 / 0.7295 / 0.4958

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
evidence, not proof of fairness or legal acceptability. Summary: `{'audit_age_band': {'demographic_parity_difference': 0.06720925451437404, 'equal_opportunity_difference': 0.045765027322404284, 'equalized_odds_difference': 0.06656054335137429}, 'audit_sex': {'demographic_parity_difference': 0.026340380607414893, 'equal_opportunity_difference': 0.03634641608839129, 'equalized_odds_difference': 0.03634641608839129}}`.

## Error analysis and robustness

The test error rate is `0.3282` across
`6000` records; false-positive and false-negative rates are reported separately with
segment sample sizes in `reports/metrics/error_segments_test.csv`. These observations are descriptive,
not causal demographic findings. Missing-value injection, internally consistent feature perturbation,
input-range rejection, and bootstrap stability are demonstrated. Missingness AP change:
`-0.0152`.

## Monitoring

PSI, KS statistics, feature-distribution changes, and prediction-distribution changes are demonstrated
with deliberately shifted base fields whose dependent engineered features are rebuilt before scoring.
Monitoring prototype using synthetic perturbations; not evidence of production operation.

## Limitations and ethical considerations

The benchmark is historical, geographically specific, and lacks a timestamp appropriate for temporal
validation. Demographic labels are coarse and context-limited. Calibration and group metrics may drift.
Real deployment would require current representative data, legal review, adverse-action governance,
human oversight, security controls, ongoing validation, and institution-specific cost analysis.

## Reproducibility

Run `make install`, then `make reproduce`. Publishable machine-readable evidence is under
`reports/metrics/`; fitted models and their runtime metadata remain Git-ignored under `artifacts/`.
