# Methodology

## Evaluation contract

The 30,000 records are split once with target stratification and seed 42: 60% training, 20% validation,
and 20% final test. The split is random because the source does not provide a defensible application/event
timestamp. Consequently, results measure in-distribution discrimination and calibration, not temporal
generalization.

Training data owns preprocessing fits, cross-validation, and hyperparameter search. Validation data owns
model choice, the calibrated-versus-uncalibrated decision, and the illustrative threshold policy. The test
split has a separate loader and is not opened by the training path. Final evaluation loads it only after
the selected model, calibration decision, and threshold have been persisted.

## Features and preprocessing

The model pipeline median-imputes and scales every numeric field. These steps live inside the estimator
pipeline, so each cross-validation fold learns its own statistics. Aggregate features summarize average
bills/payments, utilization, payment-to-bill ratio, repayment delay, and recent amount changes.

## Model selection

Dummy-prior, class-weighted logistic regression, random forest, and histogram gradient boosting form a
deliberately compact comparison. Histogram gradient boosting receives a reproducible randomized search.
Average precision is the selection metric because defaults are the minority class. The model comparison
also records ROC-AUC, threshold metrics, Brier score, and log loss.

The imbalance strategy combines stratified splitting and cross-validation, class-weighted logistic and
random-forest baselines, precision-recall evaluation, and validation-selected thresholding. Synthetic
oversampling is intentionally omitted to retain the observed prevalence for probability calibration; no
claim is made that methods such as SMOTE would never help on another dataset or objective.

## Calibration and policy

The selected base pipeline is calibrated through stratified training-fold cross-validation using sigmoid
calibration. The calibrated variant is retained only when validation Brier score improves and average
precision does not decline by more than 0.01. In this benchmark the improvement is positive but marginal,
so calibration is retained without claiming a large practical gain.

The validation threshold minimizes an illustrative cost function with false-negative cost 5 and
false-positive cost 1. This makes assumptions inspectable; it is not a real lending policy.

## Final evidence

The final test reports ROC-AUC, average precision, accuracy, precision, recall, F1, confusion matrix,
Brier score, and log loss. Percentile bootstrap intervals quantify sampling uncertainty for central
metrics. Explainability, subgroup audit, error review, missingness/perturbation checks, and monitoring
demonstrations are generated from the same locked model and configuration.

Training writes one canonical model-metadata object to two locations with distinct roles. The ignored
`artifacts/model_metadata.json` accompanies fitted runtime objects, while
`reports/metrics/model_metadata.json` is the publishable evidence copy used by the curated notebook and
reviewer-facing helpers. Consequently, notebook inspection does not require generated data or models.

Robustness and drift scenarios modify base financial fields and then invoke the same canonical feature
engineering function used during data preparation. Derived ratios, averages, trends, and delay summaries
therefore remain internally consistent with the perturbed base values.
