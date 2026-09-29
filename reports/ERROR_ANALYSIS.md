# Error Analysis

This analysis describes errors on the locked final test set at the validation-selected threshold. It
does not identify causes or establish demographic effects.

## Overall profile

- Sample size: 6,000
- Overall error rate: 0.328
- False-positive rate among non-defaults: 0.345
- False-negative rate among defaults: 0.271
- Counts: {'correct': 4031, 'false_negative': 359, 'false_positive': 1610}

## Threshold-near cases

Within ±0.05 of the illustrative threshold, the sample contains
1,913 records and 799 errors
(0.418). These ambiguous cases are useful for manual review of policy trade-offs;
they do not define an automated credit decision.

## Reproducible observations

- male / under_30 has the highest observed error rate (0.363; n=669).
- male / under_30 has the highest observed false positive rate (0.408; 212 / 519 non-default cases).
- female / 40_to_49 has the highest observed false negative rate (0.300; 54 / 180 default cases).
- Within ±0.05 of the threshold, 799 of 1913 cases are errors (0.418).

Every segment rate is accompanied by its sample size in
`reports/metrics/error_segments_test.csv`. The chart `reports/figures/error_rates_by_segment_test.png`
shows overall, false-positive, and false-negative rates separately. Differences are descriptive and
must not be interpreted as causal, inherently demographic, fair, or legally acceptable.
