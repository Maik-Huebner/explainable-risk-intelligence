# Executive Evaluation Summary

The final `tuned_hist_gradient_boosting` (sigmoid_calibrated) achieved ROC-AUC
**0.781**, average precision **0.559**, and Brier score
**0.134** on 6,000 untouched test records. At the validation-selected
threshold of **0.17**, recall was **0.729** and precision was
**0.375**.

This is a decision-support prototype, not an automated lending system. The strongest evidence is the
separation of train/validation/test responsibilities, calibrated probability evaluation, confidence
intervals, labelled local and global explanations, normalized subgroup error analysis, explicit error-cost
assumptions, and an internally consistent synthetic monitoring demonstration. The largest limitations are
the historical 2005 population, random rather than temporal validation, and absence of real deployment
outcomes.
