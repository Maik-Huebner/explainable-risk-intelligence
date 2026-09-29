# Responsible Use

Explainable Risk Intelligence is a decision-support prototype and portfolio implementation. It is not an
automated lending system, a credit score, legal advice, or evidence of regulatory compliance.

## Prohibited interpretations

- A high probability must not automatically trigger credit denial, pricing, or a lower limit.
- SHAP values must not be interpreted as causal reasons for default.
- A small subgroup gap must not be presented as proof that the system is fair.
- Historical benchmark performance must not be extrapolated to current customers or other markets.
- The monitoring demonstration must not be described as production history.

## Required controls before any real use

Representative current data, lawful-purpose review, adverse-action governance, human oversight, security,
access controls, independent model validation, calibrated institution-specific thresholds, documented
appeal paths, ongoing drift/performance/fairness monitoring, and formal change management would all be
required. This repository implements none of those organizational approvals.

