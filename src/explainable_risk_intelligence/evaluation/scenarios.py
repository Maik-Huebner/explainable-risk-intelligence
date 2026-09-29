"""Consistent feature scenarios used by robustness and drift evaluation."""

from __future__ import annotations

import pandas as pd

from explainable_risk_intelligence.data.schema import BILL_COLUMNS, PAYMENT_COLUMNS
from explainable_risk_intelligence.features.build import recompute_engineered_features


def apply_robustness_perturbation(frame: pd.DataFrame) -> pd.DataFrame:
    """Increase credit limits and reduce payments, then rebuild derived features."""

    perturbed = frame.copy()
    perturbed["LIMIT_BAL"] *= 1.10
    perturbed[PAYMENT_COLUMNS] *= 0.95
    return recompute_engineered_features(perturbed)


def apply_synthetic_drift_shift(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply a documented synthetic portfolio shift and rebuild derived features."""

    shifted = frame.copy()
    shifted["LIMIT_BAL"] *= 0.80
    shifted[BILL_COLUMNS] *= 1.15
    shifted[PAYMENT_COLUMNS] *= 0.85
    return recompute_engineered_features(shifted)
