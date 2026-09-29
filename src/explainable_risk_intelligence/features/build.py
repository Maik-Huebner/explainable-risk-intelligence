"""Feature engineering and audit-attribute separation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from explainable_risk_intelligence.data.schema import (
    BILL_COLUMNS,
    PAY_STATUS_COLUMNS,
    PAYMENT_COLUMNS,
)

TARGET = "default_next_month"
RECORD_ID = "record_id"
AUDIT_COLUMNS = ["audit_sex", "audit_age", "audit_age_band", "audit_education", "audit_marriage"]
BASE_MODEL_FEATURES = ["LIMIT_BAL", *PAY_STATUS_COLUMNS, *BILL_COLUMNS, *PAYMENT_COLUMNS]
ENGINEERED_FEATURES = [
    "avg_bill_amount",
    "avg_payment_amount",
    "credit_utilization",
    "payment_to_bill_ratio",
    "max_payment_delay",
    "delayed_month_count",
    "bill_amount_trend",
    "payment_amount_trend",
]
MODEL_FEATURES = [*BASE_MODEL_FEATURES, *ENGINEERED_FEATURES]


def recompute_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild every derived feature from the canonical base model fields."""

    missing = sorted(set(BASE_MODEL_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing base features for feature engineering: {missing}")
    result = frame.copy()
    average_bill = result[BILL_COLUMNS].mean(axis=1)
    average_payment = result[PAYMENT_COLUMNS].mean(axis=1)
    result["avg_bill_amount"] = average_bill
    result["avg_payment_amount"] = average_payment
    result["credit_utilization"] = average_bill / result["LIMIT_BAL"].clip(lower=1)
    result["payment_to_bill_ratio"] = average_payment / average_bill.abs().clip(lower=1)
    result["max_payment_delay"] = result[PAY_STATUS_COLUMNS].max(axis=1)
    result["delayed_month_count"] = (result[PAY_STATUS_COLUMNS] > 0).sum(axis=1)
    result["bill_amount_trend"] = result["BILL_AMT1"] - result["BILL_AMT6"]
    result["payment_amount_trend"] = result["PAY_AMT1"] - result["PAY_AMT6"]
    return result.replace([np.inf, -np.inf], np.nan)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build domain-motivated features while retaining audit attributes separately."""

    frame = raw.copy()
    frame = frame.rename(columns={"ID": RECORD_ID, "default payment next month": TARGET})
    frame["audit_sex"] = frame["SEX"].map({1: "male", 2: "female"}).fillna("unknown")
    frame["audit_age"] = frame["AGE"].astype(int)
    frame["audit_age_band"] = pd.cut(
        frame["AGE"],
        bins=[0, 29, 39, 49, 120],
        labels=["under_30", "30_to_39", "40_to_49", "50_plus"],
        include_lowest=True,
    ).astype("string")
    frame["audit_education"] = frame["EDUCATION"].astype(str)
    frame["audit_marriage"] = frame["MARRIAGE"].astype(str)
    frame = recompute_engineered_features(frame)

    return frame[[RECORD_ID, TARGET, *AUDIT_COLUMNS, *MODEL_FEATURES]].copy()
