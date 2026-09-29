"""Dataset and inference schema validation."""

from __future__ import annotations

import pandas as pd

TARGET_SOURCE = "default payment next month"
PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    TARGET_SOURCE,
]


def validate_raw_schema(frame: pd.DataFrame, expected_rows: int = 30_000) -> None:
    """Fail loudly when the source shape or semantics change."""

    if frame.columns.tolist() != RAW_COLUMNS:
        raise ValueError(f"Unexpected columns: {frame.columns.tolist()}")
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(frame)}")
    if frame["ID"].isna().any() or not frame["ID"].is_unique:
        raise ValueError("ID must be complete and unique")
    if frame.isna().any().any():
        raise ValueError("Official source is expected to contain no missing values")
    if set(frame[TARGET_SOURCE].unique()) != {0, 1}:
        raise ValueError("Target must be binary with values 0 and 1")
    if (frame["LIMIT_BAL"] <= 0).any():
        raise ValueError("LIMIT_BAL must be positive")
    if not frame["AGE"].between(18, 100).all():
        raise ValueError("AGE is outside the documented plausible range")


def validate_inference_features(frame: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate model input without silently accepting impossible values."""

    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    if "LIMIT_BAL" in frame and (frame["LIMIT_BAL"].dropna() <= 0).any():
        raise ValueError("LIMIT_BAL must be positive")
    non_numeric = [
        column
        for column in required_columns
        if column in frame and not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(f"Model features must be numeric: {non_numeric}")
    infinite = [
        column
        for column in required_columns
        if column in frame and frame[column].dropna().isin([float("inf"), float("-inf")]).any()
    ]
    if infinite:
        raise ValueError(f"Model features must be finite when present: {infinite}")
    present_status_columns = [column for column in PAY_STATUS_COLUMNS if column in frame]
    if present_status_columns:
        valid = frame[present_status_columns].stack().between(-2, 9).all()
        if not valid:
            raise ValueError("Repayment status must be between -2 and 9")
    present_payment_columns = [column for column in PAYMENT_COLUMNS if column in frame]
    if present_payment_columns and (frame[present_payment_columns].stack() < 0).any():
        raise ValueError("Payment amounts must be non-negative")
    if "delayed_month_count" in frame:
        delayed = frame["delayed_month_count"].dropna()
        if not delayed.between(0, len(PAY_STATUS_COLUMNS)).all():
            raise ValueError("delayed_month_count must be between 0 and 6")
