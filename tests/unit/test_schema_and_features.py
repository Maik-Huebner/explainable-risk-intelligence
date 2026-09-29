from __future__ import annotations

import pandas as pd
import pytest

from explainable_risk_intelligence.data.schema import (
    validate_inference_features,
    validate_raw_schema,
)
from explainable_risk_intelligence.features.build import (
    AUDIT_COLUMNS,
    MODEL_FEATURES,
    RECORD_ID,
    TARGET,
    build_features,
)


def test_valid_raw_schema(raw_frame: pd.DataFrame) -> None:
    validate_raw_schema(raw_frame, expected_rows=len(raw_frame))


def test_duplicate_id_is_rejected(raw_frame: pd.DataFrame) -> None:
    raw_frame.loc[1, "ID"] = raw_frame.loc[0, "ID"]
    with pytest.raises(ValueError, match="unique"):
        validate_raw_schema(raw_frame, expected_rows=len(raw_frame))


def test_wrong_target_is_rejected(raw_frame: pd.DataFrame) -> None:
    raw_frame.loc[:, "default payment next month"] = 0
    with pytest.raises(ValueError, match="binary"):
        validate_raw_schema(raw_frame, expected_rows=len(raw_frame))


def test_feature_contract_excludes_identifier_and_demographics(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame)
    assert prepared.columns.tolist() == [RECORD_ID, TARGET, *AUDIT_COLUMNS, *MODEL_FEATURES]
    assert "SEX" not in MODEL_FEATURES
    assert "AGE" not in MODEL_FEATURES
    assert RECORD_ID not in MODEL_FEATURES
    assert prepared[MODEL_FEATURES].shape[1] == len(MODEL_FEATURES)


def test_feature_engineering_has_finite_ratios(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame)
    assert prepared["credit_utilization"].notna().all()
    assert prepared["payment_to_bill_ratio"].notna().all()


def test_invalid_inference_limit_is_rejected(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame).head(1)
    prepared.loc[:, "LIMIT_BAL"] = -1
    with pytest.raises(ValueError, match="positive"):
        validate_inference_features(prepared, MODEL_FEATURES)


def test_missing_inference_feature_is_rejected(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame).drop(columns=[MODEL_FEATURES[0]])
    with pytest.raises(ValueError, match="Missing model features"):
        validate_inference_features(prepared, MODEL_FEATURES)


def test_complete_plausible_inference_row_is_accepted(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame).head(1)
    validate_inference_features(prepared, MODEL_FEATURES)


def test_payment_amount_above_status_range_is_accepted(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame).head(1)
    prepared.loc[:, "PAY_AMT2"] = 1_000
    validate_inference_features(prepared, MODEL_FEATURES)


def test_repayment_status_outside_range_is_rejected(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame).head(1)
    prepared.loc[:, "PAY_2"] = 10
    with pytest.raises(ValueError, match="Repayment status"):
        validate_inference_features(prepared, MODEL_FEATURES)


def test_negative_payment_amount_is_rejected(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame).head(1)
    prepared.loc[:, "PAY_AMT2"] = -1
    with pytest.raises(ValueError, match="Payment amounts"):
        validate_inference_features(prepared, MODEL_FEATURES)
