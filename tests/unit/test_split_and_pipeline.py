from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from explainable_risk_intelligence.data.prepare import load_train_validation_splits, split_frame
from explainable_risk_intelligence.features.build import (
    MODEL_FEATURES,
    RECORD_ID,
    TARGET,
    build_features,
)
from explainable_risk_intelligence.models.pipeline import make_model_pipeline


def test_split_is_reproducible_and_disjoint(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame)
    first = split_frame(prepared, random_seed=42)
    second = split_frame(prepared, random_seed=42)
    for name in first:
        assert first[name][RECORD_ID].tolist() == second[name][RECORD_ID].tolist()
    ids = {name: set(frame[RECORD_ID]) for name, frame in first.items()}
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])


def test_split_sizes_and_stratification(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame)
    splits = split_frame(prepared, random_seed=42)
    assert {name: len(frame) for name, frame in splits.items()} == {
        "train": 240,
        "validation": 80,
        "test": 80,
    }
    overall = prepared[TARGET].mean()
    assert all(abs(frame[TARGET].mean() - overall) < 0.02 for frame in splits.values())


def test_pipeline_handles_missing_values_and_probability_bounds(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame)
    X = prepared[MODEL_FEATURES].copy()
    X.loc[0:10, "BILL_AMT1"] = np.nan
    model = make_model_pipeline(LogisticRegression(max_iter=500))
    model.fit(X.iloc[:300], prepared[TARGET].iloc[:300])
    probabilities = model.predict_proba(X.iloc[300:])[:, 1]
    assert probabilities.shape == (100,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_preprocessor_is_fitted_only_during_fit(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame)
    model = make_model_pipeline(LogisticRegression(max_iter=500))
    assert not hasattr(model.named_steps["preprocessor"], "transformers_")
    model.fit(prepared[MODEL_FEATURES], prepared[TARGET])
    assert hasattr(model.named_steps["preprocessor"], "transformers_")


def test_training_loader_never_opens_test_split(monkeypatch: object) -> None:
    opened: list[str] = []

    def fake_read_csv(path: object) -> pd.DataFrame:
        opened.append(str(path))
        return pd.DataFrame()

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)  # type: ignore[attr-defined]
    load_train_validation_splits()
    assert len(opened) == 2
    assert opened[0].endswith("train.csv")
    assert opened[1].endswith("validation.csv")
    assert all(not path.endswith("test.csv") for path in opened)
