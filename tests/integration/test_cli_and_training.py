from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from explainable_risk_intelligence.cli import main, smoke_test
from explainable_risk_intelligence.evaluation.metrics import calculate_metrics
from explainable_risk_intelligence.explainability.analysis import shap_attributions
from explainable_risk_intelligence.features.build import MODEL_FEATURES, TARGET, build_features
from explainable_risk_intelligence.models.pipeline import make_model_pipeline


def test_cli_smoke_command(capsys: object) -> None:
    assert main(["smoke-test"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "passed" in output


def test_smoke_function_is_repeatable() -> None:
    smoke_test()
    smoke_test()


def test_small_nonlinear_training_roundtrip(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame)
    train = prepared.iloc[:300]
    test = prepared.iloc[300:]
    model = make_model_pipeline(
        HistGradientBoostingClassifier(max_iter=20, max_leaf_nodes=7, random_state=42)
    )
    model.fit(train[MODEL_FEATURES], train[TARGET])
    probabilities = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    metrics = calculate_metrics(test[TARGET].to_numpy(), probabilities)
    assert len(probabilities) == 100
    assert np.isfinite(probabilities).all()
    assert 0 <= metrics["average_precision"] <= 1


def test_hist_gradient_boosting_shap_roundtrip(raw_frame: pd.DataFrame) -> None:
    prepared = build_features(raw_frame)
    model = make_model_pipeline(
        HistGradientBoostingClassifier(max_iter=10, max_leaf_nodes=7, random_state=42)
    )
    model.fit(prepared[MODEL_FEATURES].iloc[:300], prepared[TARGET].iloc[:300])
    table, values = shap_attributions(
        model,
        prepared[MODEL_FEATURES].iloc[:20],
        prepared[MODEL_FEATURES].iloc[300:305],
    )
    assert len(table) == len(MODEL_FEATURES)
    assert values.shape == (5, len(MODEL_FEATURES))
    assert np.isfinite(values).all()
