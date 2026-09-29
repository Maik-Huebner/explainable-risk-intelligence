from __future__ import annotations

from explainable_risk_intelligence.config import ProjectPaths, load_config
from explainable_risk_intelligence.utils.io import read_json


def test_readme_metrics_match_machine_readable_test_results() -> None:
    paths = ProjectPaths()
    results = read_json(paths.metrics / "test_metrics.json")["metrics"]
    readme = (paths.root / "README.md").read_text(encoding="utf-8")
    expected = {
        "ROC-AUC": results["roc_auc"],
        "Average precision / PR-AUC": results["average_precision"],
        "Brier score": results["brier_score"],
        "Log loss": results["log_loss"],
        "Precision at policy threshold": results["precision"],
        "Recall at policy threshold": results["recall"],
        "F1 at policy threshold": results["f1"],
    }
    for label, value in expected.items():
        assert f"| {label} | {value:.3f} |" in readme
    assert f"threshold is **{results['threshold']:.2f}**" in readme


def test_model_card_metrics_match_machine_readable_test_results() -> None:
    paths = ProjectPaths()
    results = read_json(paths.metrics / "test_metrics.json")["metrics"]
    model_card = (paths.model_card / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert f"ROC-AUC: {results['roc_auc']:.4f}" in model_card
    assert f"Average precision: {results['average_precision']:.4f}" in model_card
    assert f"Brier score: {results['brier_score']:.4f}" in model_card
    assert f"Log loss: {results['log_loss']:.4f}" in model_card
    assert "positive but marginal" in model_card


def test_documented_dataset_hash_matches_configuration() -> None:
    paths = ProjectPaths()
    digest = load_config()["dataset"]["archive_sha256"]
    readme = (paths.root / "README.md").read_text(encoding="utf-8")
    assert digest in readme
    provenance = paths.raw / "provenance.json"
    if provenance.exists():
        assert digest in provenance.read_text(encoding="utf-8")
