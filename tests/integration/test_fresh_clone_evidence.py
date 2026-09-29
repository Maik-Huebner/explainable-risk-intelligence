from __future__ import annotations

import shutil
from pathlib import Path

from explainable_risk_intelligence.config import ProjectPaths
from explainable_risk_intelligence.notebook_runner import execute_notebook
from explainable_risk_intelligence.portfolio import load_portfolio_evidence


def test_curated_notebook_uses_only_published_evidence(tmp_path: Path, capsys: object) -> None:
    source = ProjectPaths().root
    shutil.copytree(source / "reports", tmp_path / "reports")
    shutil.copytree(source / "notebooks", tmp_path / "notebooks")
    shutil.copytree(source / "src", tmp_path / "src")
    shutil.copy2(source / "pyproject.toml", tmp_path / "pyproject.toml")

    runtime_metadata = tmp_path / "artifacts" / "model_metadata.json"
    published_metadata = tmp_path / "reports" / "metrics" / "model_metadata.json"
    assert not runtime_metadata.exists()
    assert published_metadata.is_file()

    evidence = load_portfolio_evidence(root=tmp_path)
    assert evidence["metadata"]["selected_model"]
    assert execute_notebook(tmp_path / "notebooks" / "01_portfolio_analysis.ipynb") == 0
    assert "Notebook execution passed" in capsys.readouterr().out  # type: ignore[attr-defined]
