from __future__ import annotations

import json
from pathlib import Path

from explainable_risk_intelligence.repository_audit import audit_repository, main


def test_audit_scans_env_citation_notebook_and_public_json(tmp_path: Path) -> None:
    openai_token = "sk-" + "a" * 40
    github_token = "gh" + "p_" + "b" * 40
    private_key_marker = "-----BEGIN " + "PRIVATE" + " KEY-----"

    (tmp_path / ".env.example").write_text(f"API_KEY={openai_token}\n", encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text(private_key_marker, encoding="utf-8")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "demo.ipynb").write_text(
        json.dumps({"cells": [{"source": [github_token]}]}), encoding="utf-8"
    )
    (tmp_path / "reports" / "metrics").mkdir(parents=True)
    (tmp_path / "reports" / "metrics" / "evidence.json").write_text(
        json.dumps({"accidental_token": openai_token}), encoding="utf-8"
    )

    findings = audit_repository(tmp_path)
    assert any(".env.example" in finding for finding in findings)
    assert any("CITATION.cff" in finding for finding in findings)
    assert any("demo.ipynb" in finding for finding in findings)
    assert any("evidence.json" in finding for finding in findings)
    assert main(tmp_path) == 1
