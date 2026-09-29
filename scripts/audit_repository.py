"""Audit this checkout through the installed project package."""

from __future__ import annotations

from pathlib import Path

from explainable_risk_intelligence.repository_audit import main

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(main(ROOT))
