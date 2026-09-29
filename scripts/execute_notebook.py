"""Execute the curated notebook through the installed project package."""

from __future__ import annotations

from pathlib import Path

from explainable_risk_intelligence.notebook_runner import main

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(main(default_notebook=ROOT / "notebooks" / "01_portfolio_analysis.ipynb"))
