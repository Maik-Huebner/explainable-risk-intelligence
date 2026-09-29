"""Configuration loading and shared project paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

PACKAGE_CONFIG = files("explainable_risk_intelligence").joinpath("resources/default.yaml")


def discover_project_root(start: Path | None = None) -> Path:
    """Find a repository root when present, otherwise use the caller's directory."""

    configured = os.getenv("ERI_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.exists() and (candidate / "src" / "explainable_risk_intelligence").exists():
            return candidate
    return current


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical paths used by every command."""

    root: Path | None = None

    def __post_init__(self) -> None:
        resolved = discover_project_root() if self.root is None else Path(self.root).resolve()
        object.__setattr__(self, "root", resolved)

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "figures"

    @property
    def metrics(self) -> Path:
        return self.root / "reports" / "metrics"

    @property
    def model_card(self) -> Path:
        return self.root / "reports" / "model-card"

    def ensure_generated_directories(self) -> None:
        for path in (
            self.raw,
            self.interim,
            self.processed,
            self.artifacts,
            self.figures,
            self.metrics,
            self.model_card,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load an explicit config or the default shipped inside the package."""

    config_source = Path(path) if path is not None else PACKAGE_CONFIG
    with config_source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_source}")
    return config
