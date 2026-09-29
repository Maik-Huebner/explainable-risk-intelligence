"""Create deterministic, stratified train/validation/test data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from explainable_risk_intelligence.config import ProjectPaths, load_config
from explainable_risk_intelligence.data.schema import validate_raw_schema
from explainable_risk_intelligence.features.build import (
    AUDIT_COLUMNS,
    MODEL_FEATURES,
    RECORD_ID,
    TARGET,
    build_features,
)
from explainable_risk_intelligence.utils.io import write_json


def load_official_workbook(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, header=1, engine="xlrd")
    validate_raw_schema(frame)
    return frame


def split_frame(
    frame: pd.DataFrame,
    random_seed: int,
    train_size: float = 0.60,
    validation_size: float = 0.20,
    test_size: float = 0.20,
) -> dict[str, pd.DataFrame]:
    """Create disjoint, reproducible splits with target stratification."""

    if not abs(train_size + validation_size + test_size - 1.0) < 1e-9:
        raise ValueError("Split proportions must sum to 1")
    train_validation, test = train_test_split(
        frame,
        test_size=test_size,
        stratify=frame[TARGET],
        random_state=random_seed,
    )
    relative_validation = validation_size / (train_size + validation_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation,
        stratify=train_validation[TARGET],
        random_state=random_seed,
    )
    return {
        "train": train.sort_values(RECORD_ID).reset_index(drop=True),
        "validation": validation.sort_values(RECORD_ID).reset_index(drop=True),
        "test": test.sort_values(RECORD_ID).reset_index(drop=True),
    }


def prepare_data(workbook_path: Path | None = None) -> dict[str, pd.DataFrame]:
    """Validate, feature-engineer, split, and persist the modelling data."""

    config = load_config()
    paths = ProjectPaths()
    paths.ensure_generated_directories()
    source = workbook_path or paths.raw / config["dataset"]["workbook_name"]
    if not source.exists():
        raise FileNotFoundError("Dataset workbook missing; run download-data first")

    raw = load_official_workbook(source)
    prepared = build_features(raw)
    split_config = config["split"]
    splits = split_frame(
        prepared,
        random_seed=config["project"]["random_seed"],
        train_size=split_config["train_size"],
        validation_size=split_config["validation_size"],
        test_size=split_config["test_size"],
    )
    prepared.to_csv(paths.interim / "prepared_full.csv", index=False)
    for name, split in splits.items():
        split.to_csv(paths.processed / f"{name}.csv", index=False)

    id_sets = {name: set(split[RECORD_ID]) for name, split in splits.items()}
    if (
        id_sets["train"] & id_sets["validation"]
        or id_sets["train"] & id_sets["test"]
        or id_sets["validation"] & id_sets["test"]
    ):
        raise RuntimeError("Split overlap detected")

    write_json(
        paths.metrics / "data_manifest.json",
        {
            "rows": len(prepared),
            "columns": len(prepared.columns),
            "target": TARGET,
            "default_rate": prepared[TARGET].mean(),
            "random_seed": config["project"]["random_seed"],
            "split_strategy": "stratified random train/validation/test; no timestamp available",
            "splits": {
                name: {"rows": len(split), "default_rate": split[TARGET].mean()}
                for name, split in splits.items()
            },
            "model_features": MODEL_FEATURES,
            "excluded_from_model": [RECORD_ID, *AUDIT_COLUMNS],
        },
    )
    return splits


def load_split(name: str) -> pd.DataFrame:
    """Load exactly one prepared split to keep data-access boundaries explicit."""

    if name not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {name}")
    paths = ProjectPaths()
    return pd.read_csv(paths.processed / f"{name}.csv")


def load_train_validation_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load development data without opening the final test split."""

    return load_split("train"), load_split("validation")


def load_test_split() -> pd.DataFrame:
    """Load the final test split only for locked-model evaluation."""

    return load_split("test")
