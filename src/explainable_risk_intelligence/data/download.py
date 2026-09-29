"""Download and verify the official UCI dataset."""

from __future__ import annotations

import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

from explainable_risk_intelligence.config import ProjectPaths, load_config
from explainable_risk_intelligence.utils.io import sha256, write_json


def download_dataset(force: bool = False) -> Path:
    """Download, hash-check, and extract the exact UCI workbook."""

    config = load_config()
    dataset = config["dataset"]
    paths = ProjectPaths()
    paths.ensure_generated_directories()
    archive_path = paths.raw / dataset["archive_name"]
    workbook_path = paths.raw / dataset["workbook_name"]
    url = os.getenv("ERI_DATASET_URL", dataset["download_url"])

    if force or not archive_path.exists():
        partial = archive_path.with_suffix(".zip.part")
        request = urllib.request.Request(
            url, headers={"User-Agent": "explainable-risk-intelligence/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            partial.write_bytes(response.read())
        partial.replace(archive_path)

    actual_hash = sha256(archive_path)
    expected_hash = dataset["archive_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            f"Dataset hash mismatch: expected {expected_hash}, got {actual_hash}. "
            "Delete the local archive only after investigating the provenance change."
        )

    with ZipFile(archive_path) as archive:
        members = archive.namelist()
        if members != [dataset["workbook_name"]]:
            raise ValueError(f"Unexpected archive contents: {members}")
        workbook_path.write_bytes(archive.read(dataset["workbook_name"]))

    write_json(
        paths.raw / "provenance.json",
        {
            "dataset": dataset["name"],
            "uci_id": dataset["uci_id"],
            "doi": dataset["doi"],
            "source_page": dataset["source_page"],
            "download_url": url,
            "license": dataset["license"],
            "archive_sha256": actual_hash,
            "archive_bytes": archive_path.stat().st_size,
            "workbook_bytes": workbook_path.stat().st_size,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return workbook_path
