from __future__ import annotations

import hashlib
import io
import json
from zipfile import ZIP_DEFLATED, ZipFile

from explainable_risk_intelligence.config import PACKAGE_CONFIG, load_config
from explainable_risk_intelligence.data import download as download_module


def test_packaged_default_config_is_available() -> None:
    assert PACKAGE_CONFIG.is_file()
    config = load_config()
    assert config["project"]["random_seed"] == 42
    assert config["dataset"]["uci_id"] == 350


def test_download_writes_verified_offline_provenance(tmp_path: object, monkeypatch: object) -> None:
    workbook_name = "fixture.xls"
    archive_buffer = io.BytesIO()
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(workbook_name, b"offline workbook fixture")
    archive_bytes = archive_buffer.getvalue()
    expected_hash = hashlib.sha256(archive_bytes).hexdigest()
    config = {
        "dataset": {
            "name": "Offline fixture",
            "uci_id": 350,
            "doi": "https://doi.org/10.24432/C55S3H",
            "source_page": "https://example.invalid/source",
            "download_url": "https://example.invalid/data.zip",
            "archive_sha256": expected_hash,
            "archive_name": "fixture.zip",
            "workbook_name": workbook_name,
            "license": "CC BY 4.0",
        }
    }

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    monkeypatch.setenv("ERI_PROJECT_ROOT", str(tmp_path))  # type: ignore[attr-defined]
    monkeypatch.setattr(download_module, "load_config", lambda: config)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        download_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(archive_bytes),
    )
    workbook = download_module.download_dataset(force=True)
    provenance = json.loads((workbook.parent / "provenance.json").read_text(encoding="utf-8"))
    assert workbook.read_bytes() == b"offline workbook fixture"
    assert provenance["archive_sha256"] == expected_hash
    assert provenance["uci_id"] == 350
