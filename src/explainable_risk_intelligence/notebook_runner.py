"""Execute notebook code cells in order without notebook runtime dependencies."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def execute_notebook(path: Path) -> int:
    """Execute all code cells from a curated notebook in its project root."""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "__notebook__"}
    original_directory = Path.cwd()
    project_root = path.resolve().parents[1]
    os.chdir(project_root)
    try:
        for index, cell in enumerate(notebook["cells"], start=1):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(original_directory)
    print(f"Notebook execution passed: {path}")
    return 0


def main(
    argv: Sequence[str] | None = None,
    default_notebook: Path = Path("notebooks/01_portfolio_analysis.ipynb"),
) -> int:
    """Parse the developer-tool CLI and execute its selected notebook."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path, default=default_notebook)
    return execute_notebook(parser.parse_args(argv).notebook)
