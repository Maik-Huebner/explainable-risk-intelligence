"""Detect common accidental leaks and oversized source-controlled files.

This deliberately small quality gate is not a replacement for a professional secret scanner.
"""

from __future__ import annotations

import re
from pathlib import Path

EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
TEXT_SUFFIXES = {
    ".cff",
    ".example",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {".gitignore", "LICENSE", "Makefile"}
SECRET_PATTERNS = {
    "private key": re.compile("BEGIN [A-Z ]*PRIVATE" + " KEY"),
    "OpenAI key": re.compile("sk-" + r"[A-Za-z0-9]{32,}"),
    "GitHub token": re.compile("gh" + r"[oprs]_[A-Za-z0-9]{30,}"),
}


def is_generated(path: Path, root: Path) -> bool:
    """Return whether a path belongs to ignored runtime data or artifacts."""

    generated_prefixes = {root / "data", root / "artifacts"}
    return any(prefix == path or prefix in path.parents for prefix in generated_prefixes)


def audit_repository(root: Path) -> list[str]:
    """Return actionable findings for a repository root."""

    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.name == ".DS_Store":
            findings.append(f"macOS metadata: {path.relative_to(root)}")
        if path.stat().st_size > 10 * 1024 * 1024 and not is_generated(path, root):
            findings.append(f"large file: {path.relative_to(root)} ({path.stat().st_size} bytes)")
        is_text_candidate = path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES
        if is_generated(path, root) or not is_text_candidate:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"possible {label}: {path.relative_to(root)}")
    return findings


def main(root: Path) -> int:
    """Run the audit and return a process-compatible exit code."""

    findings = audit_repository(root)
    if findings:
        print("\n".join(findings))
        return 1
    print(
        "Repository audit passed: no obvious secrets, .DS_Store files, or oversized source files. "
        "This lightweight gate is not a replacement for a professional secret scanner."
    )
    return 0
