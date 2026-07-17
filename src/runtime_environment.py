"""Environment recording shared by the public analysis entry points."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd


ANALYSIS_DEPENDENCIES = [
    "numpy", "pandas", "scipy", "matplotlib", "statsmodels", "openpyxl",
    "python-docx", "nbformat", "pyyaml", "pytest",
]


def record_environment(tables_dir: Path, logs_dir: Path) -> None:
    """Record interpreter, platform, dependency versions, and the full freeze."""

    rows = [
        {"component": "python", "version": platform.python_version()},
        {"component": "platform", "version": platform.platform()},
    ]
    for package in ANALYSIS_DEPENDENCIES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT INSTALLED"
        rows.append({"component": package, "version": version})
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables_dir / "environment_versions.csv", index=False, encoding="utf-8")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True,
        capture_output=True, text=True, encoding="utf-8",
    )
    (logs_dir / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
