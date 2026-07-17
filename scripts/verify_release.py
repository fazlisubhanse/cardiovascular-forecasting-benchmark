"""Run fast, non-training checks on the publication staging snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "README.md",
        root / "LICENSE",
        root / "LICENSE-CONTENT.md",
        root / "CITATION.cff",
        root / "requirements.txt",
        root / "docs" / "DATA_ACCESS.md",
        root / "docs" / "REPRODUCIBILITY.md",
        root / "results" / "FROZEN_RESULTS_MANIFEST.json",
        root / "results" / "tables" / "table_primary_core_results.csv",
    ]
    missing = [path.relative_to(root).as_posix() for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing release files: {missing}")

    mit = (root / "LICENSE").read_text(encoding="utf-8")
    scope = (root / "LICENSE-CONTENT.md").read_text(encoding="utf-8")
    if not mit.startswith("MIT License\n\nCopyright (c) 2026 The Authors"):
        raise RuntimeError("The approved MIT license or copyright notice is missing.")
    required_scope = ["Creative Commons Attribution 4.0 International", "complete 1970-2023 study dataset", "candidate NCD-RisC source files"]
    if not all(item in scope for item in required_scope):
        raise RuntimeError("The dual-license scope or data exclusions are incomplete.")

    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    if citation.get("version") != "1.0.0" or "doi" in citation or "repository-code" in citation:
        raise RuntimeError("Citation metadata do not match the approved local v1.0.0 identity.")
    if '__version__ = "1.0.0"' not in (root / "src" / "__init__.py").read_text(encoding="utf-8"):
        raise RuntimeError("The package version does not match the approved local v1.0.0 identity.")

    manifest = json.loads((root / "results" / "FROZEN_RESULTS_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("freeze_version") != "Phase2D.1" or manifest.get("scientific_results_frozen") is not True:
        raise RuntimeError("Frozen-results manifest does not identify the Phase 2D.1 freeze.")

    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".docx", ".xlsx", ".zip"}
    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden private, checkpoint, or archive files found: {forbidden}")
    if (root / "data").exists() and any((root / "data").rglob("*")):
        raise RuntimeError("The release snapshot must not contain study data files.")

    tables = list((root / "results" / "tables").glob("*.csv"))
    figures = [path for path in (root / "figures").iterdir() if path.is_file()]
    if len(tables) != 15 or len(figures) != 30:
        raise RuntimeError(f"Unexpected evidence snapshot counts: tables={len(tables)}, figures={len(figures)}")
    print("PASS: licenses, publication staging structure, and frozen evidence snapshot are internally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
