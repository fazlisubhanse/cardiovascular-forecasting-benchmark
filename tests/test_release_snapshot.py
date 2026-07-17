"""Fast, non-training checks for the public release snapshot."""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_manifest_and_evidence_counts() -> None:
    manifest = json.loads((ROOT / "results/FROZEN_RESULTS_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["freeze_version"] == "Phase2D.1"
    assert manifest["scientific_results_frozen"] is True
    assert len(list((ROOT / "results/tables").glob("*.csv"))) == 15
    assert len([path for path in (ROOT / "figures").iterdir() if path.is_file()]) == 30


def test_no_dataset_or_model_checkpoint_is_staged() -> None:
    assert not (ROOT / "data").exists()
    prohibited = {".pt", ".pth", ".ckpt", ".docx", ".xlsx", ".zip"}
    assert not [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in prohibited]


def test_citation_has_no_unverified_persistent_identifier() -> None:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation = yaml.safe_load(text)
    assert citation["version"] == "1.0.0"
    assert "orcid:" not in text.lower()
    assert "doi:" not in text.lower()
    assert "repository-code:" not in text.lower()
    assert '__version__ = "1.0.0"' in (ROOT / "src/__init__.py").read_text(encoding="utf-8")


def test_dual_license_scope_and_exclusions() -> None:
    mit = (ROOT / "LICENSE").read_text(encoding="utf-8")
    scope = (ROOT / "LICENSE-CONTENT.md").read_text(encoding="utf-8")
    assert mit.startswith("MIT License\n\nCopyright (c) 2026 The Authors")
    assert "Creative Commons Attribution 4.0 International" in scope
    assert "complete 1970-2023 study dataset" in scope
    assert "candidate NCD-RisC source files" in scope
