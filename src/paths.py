"""Centralized project paths used by the Phase 1 audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project paths and required output directories."""

    root: Path
    config: Path
    raw_csv: Path
    validated_csv: Path
    manuscript: Path
    reviewer_comments: Path
    legacy_zip: Path
    reference_dir: Path
    outputs_dir: Path
    figures_dir: Path
    tables_dir: Path
    logs_dir: Path
    reports_dir: Path
    legacy_extracted_dir: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "ProjectPaths":
        """Build paths relative to *root* or the repository root."""

        resolved = (root or Path(__file__).resolve().parents[1]).resolve()
        return cls(
            root=resolved,
            config=resolved / "configs" / "audit.yaml",
            raw_csv=resolved / "data" / "raw" / "HeartDiseasesDataset.csv",
            validated_csv=resolved / "data" / "validated" / "heart_disease_series_validated.csv",
            manuscript=resolved / "reference" / "99686-1460173-1-ED.docx",
            reviewer_comments=resolved / "reference" / "reviewer_comments.txt",
            legacy_zip=resolved / "reference" / "KhanJe_1stpaper.zip",
            reference_dir=resolved / "reference",
            outputs_dir=resolved / "outputs",
            figures_dir=resolved / "outputs" / "figures",
            tables_dir=resolved / "outputs" / "tables",
            logs_dir=resolved / "outputs" / "logs",
            reports_dir=resolved / "reports",
            legacy_extracted_dir=resolved / "outputs" / "legacy_extracted",
        )

    def create_output_directories(self) -> None:
        """Create generated-data directories without touching references."""

        for path in (
            self.validated_csv.parent,
            self.figures_dir,
            self.tables_dir,
            self.logs_dir,
            self.reports_dir,
            self.legacy_extracted_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
