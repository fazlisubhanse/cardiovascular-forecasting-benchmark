"""Publication-safe project paths for dataset validation and analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Paths used by the public data-validation workflow."""

    root: Path
    config: Path
    raw_csv: Path
    validated_csv: Path
    outputs_dir: Path
    figures_dir: Path
    tables_dir: Path
    logs_dir: Path
    reports_dir: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "ProjectPaths":
        resolved = (root or Path(__file__).resolve().parents[1]).resolve()
        return cls(
            root=resolved,
            config=resolved / "configs" / "data_validation.yaml",
            raw_csv=resolved / "data" / "raw" / "input_series.csv",
            validated_csv=resolved / "data" / "validated" / "heart_disease_series_validated.csv",
            outputs_dir=resolved / "outputs",
            figures_dir=resolved / "outputs" / "figures",
            tables_dir=resolved / "outputs" / "tables",
            logs_dir=resolved / "outputs" / "logs",
            reports_dir=resolved / "reports",
        )

    def create_output_directories(self) -> None:
        for path in (
            self.validated_csv.parent,
            self.figures_dir,
            self.tables_dir,
            self.logs_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
