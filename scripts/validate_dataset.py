"""Validate an author-authorized local copy of the study dataset."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_loader import load_raw_data
from src.data_validation import validate_data
from src.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/input_series.csv"),
        help="Local CSV obtained through an authorized data-access route.",
    )
    args = parser.parse_args()
    root = ROOT
    input_path = (root / args.input).resolve() if not args.input.is_absolute() else args.input.resolve()
    try:
        input_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("For reproducible relative-path records, place the input inside this repository.") from exc

    paths = replace(ProjectPaths.from_root(root), raw_csv=input_path)
    paths.create_output_directories()
    config = load_config(root / "configs" / "data_validation.yaml")
    raw = load_raw_data(input_path, config)
    result = validate_data(raw, config, paths)
    summary = result["summary"]
    print(
        f"PASS: {summary['rows']} rows, {summary['first_year']}-{summary['last_year']}, "
        f"SHA-256 {summary['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
