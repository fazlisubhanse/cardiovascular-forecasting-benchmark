"""Read-only validation for the JMIR Round-2 publication figure package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

from generate_round2_figures import FIGURES, ROOT, DEFAULT_OUTPUT, sha256, verify_inputs


def validate(output: Path | None = None) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, observed: object, expected: object) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL",
                       "observed": observed, "expected": expected})

    before = verify_inputs()
    output = DEFAULT_OUTPUT if output is None else output
    expected_figure_paths = [output / f"{stem}.{suffix}" for stem in FIGURES.values() for suffix in ["svg", "pdf", "png"]]
    check("all_12_safe_figure_files_exist", all(path.exists() for path in expected_figure_paths),
          sum(path.exists() for path in expected_figure_paths), 12)

    valid_svg = 0; valid_pdf = 0; valid_png = 0; png_dpi = []
    for path in expected_figure_paths:
        if path.suffix == ".svg":
            root = ET.parse(path).getroot()
            if root.tag.endswith("svg"):
                valid_svg += 1
        elif path.suffix == ".pdf":
            if path.read_bytes()[:5] == b"%PDF-":
                valid_pdf += 1
        else:
            with Image.open(path) as image:
                dpi = image.info.get("dpi", (0, 0))
                png_dpi.append(dpi)
                if image.width >= 3000 and image.height >= 1800 and min(dpi) >= 590:
                    valid_png += 1
    check("svg_files_parse", valid_svg == 4, valid_svg, 4)
    check("pdf_files_parse", valid_pdf == 4, valid_pdf, 4)
    check("png_files_high_resolution", valid_png == 4, valid_png, 4)

    manifest_path = output / "FIGURE_MANIFEST.csv"
    manifest = pd.read_csv(manifest_path)
    manifest_bad = []
    for row in manifest.itertuples(index=False):
        path = ROOT / row.path
        if not path.exists() or sha256(path) != row.sha256:
            manifest_bad.append(row.path)
    check("figure_manifest_hashes", not manifest_bad, manifest_bad, [])

    checksum_path = output / "OUTPUT_SHA256SUMS.txt"
    checksum_bad = []; checksum_count = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1); checksum_count += 1
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            checksum_bad.append(relative)
    check("output_checksum_manifest", not checksum_bad, checksum_bad, [])

    rankings = pd.read_csv(ROOT / "results/round2/primary_rankings.csv")
    winner_rows = rankings.loc[rankings["rank"].eq(1)]
    expected_winners = {
        ("ASPH", 1): "ARIMA", ("ASPH", 2): "ARIMA", ("ASPH", 5): "ARIMA",
        ("MTC", 1): "ARIMA", ("MTC", 2): "ARIMA", ("MTC", 5): "ETS",
    }
    observed_winners = {(row.target, int(row.horizon)): row.model_label for row in winner_rows.itertuples()}
    check("frozen_winners_preserved", observed_winners == expected_winners, observed_winners, expected_winners)
    check("nineteen_primary_specifications_per_cell",
          bool((rankings.groupby(["target", "horizon"]).size() == 19).all()),
          rankings.groupby(["target", "horizon"]).size().to_dict(), "19 in each cell")

    coverage = pd.read_csv(ROOT / "results/round2/interval_coverage.csv")
    summary = pd.read_csv(ROOT / "results/round2/tables/table_interval_coverage_summary.csv")
    capable = coverage.loc[coverage.valid_interval_n.gt(0)].copy()
    capable["covered"] = np.rint(capable.empirical_coverage * capable.valid_interval_n)
    pooled = capable.groupby(["target", "horizon", "nominal_coverage"]).agg(
        covered=("covered", "sum"), valid=("valid_interval_n", "sum")
    ).reset_index()
    pooled["recomputed"] = pooled.covered / pooled.valid
    merged = summary.merge(pooled, on=["target", "horizon", "nominal_coverage"], validate="one_to_one")
    pooled_match = np.allclose(
        merged.pooled_coverage_across_valid_model_forecast_interval_predictions,
        merged.recomputed, atol=1e-15, rtol=0,
    )
    check("pooled_coverage_recomputed", bool(pooled_match), bool(pooled_match), True)

    winner_coverage = pd.read_csv(ROOT / "results/round2/tables/table_winner_interval_coverage.csv")
    mtc_h5 = winner_coverage.loc[winner_coverage.target.eq("MTC") & winner_coverage.horizon.eq(5)].iloc[0]
    check("mtc_h5_ets_zero_coverage",
          bool(mtc_h5.winner == "ETS" and mtc_h5.empirical_80_coverage == 0 and mtc_h5.empirical_95_coverage == 0),
          [mtc_h5.winner, mtc_h5.empirical_80_coverage, mtc_h5.empirical_95_coverage], ["ETS", 0, 0])

    ablation = pd.read_csv(ROOT / "results/round2/ablation_results.csv")
    inference = pd.read_csv(ROOT / "results/round2/inference.csv")
    check("ablation_inference_complete", len(ablation) == 12 and len(inference) == 12,
          [len(ablation), len(inference)], [12, 12])
    disabled = inference.loc[inference.validity_flag.eq("disabled_a_priori_descriptive_only")]
    check("asph_h5_inference_disabled", len(disabled) == 2 and disabled.target.eq("ASPH").all()
          and disabled.horizon.eq(5).all() and disabled.holm_adjusted_p_value.isna().all(),
          len(disabled), 2)

    captions = (output / "FIGURE_CAPTIONS.md").read_text(encoding="utf-8")
    provenance = (output / "FIGURE_PROVENANCE.md").read_text(encoding="utf-8")
    check("captions_complete", all(title in captions for title in ["Figure 2.", "Figure 3.", "Figure 4.", "Multimedia Appendix Figure 1."]),
          "all titles present", "all titles present")
    check("source_series_figure_not_generated", "## Figure 1." not in captions
          and not any(output.glob("figure_1_provenance_framework.*")),
          "absent", "absent")
    check("per_figure_provenance_complete", all(figure_id in provenance for figure_id in FIGURES),
          "all figure IDs present", "all figure IDs present")

    after = verify_inputs()
    check("frozen_inputs_unchanged", before == after, before == after, True)
    failed = [row for row in checks if row["status"] == "FAIL"]
    return {
        "status": "PASS" if not failed else "FAIL",
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_checks": failed,
        "checksum_entries_verified": checksum_count,
        "png_dpi_observed": png_dpi,
        "model_training_executed": False,
        "forecast_generation_executed": False,
        "frozen_inputs_unchanged": before == after,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    selected_output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    result = validate(selected_output)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["status"] == "PASS" else 1)
