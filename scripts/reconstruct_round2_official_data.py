"""Reconstruct the official China-male cardiovascular series from user downloads.

This Mode-A release script embeds no source observations. It verifies the three
official input hashes, selects the prespecified rows and columns, validates the
independent WHO rounding cross-check, and writes the 69-row private local
analysis dataset. It does not use the excluded legacy dataset and performs no
forecasting, interpolation, extrapolation, imputation, smoothing, splicing, or
synthetic extension.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "china_male_official_cardiovascular_series.csv"

WHO_FILENAME = "608DE39_ALL_LATEST.csv"
HYP_FILENAME = "NCD-RisC_Lancet_2021_Hypertension_age_standardised_countries.csv"
CHOL_FILENAME = "NCD_RisC_Nature_2020_Cholesterol_age_standardised_countries.csv"

EXPECTED_HASHES = {
    WHO_FILENAME: "767d6fe20ad8116c2ca856dccb504ca3c55668d2316bbdb1082a7d4b9d10bffd",
    HYP_FILENAME: "5b93b95db281acddce4b80feec4523ddc928cff6b236d93fa449566d070e5517",
    CHOL_FILENAME: "e24ac13cebf9e80ecc3bd42c8fec0a8e5eb63da34e464d1fb3cc64f5cb18b05b",
}
EXPECTED_OUTPUT_HASH = "4575b29feee41e3832c9c2e51c9c0f8f0e312d5e7f4df9b87ce75e13e705c3e2"

HYP_URL = (
    "https://www.ncdrisc.org/downloads/hypertension/"
    "NCD-RisC_Lancet_2021_Hypertension_age_standardised_countries.csv"
)
CHOL_URL = (
    "https://www.ncdrisc.org/downloads/chol/"
    "NCD_RisC_Nature_2020_Cholesterol_age_standardised_countries.csv"
)
HYP_PUBLICATION = (
    "NCD Risk Factor Collaboration (NCD-RisC). Worldwide trends in hypertension "
    "prevalence and progress in treatment and control from 1990 to 2019: a pooled "
    "analysis of 1,201 population-representative studies with 104 million participants. "
    "The Lancet. 2021;398:957–980. doi:10.1016/S0140-6736(21)01330-1."
)
CHOL_PUBLICATION = (
    "NCD Risk Factor Collaboration (NCD-RisC). Repositioning of the global epicentre "
    "of non-optimal cholesterol. Nature. 2020;582:73–77. "
    "doi:10.1038/s41586-020-2338-1."
)

FIELDS = [
    "target", "country", "sex", "age_population", "year", "source_value",
    "analysis_value", "unit", "lower_interval", "upper_interval",
    "source_organization", "source_dataset", "source_publication", "source_url",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header: {path}")
        return list(reader)


def select_sources(raw_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    for filename, expected in EXPECTED_HASHES.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required official download missing: {path}")
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"SHA-256 mismatch for {filename}: {observed}")

    who = read_rows(raw_dir / WHO_FILENAME)
    hyp = read_rows(raw_dir / HYP_FILENAME)
    chol = read_rows(raw_dir / CHOL_FILENAME)
    return who, hyp, chol


def reconstruct(hyp: list[dict[str, str]], chol: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    hyp_rows = sorted(
        (
            row for row in hyp
            if row["Country/Region/World"] == "China"
            and row["ISO"] == "CHN"
            and row["Sex"] == "Men"
            and row["Age"] == "Age standardised (30-79 years)"
            and 1990 <= int(row["Year"]) <= 2019
        ),
        key=lambda row: int(row["Year"]),
    )
    for row in hyp_rows:
        output.append({
            "target": "ASPH",
            "country": "China",
            "sex": "Male",
            "age_population": "30–79 years",
            "year": int(row["Year"]),
            "source_value": row["Prevalence of hypertension"],
            "analysis_value": row["Prevalence of hypertension"],
            "unit": "proportion",
            "lower_interval": row["Prevalence of hypertension lower 95% uncertainty interval"],
            "upper_interval": row["Prevalence of hypertension upper 95% uncertainty interval"],
            "source_organization": "NCD Risk Factor Collaboration (NCD-RisC)",
            "source_dataset": "NCD-RisC Lancet 2021 age-standardised hypertension country estimates",
            "source_publication": HYP_PUBLICATION,
            "source_url": HYP_URL,
        })

    central = "Mean total cholesterol (mmol/L)"
    lower = "Mean total cholesterol lower 95% uncertainty interval (mmol/L)"
    upper = "Mean total cholesterol upper 95% uncertainty interval (mmol/L)"
    chol_rows = sorted(
        (
            row for row in chol
            if row["Country/Region/World"] == "China"
            and row["ISO"] == "CHN"
            and row["Sex"] == "Men"
            and 1980 <= int(row["Year"]) <= 2018
        ),
        key=lambda row: int(row["Year"]),
    )
    for row in chol_rows:
        output.append({
            "target": "MTC",
            "country": "China",
            "sex": "Male",
            "age_population": "18+ years",
            "year": int(row["Year"]),
            "source_value": row[central],
            "analysis_value": row[central],
            "unit": "mmol/L",
            "lower_interval": row[lower],
            "upper_interval": row[upper],
            "source_organization": "NCD Risk Factor Collaboration (NCD-RisC)",
            "source_dataset": "NCD-RisC Nature 2020 age-standardised country estimates",
            "source_publication": CHOL_PUBLICATION,
            "source_url": CHOL_URL,
        })
    return output


def validate(rows: list[dict[str, object]], who: list[dict[str, str]]) -> None:
    asph = [row for row in rows if row["target"] == "ASPH"]
    mtc = [row for row in rows if row["target"] == "MTC"]
    if [int(row["year"]) for row in asph] != list(range(1990, 2020)):
        raise AssertionError("ASPH is not exactly one annual row for 1990–2019")
    if [int(row["year"]) for row in mtc] != list(range(1980, 2019)):
        raise AssertionError("MTC is not exactly one annual row for 1980–2018")
    keys = [(str(row["target"]), int(row["year"])) for row in rows]
    if len(rows) != 69 or len(keys) != len(set(keys)):
        raise AssertionError("Expected 69 unique target-year rows")

    numeric = ("source_value", "analysis_value", "lower_interval", "upper_interval")
    if any(not math.isfinite(float(row[field])) for row in rows for field in numeric):
        raise AssertionError("Missing or nonfinite source number")
    if any(not (Decimal(0) <= Decimal(str(row[field])) <= Decimal(1)) for row in asph for field in numeric):
        raise AssertionError("ASPH value outside [0,1]")
    if any(not (Decimal(0) < Decimal(str(row["analysis_value"])) < Decimal(20)) for row in mtc):
        raise AssertionError("MTC value outside the prespecified plausibility range")
    if any(not (Decimal(str(row["lower_interval"])) <= Decimal(str(row["analysis_value"])) <= Decimal(str(row["upper_interval"]))) for row in rows):
        raise AssertionError("Central estimate outside its uncertainty interval")

    who_selected = {
        int(row["DIM_TIME"]): Decimal(row["RATE_PER_100_N"])
        for row in who
        if row["GEO_NAME_SHORT"] == "China"
        and row["DIM_GEO_CODE_TYPE"] == "COUNTRY"
        and row["DIM_GEO_CODE_M49"] == "156"
        and row["DIM_SEX"] == "MALE"
        and 1990 <= int(row["DIM_TIME"]) <= 2019
    }
    if set(who_selected) != set(range(1990, 2020)):
        raise AssertionError("WHO cross-check does not contain 1990–2019 exactly once")
    matches = 0
    for row in asph:
        expected_percent = (Decimal(str(row["analysis_value"])) * Decimal(100)).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
        matches += expected_percent == who_selected[int(row["year"])]
    if matches != 30:
        raise AssertionError(f"WHO rounding cross-check failed: {matches}/30")


def write_output(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    observed = sha256(output)
    if observed != EXPECTED_OUTPUT_HASH:
        raise AssertionError(f"Unexpected reconstructed dataset SHA-256: {observed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    who, hyp, chol = select_sources(args.raw_dir.resolve())
    rows = reconstruct(hyp, chol)
    validate(rows, who)
    write_output(rows, args.output.resolve())
    print("PASS: 69 unique official rows reconstructed (ASPH 30; MTC 39).")
    print(f"Processed SHA-256: {EXPECTED_OUTPUT_HASH}")
    print("WHO one-decimal-percentage rounding cross-check: 30/30.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
