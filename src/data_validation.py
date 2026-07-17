"""Raw-data integrity, descriptive, change, and stationarity audits."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from .data_loader import standardized_data, write_validated_copy
from .paths import ProjectPaths

LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    """Calculate a streaming SHA-256 digest without modifying a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(directory: Path) -> pd.DataFrame:
    """Hash every file beneath a directory using stable relative paths."""

    rows = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(directory).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows, columns=["relative_path", "sha256", "size_bytes"])


def _stationarity_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in ("asph", "mtc"):
        values = frame[column].to_numpy(dtype=float)
        adf = adfuller(values, autolag="AIC")
        try:
            kpss_result = kpss(values, regression="ct", nlags="auto")
            kpss_stat, kpss_p, kpss_lags = kpss_result[0], kpss_result[1], kpss_result[2]
            kpss_error = ""
        except (ValueError, OverflowError) as exc:
            kpss_stat, kpss_p, kpss_lags, kpss_error = np.nan, np.nan, np.nan, str(exc)
        rows.extend(
            [
                {
                    "series": column,
                    "test": "ADF",
                    "statistic": adf[0],
                    "p_value": adf[1],
                    "lags": adf[2],
                    "null_hypothesis": "unit root (nonstationary)",
                    "limited_power_note": "Exploratory only; n=54 provides limited statistical power.",
                    "error": "",
                },
                {
                    "series": column,
                    "test": "KPSS-trend",
                    "statistic": kpss_stat,
                    "p_value": kpss_p,
                    "lags": kpss_lags,
                    "null_hypothesis": "trend stationary",
                    "limited_power_note": "Exploratory only; n=54 provides limited statistical power.",
                    "error": kpss_error,
                },
            ]
        )
    return rows


def validate_data(raw: pd.DataFrame, config: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
    """Validate all hard integrity constraints and write reproducible audit tables."""

    expected = config["expected"]
    frame = standardized_data(raw, config)
    digest = sha256_file(paths.raw_csv)
    hash_table = pd.DataFrame(
        [{"relative_path": paths.raw_csv.relative_to(paths.root).as_posix(), "sha256": digest,
          "size_bytes": paths.raw_csv.stat().st_size}]
    )
    hash_table.to_csv(paths.tables_dir / "raw_file_hash.csv", index=False, encoding="utf-8")
    if digest.lower() != str(expected["raw_sha256"]).lower():
        raise ValueError(f"Raw-data SHA-256 mismatch: observed {digest}")

    years = frame["year"]
    expected_years = list(range(expected["first_year"], expected["last_year"] + 1))
    failures: list[str] = []
    checks = {
        "row_count": len(frame) == expected["row_count"],
        "column_count": len(frame.columns) == 5,
        "no_missing_targets": not frame[["asph", "mtc"]].isna().any().any(),
        "no_duplicate_rows": not frame.duplicated().any(),
        "no_duplicate_years": not years.duplicated().any(),
        "exact_year_sequence": sorted(years.astype(int).tolist()) == expected_years,
        "chronological_order": years.astype(int).tolist() == expected_years,
        "finite_targets": bool(np.isfinite(frame[["asph", "mtc"]].to_numpy(dtype=float)).all()),
        "asph_in_unit_interval": bool(frame["asph"].between(0.0, 1.0, inclusive="both").all()),
        "mtc_positive": bool((frame["mtc"] > 0).all()),
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    if failures:
        raise ValueError("Raw-data integrity failure(s): " + ", ".join(failures))

    write_validated_copy(frame, paths.validated_csv)
    descriptive = frame[["year", "asph", "mtc"]].describe().T.reset_index(names="variable")
    descriptive.to_csv(paths.tables_dir / "descriptive_statistics.csv", index=False, encoding="utf-8")

    differences = frame[["year", "asph", "mtc"]].copy()
    differences["asph_difference"] = frame["asph"].diff()
    differences["mtc_difference"] = frame["mtc"].diff()
    differences.to_csv(paths.tables_dir / "annual_differences.csv", index=False, encoding="utf-8")

    extremes: list[dict[str, Any]] = []
    breaks: list[dict[str, Any]] = []
    for column in ("asph", "mtc"):
        diff = frame[column].diff()
        for direction, index in (("largest_positive", diff.idxmax()), ("largest_negative", diff.idxmin())):
            extremes.append(
                {"series": column, "direction": direction, "year": int(frame.loc[index, "year"]),
                 "change": float(diff.loc[index])}
            )
        valid = diff.dropna()
        scale = valid.std(ddof=1)
        zscores = (valid - valid.mean()) / scale if scale > 0 else valid * np.nan
        selected = zscores.abs().sort_values(ascending=False).head(5)
        for index, zscore in selected.items():
            breaks.append(
                {"series": column, "year": int(frame.loc[index, "year"]), "change": float(diff.loc[index]),
                 "difference_z_score": float(zscore), "flagged_abs_z_ge_2": bool(abs(zscore) >= 2)}
            )
    pd.DataFrame(extremes).to_csv(paths.tables_dir / "largest_annual_changes.csv", index=False, encoding="utf-8")
    pd.DataFrame(breaks).to_csv(paths.tables_dir / "possible_abrupt_changes.csv", index=False, encoding="utf-8")

    train = frame.loc[frame["year"] <= expected["training_last_year"]]
    test = frame.loc[frame["year"] > expected["training_last_year"]]
    exceedance_rows = []
    for column in ("asph", "mtc"):
        maximum = float(train[column].max())
        for _, row in test.iterrows():
            exceedance_rows.append(
                {"series": column, "year": int(row["year"]), "value": float(row[column]),
                 "training_maximum_through_2015": maximum, "exceeds_training_maximum": bool(row[column] > maximum)}
            )
    exceedance = pd.DataFrame(exceedance_rows)
    exceedance.to_csv(paths.tables_dir / "training_range_exceedance.csv", index=False, encoding="utf-8")

    stationarity = pd.DataFrame(_stationarity_rows(frame))
    stationarity.to_csv(paths.tables_dir / "stationarity_tests.csv", index=False, encoding="utf-8")
    dictionary = pd.DataFrame(
        [
            {"name": "sex", "source_name": "Sex", "dtype": str(frame["sex"].dtype), "description": "Reported sex category", "unit": "category"},
            {"name": "year", "source_name": "Year", "dtype": str(frame["year"].dtype), "description": "Annual observation year", "unit": "calendar year"},
            {"name": "age_group", "source_name": "Age", "dtype": str(frame["age_group"].dtype), "description": "Reported age group", "unit": "category"},
            {"name": "asph", "source_name": expected["raw_columns"][3], "dtype": str(frame["asph"].dtype), "description": "Age-standardised prevalence of hypertension", "unit": "proportion"},
            {"name": "mtc", "source_name": expected["raw_columns"][4], "dtype": str(frame["mtc"].dtype), "description": "Mean total cholesterol", "unit": "mmol/L"},
        ]
    )
    dictionary.to_csv(paths.tables_dir / "data_dictionary.csv", index=False, encoding="utf-8")

    summary = {
        "sha256": digest,
        "size_bytes": paths.raw_csv.stat().st_size,
        "rows": len(raw),
        "columns": len(raw.columns),
        "column_names": raw.columns.tolist(),
        "dtypes": {name: str(dtype) for name, dtype in raw.dtypes.items()},
        "missing_values": {name: int(value) for name, value in raw.isna().sum().items()},
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_years": int(years.duplicated().sum()),
        "first_year": int(years.min()),
        "last_year": int(years.max()),
        "sex_values": sorted(frame["sex"].astype(str).unique().tolist()),
        "age_values": sorted(frame["age_group"].astype(str).unique().tolist()),
        "checks": checks,
        "exceedance_counts": exceedance.groupby("series")["exceeds_training_maximum"].sum().astype(int).to_dict(),
        "stationarity": stationarity.to_dict(orient="records"),
    }
    LOGGER.info("Validated %d rows spanning %d-%d", len(frame), years.min(), years.max())
    return {"frame": frame, "summary": summary, "extremes": extremes, "breaks": breaks, "exceedance": exceedance}
