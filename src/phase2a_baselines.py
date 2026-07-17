"""Single entry point for leakage-free Phase 2A classical baseline evaluation."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtesting.engine import run_backtest, validate_forecast_records
from .backtesting.origins import generate_forecast_origin_manifest
from .config import load_phase2a_config
from .data_validation import sha256_file
from .evaluation.aggregation import aggregate_point_metrics
from .evaluation.intervals import aggregate_interval_metrics, validate_intervals
from .evaluation.statistical_tests import compare_models
from .forecasting import (
    ArimaForecaster,
    DriftForecaster,
    EtsForecaster,
    LinearTrendForecaster,
    NaiveForecaster,
    ThetaForecaster,
)
from .runtime_environment import record_environment
from .reporting.phase2a_figures import generate_phase2a_figures
from .reporting.phase2a_report import (
    write_failure_report,
    write_method_validation_report,
    write_results_report,
)
from .reporting.phase2a_tables import aggregate_computational_cost, build_failure_table, runtime_environment

LOGGER = logging.getLogger(__name__)


def _output_paths(root: Path) -> dict[str, Path]:
    base = root / "outputs" / "phase2a"
    return {
        "base": base,
        "forecasts": base / "forecasts",
        "tables": base / "tables",
        "figures": base / "figures",
        "logs": base / "logs",
        "models": base / "models",
        "reports": root / "reports",
    }


def _prepare_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    base = paths["base"].resolve()
    if base.exists() and overwrite:
        root_outputs = base.parents[0]
        if base.name != "phase2a" or root_outputs.name != "outputs":
            raise RuntimeError(f"Refusing to clear unexpected Phase 2A output path: {base}")
        shutil.rmtree(base)
    for name in ("forecasts", "tables", "figures", "logs", "models", "reports"):
        paths[name].mkdir(parents=True, exist_ok=True)


def _configure_logging(path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _protected_manifest(root: Path) -> pd.DataFrame:
    """Hash the user-supplied validated dataset and frozen configurations."""

    protected: list[Path] = []
    for relative in ("data/validated", "configs"):
        directory = root / relative
        if directory.exists():
            protected.extend(path for path in directory.rglob("*") if path.is_file() and not path.name.startswith("PHASE2A_"))
    rows = [
        {"relative_path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(set(protected))
    ]
    return pd.DataFrame(rows)


def _verify_input(config: dict[str, Any], root: Path) -> tuple[pd.DataFrame, str]:
    required = [root / config["data"]["path"]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required validated dataset missing: " + ", ".join(missing))
    validated_path = root / config["data"]["path"]
    digest = sha256_file(validated_path)
    if digest.lower() != config["data"]["expected_sha256"].lower():
        raise ValueError(f"Validated-data SHA-256 mismatch: observed {digest}")
    return _protected_manifest(root), digest


def _load_validated(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8", float_precision="round_trip")
    expected_columns = ["sex", "year", "age_group", "asph", "mtc"]
    if data.columns.tolist() != expected_columns:
        raise ValueError(f"Validated-data schema changed: {data.columns.tolist()}")
    if data["year"].tolist() != list(range(1970, 2024)):
        raise ValueError("Validated years must be exactly 1970-2023 in chronological order.")
    if data[["asph", "mtc"]].isna().any().any() or not np.isfinite(data[["asph", "mtc"]]).all().all():
        raise ValueError("Validated targets contain missing or nonfinite values.")
    return data


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def run_phase2a(root: Path | None = None) -> dict[str, Any]:
    """Execute the complete Phase 2A baseline pipeline."""

    project_root = (root or Path(__file__).resolve().parents[1]).resolve()
    config = load_phase2a_config(project_root / "configs" / "phase2a_baselines.yaml")
    output = _output_paths(project_root)
    protected_before, validated_hash = _verify_input(config, project_root)
    _prepare_outputs(output, bool(config["runtime"]["overwrite_phase2a_outputs"]))
    _configure_logging(output["logs"] / "phase2a_baselines.log")
    LOGGER.info("Starting Phase 2A baseline evaluation")
    record_environment(project_root / "outputs" / "tables", output["logs"])
    _write_csv(protected_before, output["tables"] / "phase1_protected_hashes_before.csv")

    data = _load_validated(project_root / config["data"]["path"])
    manifest = generate_forecast_origin_manifest(data, config)
    _write_csv(manifest, output["tables"] / "forecast_origin_manifest.csv")
    minimum_residuals = int(config["intervals"]["minimum_residual_count"])
    models = [
        NaiveForecaster(minimum_residuals), DriftForecaster(minimum_residuals), LinearTrendForecaster(),
        EtsForecaster(config["ets"], minimum_residuals), ArimaForecaster(config["arima"]),
        ThetaForecaster(minimum_residuals),
    ]
    forecasts, arima_candidates, ets_candidates = run_backtest(data, manifest, models, config)
    forecast_path = output["forecasts"] / "baseline_forecasts_long.csv"
    _write_csv(forecasts, forecast_path)
    _write_csv(arima_candidates, output["tables"] / "arima_candidate_search.csv")
    _write_csv(ets_candidates, output["tables"] / "ets_candidate_search.csv")

    stored_forecasts = pd.read_csv(forecast_path, encoding="utf-8", float_precision="round_trip")
    validate_forecast_records(stored_forecasts, manifest, [model.name for model in models])
    all_metrics, diagnostic_metrics = aggregate_point_metrics(stored_forecasts, manifest)
    primary_metrics = all_metrics.loc[all_metrics["analysis_window"] == "primary_2000_2023"].reset_index(drop=True)
    recent_metrics = all_metrics.loc[all_metrics["analysis_window"] == "recent_2016_2023"].reset_index(drop=True)
    _write_csv(primary_metrics, output["tables"] / "baseline_metrics_primary.csv")
    _write_csv(recent_metrics, output["tables"] / "baseline_metrics_recent_window.csv")
    _write_csv(diagnostic_metrics, output["tables"] / "baseline_metrics_diagnostics.csv")

    interval_metrics = aggregate_interval_metrics(stored_forecasts)
    interval_violations = validate_intervals(stored_forecasts)
    _write_csv(interval_metrics, output["tables"] / "baseline_interval_metrics.csv")
    _write_csv(interval_violations, output["tables"] / "interval_record_violations.csv")
    dm_tests, bootstrap = compare_models(stored_forecasts, all_metrics, config)
    _write_csv(dm_tests, output["tables"] / "dm_tests_hln.csv")
    _write_csv(bootstrap, output["tables"] / "paired_bootstrap_metric_differences.csv")

    costs = aggregate_computational_cost(stored_forecasts)
    runtime = runtime_environment()
    failures = build_failure_table(stored_forecasts)
    _write_csv(costs, output["tables"] / "baseline_computational_cost.csv")
    _write_csv(runtime, output["tables"] / "runtime_environment.csv")
    _write_csv(failures, output["tables"] / "model_fit_failures.csv")
    generate_phase2a_figures(data, stored_forecasts, primary_metrics, recent_metrics, interval_metrics, costs, output["figures"])

    write_method_validation_report(
        output["reports"] / "PHASE2A_METHOD_VALIDATION.md", manifest, config, validated_hash
    )
    write_failure_report(
        output["reports"] / "PHASE2A_FAILURE_LOG.md", failures, arima_candidates, ets_candidates, interval_violations
    )
    write_results_report(
        output["reports"] / "PHASE2A_BASELINE_RESULTS.md", primary_metrics, recent_metrics,
        interval_metrics, dm_tests, bootstrap, costs, failures,
    )

    protected_after = _protected_manifest(project_root)
    _write_csv(protected_after, output["tables"] / "phase1_protected_hashes_after.csv")
    if not protected_before.equals(protected_after):
        raise RuntimeError("A protected validated-data or configuration file changed during Phase 2A.")
    if sha256_file(project_root / config["data"]["path"]) != validated_hash:
        raise RuntimeError("Validated-data hash changed during Phase 2A.")

    primary_records = stored_forecasts.loc[stored_forecasts["analysis_window"] == "primary_2000_2023"]
    summary = {
        "data_hash_status": "PASS",
        "targets": len(config["data"]["targets"]), "models": len(models),
        "horizons": len(config["backtest"]["horizons"]), "expected_records": len(manifest) * len(models),
        "successful_records": int(stored_forecasts["fit_status"].eq("success").sum()),
        "failed_records": int((~stored_forecasts["fit_status"].eq("success")).sum()),
        "report": output["reports"] / "PHASE2A_BASELINE_RESULTS.md",
    }
    print("\nPHASE 2A EXECUTION SUMMARY")
    print(f"Data hash status: {summary['data_hash_status']}")
    print(f"Targets / models / horizons: {summary['targets']} / {summary['models']} / {summary['horizons']}")
    print(f"Expected total forecast records: {summary['expected_records']}")
    print(f"Successful / failed forecast records: {summary['successful_records']} / {summary['failed_records']}")
    for (target, horizon), group in primary_metrics.groupby(["target", "horizon"], sort=True):
        eligible = group.loc[group["ranking_eligible"].astype(bool)]
        best_rmse = eligible.nsmallest(1, "rmse").iloc[0]
        best_mase = eligible.nsmallest(1, "mase").iloc[0]
        print(f"{target} h={horizon}: best RMSE={best_rmse['model']} ({best_rmse['rmse']:.8g}); best MASE={best_mase['model']} ({best_mase['mase']:.6g})")
    primary_intervals = interval_metrics.loc[interval_metrics["analysis_window"] == "primary_2000_2023"]
    for level in (0.80, 0.95):
        subset = primary_intervals.loc[np.isclose(primary_intervals["nominal_level"], level) & (primary_intervals["valid_interval_n"] > 0)]
        print(f"Mean {int(level*100)}% interval coverage across cells: {subset['coverage_probability'].mean():.3f}")
    print("DM caveat: non-rejection is not evidence of equivalence; paired annual samples have limited power.")
    print(f"Main report: {summary['report']}")
    return summary


def main() -> int:
    """CLI wrapper returning nonzero status for integrity or execution failures."""

    try:
        run_phase2a()
    except Exception:
        logging.getLogger(__name__).exception("Phase 2A failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
