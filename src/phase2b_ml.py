"""Single entry point for Phase 2B nested classical machine learning."""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from .backtesting.origins import generate_forecast_origin_manifest
from .phase2b_config import load_phase2b_config
from .data_validation import sha256_file
from .evaluation.aggregation import aggregate_point_metrics
from .evaluation.intervals import aggregate_interval_metrics, validate_intervals
from .evaluation.statistical_tests import compare_models
from .evaluation.phase2b_statistical_tests import compare_models_to_fixed_references
from .features.supervised import build_sample_manifest
from .runtime_environment import ANALYSIS_DEPENDENCIES
from .phase2a_baselines import _load_validated
from .phase2b_engine import NestedMLRunner
from .reporting.phase2b_figures import generate_phase2b_figures
from .reporting.phase2b_report import write_phase2b_reports

LOGGER = logging.getLogger(__name__)


def _paths(root: Path) -> dict[str, Path]:
    base = root / "outputs" / "phase2b"
    return {"base": base, "forecasts": base/"forecasts", "tables": base/"tables",
            "figures": base/"figures", "logs": base/"logs", "models": base/"models",
            "reports": root/"reports"}


def _prepare(output: dict[str, Path], overwrite: bool) -> None:
    base = output["base"].resolve()
    if base.exists() and overwrite:
        if base.name != "phase2b" or base.parent.name != "outputs":
            raise RuntimeError(f"Refusing to clear unexpected path: {base}")
        shutil.rmtree(base)
    for key in ("forecasts","tables","figures","logs","models","reports"):
        output[key].mkdir(parents=True, exist_ok=True)


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _protected_manifest(root: Path) -> pd.DataFrame:
    """Hash frozen Phase 1/2A inputs and scientific artifacts, excluding shared env snapshots."""
    files: list[Path] = []
    for relative in ("data/validated", "configs", "outputs/phase2a"):
        directory = root / relative
        if directory.exists(): files.extend(p for p in directory.rglob("*") if p.is_file())
    table_dir = root/"outputs"/"tables"
    if table_dir.exists(): files.extend(p for p in table_dir.rglob("*") if p.is_file() and p.name != "environment_versions.csv")
    phase1_log = root/"outputs"/"logs"/"phase1_audit.log"
    if phase1_log.is_file(): files.append(phase1_log)
    report_dir = root/"reports"
    if report_dir.exists(): files.extend(p for p in report_dir.glob("*") if p.is_file() and not p.name.startswith("PHASE2B_"))
    rows=[{"relative_path":p.relative_to(root).as_posix(),"sha256":sha256_file(p),"size_bytes":p.stat().st_size}
          for p in sorted(set(files))]
    return pd.DataFrame(rows)


def _hash_comparison(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    merged=before.merge(after,on="relative_path",how="outer",suffixes=("_before","_after"),indicator=True)
    merged["hash_match"]=(merged["_merge"]=="both") & (merged["sha256_before"]==merged["sha256_after"])
    return merged


def _environment(root: Path, output: dict[str, Path]) -> dict[str,str]:
    phase2b_dependencies=[*ANALYSIS_DEPENDENCIES,"scikit-learn","xgboost"]
    freeze=subprocess.run([sys.executable,"-m","pip","freeze"],check=True,capture_output=True,text=True,encoding="utf-8").stdout
    (output["logs"]/"phase2b_pip_freeze.txt").write_text(freeze,encoding="utf-8")
    (root/"requirements-phase2b-lock.txt").write_text(freeze,encoding="utf-8")
    versions={"python":platform.python_version()}
    for package in phase2b_dependencies:
        versions[package]=importlib.metadata.version(package)
    _write(pd.DataFrame([{"component":"python","version":platform.python_version()},
                         {"component":"platform","version":platform.platform()},
                         *[{"component":k,"version":v} for k,v in versions.items() if k != "python"]]),
           root/"outputs"/"tables"/"environment_versions.csv")
    _write(pd.DataFrame([{"component":k,"version":v} for k,v in versions.items()]), output["tables"]/"software_versions.csv")
    return versions


def _raw_diagnostics(raw: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for row in raw.itertuples(index=False):
        history=data.loc[data.year<=row.origin_year,row.target].to_numpy(dtype=float)
        minimum,maximum=float(history.min()),float(history.max())
        rows.append({**row._asdict(),"training_min":minimum,"training_max":maximum,
                     "actual_exceeds_training_max":bool(row.actual>maximum),
                     "actual_below_training_min":bool(row.actual<minimum),
                     "forecast_exceeds_training_max":bool(row.point_forecast>maximum),
                     "forecast_below_training_min":bool(row.point_forecast<minimum),
                     "ceiling_gap":float(max(0,row.point_forecast-maximum)),
                     "shortfall_below_floor":float(max(0,minimum-row.point_forecast))})
    return pd.DataFrame(rows)


def _run_independent_cell(payload):
    """Execute one target/horizon cell; every estimator still receives n_jobs=1."""
    data, manifest, config, target, horizon = payload
    runner=NestedMLRunner(data,manifest,config)
    ml,raw=runner.run()
    return {
        "target":target,"horizon":horizon,"ml":ml,"raw":raw,
        "candidates":runner.candidates,"selected":runner.selected,"calibration":runner.calibration,
        "trends":runner.trends,"scalers":runner.scalers,"timings":runner.timings,"failures":runner.failures,
    }


def run_phase2b(root: Path | None = None) -> dict[str, Any]:
    project=(root or Path(__file__).resolve().parents[1]).resolve()
    config=load_phase2b_config(project/"configs"/"phase2b_ml.yaml")
    data_path=project/config["data"]["path"]
    digest=sha256_file(data_path)
    if digest.lower()!=config["data"]["expected_sha256"].lower():
        raise ValueError(f"Validated-data SHA-256 mismatch: {digest}")
    required=[project/"outputs"/"phase2a"/"forecasts"/"baseline_forecasts_long.csv",
              project/"outputs"/"phase2a"/"tables"/"baseline_metrics_primary.csv"]
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise FileNotFoundError("Phase 2A artifacts required: "+", ".join(missing))
    before=_protected_manifest(project)
    output=_paths(project); _prepare(output,bool(config["runtime"]["overwrite_phase2b_outputs"]))
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        handlers=[logging.FileHandler(output["logs"]/"phase2b_ml.log",mode="w",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)
    LOGGER.info("Starting Phase 2B nested classical-ML evaluation")
    _write(before,output["tables"]/"phase1_phase2a_protected_hashes_before.csv")
    versions=_environment(project,output)
    data=_load_validated(data_path)
    manifest=generate_forecast_origin_manifest(data,config)
    _write(manifest,output["tables"]/"forecast_origin_manifest.csv")
    sample_manifest=build_sample_manifest(data,manifest,list(config["models"]),
                                          config["representations"]["primary"]+config["representations"]["diagnostic"],config["lookbacks"])
    _write(sample_manifest,output["tables"]/"supervised_sample_manifest.csv")
    payloads=[]
    for target in config["data"]["targets"]:
        for horizon in config["backtest"]["horizons"]:
            cell_manifest=manifest.loc[(manifest.target==target)&(manifest.horizon==horizon)].copy()
            payloads.append((data,cell_manifest,config,target,horizon))
    results=[]
    with ProcessPoolExecutor(max_workers=min(3,len(payloads))) as pool:
        for result in pool.map(_run_independent_cell,payloads):
            LOGGER.info("Completed nested cell target=%s horizon=%s",result["target"],result["horizon"])
            results.append(result)
    ml=pd.concat([result["ml"] for result in results],ignore_index=True).sort_values(
        ["analysis_window","target","model","horizon","origin_year"]).reset_index(drop=True)
    raw=pd.concat([result["raw"] for result in results],ignore_index=True).sort_values(
        ["analysis_window","target","model","horizon","origin_year"]).reset_index(drop=True)
    runner=SimpleNamespace(**{name:sum((result[name] for result in results),[])
                              for name in ("candidates","selected","calibration","trends","scalers","timings","failures")})
    if len(ml)!=546 or len(raw)!=546:
        raise RuntimeError(f"Expected 546 primary+recent records per family; got ML={len(ml)}, raw={len(raw)}")
    raw=_raw_diagnostics(raw,data)
    _write(ml,output["forecasts"]/"ml_forecasts_long.csv")
    _write(raw,output["forecasts"]/"raw_level_ablation_forecasts_long.csv")
    candidates=pd.DataFrame(runner.candidates); selected=pd.DataFrame(runner.selected)
    calibration=pd.DataFrame(runner.calibration); trends=pd.DataFrame(runner.trends)
    scalers=pd.DataFrame(runner.scalers); timings=pd.DataFrame(runner.timings)
    failures=pd.DataFrame(runner.failures,columns=["target","model","horizon","origin_year","error"])
    for frame,name in ((candidates,"nested_candidate_evaluations.csv"),(selected,"selected_configurations.csv"),
                       (calibration,"conformal_calibration_records.csv"),(trends,"outer_trend_coefficients.csv"),
                       (scalers,"svr_scaler_audit.csv"),(timings,"computational_timings.csv"),(failures,"model_fit_failures.csv")):
        _write(frame,output["tables"]/name)
    ml_metrics,ml_diag=aggregate_point_metrics(ml,manifest)
    raw_metrics,raw_metric_diag=aggregate_point_metrics(raw,manifest)
    interval_metrics=aggregate_interval_metrics(ml); interval_violations=validate_intervals(ml)
    _write(ml_metrics.loc[ml_metrics.analysis_window=="primary_2000_2023"],output["tables"]/"ml_metrics_primary.csv")
    _write(ml_metrics.loc[ml_metrics.analysis_window=="recent_2016_2023"],output["tables"]/"ml_metrics_recent_window.csv")
    _write(ml_diag,output["tables"]/"ml_metrics_diagnostics.csv")
    _write(raw_metrics,output["tables"]/"raw_level_ablation_metrics.csv")
    _write(raw_metric_diag,output["tables"]/"raw_level_ablation_metric_diagnostics.csv")
    _write(interval_metrics,output["tables"]/"ml_interval_metrics.csv")
    _write(interval_violations,output["tables"]/"interval_record_violations.csv")
    phase2a=pd.read_csv(required[0],float_precision="round_trip")
    combined=pd.concat([phase2a,ml],ignore_index=True,sort=False)
    combined_metrics,combined_diag=aggregate_point_metrics(combined,manifest)
    _write(combined,output["forecasts"]/"combined_phase2a_phase2b_forecasts_long.csv")
    _write(combined_metrics,output["tables"]/"combined_model_metrics.csv")
    _write(combined_diag,output["tables"]/"combined_model_metric_diagnostics.csv")
    baseline_metrics=aggregate_point_metrics(phase2a,manifest)[0]
    references=baseline_metrics.loc[baseline_metrics.groupby(["analysis_window","target","horizon"]).rmse.idxmin(),
                                    ["analysis_window","target","horizon","model"]].rename(columns={"model":"reference_model"})
    dm_a,boot_a=compare_models_to_fixed_references(combined,combined_metrics,references,list(config["models"]),config)
    dm_b,boot_b=compare_models(combined,combined_metrics,config)
    for frame,name in ((dm_a,"dm_ml_vs_best_phase2a.csv"),(boot_a,"bootstrap_ml_vs_best_phase2a.csv"),
                       (dm_b,"dm_all_vs_best_overall.csv"),(boot_b,"bootstrap_all_vs_best_overall.csv")):
        _write(frame,output["tables"]/name)
    rep_frequency=selected.groupby(["selection_family","target","model","horizon","representation"]).size().reset_index(name="selection_n")
    look_frequency=selected.groupby(["selection_family","target","model","horizon","lookback"]).size().reset_index(name="selection_n")
    timing_summary=timings.groupby(["selection_family","target","model","horizon"])[["feature_seconds","selection_seconds","fit_seconds","forecast_seconds","interval_seconds","total_seconds"]].agg(["sum","mean"]).reset_index()
    timing_summary.columns=["_".join(str(x) for x in col if x).rstrip("_") for col in timing_summary.columns]
    _write(rep_frequency,output["tables"]/"representation_selection_frequency.csv")
    _write(look_frequency,output["tables"]/"lookback_selection_frequency.csv")
    _write(timing_summary,output["tables"]/"computational_timing_summary.csv")
    mtc_summary=raw.loc[(raw.analysis_window=="primary_2000_2023")&(raw.target=="mtc")].groupby(["model","horizon"])[["actual_exceeds_training_max","forecast_exceeds_training_max","forecast_below_training_min","ceiling_gap","shortfall_below_floor"]].agg(["sum","mean"]).reset_index()
    mtc_summary.columns=["_".join(str(x) for x in col if x).rstrip("_") for col in mtc_summary.columns]
    _write(mtc_summary,output["tables"]/"mtc_raw_level_extrapolation_summary.csv")
    figures=generate_phase2b_figures(data,ml,raw,combined_metrics,interval_metrics,selected,timings,output["figures"])
    if len(figures)!=14: raise RuntimeError(f"Expected 14 Phase 2B figures, generated {len(figures)}")
    after=_protected_manifest(project); comparison=_hash_comparison(before,after)
    _write(after,output["tables"]/"phase1_phase2a_protected_hashes_after.csv")
    _write(comparison,output["tables"]/"phase1_phase2a_protected_hash_comparison.csv")
    if not comparison["hash_match"].all(): raise RuntimeError("A protected Phase 1/2A input or scientific artifact changed.")
    write_phase2b_reports(output["reports"],combined_metrics,interval_metrics,selected,raw,failures,
                          comparison,versions,candidates,interval_violations,dm_a,dm_b,timings)
    primary=ml.loc[ml.analysis_window=="primary_2000_2023"]
    summary={"data_hash_status":"PASS","primary_records":len(primary),"recent_records":len(ml)-len(primary),
             "raw_records":len(raw),"candidate_rows":len(candidates),"failures":len(failures),
             "figures":len(figures),"report":output["reports"]/"PHASE2B_ML_RESULTS.md"}
    print("\nPHASE 2B EXECUTION SUMMARY")
    print(f"Data hash status: PASS ({digest})")
    print(f"ML primary / recent records: {summary['primary_records']} / {summary['recent_records']}")
    print(f"Raw ablation records: {summary['raw_records']}; candidate evaluations: {summary['candidate_rows']}")
    print(f"Outer failures: {summary['failures']}; figures: {summary['figures']}")
    print("Non-rejection in paired tests is not evidence of equal performance; power is limited.")
    print(f"Main report: {summary['report']}")
    return summary


def main() -> int:
    try: run_phase2b()
    except Exception:
        logging.getLogger(__name__).exception("Phase 2B failed"); return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
