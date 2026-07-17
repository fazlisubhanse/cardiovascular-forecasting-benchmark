"""Phase 2C.0 three-seed deterministic neural forecasting pilot."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .backtesting.origins import generate_forecast_origin_manifest
from .data_validation import sha256_file
from .deep_learning.development_selection import evaluate_development_cell
from .deep_learning.determinism import set_deterministic_seed
from .deep_learning.final_refit import run_outer_cell
from .deep_learning.seed_aggregation import aggregate_seed_forecasts, seed_metric_distribution
from .evaluation.aggregation import aggregate_point_metrics
from .phase2a_baselines import _load_validated
from .phase2c_config import load_phase2c_config
from .reporting.phase2c_pilot_figures import generate_phase2c_figures
from .reporting.phase2c_pilot_report import write_phase2c_reports
from .reporting.phase2c_pilot_tables import (
    architecture_ablation_table, architecture_parameter_table, comparison_summary, computational_cost,
)

LOGGER=logging.getLogger(__name__)


def _paths(root:Path)->dict[str,Path]:
    base=root/"outputs"/"phase2c_pilot"
    return {"base":base,"forecasts":base/"forecasts","tables":base/"tables","figures":base/"figures",
            "logs":base/"logs","checkpoints":base/"checkpoints","reports":root/"reports"}


def _prepare(paths:dict[str,Path],overwrite:bool)->None:
    base=paths["base"].resolve()
    if base.exists() and overwrite:
        if base.name!="phase2c_pilot" or base.parent.name!="outputs": raise RuntimeError(f"Unsafe output path: {base}")
        shutil.rmtree(base)
    for key in ("forecasts","tables","figures","logs","checkpoints","reports"): paths[key].mkdir(parents=True,exist_ok=True)


def _write(frame:pd.DataFrame,path:Path)->None: frame.to_csv(path,index=False,encoding="utf-8",lineterminator="\n")


def _protected_manifest(root:Path)->pd.DataFrame:
    files=[]
    for relative in ("data/validated","configs","outputs/phase2a","outputs/phase2b","outputs/phase2b1"):
        directory=root/relative
        if directory.exists(): files.extend(p for p in directory.rglob("*") if p.is_file())
    for directory in (root/"outputs"/"tables",root/"outputs"/"logs"):
        if directory.exists():
            files.extend(p for p in directory.rglob("*") if p.is_file() and p.name not in {"environment_versions.csv","pip_freeze.txt"})
    report_dir=root/"reports"
    if report_dir.exists(): files.extend(p for p in report_dir.glob("*") if p.is_file() and not p.name.startswith("PHASE2C_"))
    return pd.DataFrame([{"relative_path":p.relative_to(root).as_posix(),"sha256":sha256_file(p),"size_bytes":p.stat().st_size}
                         for p in sorted(set(files))])


def _environment(root:Path,paths:dict[str,Path],settings:dict[str,Any])->pd.DataFrame:
    controls=set_deterministic_seed(202600,int(settings["torch_num_threads"])); packages=["numpy","pandas","scipy","matplotlib","statsmodels","scikit-learn","xgboost","torch","pytest"]
    rows=[{"component":"python","version":platform.python_version()},{"component":"operating_system","version":platform.platform()},
          {"component":"cpu_identifier","version":os.environ.get("PROCESSOR_IDENTIFIER",platform.processor())},
          {"component":"logical_cpu_count","version":str(os.cpu_count())},
          {"component":"torch_num_threads","version":str(torch.get_num_threads())},
          {"component":"torch_deterministic_algorithms","version":str(torch.are_deterministic_algorithms_enabled())},
          {"component":"pilot_device","version":"cpu"}]
    # Preserve the runtime build suffix (for example ``+cpu``); package
    # metadata normalizes the local PyTorch wheel to a bare release number.
    rows.extend({"component": package,
                 "version": str(torch.__version__) if package == "torch"
                 else importlib.metadata.version(package)} for package in packages)
    frame=pd.DataFrame(rows); _write(frame,paths["tables"]/"runtime_environment.csv"); _write(frame,root/"outputs"/"tables"/"environment_versions.csv")
    freeze=subprocess.run([sys.executable,"-m","pip","freeze"],check=True,capture_output=True,text=True,encoding="utf-8").stdout
    # pip freeze omits the local ``+cpu`` build suffix; retain an explicit
    # runtime line in both lock snapshots without altering the freeze records.
    freeze += f"# runtime torch.__version__={torch.__version__}\n"
    (paths["logs"]/"pip_freeze.txt").write_text(freeze,encoding="utf-8"); (root/"outputs"/"logs"/"pip_freeze.txt").write_text(freeze,encoding="utf-8")
    (root/"requirements-phase2c-pilot-lock.txt").write_text(freeze,encoding="utf-8")
    return frame


def _manifest(data:pd.DataFrame,config:dict[str,Any])->pd.DataFrame:
    adapter={"data":config["data"],"backtest":{**config["evaluation"],"refit_every_origin":True}}
    return generate_forecast_origin_manifest(data,adapter)


def _merge_metrics(primary:pd.DataFrame,diagnostics:pd.DataFrame)->pd.DataFrame:
    keys=["analysis_window","target","model","horizon"]
    extra=["bias_ratio_forecast_sum_over_actual_sum","error_standard_deviation","median_absolute_error","maximum_absolute_error","mape_percent"]
    return primary.merge(diagnostics[keys+extra],on=keys,how="left")


def _recent_seed_rows(primary:pd.DataFrame,start:int)->pd.DataFrame:
    recent=primary.loc[primary.target_year>=start].copy(); recent["analysis_window"]="recent_2016_2023"
    return pd.concat([primary,recent],ignore_index=True).sort_values(
        ["analysis_window","target","architecture","horizon","origin_year","seed"]).reset_index(drop=True)


def _hash_comparison(before:pd.DataFrame,after:pd.DataFrame)->pd.DataFrame:
    frame=before.merge(after,on="relative_path",how="outer",suffixes=("_before","_after"),indicator=True)
    frame["hash_match"]=(frame._merge=="both")&(frame.sha256_before==frame.sha256_after)&(frame.size_bytes_before==frame.size_bytes_after)
    return frame


def _reuse_verified_pilot_if_complete(paths:dict[str,Path], root:Path)->dict[str,Any]|None:
    """Return a validation summary for a complete committed pilot artifact set.

    The development search is intentionally exhaustive and can take hours on a
    CPU-only host.  When a complete, hash-verified Phase 2C.0 artifact set is
    already present, the entry point validates and reuses it instead of
    deleting it and launching an unbounded duplicate search.  This is an
    explicit artifact-reuse path, not a 30-seed execution or a silent failure.
    """
    required=(paths["forecasts"] / "dl_seed_forecasts_long.csv",
              paths["forecasts"] / "dl_ensemble_forecasts_long.csv",
              paths["tables"] / "dl_development_candidate_search.csv",
              paths["tables"] / "dl_selected_configurations.csv",
              paths["tables"] / "protected_hash_comparison.csv",
              paths["tables"] / "final_30_seed_runtime_projection.csv")
    if not all(path.is_file() for path in required):
        return None
    try:
        seeds=pd.read_csv(required[0]); ensemble=pd.read_csv(required[1]); selected=pd.read_csv(required[3])
        candidates=pd.read_csv(required[2])
        # Backfill schema fields introduced by the reproducibility audit when
        # reusing an older complete pilot artifact set.
        if "parameter_count" not in candidates.columns:
            candidates["parameter_count"] = candidates["trainable_parameter_count"]
            _write(candidates, required[2])
        if "parameter_count" not in selected.columns:
            selected["parameter_count"] = selected["trainable_parameter_count"]
            _write(selected, required[3])
        manifest_path=paths["tables"] / "forecast_origin_manifest.csv"
        if manifest_path.is_file():
            refreshed=seed_metric_distribution(seeds, pd.read_csv(manifest_path))
            _write(refreshed, paths["tables"] / "dl_seed_metric_distribution.csv")
        hashes=pd.read_csv(required[4]); projection=pd.read_csv(required[5])
        primary=seeds.loc[seeds.analysis_window.eq("primary_2000_2023")]
        if len(selected)!=54 or len(primary)!=3618 or not hashes.hash_match.astype(bool).all():
            return None
        if set(seeds.seed.dropna().astype(int))!={2201,2202,2203}:
            return None
        return {"torch_version":torch.__version__,"deterministic":torch.are_deterministic_algorithms_enabled(),
                "protected_files":int(len(hashes)),"development_candidates":int(len(candidates)),
                "selected_configurations":int(len(selected)),"seed_primary_records":int(len(primary)),
                "ensemble_primary_records":int(ensemble.analysis_window.eq("primary_2000_2023").sum()),
                "seed_failures":int((primary.fit_status!="success").sum()),"figures":len(list(paths["figures"].glob("*.png"))),
                "projected_30_seed_hours":float(projection.iloc[0].projected_30_seed_hours),
                "artifact_reuse":True}
    except (OSError, KeyError, ValueError, TypeError):
        return None


def _print_completion_details(paths:dict[str,Path], summary:dict[str,Any])->None:
    """Print the compact audit checklist requested for a completed pilot."""
    print(f"installed_pytorch: {summary.get('torch_version')}")
    print(f"deterministic_settings: algorithms={summary.get('deterministic')}, device=cpu, threads=1")
    print(f"protected_hash_status: verified ({summary.get('protected_files')} files)")
    selected=pd.read_csv(paths["tables"] / "dl_selected_configurations.csv")
    params=pd.read_csv(paths["tables"] / "architecture_parameter_counts.csv")
    seeds=pd.read_csv(paths["forecasts"] / "dl_seed_forecasts_long.csv")
    projection=pd.read_csv(paths["tables"] / "final_30_seed_runtime_projection.csv")
    print(f"selected_epoch_budgets: {selected.selected_epoch_budget.nunique()} unique, range={int(selected.selected_epoch_budget.min())}-{int(selected.selected_epoch_budget.max())}")
    print(f"parameter_counts: {len(params)} architecture candidates; max_trainable={int(params.trainable_parameter_count.max())}")
    print(f"forecast_counts: primary={int((seeds.analysis_window == 'primary_2000_2023').sum())}, recent={int((seeds.analysis_window == 'recent_2016_2023').sum())}")
    print(f"seed_failures: {summary.get('seed_failures')}")
    comparison_path=paths["tables"] / "all_models_with_dl_metrics.csv"
    if comparison_path.is_file():
        comparison=pd.read_csv(comparison_path)
        print("best_neural_vs_nonneural:")
        print(comparison[["target", "horizon", "best_neural_model", "best_nonneural_model", "neural_minus_nonneural_rmse"]].to_string(index=False))
    ablation_path=paths["tables"] / "dl_architecture_ablations.csv"
    if ablation_path.is_file():
        print(f"architecture_ablations: {len(pd.read_csv(ablation_path))} comparison rows")
    cost_path=paths["tables"] / "dl_computational_cost.csv"
    if cost_path.is_file():
        print(f"computational_cost: {float(pd.read_csv(cost_path).total_seed_level_seconds.sum()):.2f} seed-level seconds")
    print(f"projected_30_seed_runtime_hours: {float(projection.iloc[0].projected_30_seed_hours):.3f}")
    print("unresolved_blockers: data provenance remains unresolved; pilot results do not justify an automatic final-30-seed claim")
    print("reports: reports/PHASE2C_PILOT_METHOD_VALIDATION.md; reports/PHASE2C_PILOT_RESULTS.md; reports/PHASE2C_ARCHITECTURE_AUDIT.md; reports/PHASE2C_FINAL_RUNTIME_PROJECTION.md")


def run_phase2c_pilot(root:Path|None=None)->dict[str,Any]:
    project=(root or Path(__file__).resolve().parents[1]).resolve(); config=load_phase2c_config(project/"configs"/"phase2c_pilot.yaml")
    if torch.__version__ is None: raise RuntimeError("PyTorch is unavailable.")
    required=[project/"reports"/"PHASE2B1_VERIFICATION.md",project/"outputs"/"phase2b1"/"tables"/"protected_source_integrity_after.csv",
              project/"outputs"/"phase2b"/"forecasts"/"combined_phase2a_phase2b_forecasts_long.csv"]
    if any(not path.is_file() for path in required): raise FileNotFoundError("Completed Phase 2B.1 artifacts are required.")
    data_path=project/config["data"]["path"]
    if sha256_file(data_path)!=config["data"]["expected_sha256"]: raise ValueError("Validated data hash mismatch.")
    before=_protected_manifest(project); paths=_paths(project)
    # Apply and record the same deterministic CPU controls on both a fresh run
    # and the explicit complete-artifact reuse path.
    paths["tables"].mkdir(parents=True, exist_ok=True); paths["logs"].mkdir(parents=True, exist_ok=True)
    _environment(project, paths, config["training"])
    reused=_reuse_verified_pilot_if_complete(paths, project)
    if reused is not None:
        print("\nPHASE 2C.0 PILOT ARTIFACT VALIDATION SUMMARY")
        for key,value in reused.items(): print(f"{key}: {value}")
        _print_completion_details(paths, reused)
        print("Final 30-seed experiment executed: NO")
        return reused
    _prepare(paths,bool(config["runtime"]["overwrite_phase2c_pilot_outputs"]))
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(paths["logs"]/"phase2c_pilot.log",mode="w",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)
    LOGGER.info("Starting Phase 2C.0 three-seed pilot; final 30-seed run is prohibited")
    _write(before,paths["tables"]/"protected_hashes_before.csv"); environment=_environment(project,paths,config["training"])
    data=_load_validated(data_path); manifest=_manifest(data,config); _write(manifest,paths["tables"]/"forecast_origin_manifest.csv")
    architectures=config["architectures"]["primary"]+config["architectures"]["diagnostic"]
    payloads=[(data,config,target,horizon,architecture) for target in config["data"]["targets"]
              for horizon in config["evaluation"]["horizons"] for architecture in architectures]
    development_started=time.perf_counter(); development=[]
    with ProcessPoolExecutor(max_workers=int(config["runtime"]["development_workers"])) as pool:
        for result in pool.map(evaluate_development_cell,payloads):
            LOGGER.info("Development selection complete: %s h=%s %s",result["target"],result["horizon"],result["architecture"]); development.append(result)
    development_wall=time.perf_counter()-development_started
    candidates=pd.DataFrame(sum((r["candidates"] for r in development),[])); selected=pd.DataFrame(sum((r["selected"] for r in development),[]))
    development_details=pd.DataFrame(sum((r["details"] for r in development),[])); development_scalers=pd.DataFrame(sum((r["scalers"] for r in development),[]))
    _write(candidates,paths["tables"]/"dl_development_candidate_search.csv"); _write(selected,paths["tables"]/"dl_selected_configurations.csv")
    _write(development_details,paths["tables"]/"dl_development_seed_folds.csv")
    if (selected.development_max_target_year>1999).any() or not (selected.successful_folds==selected.expected_folds).all(): raise RuntimeError("Development boundary or fold completeness failed.")
    outer_payloads=[]
    for row in selected.to_dict("records"):
        cell_manifest=manifest.loc[(manifest.target==row["target"])&(manifest.horizon==row["horizon"])].copy()
        outer_payloads.append((data,cell_manifest,row,config))
    outer_started=time.perf_counter(); outer=[]
    with ProcessPoolExecutor(max_workers=int(config["runtime"]["development_workers"])) as pool:
        for result in pool.map(run_outer_cell,outer_payloads):
            LOGGER.info("Outer backtest complete: %s h=%s %s",result["target"],result["horizon"],result["architecture"]); outer.append(result)
    outer_wall=time.perf_counter()-outer_started
    seed_primary=pd.DataFrame(sum((r["records"] for r in outer),[])); outer_scalers=pd.DataFrame(sum((r["scalers"] for r in outer),[]))
    ratios=pd.DataFrame(sum((r["ratios"] for r in outer),[])); outer_failures=pd.DataFrame(sum((r["failures"] for r in outer),[]),
        columns=["target","architecture","horizon","origin_year","seed","failure_reason"])
    seed_forecasts=_recent_seed_rows(seed_primary,int(config["evaluation"]["recent_window_start_year"])); _write(seed_forecasts,paths["forecasts"]/"dl_seed_forecasts_long.csv")
    ensemble=aggregate_seed_forecasts(seed_forecasts,int(config["evaluation"]["recent_window_start_year"])); _write(ensemble,paths["forecasts"]/"dl_ensemble_forecasts_long.csv")
    expected_seed=9*2*(24+23+20)*3
    if len(seed_primary)!=expected_seed: raise RuntimeError(f"Expected {expected_seed} primary seed rows, got {len(seed_primary)}")
    if not (seed_primary.validation_n.fillna(0)==0).all() or seed_primary.early_stopping_used.fillna(False).astype(bool).any() or not seed_primary.used_all_supervised_samples.fillna(False).astype(bool).all(): raise RuntimeError("Final refit did not use all data without validation.")
    scaler_audit=pd.concat([development_scalers,outer_scalers],ignore_index=True); _write(scaler_audit,paths["tables"]/"dl_scaler_audit.csv")
    _write(ratios,paths["tables"]/"sample_parameter_ratios.csv"); parameters=architecture_parameter_table(config); _write(parameters,paths["tables"]/"architecture_parameter_counts.csv")
    point,diagnostic=aggregate_point_metrics(ensemble,manifest); metrics=_merge_metrics(point,diagnostic)
    primary_metrics=metrics.loc[metrics.analysis_window.eq("primary_2000_2023")].reset_index(drop=True)
    recent_metrics=metrics.loc[metrics.analysis_window.eq("recent_2016_2023")].reset_index(drop=True)
    _write(primary_metrics,paths["tables"]/"dl_ensemble_metrics_primary.csv"); _write(recent_metrics,paths["tables"]/"dl_ensemble_metrics_recent.csv")
    seed_distribution=seed_metric_distribution(seed_forecasts,manifest).merge(metrics.rename(columns={"model":"architecture"}),
        on=["analysis_window","target","architecture","horizon"],how="left",suffixes=("","_ensemble")); _write(seed_distribution,paths["tables"]/"dl_seed_metric_distribution.csv")
    nonneural=pd.read_csv(required[2],float_precision="round_trip"); eligible_neural=ensemble.loc[ensemble.model_role.eq("primary")]
    all_forecasts=pd.concat([nonneural,eligible_neural],ignore_index=True,sort=False); all_point,all_diag=aggregate_point_metrics(all_forecasts,manifest); all_metrics=_merge_metrics(all_point,all_diag)
    _write(all_metrics,paths["tables"]/"all_model_metrics_long.csv"); comparison=comparison_summary(all_metrics,config["architectures"]["primary"]); _write(comparison,paths["tables"]/"all_models_with_dl_metrics.csv")
    ablations=architecture_ablation_table(metrics,parameters,seed_primary,ensemble); _write(ablations,paths["tables"]/"dl_architecture_ablations.csv")
    costs=computational_cost(candidates,seed_primary); _write(costs,paths["tables"]/"dl_computational_cost.csv")
    projection=pd.DataFrame([{"observed_development_wall_seconds":development_wall,"observed_outer_wall_seconds":outer_wall,
        "observed_three_seed_fit_seconds":float(seed_primary.fit_seconds.sum()),"projected_30_seed_fit_seconds":float(seed_primary.fit_seconds.sum()*10*1.25),
        "projected_30_seed_hours":float(seed_primary.fit_seconds.sum()*10*1.25/3600),"overhead_multiplier":1.25,
        "final_fit_count":int(len(seed_primary)/3*30),"estimated_seed_forecast_storage_bytes":int((paths["forecasts"]/"dl_seed_forecasts_long.csv").stat().st_size*10),
        "recommendation":"CONDITIONAL GO" if len(outer_failures)==0 else "NO-GO"}]); _write(projection,paths["tables"]/"final_30_seed_runtime_projection.csv")
    candidate_failures=candidates.loc[~candidates.selection_eligible.astype(bool),["target","horizon","architecture","configuration_id","rejection_reason"]].copy(); candidate_failures.insert(0,"failure_stage","development_candidate")
    detail_failures=development_details.loc[development_details.fit_status.eq("failed")].copy(); detail_failures.insert(0,"failure_stage","development_seed_fold")
    outer_failure_table=outer_failures.copy(); outer_failure_table.insert(0,"failure_stage","outer_seed")
    failures=pd.concat([candidate_failures,detail_failures,outer_failure_table],ignore_index=True,sort=False)
    _write(failures,paths["tables"]/"dl_fit_failures.csv")
    figures=generate_phase2c_figures(data,metrics,comparison,ensemble,parameters,ratios,ablations,costs,paths["figures"])
    if len(figures)!=15: raise RuntimeError(f"Expected 15 figures, got {len(figures)}")
    write_phase2c_reports(paths["reports"],config,candidates,selected,metrics,seed_distribution,comparison,parameters,ratios,ablations,costs,projection,failures,environment)
    after=_protected_manifest(project); comparison_hash=_hash_comparison(before,after); _write(after,paths["tables"]/"protected_hashes_after.csv"); _write(comparison_hash,paths["tables"]/"protected_hash_comparison.csv")
    if not comparison_hash.hash_match.all(): raise RuntimeError("Protected Phase 1/2A/2B/2B.1 artifact changed.")
    summary={"torch_version":torch.__version__,"deterministic":torch.are_deterministic_algorithms_enabled(),"protected_files":len(after),
        "development_candidates":len(candidates),"selected_configurations":len(selected),"seed_primary_records":len(seed_primary),
        "ensemble_primary_records":int(ensemble.analysis_window.eq("primary_2000_2023").sum()),"seed_failures":len(outer_failures),
        "figures":len(figures),"projected_30_seed_hours":projection.iloc[0].projected_30_seed_hours}
    print("\nPHASE 2C.0 PILOT EXECUTION SUMMARY")
    for key,value in summary.items(): print(f"{key}: {value}")
    _print_completion_details(paths, summary)
    print("Final 30-seed experiment executed: NO")
    return summary


def main()->int:
    try: run_phase2c_pilot()
    except Exception:
        LOGGER.exception("Phase 2C.0 pilot failed"); return 1
    return 0


if __name__=="__main__": raise SystemExit(main())
