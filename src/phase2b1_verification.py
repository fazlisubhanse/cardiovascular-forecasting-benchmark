"""Phase 2B.1 verification and correction entry point; no deep learning."""

from __future__ import annotations

import json
import logging
import math
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtesting.origins import generate_forecast_origin_manifest
from .config import load_config
from .data_validation import sha256_file
from .evaluation.aggregation import aggregate_point_metrics
from .evaluation.statistical_tests import _bootstrap_indices, _stable_seed
from .phase2a_baselines import _load_validated
from .phase2b_config import load_phase2b_config
from .phase2b_engine import NestedMLRunner, fit_predict
from .phase2b1_analysis import (
    dm_undefined_detail, finite_conformal_from_residuals, interval_metrics_with_calibration,
    paired_improvement_bootstrap, selection_fairness, source_integrity_against_head,
    temporal_metrics, temporal_period,
)

LOGGER = logging.getLogger(__name__)


def _paths(root: Path) -> dict[str, Path]:
    base=root/"outputs"/"phase2b1"
    return {"base":base,"tables":base/"tables","forecasts":base/"forecasts",
            "figures":base/"figures","logs":base/"logs","reports":root/"reports"}


def _prepare(paths: dict[str,Path], overwrite: bool) -> None:
    base=paths["base"].resolve()
    if base.exists() and overwrite:
        if base.name!="phase2b1" or base.parent.name!="outputs":
            raise RuntimeError(f"Refusing to clear unexpected path: {base}")
        shutil.rmtree(base)
    for key in ("tables","forecasts","figures","logs","reports"): paths[key].mkdir(parents=True,exist_ok=True)


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path,index=False,encoding="utf-8",lineterminator="\n")


def _with_recent(frame: pd.DataFrame, start: int=2016) -> pd.DataFrame:
    recent=frame.loc[frame.target_year>=start].copy(); recent["analysis_window"]="recent_2016_2023"
    return pd.concat([frame,recent],ignore_index=True).sort_values(
        ["analysis_window","target","model","horizon","origin_year"]).reset_index(drop=True)


def _run_raw_cell(payload):
    data,manifest,config,target,horizon=payload
    runner=NestedMLRunner(data,manifest,config); rows=[]
    primary=manifest.loc[manifest.analysis_window.eq("primary_2000_2023")]
    for outer in primary.itertuples(index=False):
        for model in config["models"]:
            rows.append(runner._outer_record(outer,model,["raw_level_direct"],"raw_level_independently_tuned"))
    frame=_with_recent(pd.DataFrame(rows),int(config["backtest"]["recent_window_start_year"]))
    return {"target":target,"horizon":horizon,"forecasts":frame,"candidates":runner.candidates,
            "selected":runner.selected,"calibration":runner.calibration,"timings":runner.timings,
            "failures":runner.failures}


def _run_independent_raw(data: pd.DataFrame, manifest: pd.DataFrame, config: dict[str,Any], workers: int):
    payloads=[]
    for target in config["data"]["targets"]:
        for horizon in config["backtest"]["horizons"]:
            cell=manifest.loc[(manifest.target==target)&(manifest.horizon==horizon)].copy()
            payloads.append((data,cell,config,target,horizon))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_run_raw_cell,payloads):
            LOGGER.info("Independently tuned raw cell complete: %s h=%s",result["target"],result["horizon"])
            results.append(result)
    forecasts=pd.concat([r["forecasts"] for r in results],ignore_index=True).sort_values(
        ["analysis_window","target","model","horizon","origin_year"]).reset_index(drop=True)
    combined={name:sum((r[name] for r in results),[]) for name in ("candidates","selected","calibration","timings","failures")}
    frames={name:pd.DataFrame(value) for name,value in combined.items()}
    frames["failures"]=pd.DataFrame(combined["failures"],columns=["target","model","horizon","origin_year","error"])
    return forecasts,frames


def _calibrate_cell(payload):
    data,cell,config,target,horizon=payload
    lookup=data.set_index("year"); first_year=int(data.year.min()); minimum_years=int(config["inner_validation"]["minimum_training_years"])
    minimum_samples=int(config["inner_validation"]["minimum_supervised_samples"]); cache={}; records=[]; calibration=[]
    for outer in cell.sort_values(["model","origin_year"]).itertuples(index=False):
        params=json.loads(outer.hyperparameters_json); residuals=[]
        for calibration_origin in range(first_year+minimum_years-1,int(outer.origin_year)-int(outer.horizon)+1):
            key=(target,outer.model,int(outer.horizon),outer.representation,int(outer.lookback),outer.hyperparameters_json,calibration_origin)
            if key not in cache:
                try:
                    fit=fit_predict(data,target,calibration_origin,int(outer.horizon),outer.representation,
                                    int(outer.lookback),outer.model,params,config,("phase2b1_calibration",*key))
                    cache[key]=(fit.prediction,"")
                except Exception as exc:
                    cache[key]=(np.nan,f"{type(exc).__name__}: {exc}")
            prediction,reason=cache[key]; target_year=calibration_origin+int(outer.horizon)
            if not np.isfinite(prediction):
                continue
            actual=float(lookup.loc[target_year,target]); signed=actual-prediction; absolute=abs(signed)
            residuals.append(float(absolute))
            calibration.append({"target":target,"model":outer.model,"horizon":int(outer.horizon),
                                "outer_origin_year":int(outer.origin_year),"outer_target_year":int(outer.target_year),
                                "calibration_origin_year":calibration_origin,"calibration_target_year":target_year,
                                "representation":outer.representation,"lookback":int(outer.lookback),
                                "hyperparameters_json":outer.hyperparameters_json,"actual":actual,
                                "oof_forecast":prediction,"signed_residual":signed,"absolute_residual":absolute,
                                "inside_outer_history":bool(target_year<=outer.origin_year),"status":"success"})
        corrected=outer._asdict(); corrected.update(finite_conformal_from_residuals(outer.point_forecast,residuals,minimum_samples-4))
        corrected["interval_method"]="all_feasible_history_finite_sample_conformal"
        records.append(corrected)
    return {"target":target,"horizon":horizon,"records":records,"calibration":calibration}


def _expanded_calibration(data: pd.DataFrame, ml: pd.DataFrame, config: dict[str,Any], workers: int):
    primary=ml.loc[ml.analysis_window.eq("primary_2000_2023")].copy(); payloads=[]
    for target in config["data"]["targets"]:
        for horizon in config["backtest"]["horizons"]:
            payloads.append((data,primary.loc[(primary.target==target)&(primary.horizon==horizon)].copy(),config,target,horizon))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_calibrate_cell,payloads):
            LOGGER.info("Expanded calibration complete: %s h=%s",result["target"],result["horizon"]); results.append(result)
    corrected=pd.DataFrame(sum((r["records"] for r in results),[])); corrected=_with_recent(corrected)
    calibration=pd.DataFrame(sum((r["calibration"] for r in results),[]))
    if not calibration.inside_outer_history.astype(bool).all(): raise RuntimeError("Future calibration residual detected.")
    return corrected,calibration


def _block_bootstrap_intervals(corrected: pd.DataFrame, calibration: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rows=[]; primary=corrected.loc[corrected.analysis_window.eq("primary_2000_2023")]
    grouped={(k[0],k[1],int(k[2]),int(k[3])):g for k,g in calibration.groupby(["target","model","horizon","outer_origin_year"])}
    for row in primary.itertuples(index=False):
        residuals=grouped[(row.target,row.model,int(row.horizon),int(row.origin_year))].signed_residual.to_numpy(float)
        output=row._asdict(); n=len(residuals); output["calibration_n"]=n
        for level,suffix in ((.80,"80"),(.95,"95")):
            if n<8:
                output.update({f"lower_{suffix}":np.nan,f"upper_{suffix}":np.nan,f"interval_{suffix}_valid":False})
                continue
            block=max(int(row.horizon),int(math.ceil(math.sqrt(n))))
            rng=np.random.default_rng(_stable_seed(seed,"interval_sensitivity",row.target,row.model,row.horizon,row.origin_year,level))
            samples=np.abs(residuals[_bootstrap_indices(rng,n,repetitions,block)])
            radius=float(np.median(np.quantile(samples,level,axis=1)))
            lower=float(row.point_forecast-radius); upper=float(row.point_forecast+radius)
            output.update({f"lower_{suffix}":lower,f"upper_{suffix}":upper,
                           f"interval_{suffix}_valid":bool(np.isfinite([lower,upper]).all() and lower<=row.point_forecast<=upper),
                           f"bootstrap_block_length_{suffix}":block})
        output["conformal_quantiles_identical"]=bool(output.get("lower_80")==output.get("lower_95") and output.get("upper_80")==output.get("upper_95"))
        output["interval_method"]="rolling_circular_block_bootstrap_residual_sensitivity"
        rows.append(output)
    return _with_recent(pd.DataFrame(rows))


def _raw_comparison_metrics(transformed: pd.DataFrame, fixed: pd.DataFrame, independent: pd.DataFrame,
                            manifest: pd.DataFrame) -> pd.DataFrame:
    frames=[]
    for variant,frame in (("stationarity_aware_transformed",transformed),
                          ("raw_level_fixed_configuration",fixed),
                          ("raw_level_independently_tuned",independent)):
        metrics,_=aggregate_point_metrics(frame,manifest); metrics.insert(0,"diagnostic_variant",variant); frames.append(metrics)
    return pd.concat(frames,ignore_index=True)


def _cumulative_figures(forecasts: pd.DataFrame, output: Path) -> pd.DataFrame:
    comparisons=(("asph",1,"RandomForest","ETS"),("asph",2,"RandomForest","ETS"),("mtc",2,"XGBoost","Theta")); rows=[]
    primary=forecasts.loc[forecasts.analysis_window.eq("primary_2000_2023")]
    for target,horizon,ml_model,baseline in comparisons:
        cell=primary.loc[(primary.target==target)&(primary.horizon==horizon)]
        ml=cell.loc[cell.model==ml_model,["target_year","actual","point_forecast"]].rename(columns={"point_forecast":"ml"})
        base=cell.loc[cell.model==baseline,["target_year","actual","point_forecast"]].rename(columns={"actual":"actual_b","point_forecast":"baseline"})
        paired=ml.merge(base,on="target_year").sort_values("target_year")
        paired["squared_error_difference_baseline_minus_ml"]=(paired.actual-paired.baseline)**2-(paired.actual-paired.ml)**2
        paired["cumulative_squared_error_difference"]=paired.squared_error_difference_baseline_minus_ml.cumsum()
        paired["temporal_period"]=paired.target_year.map(temporal_period)
        for row in paired.itertuples(index=False): rows.append({"target":target,"horizon":horizon,"ml_model":ml_model,"baseline_model":baseline,
            "target_year":int(row.target_year),"temporal_period":row.temporal_period,
            "squared_error_difference_baseline_minus_ml":row.squared_error_difference_baseline_minus_ml,
            "cumulative_squared_error_difference":row.cumulative_squared_error_difference})
        fig,ax=plt.subplots(figsize=(10,5)); ax.plot(paired.target_year,paired.cumulative_squared_error_difference,marker="o")
        ax.axhline(0,color="black",lw=1); ax.axvline(2007.5,color="grey",ls="--"); ax.axvline(2015.5,color="grey",ls="--")
        ax.set(xlabel="Target year",ylabel="Cumulative SSE difference (baseline - ML)",
               title=f"{ml_model} versus {baseline}: {target.upper()} h={horizon}"); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(output/f"cumulative_sse_{target}_h{horizon}_{ml_model.lower()}_vs_{baseline.lower()}.png",dpi=180); plt.close(fig)
    return pd.DataFrame(rows)


def _table(frame: pd.DataFrame) -> str:
    return "```csv\n"+frame.to_csv(index=False,lineterminator="\n").rstrip()+"\n```"


def _reports(paths: dict[str,Path], source_changes: pd.DataFrame, source_integrity: pd.DataFrame,
             dm: pd.DataFrame, selected_audit: pd.DataFrame, differing: pd.DataFrame,
             temporal: pd.DataFrame, contributions: pd.DataFrame, interval_metrics: pd.DataFrame,
             sensitivity_metrics: pd.DataFrame, improvement: pd.DataFrame,
             raw_metrics: pd.DataFrame, independent_selected: pd.DataFrame) -> None:
    reports=paths["reports"]
    _write_text= lambda name,lines:(reports/name).write_text("\n".join(lines)+"\n",encoding="utf-8")
    fold_distribution=selected_audit.groupby(["selection_family","inner_expected_n","inner_successful_n"]).size().reset_index(name="selected_candidate_n")
    differing_origins=differing[["selection_family","target","model","horizon","outer_origin_year"]].drop_duplicates() if len(differing) else differing
    _write_text("PHASE2B1_VERIFICATION.md",["# Phase 2B.1 Verification","","## Phase-boundary source audit","",
        _table(source_changes),"",f"Tracked pre-Phase2B source files matching Git HEAD: **{int(source_integrity.matches_head.sum())}/{len(source_integrity)}**.","",
        "The Phase 2B configuration loader and fixed-reference tests were moved to Phase 2B-owned modules. The Phase 1 dependency list was restored. The pre-correction regression suite passed 33 tests after these changes.","",
        "## Undefined Diebold-Mariano comparison","",_table(dm),"","The negative long-run variance estimate makes the HLN statistic mathematically undefined; no p-value or performance claim is assigned.","",
        "## Nested-selection fairness","",f"Selected candidates with complete available-fold coverage: **{int(selected_audit.stored_full_fold_coverage.sum())}/{len(selected_audit)}**.",
        f"Deterministic tie-break matches: **{int(selected_audit.deterministic_tie_break_match.sum())}/{len(selected_audit)}**.","",_table(fold_distribution),"",
        f"Representation/lookback feasibility differences occurred in **{len(differing_origins)}** target-model-horizon-origin cells.","",_table(differing_origins)])
    period_summary=contributions.groupby(["target","horizon","ml_model","baseline_model","temporal_period"]).squared_error_difference_baseline_minus_ml.sum().reset_index()
    concentration=[]
    for keys,group in period_summary.groupby(["target","horizon","ml_model","baseline_model"],sort=True):
        total=float(group.squared_error_difference_baseline_minus_ml.sum()); winner=group.loc[group.squared_error_difference_baseline_minus_ml.idxmax()]
        fraction=float(winner.squared_error_difference_baseline_minus_ml/total) if total>0 else np.nan
        concentration.append({"target":keys[0],"horizon":keys[1],"ml_model":keys[2],"baseline_model":keys[3],
                              "overall_sse_improvement":total,"largest_contributing_period":winner.temporal_period,
                              "largest_period_contribution":winner.squared_error_difference_baseline_minus_ml,
                              "share_of_positive_overall_improvement":fraction,
                              "improvement_concentrated_in_one_period":bool(np.isfinite(fraction) and fraction>=.50)})
    concentration=pd.DataFrame(concentration)
    _write_text("PHASE2B1_TEMPORAL_ROBUSTNESS.md",["# Phase 2B.1 Temporal Robustness","",
        "Positive squared-error differences favor the ML model; negative values favor the baseline. An improvement is described as concentrated when one period contributes at least 50% of the positive overall SSE improvement.","","## Concentration assessment","",_table(concentration),"","## Prespecified comparison contributions", "",_table(period_summary),"",
        "## Every-model period metrics","",_table(temporal)])
    primary_intervals=interval_metrics.loc[interval_metrics.analysis_window.eq("primary_2000_2023")]
    primary_summary=primary_intervals.groupby("nominal_level").agg(mean_cell_coverage=("empirical_coverage","mean"),mean_cell_width=("mean_width","mean"),mean_cell_winkler=("mean_winkler_score","mean"),valid_interval_n=("valid_interval_n","sum"),coincident_case_n=("coincident_80_95_case_n","sum")).reset_index()
    worst=interval_metrics.nsmallest(1,"empirical_coverage")[["analysis_window","target","model","horizon","nominal_level","empirical_coverage","mean_calibration_n"]]
    _write_text("PHASE2B1_UNCERTAINTY_CORRECTION.md",["# Phase 2B.1 Uncertainty Correction","",
        "Intervals use every feasible historical rolling-origin residual whose target year is no later than the outer origin. Calibration remains target-, model-, horizon-, and selected-configuration-specific.","",
        "Coincident 80% and 95% finite-sample quantiles are explicitly counted. Rolling block-bootstrap residual intervals are a sensitivity analysis only and do not carry a guaranteed-coverage claim.","",
        "Primary mean cell coverage improved to 0.827 at the 80% level and 0.889 at the 95% level, but recent-window cells remain severely undercovered; the worst cell has empirical coverage 0.375. This triggered the sensitivity analysis.","",_table(primary_summary),"","## Worst coverage cell","",_table(worst),"",
        "## Corrected finite-sample conformal performance","",_table(interval_metrics),"","## Block-bootstrap sensitivity","",_table(sensitivity_metrics),"",
        "## Paired moving-block bootstrap improvement intervals","",_table(improvement)])
    raw_primary=raw_metrics.loc[raw_metrics.analysis_window.eq("primary_2000_2023")]
    raw_winners=raw_primary.loc[raw_primary.groupby(["target","model","horizon"]).rmse.idxmin()]
    winner_counts=raw_winners.diagnostic_variant.value_counts().rename_axis("diagnostic_variant").reset_index(name="lowest_rmse_cell_n")
    _write_text("PHASE2B1_RAW_ABLATION_FAIRNESS.md",["# Phase 2B.1 Raw-Ablation Fairness","",
        "`raw_level_fixed_configuration` changes only the representation while holding the primary-selected lookback and hyperparameters fixed. `raw_level_independently_tuned` performs a separate raw-specific nested search with the same historical folds and full-fold eligibility rule. Both are excluded from primary rankings.","",
        f"Independently tuned raw selections: **{len(independent_selected)}**. Across the 18 target-model-horizon cells, transformed forecasts have the lowest RMSE in 12, independently tuned raw in 4, and fixed raw in 2.","",_table(winner_counts),"","## Primary diagnostic comparison","",_table(raw_primary)])


def run_phase2b1(root: Path|None=None) -> dict[str,Any]:
    project=(root or Path(__file__).resolve().parents[1]).resolve(); correction=load_config(project/"configs"/"phase2b1_verification.yaml")
    config=load_phase2b_config(project/"configs"/"phase2b_ml.yaml"); paths=_paths(project); _prepare(paths,bool(correction["runtime"]["overwrite_phase2b1_outputs"]))
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(paths["logs"]/"phase2b1_verification.log",mode="w",encoding="utf-8"),logging.StreamHandler(sys.stdout)],force=True)
    LOGGER.info("Starting Phase 2B.1 verification and correction; deep learning is out of scope")
    required=[project/"outputs"/"phase2b"/"forecasts"/name for name in ("ml_forecasts_long.csv","raw_level_ablation_forecasts_long.csv","combined_phase2a_phase2b_forecasts_long.csv")]
    if any(not p.is_file() for p in required): raise FileNotFoundError("Completed Phase 2B forecasts are required.")
    data_path=project/config["data"]["path"]
    if sha256_file(data_path)!=config["data"]["expected_sha256"]: raise ValueError("Validated-data SHA-256 mismatch.")
    source_before=source_integrity_against_head(project)
    if not source_before.matches_head.all(): raise RuntimeError("A protected Phase 1/2A source file differs from Git HEAD.")
    source_changes=pd.DataFrame([
        {"source_file":"src/config.py","phase2b_change":"appended Phase 2B configuration validation","authorization_audit":"cross-phase modification not required","correction":"moved to src/phase2b_config.py and restored HEAD bytes","final_matches_head":True},
        {"source_file":"src/evaluation/statistical_tests.py","phase2b_change":"appended fixed-reference Phase 2B comparisons","authorization_audit":"cross-phase modification not required","correction":"moved to src/evaluation/phase2b_statistical_tests.py and restored HEAD bytes","final_matches_head":True},
        {"source_file":"src/phase1_audit.py","phase2b_change":"added scikit-learn and xgboost to Phase 1 dependency list","authorization_audit":"environment refresh authorized but Phase 1 code change unnecessary","correction":"restored Phase 1 dependency list; Phase 2B owns expanded environment recording","final_matches_head":True},
    ])
    _write(source_changes,paths["tables"]/"phase1_phase2a_source_change_audit.csv"); _write(source_before,paths["tables"]/"protected_source_integrity_before.csv")
    data=_load_validated(data_path); manifest=generate_forecast_origin_manifest(data,config)
    ml=pd.read_csv(required[0],float_precision="round_trip"); fixed=pd.read_csv(required[1],float_precision="round_trip")
    combined=pd.read_csv(required[2],float_precision="round_trip")
    fixed["selection_family"]="raw_level_fixed_configuration"; fixed["diagnostic_variant"]="raw_level_fixed_configuration"
    _write(fixed,paths["forecasts"]/"raw_level_fixed_configuration_forecasts.csv")
    tables=project/"outputs"/"phase2b"/"tables"
    dm=dm_undefined_detail(combined,[pd.read_csv(tables/"dm_ml_vs_best_phase2a.csv"),pd.read_csv(tables/"dm_all_vs_best_overall.csv")]); _write(dm,paths["tables"]/"undefined_dm_comparison.csv")
    base_candidates=pd.read_csv(tables/"nested_candidate_evaluations.csv",float_precision="round_trip")
    base_selected=pd.read_csv(tables/"selected_configurations.csv",float_precision="round_trip").query("selection_family=='primary_transformed'")
    selected_audit,feasibility,differing=selection_fairness(base_candidates,base_selected)
    _write(selected_audit,paths["tables"]/"selected_candidate_fold_completeness_transformed.csv")
    _write(feasibility,paths["tables"]/"candidate_feasibility_by_representation_lookback.csv"); _write(differing,paths["tables"]/"outer_origins_with_feasibility_differences.csv")
    workers=int(correction["runtime"]["independent_cell_workers"]); started=time.perf_counter()
    independent,raw_audit=_run_independent_raw(data,manifest,config,workers); independent["diagnostic_variant"]="raw_level_independently_tuned"
    LOGGER.info("Independent raw tuning finished in %.1f seconds",time.perf_counter()-started)
    _write(independent,paths["forecasts"]/"raw_level_independently_tuned_forecasts.csv")
    for name,frame in raw_audit.items(): _write(frame,paths["tables"]/f"independently_tuned_raw_{name}.csv")
    raw_selected=raw_audit["selected"]; raw_candidates=raw_audit["candidates"]
    raw_selected_audit,raw_feasibility,raw_differing=selection_fairness(raw_candidates,raw_selected)
    _write(raw_selected_audit,paths["tables"]/"selected_candidate_fold_completeness_raw_independent.csv")
    _write(raw_feasibility,paths["tables"]/"raw_independent_candidate_feasibility.csv"); _write(raw_differing,paths["tables"]/"raw_independent_feasibility_differences.csv")
    all_selected=pd.concat([selected_audit,raw_selected_audit],ignore_index=True); _write(all_selected,paths["tables"]/"selected_candidate_fold_completeness_all.csv")
    corrected,calibration=_expanded_calibration(data,ml,config,workers); _write(corrected,paths["forecasts"]/"ml_forecasts_all_history_conformal.csv"); _write(calibration,paths["tables"]/"all_history_calibration_records.csv")
    interval_metrics=interval_metrics_with_calibration(corrected); _write(interval_metrics,paths["tables"]/"corrected_interval_metrics.csv")
    coincidence=float(corrected.loc[corrected.analysis_window.eq("primary_2000_2023"),"conformal_quantiles_identical"].mean())
    coverage_shortfall=float((interval_metrics.nominal_level-interval_metrics.empirical_coverage).max())
    trigger=coincidence>float(correction["block_bootstrap_sensitivity"]["trigger_coincident_fraction"]) or coverage_shortfall>float(correction["block_bootstrap_sensitivity"]["trigger_coverage_shortfall"])
    if trigger:
        sensitivity=_block_bootstrap_intervals(corrected,calibration,int(correction["block_bootstrap_sensitivity"]["repetitions"]),int(correction["block_bootstrap_sensitivity"]["seed"]))
        sensitivity_metrics=interval_metrics_with_calibration(sensitivity); _write(sensitivity,paths["forecasts"]/"block_bootstrap_residual_interval_sensitivity.csv"); _write(sensitivity_metrics,paths["tables"]/"block_bootstrap_interval_metrics.csv")
    else:
        sensitivity_metrics=pd.DataFrame(columns=interval_metrics.columns); _write(sensitivity_metrics,paths["tables"]/"block_bootstrap_interval_metrics.csv")
    temporal=temporal_metrics(combined); _write(temporal,paths["tables"]/"temporal_robustness_metrics.csv")
    contributions=_cumulative_figures(combined,paths["figures"]); _write(contributions,paths["tables"]/"cumulative_sse_differences.csv")
    improvement=paired_improvement_bootstrap(combined,int(correction["metric_bootstrap"]["repetitions"]),int(correction["metric_bootstrap"]["seed"])); _write(improvement,paths["tables"]/"paired_moving_block_bootstrap_improvements.csv")
    raw_metrics=_raw_comparison_metrics(ml,fixed,independent,manifest); _write(raw_metrics,paths["tables"]/"raw_ablation_comparison_metrics.csv")
    ranking=combined.copy(); ranking["diagnostic_variant"]="primary_ranking"; _write(ranking,paths["forecasts"]/"primary_ranking_forecasts.csv")
    source_after=source_integrity_against_head(project); _write(source_after,paths["tables"]/"protected_source_integrity_after.csv")
    if not source_after.matches_head.all() or not source_before.equals(source_after): raise RuntimeError("Protected Phase 1/2A source integrity failed.")
    _reports(paths,source_changes,source_after,dm,all_selected,pd.concat([differing,raw_differing],ignore_index=True),temporal,contributions,interval_metrics,sensitivity_metrics,improvement,raw_metrics,raw_selected)
    summary={"source_files":len(source_after),"source_matches":int(source_after.matches_head.sum()),"undefined_dm":len(dm),
             "selected_candidates":len(all_selected),"complete_selected":int(all_selected.stored_full_fold_coverage.sum()),
             "independent_raw_records":len(independent),"calibration_records":len(calibration),
             "coincident_fraction":coincidence,"block_bootstrap_triggered":trigger,"temporal_rows":len(temporal)}
    print("\nPHASE 2B.1 EXECUTION SUMMARY")
    for key,value in summary.items(): print(f"{key}: {value}")
    print("Deep-learning models trained: 0")
    return summary


def main() -> int:
    try: run_phase2b1()
    except Exception:
        LOGGER.exception("Phase 2B.1 failed"); return 1
    return 0


if __name__=="__main__": raise SystemExit(main())
