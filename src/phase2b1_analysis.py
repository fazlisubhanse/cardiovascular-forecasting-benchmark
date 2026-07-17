"""Auditable Phase 2B.1 verification and correction analyses."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluation.metrics import directional_accuracy, mase, mean_absolute_error, root_mean_squared_error
from .evaluation.statistical_tests import _bootstrap_indices, _stable_seed
from .forecasting.conformal import conformal_bounds
from .tuning.forward_search import complexity_key


PERIODS = (("2000-2007", 2000, 2007), ("2008-2015", 2008, 2015), ("2016-2023", 2016, 2023))


def temporal_period(year: int) -> str:
    for name, start, end in PERIODS:
        if start <= int(year) <= end:
            return name
    raise ValueError(f"Forecast target year outside Phase 2B.1 periods: {year}")


def temporal_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate requested metrics for every model/cell and nonoverlapping period."""
    primary = forecasts.loc[forecasts.analysis_window.eq("primary_2000_2023")].copy()
    primary["temporal_period"] = primary.target_year.map(temporal_period)
    rows = []
    for keys, group in primary.groupby(["temporal_period", "target", "model", "horizon"], sort=True):
        actual = group.actual.to_numpy(dtype=float); prediction = group.point_forecast.to_numpy(dtype=float)
        errors = actual - prediction
        rows.append({
            "temporal_period": keys[0], "target": keys[1], "model": keys[2], "horizon": int(keys[3]),
            "forecast_count": len(group), "rmse": root_mean_squared_error(actual, prediction),
            "mae": mean_absolute_error(actual, prediction),
            "mase": mase(errors, group.in_sample_naive_mae.to_numpy(dtype=float)),
            "mean_error": float(np.mean(errors)),
            "directional_accuracy_percent": directional_accuracy(
                actual, prediction, group.last_observed_value.to_numpy(dtype=float)),
            "first_target_year": int(group.target_year.min()), "last_target_year": int(group.target_year.max()),
        })
    return pd.DataFrame(rows)


def dm_undefined_detail(forecasts: pd.DataFrame, dm_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Reconstruct every unique undefined DM comparison and its variance components."""
    undefined = []
    for table in dm_tables:
        subset = table.loc[table.interpretation.eq("test_undefined")].copy()
        for row in subset.itertuples(index=False):
            reference = getattr(row, "reference_model", getattr(row, "reference_best_rmse_model", None))
            key = (row.analysis_window, row.target, int(row.horizon), str(reference), row.comparison_model)
            if key in [item[0] for item in undefined]:
                continue
            undefined.append((key, row.undefined_reason))
    rows = []
    for (window, target, horizon, reference, alternative), reason in undefined:
        cell = forecasts.loc[(forecasts.analysis_window == window) & (forecasts.target == target)
                             & (forecasts.horizon == horizon)]
        ref = cell.loc[cell.model == reference, ["origin_year", "actual", "point_forecast"]].rename(
            columns={"point_forecast": "reference_forecast"})
        alt = cell.loc[cell.model == alternative, ["origin_year", "actual", "point_forecast"]].rename(
            columns={"actual": "alternative_actual", "point_forecast": "alternative_forecast"})
        paired = ref.merge(alt, on="origin_year")
        actual = paired.actual.to_numpy(dtype=float)
        differential = np.square(actual - paired.alternative_forecast.to_numpy(dtype=float)) - np.square(
            actual - paired.reference_forecast.to_numpy(dtype=float))
        centered = differential - differential.mean(); count = len(differential)
        gamma0 = float(centered @ centered / count); autocovariances = []
        long_run_variance = gamma0
        for lag in range(1, horizon):
            gamma = float(centered[lag:] @ centered[:-lag] / count)
            autocovariances.append(gamma); long_run_variance += 2 * gamma
        rows.append({
            "analysis_window": window, "target": target, "horizon": horizon,
            "reference_model": reference, "comparison_model": alternative, "paired_n": count,
            "mean_loss_differential_alt_minus_reference": float(differential.mean()),
            "gamma0": gamma0, "lag_autocovariances_json": json.dumps(autocovariances),
            "long_run_variance_estimate": long_run_variance,
            "variance_of_mean_estimate": long_run_variance / count,
            "machine_epsilon": float(np.finfo(float).eps), "undefined_reason": reason,
        })
    return pd.DataFrame(rows)


def _candidate_key(row: pd.Series) -> tuple:
    params = json.loads(row.hyperparameters_json)
    return (float(row.inner_rmse), float(row.inner_mae), int(row.lookback),
            complexity_key(str(row.model), params), str(row.representation), str(row.hyperparameters_json))


def selection_fairness(candidates: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Verify full-fold eligibility, deterministic winners, and feasibility differences."""
    candidates = candidates.copy()
    candidates["full_available_fold_coverage"] = candidates.inner_successful_n == candidates.inner_expected_n
    candidates["selection_eligible"] = candidates.candidate_status.eq("success") & candidates.full_available_fold_coverage
    selected = selected.copy()
    selected["full_available_fold_coverage"] = selected.inner_successful_n == selected.inner_expected_n
    selected["selection_eligible"] = selected.candidate_status.eq("success") & selected.full_available_fold_coverage
    keys = ["selection_family", "target", "model", "horizon", "outer_origin_year"]
    audits = []
    for group_key, group in candidates.groupby(keys, sort=True):
        eligible = group.loc[group.selection_eligible]
        winner = eligible.loc[min(eligible.index, key=lambda index: _candidate_key(eligible.loc[index]))]
        stored = selected
        for column, value in zip(keys, group_key): stored = stored.loc[stored[column] == value]
        if len(stored) != 1:
            raise RuntimeError(f"Expected one stored selection for {group_key}, got {len(stored)}")
        stored = stored.iloc[0]
        match = (stored.representation == winner.representation and int(stored.lookback) == int(winner.lookback)
                 and stored.hyperparameters_json == winner.hyperparameters_json)
        audits.append({**dict(zip(keys, group_key)), "eligible_candidate_n": len(eligible),
                       "stored_full_fold_coverage": bool(stored.full_available_fold_coverage),
                       "recomputed_representation": winner.representation,
                       "recomputed_lookback": int(winner.lookback),
                       "recomputed_hyperparameters_json": winner.hyperparameters_json,
                       "deterministic_tie_break_match": bool(match)})
    selected_audit = selected.merge(pd.DataFrame(audits), on=keys, how="left")
    feasibility = candidates.groupby(keys + ["representation", "lookback"], sort=True).agg(
        candidate_n=("hyperparameters_json", "size"), full_coverage_candidate_n=("selection_eligible", "sum"),
        minimum_successful_folds=("inner_successful_n", "min"), maximum_successful_folds=("inner_successful_n", "max"),
        available_fold_n=("inner_expected_n", "max")).reset_index()
    feasibility["full_coverage_fraction"] = feasibility.full_coverage_candidate_n / feasibility.candidate_n
    outer_keys = keys
    differing_keys = []
    for key, group in feasibility.groupby(outer_keys, sort=True):
        if group.full_coverage_fraction.nunique() > 1:
            differing_keys.append(dict(zip(outer_keys, key)))
    differing = feasibility.merge(pd.DataFrame(differing_keys), on=outer_keys, how="inner") if differing_keys else feasibility.iloc[0:0]
    return selected_audit, feasibility, differing


def source_integrity_against_head(root: Path) -> pd.DataFrame:
    """Compare every tracked pre-Phase2B Python source file with its Git HEAD bytes."""
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD", "src"], cwd=root,
                           check=True, capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
    rows = []
    for relative in sorted(name for name in names if name.endswith(".py")):
        expected = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=root, check=True, capture_output=True).stdout
        path = root / relative; observed = path.read_bytes()
        rows.append({"relative_path": relative, "head_sha256": hashlib.sha256(expected).hexdigest(),
                     "working_sha256": hashlib.sha256(observed).hexdigest(),
                     "matches_head": expected == observed})
    return pd.DataFrame(rows)


def interval_metrics_with_calibration(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Evaluate corrected intervals by every requested dimension."""
    rows = []
    for keys, group in forecasts.groupby(["analysis_window", "target", "model", "horizon"], sort=True):
        for level, suffix in ((.80, "80"), (.95, "95")):
            valid = group.loc[group[f"interval_{suffix}_valid"].astype(bool)].copy()
            actual = valid.actual.to_numpy(float); lower = valid[f"lower_{suffix}"].to_numpy(float)
            upper = valid[f"upper_{suffix}"].to_numpy(float); alpha = 1-level
            widths = upper-lower; covered = (actual>=lower)&(actual<=upper)
            scores = widths + np.where(actual<lower,2/alpha*(lower-actual),0)+np.where(actual>upper,2/alpha*(actual-upper),0)
            rows.append({"analysis_window":keys[0],"target":keys[1],"model":keys[2],"horizon":int(keys[3]),
                         "nominal_level":level,"forecast_n":len(group),"valid_interval_n":len(valid),
                         "empirical_coverage":float(np.mean(covered)) if len(valid) else np.nan,
                         "mean_width":float(np.mean(widths)) if len(valid) else np.nan,
                         "mean_winkler_score":float(np.mean(scores)) if len(valid) else np.nan,
                         "mean_calibration_n":float(group.calibration_n.mean()),
                         "minimum_calibration_n":int(group.calibration_n.min()),
                         "maximum_calibration_n":int(group.calibration_n.max()),
                         "coincident_80_95_case_n":int(group.conformal_quantiles_identical.sum())})
    return pd.DataFrame(rows)


def paired_improvement_bootstrap(forecasts: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    """Moving-block bootstrap CIs for the three prespecified ML improvements."""
    comparisons = (("asph",1,"RandomForest","ETS"),("asph",2,"RandomForest","ETS"),("mtc",2,"XGBoost","Theta"))
    primary=forecasts.loc[forecasts.analysis_window.eq("primary_2000_2023")]
    rows=[]
    for target,horizon,ml_model,baseline_model in comparisons:
        cell=primary.loc[(primary.target==target)&(primary.horizon==horizon)]
        ml=cell.loc[cell.model==ml_model,["origin_year","actual","point_forecast"]].rename(columns={"point_forecast":"ml"})
        baseline=cell.loc[cell.model==baseline_model,["origin_year","actual","point_forecast"]].rename(columns={"actual":"actual_b","point_forecast":"baseline"})
        paired=ml.merge(baseline,on="origin_year"); actual=paired.actual.to_numpy(float)
        eml=actual-paired.ml.to_numpy(float); ebase=actual-paired.baseline.to_numpy(float); n=len(paired)
        block=max(horizon,int(math.ceil(math.sqrt(n))))
        rng=np.random.default_rng(_stable_seed(seed,"phase2b1",target,horizon,ml_model,baseline_model))
        indices=_bootstrap_indices(rng,n,repetitions,block)
        rmse=np.sqrt(np.mean(ebase[indices]**2,axis=1))-np.sqrt(np.mean(eml[indices]**2,axis=1))
        mae=np.mean(np.abs(ebase[indices]),axis=1)-np.mean(np.abs(eml[indices]),axis=1)
        rows.append({"target":target,"horizon":horizon,"ml_model":ml_model,"baseline_model":baseline_model,
                     "paired_n":n,"block_length":block,"bootstrap_repetitions":repetitions,
                     "rmse_improvement_baseline_minus_ml":float(np.sqrt(np.mean(ebase**2))-np.sqrt(np.mean(eml**2))),
                     "rmse_improvement_ci_lower_95":float(np.quantile(rmse,.025)),"rmse_improvement_ci_upper_95":float(np.quantile(rmse,.975)),
                     "mae_improvement_baseline_minus_ml":float(np.mean(np.abs(ebase))-np.mean(np.abs(eml))),
                     "mae_improvement_ci_lower_95":float(np.quantile(mae,.025)),"mae_improvement_ci_upper_95":float(np.quantile(mae,.975))})
    return pd.DataFrame(rows)


def finite_conformal_from_residuals(point: float, residuals: list[float], minimum_n: int = 8) -> dict[str, Any]:
    """Build both conformal levels and explicitly flag indistinguishable quantiles."""
    result: dict[str, Any] = {"calibration_n": len(residuals)}
    for level,suffix in ((.80,"80"),(.95,"95")):
        lo,hi,q,rank,valid=conformal_bounds(point,residuals,level,minimum_n)
        result.update({f"lower_{suffix}":lo,f"upper_{suffix}":hi,f"conformal_q_{suffix}":q,
                       f"conformal_rank_{suffix}":rank,f"interval_{suffix}_valid":valid})
    identical=bool(np.isfinite([result["conformal_q_80"],result["conformal_q_95"]]).all()
                   and result["conformal_q_80"]==result["conformal_q_95"])
    result["conformal_quantiles_identical"]=identical
    result["conformal_level_limitation"]="finite_sample_quantiles_identical" if identical else ""
    return result
