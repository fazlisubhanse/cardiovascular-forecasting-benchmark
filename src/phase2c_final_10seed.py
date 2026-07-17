"""Restart-safe final ten-seed confirmation experiment for Phase 2C.1.

All model-development decisions are read from the verified Phase 2C.0
selection table.  This module never performs hyperparameter search and never
uses evaluation-period results for configuration selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .backtesting.origins import generate_forecast_origin_manifest
from .data_validation import sha256_file
from .deep_learning.datasets import build_scaled_fold
from .deep_learning.determinism import set_deterministic_seed
from .deep_learning.parameter_audit import parameter_counts
from .deep_learning.training import final_fixed_epoch_fit
from .evaluation.aggregation import aggregate_point_metrics
from .evaluation.statistical_tests import compare_models
from .phase2a_baselines import _load_validated
from .phase2c_pilot import _hash_comparison, _protected_manifest, _write

LOGGER = logging.getLogger(__name__)
PRIMARY_ARCHITECTURES = ["LSTM", "GRU", "BiLSTM", "CausalCNN", "CNN_LSTM", "CNN_GRU", "CNN_BiLSTM_Attention_Compact"]
DIAGNOSTIC_ARCHITECTURES = ["CNN_BiLSTM_Compact_NoAttention", "CNN_BiLSTM_Attention_Original23K"]
ALL_ARCHITECTURES = PRIMARY_ARCHITECTURES + DIAGNOSTIC_ARCHITECTURES
SEEDS = [3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110]
KEY_COLUMNS = ["target", "architecture", "horizon", "origin_year", "seed"]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config_hash(configuration_json: str) -> str:
    return _sha256_bytes(str(configuration_json).encode("utf-8"))


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _training_data_hash(data: pd.DataFrame, target: str, origin_year: int) -> str:
    rows = data.loc[data.year <= int(origin_year), ["year", target]].copy()
    payload = rows.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return _sha256_bytes(payload)


def _load_config(root: Path) -> dict[str, Any]:
    path = root / "configs" / "phase2c_final_10seed.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("seeds") != SEEDS:
        raise ValueError("Phase 2C.1 must use exactly the ten prespecified seeds.")
    if config["evaluation"] != {"initial_train_end_year": 1999, "final_year": 2023,
                                  "horizons": [1, 2, 5], "recent_window_start_year": 2016}:
        raise ValueError("Phase 2C.1 evaluation design does not match Phase 2C.0.")
    return config


def _paths(root: Path) -> dict[str, Path]:
    base = root / "outputs" / "phase2c_final"
    return {"base": base, "progress": base / "progress", "cell_progress": base / "progress" / "cells",
            "forecasts": base / "forecasts", "tables": base / "tables", "checkpoints": base / "checkpoints",
            "logs": base / "logs", "reports": root / "reports"}


def _prepare(paths: dict[str, Path]) -> None:
    for path in paths.values():
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: ("" if pd.isna(value) else value) for key, value in row.items()})


def _cell_id(target: str, horizon: int, architecture: str) -> str:
    return f"{target}_h{int(horizon)}_{architecture}"


def _fit_key(row: dict[str, Any] | pd.Series) -> tuple[Any, ...]:
    return (str(row["target"]), str(row["architecture"]), int(row["horizon"]), int(row["origin_year"]), int(row["seed"]))


def _valid_completed(row: pd.Series, selected: dict[str, Any], data_hash: str) -> bool:
    return (str(row.get("fit_status")) == "success" and np.isfinite(float(row.get("point_forecast")))
            and str(row.get("configuration_hash")) == _config_hash(selected["configuration_json"])
            and str(row.get("training_data_index_hash")) == data_hash
            and len(str(row.get("model_state_sha256", ""))) == 64)


def _checkpoint_name(target: str, horizon: int, architecture: str, seed: int) -> str:
    return f"{target}_h{horizon}_{architecture}_seed{seed}.pt"


def _run_cell(payload: tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], str, str]) -> dict[str, Any]:
    """Fit one frozen target-horizon-architecture cell with incremental writes."""
    data, manifest, selected, config, cell_progress, checkpoint_dir = payload
    target, horizon, architecture = selected["target"], int(selected["horizon"]), selected["architecture"]
    spec = json.loads(selected["configuration_json"]); config_hash = _config_hash(selected["configuration_json"])
    progress_path, checkpoint_path = Path(cell_progress), Path(checkpoint_dir)
    existing = pd.read_csv(progress_path).to_dict("records") if progress_path.exists() else []
    existing_by_key = {_fit_key(r): r for r in existing}
    lookup = data.set_index("year"); primary = manifest.loc[manifest.analysis_window.eq("primary_2000_2023")]
    completed = resumed = failures = 0
    for outer in primary.sort_values("origin_year").itertuples(index=False):
        training_hash = _training_data_hash(data, target, int(outer.origin_year))
        try:
            fold = build_scaled_fold(data, target, int(outer.origin_year), horizon, int(spec["lookback"]), spec["representation"])
        except Exception as exc:
            for seed in SEEDS:
                key = (target, architecture, horizon, int(outer.origin_year), seed)
                if key in existing_by_key: continue
                row = {"target": target, "architecture": architecture, "horizon": horizon, "origin_year": int(outer.origin_year),
                       "target_year": int(outer.target_year), "seed": seed, "configuration_id": selected["configuration_id"],
                       "configuration_hash": config_hash, "training_data_index_hash": training_hash,
                       "selected_epoch_count": int(selected["selected_epoch_budget"]), "point_forecast": np.nan,
                       "actual": float(lookup.loc[outer.target_year, target]), "fit_status": "failed",
                       "fit_warning": f"{type(exc).__name__}: {exc}", "fit_seconds": 0.0, "forecast_seconds": 0.0,
                       "final_training_loss": np.nan, "model_state_sha256": "", "representative_checkpoint": ""}
                _append_csv(progress_path, row); existing_by_key[key] = row; failures += 1
            continue
        for seed in SEEDS:
            key = (target, architecture, horizon, int(outer.origin_year), seed)
            old = existing_by_key.get(key)
            if old is not None and _valid_completed(pd.Series(old), selected, training_hash):
                resumed += 1; continue
            started = time.perf_counter(); warning = ""; checkpoint_name = ""
            try:
                result, model = final_fixed_epoch_fit(fold, architecture, spec["size"], float(spec["dropout"]),
                    float(spec["learning_rate"]), float(spec["weight_decay"]), int(selected["selected_epoch_budget"]), seed, config["training"])
                with torch.no_grad():
                    loss = float(torch.mean((model(fold.train_X) - fold.train_y) ** 2).item())
                transformed, prediction = fold.reconstruct(result.scaled_prediction)
                state_hash = _state_hash(model); status = "success"
                if int(outer.origin_year) == int(primary.origin_year.max()):
                    checkpoint_name = _checkpoint_name(target, horizon, architecture, seed)
                    checkpoint_file = checkpoint_path / checkpoint_name
                    metadata = {"architecture": architecture, "seed": seed, "target": target, "horizon": horizon,
                                "origin": int(outer.origin_year), "selected_configuration": selected["configuration_json"],
                                "configuration_id": selected["configuration_id"], "epoch_count": int(selected["selected_epoch_budget"]),
                                "data_hash": training_hash, "state_dict_sha256": state_hash}
                    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, checkpoint_file)
            except Exception as exc:
                result = None; transformed = prediction = np.nan; loss = np.nan; state_hash = ""; status = "failed"
                warning = f"{type(exc).__name__}: {exc}"; failures += 1
            row = {"target": target, "architecture": architecture, "model": architecture,
                   "model_role": "primary" if architecture in PRIMARY_ARCHITECTURES else "diagnostic_ablation",
                   "horizon": horizon, "origin_year": int(outer.origin_year), "target_year": int(outer.target_year),
                   "training_start_year": int(outer.training_start_year), "training_end_year": int(outer.training_end_year),
                   "training_n": int(outer.training_n), "supervised_training_n": len(fold.direct.y),
                   "representation": spec["representation"], "lookback": int(spec["lookback"]),
                   "configuration_id": selected["configuration_id"], "configuration_json": selected["configuration_json"],
                   "configuration_hash": config_hash, "training_data_index_hash": training_hash,
                   "parameter_count": int(selected["parameter_count"]), "selected_epoch_count": int(selected["selected_epoch_budget"]),
                   "seed": seed, "actual": float(lookup.loc[outer.target_year, target]),
                   "predicted_transformed_target": transformed, "point_forecast": prediction,
                   "absolute_error": abs(float(lookup.loc[outer.target_year, target]) - prediction) if np.isfinite(prediction) else np.nan,
                   "squared_error": (float(lookup.loc[outer.target_year, target]) - prediction) ** 2 if np.isfinite(prediction) else np.nan,
                   "fit_status": status, "fit_warning": warning, "fit_seconds": result.fit_seconds if result else time.perf_counter() - started,
                   "forecast_seconds": result.forecast_seconds if result else np.nan, "final_training_loss": loss,
                   "model_state_sha256": state_hash, "representative_checkpoint": checkpoint_name,
                   "last_observed_value": float(data.loc[data.year <= outer.origin_year, target].iloc[-1]),
                   "in_sample_naive_mae": float(np.mean(np.abs(np.diff(data.loc[data.year <= outer.origin_year, target].to_numpy(float))))),
                   "in_sample_naive_mse": float(np.mean(np.square(np.diff(data.loc[data.year <= outer.origin_year, target].to_numpy(float))))) ,
                   "validation_n": 0, "early_stopping_used": False, "used_all_supervised_samples": True,
                   "analysis_window": "primary_2000_2023"}
            _append_csv(progress_path, row); existing_by_key[key] = row; completed += int(status == "success")
    return {"cell_id": _cell_id(target, horizon, architecture), "completed": completed, "resumed": resumed, "failures": failures}


def _protected_final_manifest(root: Path) -> pd.DataFrame:
    frame = _protected_manifest(root)
    directory = root / "outputs" / "phase2c_pilot"
    if directory.exists():
        extra = pd.DataFrame([{"relative_path": p.relative_to(root).as_posix(), "sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                              for p in sorted(directory.rglob("*")) if p.is_file()])
        frame = pd.concat([frame, extra], ignore_index=True).drop_duplicates("relative_path").sort_values("relative_path").reset_index(drop=True)
    return frame


def _merge_cell_progress(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    progress_files = list(paths["cell_progress"].glob("*.csv")) if paths["cell_progress"].exists() else []
    frames = [pd.read_csv(p) for p in progress_files if p.stat().st_size > 0]
    if not frames: return pd.DataFrame(), pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(KEY_COLUMNS, keep="last").sort_values(KEY_COLUMNS).reset_index(drop=True)
    return merged, merged.copy()


def _write_ensemble(forecasts: pd.DataFrame, path: Path) -> pd.DataFrame:
    primary = forecasts.loc[forecasts.analysis_window.eq("primary_2000_2023")].copy()
    keys = ["target", "architecture", "model_role", "horizon", "origin_year", "target_year"]
    rows = []
    for key, group in primary.groupby(keys, sort=True):
        vals = group.loc[group.fit_status.eq("success"), "point_forecast"].to_numpy(float)
        row = group.iloc[0].to_dict(); row.update({"model": key[1], "model_family": "deep_learning_final_10seed",
            "seed_success_n": len(vals), "seed_expected_n": 10, "fit_status": "success" if len(vals) == 10 else "incomplete",
            "point_forecast": float(np.mean(vals)) if len(vals) else np.nan, "seed_mean_prediction": float(np.mean(vals)) if len(vals) else np.nan,
            "seed_median_prediction": float(np.median(vals)) if len(vals) else np.nan,
            "seed_prediction_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "seed_min_prediction": float(np.min(vals)) if len(vals) else np.nan, "seed_max_prediction": float(np.max(vals)) if len(vals) else np.nan,
            "seed_iqr_prediction": float(np.quantile(vals, .75) - np.quantile(vals, .25)) if len(vals) else np.nan})
        rows.append(row)
    primary_frame = pd.DataFrame(rows); primary_frame["analysis_window"] = "primary_2000_2023"
    recent = primary_frame.loc[primary_frame.target_year >= 2016].copy(); recent["analysis_window"] = "recent_2016_2023"
    result = pd.concat([primary_frame, recent], ignore_index=True).sort_values(["analysis_window", "target", "architecture", "horizon", "origin_year"])
    _write(result, path); return result


def _metric_table(forecasts: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecasts = forecasts.copy(); forecasts["model"] = forecasts["architecture"]
    return aggregate_point_metrics(forecasts, manifest)


def _seed_variability(forecasts: pd.DataFrame, manifest: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    for seed, frame in forecasts.groupby("seed"):
        p, _ = _metric_table(frame, manifest); p["seed"] = int(seed); rows.append(p)
    long = pd.concat(rows, ignore_index=True); measures = ["rmse", "mae", "mase", "rmsse", "smape_percent", "mean_error", "median_error", "directional_accuracy_percent"]
    output_rows = []
    rng = np.random.default_rng(20260810)
    for keys, group in long.groupby(["analysis_window", "target", "model", "horizon"], sort=True):
        row = {"analysis_window": keys[0], "target": keys[1], "architecture": keys[2], "horizon": int(keys[3])}
        for measure in measures:
            vals = group[measure].to_numpy(float); boot = rng.choice(vals, size=(5000, len(vals)), replace=True).mean(axis=1)
            row.update({f"mean_seed_{measure}": float(np.mean(vals)), f"std_seed_{measure}": float(np.std(vals, ddof=1)),
                        f"median_seed_{measure}": float(np.median(vals)), f"min_seed_{measure}": float(np.min(vals)),
                        f"max_seed_{measure}": float(np.max(vals)), f"bootstrap_ci_lower_95_{measure}": float(np.quantile(boot, .025)),
                        f"bootstrap_ci_upper_95_{measure}": float(np.quantile(boot, .975))})
        output_rows.append(row)
    result = pd.DataFrame(output_rows); _write(result, output); return result


def _family(model: str) -> str:
    if model in {"Naive", "Drift", "LinearTrend", "ETS", "ARIMA", "Theta"}: return "statistical"
    if model in {"SVR", "RandomForest", "XGBoost"}: return "classical_ml"
    return "neural" if model in PRIMARY_ARCHITECTURES else "diagnostic_ablation"


def _comparison_table(metrics: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    for (window, target, horizon), group in metrics.groupby(["analysis_window", "target", "horizon"], sort=True):
        eligible = group.loc[group.ranking_eligible.astype(bool)].copy(); best = eligible.sort_values("rmse").iloc[0]
        best_rmse = float(best.rmse)
        for _, row in eligible.iterrows():
            rows.append({"analysis_window": window, "target": target, "horizon": int(horizon), "model": row.model,
                         "model_family": _family(row.model), "rmse": row.rmse, "mae": row.mae, "mase": row.mase,
                         "relative_rmse_difference_from_best": (row.rmse - best_rmse) / best_rmse if best_rmse else np.nan,
                         "best_model": best.model})
    result = pd.DataFrame(rows); _write(result, output); return result


def _stability(pilot: pd.DataFrame, final: pd.DataFrame, output: Path) -> pd.DataFrame:
    p = pilot.rename(columns={"model": "architecture", "rmse": "pilot_rmse"})[["target", "architecture", "horizon", "pilot_rmse"]]
    f = final.rename(columns={"model": "architecture", "rmse": "ten_seed_rmse"})[["target", "architecture", "horizon", "ten_seed_rmse"]]
    result = p.merge(f, on=["target", "architecture", "horizon"]); result["absolute_difference"] = result.ten_seed_rmse - result.pilot_rmse
    result["percentage_difference"] = 100 * result.absolute_difference / result.pilot_rmse
    result["pilot_rank"] = result.groupby(["target", "horizon"]).pilot_rmse.rank(method="min")
    result["ten_seed_rank"] = result.groupby(["target", "horizon"]).ten_seed_rmse.rank(method="min")
    result["ranking_changed"] = result.pilot_rank != result.ten_seed_rank
    _write(result, output); return result


def _ablation_table(metrics: pd.DataFrame, variability: pd.DataFrame, selected: pd.DataFrame, forecasts: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []; primary = metrics.loc[metrics.analysis_window.eq("primary_2000_2023")]
    for label, other in [("attention", "CNN_BiLSTM_Compact_NoAttention"), ("capacity", "CNN_BiLSTM_Attention_Original23K")]:
        for (target, horizon), group in primary.groupby(["target", "horizon"]):
            ref = group[group.model.eq("CNN_BiLSTM_Attention_Compact")].iloc[0]; cmp = group[group.model.eq(other)].iloc[0]
            v = variability.loc[(variability.analysis_window == "primary_2000_2023") & (variability.target == target) & (variability.horizon == horizon)]
            vr = v[v.architecture.eq(ref.model)].iloc[0]; vc = v[v.architecture.eq(other)].iloc[0]
            sr = forecasts[(forecasts.target == target) & (forecasts.horizon == horizon) & forecasts.architecture.eq(ref.model)].fit_seconds.sum()
            sc = forecasts[(forecasts.target == target) & (forecasts.horizon == horizon) & forecasts.architecture.eq(other)].fit_seconds.sum()
            pr = int(selected[selected.architecture.eq(ref.model)].parameter_count.iloc[0]); pc = int(selected[selected.architecture.eq(other)].parameter_count.iloc[0])
            rows.append({"ablation": label, "target": target, "horizon": int(horizon), "rmse_difference_comparison_minus_reference": cmp.rmse - ref.rmse,
                         "mae_difference_comparison_minus_reference": cmp.mae - ref.mae, "mean_seed_rmse_sd_reference": vr.std_seed_rmse,
                         "mean_seed_rmse_sd_comparison": vc.std_seed_rmse, "parameter_count_reference": pr, "parameter_count_comparison": pc,
                         "parameter_count_ratio": pc / pr, "training_time_ratio": sc / sr if sr else np.nan})
    result = pd.DataFrame(rows); _write(result, output); return result


def run(root: Path | None = None) -> dict[str, Any]:
    root = (root or Path(__file__).resolve().parents[1]).resolve(); config = _load_config(root); paths = _paths(root); _prepare(paths)
    set_deterministic_seed(20260810, 1)
    data_path = root / config["data"]["path"]
    if sha256_file(data_path) != config["data"]["expected_sha256"]: raise ValueError("Validated data hash mismatch.")
    required = [root / "reports" / "PHASE2B1_VERIFICATION.md", root / "outputs" / "phase2b1" / "tables" / "protected_source_integrity_after.csv"]
    if any(not p.is_file() for p in required): raise FileNotFoundError("Earlier phase completion artifacts are required.")
    before = _protected_final_manifest(root); _write(before, paths["tables"] / "protected_hashes_before.csv")
    data = _load_validated(data_path); manifest = generate_forecast_origin_manifest(data, {"data": {"targets": config["data"]["targets"], "year_column": "year"}, "backtest": {**config["evaluation"], "strategy": "expanding", "refit_every_origin": True}})
    selected = pd.read_csv(root / config["data"]["selected_configurations"]); selected = selected.loc[selected.selected.astype(bool)].copy()
    if len(selected) != 54 or set(selected.architecture) != set(ALL_ARCHITECTURES): raise ValueError("Frozen Phase 2C.0 selection table is invalid.")
    payloads = []
    for row in selected.to_dict("records"):
        cell = manifest[(manifest.target == row["target"]) & (manifest.horizon == int(row["horizon"]))]
        payloads.append((data, cell, row, {"training": {"torch_num_threads": 1, "batch_size": 8, "gradient_clip_norm": 1.0}, "seeds": SEEDS}, str(paths["cell_progress"] / f"{_cell_id(row['target'], int(row['horizon']), row['architecture'])}.csv"), str(paths["checkpoints"])))
    from concurrent.futures import ProcessPoolExecutor
    started = time.perf_counter(); summaries = []
    with ProcessPoolExecutor(max_workers=int(config["runtime"]["workers"])) as pool:
        for result in pool.map(_run_cell, payloads):
            summaries.append(result); print(f"completed cell {result['cell_id']} completed={result['completed']} resumed={result['resumed']} failures={result['failures']}", flush=True)
    forecasts, _ = _merge_cell_progress(paths); _write(forecasts, paths["progress"] / "completed_fit_manifest.csv")
    if len(forecasts) != 9 * 2 * (24 + 23 + 20) * 10: raise RuntimeError(f"Expected 12060 completed fit rows, found {len(forecasts)}")
    forecasts = forecasts.sort_values(KEY_COLUMNS).reset_index(drop=True); _write(forecasts, paths["forecasts"] / "dl_10seed_forecasts_long.csv")
    ensemble = _write_ensemble(forecasts, paths["forecasts"] / "dl_10seed_ensemble_forecasts.csv")
    final_manifest = manifest.copy(); primary_manifest = final_manifest[final_manifest.analysis_window.eq("primary_2000_2023")]
    ens_primary, ens_diag = _metric_table(ensemble, final_manifest); ens_metrics = ens_primary.merge(ens_diag[["analysis_window", "target", "model", "horizon", "error_standard_deviation", "median_absolute_error", "maximum_absolute_error"]], on=["analysis_window", "target", "model", "horizon"])
    _write(ens_metrics[ens_metrics.analysis_window.eq("primary_2000_2023")], paths["tables"] / "dl_10seed_metrics_primary.csv"); _write(ens_metrics[ens_metrics.analysis_window.eq("recent_2016_2023")], paths["tables"] / "dl_10seed_metrics_recent.csv")
    variability = _seed_variability(forecasts, final_manifest, paths["tables"] / "dl_10seed_seed_variability.csv")
    pilot = pd.read_csv(root / "outputs" / "phase2c_pilot" / "tables" / "dl_ensemble_metrics_primary.csv"); _stability(pilot, ens_metrics[ens_metrics.analysis_window.eq("primary_2000_2023")], paths["tables"] / "pilot_vs_10seed_stability.csv")
    nonneural = pd.read_csv(root / "outputs" / "phase2b" / "forecasts" / "combined_phase2a_phase2b_forecasts_long.csv"); combined = pd.concat([nonneural, ensemble], ignore_index=True, sort=False); all_metrics, all_diag = _metric_table(combined, final_manifest); all_metrics = all_metrics.merge(all_diag[["analysis_window", "target", "model", "horizon", "error_standard_deviation", "median_absolute_error", "maximum_absolute_error"]], on=["analysis_window", "target", "model", "horizon"])
    comparison = _comparison_table(all_metrics, paths["tables"] / "final_all_model_comparison.csv"); _write(comparison[comparison.analysis_window.eq("recent_2016_2023")], paths["tables"] / "final_recent_model_comparison.csv")
    _ablation_table(ens_metrics, variability, selected, forecasts, paths["tables"] / "final_architecture_ablations.csv")
    stats_config = {"statistics": {"alpha": .05, "bootstrap_repetitions": 5000, "bootstrap_seed": 20260810}}
    dm, boot = compare_models(combined, all_metrics, stats_config); dm["comparison_scope"] = "overall_winner_against_all_models"; boot["comparison_scope"] = "overall_winner_against_all_models"; _write(dm, paths["tables"] / "final_statistical_dm_tests.csv"); _write(boot, paths["tables"] / "final_statistical_bootstrap.csv")
    elapsed = time.perf_counter() - started; checkpoint_files = list(paths["checkpoints"].glob("*.pt")); projection = pd.read_csv(root / "outputs" / "phase2c_pilot" / "tables" / "final_30_seed_runtime_projection.csv")
    cost = pd.DataFrame([{"total_runtime_seconds": elapsed, "pilot_projected_30_seed_hours": float(projection.iloc[0].projected_30_seed_hours), "checkpoint_count": len(checkpoint_files), "checkpoint_storage_bytes": sum(p.stat().st_size for p in checkpoint_files), "fit_count": len(forecasts), "median_fit_seconds": float(forecasts.fit_seconds.median()), "maximum_fit_seconds": float(forecasts.fit_seconds.max())}]); _write(cost, paths["tables"] / "final_computational_cost.csv")
    resume = {"expected_fit_count": 12060, "completed_fit_count": len(forecasts), "failure_count": int((forecasts.fit_status != "success").sum()), "last_run_seconds": elapsed, "updated_utc": pd.Timestamp.utcnow().isoformat()}; (paths["progress"] / "resume_state.json").write_text(json.dumps(resume, indent=2), encoding="utf-8")
    after = _protected_final_manifest(root); _write(after, paths["tables"] / "protected_hashes_after.csv"); comparison_hash = _hash_comparison(before, after); _write(comparison_hash, paths["tables"] / "protected_hash_comparison.csv")
    if not comparison_hash.hash_match.all(): raise RuntimeError("Protected artifacts changed during Phase 2C.1.")
    _write(pd.DataFrame([{"python": platform.python_version(), "torch": torch.__version__, "torch_num_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "device": "cpu"}]), paths["tables"] / "runtime_environment.csv")
    report_dir = paths["reports"]
    (report_dir / "PHASE2C_FINAL_10SEED_RESULTS.md").write_text("# Phase 2C.1 Final 10-Seed Results\n\nThe final experiment uses ten prespecified seeds and frozen Phase 2C.0 configurations. Results are confirmatory robustness estimates, not a new hyperparameter search.\n\nAll primary fit records are stored in `outputs/phase2c_final/forecasts/dl_10seed_forecasts_long.csv`; failures remain explicit.\n", encoding="utf-8")
    (report_dir / "PHASE2C_FINAL_ARCHITECTURE_ABLATION.md").write_text("# Phase 2C.1 Architecture Ablation\n\nAttention and capacity comparisons are in `outputs/phase2c_final/tables/final_architecture_ablations.csv`. Differences are descriptive and are not treated as evidence of equivalence.\n", encoding="utf-8")
    (report_dir / "PHASE2C_FINAL_REPRODUCIBILITY.md").write_text(f"# Phase 2C.1 Reproducibility\n\nSeeds: {SEEDS}. Frozen selections come from Phase 2C.0. Expected fits: 12060. Completed fits: {len(forecasts)}. Representative checkpoints: {len(checkpoint_files)}. State hashes and resume metadata are stored in the progress manifest.\n", encoding="utf-8")
    (report_dir / "PHASE2C_FINAL_PILOT_STABILITY.md").write_text("# Phase 2C.1 Pilot Stability\n\nPilot-versus-ten-seed RMSE differences and rank changes are in `outputs/phase2c_final/tables/pilot_vs_10seed_stability.csv`.\n", encoding="utf-8")
    print(f"PyTorch {torch.__version__}; deterministic={torch.are_deterministic_algorithms_enabled()}; completed={len(forecasts)}/12060; checkpoints={len(checkpoint_files)}; runtime_seconds={elapsed:.2f}; final 30-seed run: NO")
    return {"completed_fit_count": len(forecasts), "checkpoint_count": len(checkpoint_files), "runtime_seconds": elapsed}


def main() -> int:
    try:
        run()
    except Exception:
        LOGGER.exception("Phase 2C.1 final ten-seed run failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
