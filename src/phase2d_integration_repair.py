"""Non-training Phase 2D.0 integration repair.

This module reads completed Phase 2A/2B/2C.1 artifacts and regenerates only
cross-family comparison tables.  It never instantiates or fits a model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluation.aggregation import aggregate_point_metrics
from .evaluation.statistical_tests import _bootstrap_indices, _dm_hln, _holm_adjust, _stable_seed

STATISTICAL = ["Naive", "Drift", "LinearTrend", "ETS", "ARIMA", "Theta"]
CLASSICAL_ML = ["SVR", "RandomForest", "XGBoost"]
PRIMARY_NEURAL = [
    "LSTM", "GRU", "BiLSTM", "CausalCNN", "CNN_LSTM", "CNN_GRU",
    "CNN_BiLSTM_Attention_Compact",
]
DIAGNOSTIC = ["CNN_BiLSTM_Compact_NoAttention", "CNN_BiLSTM_Attention_Original23K"]
MODEL_FAMILY = {
    **{name: "statistical" for name in STATISTICAL},
    **{name: "classical_ml" for name in CLASSICAL_ML},
    **{name: "primary_neural" for name in PRIMARY_NEURAL},
    **{name: "diagnostic_ablation" for name in DIAGNOSTIC},
}
MODEL_ROLE = {
    **{name: "statistical" for name in STATISTICAL},
    **{name: "classical_ml" for name in CLASSICAL_ML},
    **{name: "primary_neural" for name in PRIMARY_NEURAL},
    **{name: "diagnostic_ablation" for name in DIAGNOSTIC},
}
WINDOWS = ["primary_2000_2023", "recent_2016_2023"]
METRIC_COLUMNS = [
    "analysis_window", "target", "model", "horizon", "expected_n", "successful_n",
    "failed_n", "completeness_percent", "ranking_eligible", "rmse", "mae", "mase",
    "rmsse", "smape_percent", "mean_error", "median_error", "directional_accuracy_percent",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(root: Path) -> dict[str, Path]:
    base = root / "outputs" / "phase2c_final"
    return {
        "base": base,
        "tables": base / "tables",
        "forecasts": base / "forecasts",
        "primary_forecasts": base / "forecasts" / "dl_10seed_forecasts_long.csv",
        "ensemble_forecasts": base / "forecasts" / "dl_10seed_ensemble_forecasts.csv",
        "fit_manifest": base / "progress" / "completed_fit_manifest.csv",
        "nonneural_forecasts": root / "outputs" / "phase2b" / "forecasts" / "combined_phase2a_phase2b_forecasts_long.csv",
        "nonneural_metrics": root / "outputs" / "phase2b" / "tables" / "combined_model_metrics.csv",
        "neural_primary_metrics": base / "tables" / "dl_10seed_metrics_primary.csv",
        "neural_recent_metrics": base / "tables" / "dl_10seed_metrics_recent.csv",
        "manifest": root / "outputs" / "phase2c_pilot" / "tables" / "forecast_origin_manifest.csv",
    }


def _model_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["model"] = frame["model"].astype(str)
    unknown = sorted(set(frame["model"]) - set(MODEL_FAMILY))
    if unknown:
        raise ValueError(f"Unknown canonical model names: {unknown}")
    frame["canonical_model_name"] = frame["model"]
    frame["model_family"] = frame["model"].map(MODEL_FAMILY)
    frame["model_role"] = frame["model"].map(MODEL_ROLE)
    frame["headline_eligible"] = ~frame["model"].isin(DIAGNOSTIC)
    return frame


def _load_metrics(paths: dict[str, Path]) -> pd.DataFrame:
    nonneural = _model_metadata(pd.read_csv(paths["nonneural_metrics"]))
    primary = _model_metadata(pd.read_csv(paths["neural_primary_metrics"]))
    recent = _model_metadata(pd.read_csv(paths["neural_recent_metrics"]))
    metrics = pd.concat([nonneural, primary, recent], ignore_index=True, sort=False)
    metrics = metrics.loc[:, list(dict.fromkeys(METRIC_COLUMNS + [
        "canonical_model_name", "model_family", "model_role", "headline_eligible",
    ]))]
    if metrics.duplicated(["analysis_window", "target", "horizon", "model"]).any():
        raise ValueError("Duplicate cross-family metric keys detected.")
    return metrics


def _load_forecasts(paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    protected_inputs = [paths["primary_forecasts"], paths["ensemble_forecasts"], paths["fit_manifest"]]
    hashes_before = {str(path): _sha256(path) for path in protected_inputs}
    nonneural = _model_metadata(pd.read_csv(paths["nonneural_forecasts"]))
    neural = _model_metadata(pd.read_csv(paths["ensemble_forecasts"]))
    forecasts = pd.concat([nonneural, neural], ignore_index=True, sort=False)
    if forecasts.duplicated(["analysis_window", "target", "horizon", "model", "target_year"]).any():
        raise ValueError("Duplicate cross-family forecast keys detected.")
    return forecasts, hashes_before


def _headline_winner(group: pd.DataFrame, family: str | None = None) -> pd.Series:
    eligible = group.loc[group["headline_eligible"].astype(bool) & group["ranking_eligible"].astype(bool)].copy()
    if family is not None:
        eligible = eligible.loc[eligible["model_family"].eq(family)]
    if eligible.empty:
        raise ValueError(f"No eligible {family or 'overall'} winner for {group[['target', 'horizon']].iloc[0].to_dict()}")
    return eligible.sort_values(["rmse", "model"]).iloc[0]


def _comparison_table(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    for (window, target, horizon), group in metrics.groupby(["analysis_window", "target", "horizon"], sort=True):
        overall = _headline_winner(group)
        stat = _headline_winner(group, "statistical")
        classical = _headline_winner(group, "classical_ml")
        neural = _headline_winner(group, "primary_neural")
        winners.append({
            "analysis_window": window, "target": target, "horizon": int(horizon),
            "best_statistical_model": stat.model, "best_statistical_rmse": stat.rmse,
            "best_statistical_mae": stat.mae, "best_statistical_mase": stat.mase,
            "best_classical_ml_model": classical.model, "best_classical_ml_rmse": classical.rmse,
            "best_classical_ml_mae": classical.mae, "best_classical_ml_mase": classical.mase,
            "best_primary_neural_model": neural.model, "best_primary_neural_rmse": neural.rmse,
            "best_primary_neural_mae": neural.mae, "best_primary_neural_mase": neural.mase,
            "best_overall_primary_model": overall.model, "best_overall_primary_rmse": overall.rmse,
            "best_overall_primary_mae": overall.mae, "best_overall_primary_mase": overall.mase,
        })
        for _, row in group.iterrows():
            rows.append({
                **row.to_dict(),
                "best_model": overall.model,
                "relative_rmse_difference_from_best": (float(row.rmse) - float(overall.rmse)) / float(overall.rmse),
            })
    comparison = pd.DataFrame(rows).sort_values(["analysis_window", "target", "horizon", "model"]).reset_index(drop=True)
    winner_frame = pd.DataFrame(winners).sort_values(["analysis_window", "target", "horizon"]).reset_index(drop=True)
    return comparison, winner_frame


def _paired_forecasts(forecasts: pd.DataFrame, window: str, target: str, horizon: int, neural: str, nonneural: str) -> pd.DataFrame:
    subset = forecasts.loc[
        (forecasts.analysis_window == window) & (forecasts.target == target)
        & (forecasts.horizon == int(horizon)) & forecasts.model.isin([neural, nonneural])
    ].copy()
    rows = []
    for model in [neural, nonneural]:
        part = subset.loc[subset.model.eq(model), ["target_year", "actual", "point_forecast"]].copy()
        if part.target_year.duplicated().any():
            raise ValueError(f"Duplicate target years for {window}/{target}/h{horizon}/{model}.")
        if part.point_forecast.isna().any():
            raise ValueError(f"Missing predictions for {window}/{target}/h{horizon}/{model}.")
        part = part.rename(columns={"actual": f"actual_{model}", "point_forecast": f"forecast_{model}"})
        rows.append(part)
    paired = rows[0].merge(rows[1], on="target_year", how="inner")
    if len(paired) != len(rows[0]) or len(paired) != len(rows[1]):
        raise ValueError(f"Paired target-year mismatch for {window}/{target}/h{horizon}.")
    if not np.allclose(paired[f"actual_{neural}"], paired[f"actual_{nonneural}"], rtol=0, atol=0):
        raise ValueError("Paired actual values differ across compared models.")
    return paired.sort_values("target_year").reset_index(drop=True)


def _cross_family_statistics(metrics: pd.DataFrame, forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dm_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        pairs: list[tuple[int, float]] = []
        pending: list[dict[str, Any]] = []
        for (current_window, target, horizon), group in metrics.groupby(["analysis_window", "target", "horizon"], sort=True):
            if current_window != window:
                continue
            neural = _headline_winner(group, "primary_neural")
            nonneural = group.loc[group["headline_eligible"] & group.model_family.isin(["statistical", "classical_ml"])].sort_values(["rmse", "model"]).iloc[0]
            paired = _paired_forecasts(forecasts, window, target, int(horizon), neural.model, nonneural.model)
            expected_n = int(len(paired))
            error_neural = paired[f"actual_{neural.model}"].to_numpy(float) - paired[f"forecast_{neural.model}"].to_numpy(float)
            error_non = paired[f"actual_{nonneural.model}"].to_numpy(float) - paired[f"forecast_{nonneural.model}"].to_numpy(float)
            loss_difference = np.square(error_neural) - np.square(error_non)
            statistic, raw_p, reason = _dm_hln(loss_difference, int(horizon))
            pending.append({
                "analysis_window": window, "target": target, "horizon": int(horizon),
                "neural_model": neural.model, "non_neural_model": nonneural.model,
                "paired_n": expected_n, "rmse_difference_neural_minus_nonneural": float(neural.rmse - nonneural.rmse),
                "mae_difference_neural_minus_nonneural": float(neural.mae - nonneural.mae),
                "dm_hln_statistic": statistic, "raw_p_value": raw_p,
                "undefined_reason": reason, "autocovariance_truncation_lag": int(horizon) - 1,
            })
            repetitions = 5000
            block_length = 1 if int(horizon) == 1 else max(int(horizon), int(np.ceil(np.sqrt(expected_n))))
            seed = _stable_seed(20260810, window, target, horizon, neural.model, nonneural.model)
            rng = np.random.default_rng(seed)
            indices = _bootstrap_indices(rng, expected_n, repetitions, block_length)
            neural_errors = error_neural[indices]
            non_errors = error_non[indices]
            rmse_bootstrap = np.sqrt(np.mean(np.square(neural_errors), axis=1)) - np.sqrt(np.mean(np.square(non_errors), axis=1))
            mae_bootstrap = np.mean(np.abs(neural_errors), axis=1) - np.mean(np.abs(non_errors), axis=1)
            for repetition, (rmse_value, mae_value) in enumerate(zip(rmse_bootstrap, mae_bootstrap)):
                distribution_rows.append({
                    "analysis_window": window, "target": target, "horizon": int(horizon),
                    "neural_model": neural.model, "non_neural_model": nonneural.model,
                    "repetition": repetition, "rmse_difference_neural_minus_nonneural": float(rmse_value),
                    "mae_difference_neural_minus_nonneural": float(mae_value),
                })
            pending[-1].update({
                "rmse_bootstrap_ci_lower_95": float(np.quantile(rmse_bootstrap, .025)),
                "rmse_bootstrap_ci_upper_95": float(np.quantile(rmse_bootstrap, .975)),
                "mae_bootstrap_ci_lower_95": float(np.quantile(mae_bootstrap, .025)),
                "mae_bootstrap_ci_upper_95": float(np.quantile(mae_bootstrap, .975)),
                "bootstrap_repetitions": repetitions, "bootstrap_type": "paired circular moving-block" if block_length > 1 else "paired iid",
                "bootstrap_block_length": block_length, "bootstrap_seed": seed,
                "bootstrap_rmse_mean": float(np.mean(rmse_bootstrap)), "bootstrap_mae_mean": float(np.mean(mae_bootstrap)),
            })
        finite = [float(row["raw_p_value"]) for row in pending if np.isfinite(row["raw_p_value"])]
        adjusted = _holm_adjust(finite)
        position = 0
        for row in pending:
            p_value = row["raw_p_value"]
            row["holm_adjusted_p_value"] = adjusted[position] if np.isfinite(p_value) else np.nan
            if not np.isfinite(p_value):
                row["interpretation"] = "test_undefined"
            elif row["holm_adjusted_p_value"] < 0.05:
                row["interpretation"] = "statistically_detected_difference"
            else:
                row["interpretation"] = "no_detected_difference_with_limited_power"
            if np.isfinite(p_value):
                position += 1
            dm_rows.append(row)
            bootstrap_rows.append({key: value for key, value in row.items() if key not in {"dm_hln_statistic", "raw_p_value", "holm_adjusted_p_value", "undefined_reason", "interpretation", "autocovariance_truncation_lag"}})
    dm = pd.DataFrame(dm_rows).sort_values(["analysis_window", "target", "horizon"]).reset_index(drop=True)
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values(["analysis_window", "target", "horizon"]).reset_index(drop=True)
    distribution = pd.DataFrame(distribution_rows).sort_values(["analysis_window", "target", "horizon", "repetition"]).reset_index(drop=True)
    return dm, bootstrap, distribution


def run(root: Path | None = None) -> dict[str, Any]:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    paths = _paths(root)
    paths["tables"].mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics(paths)
    forecasts, input_hashes = _load_forecasts(paths)
    manifest = pd.read_csv(paths["manifest"])
    # This call is deliberately read-only and verifies that all source rows are aggregable.
    aggregate_point_metrics(forecasts, manifest)
    comparison, winners = _comparison_table(metrics)
    dm, bootstrap, distribution = _cross_family_statistics(metrics, forecasts)
    comparison.to_csv(paths["tables"] / "final_all_model_comparison.csv", index=False)
    comparison.loc[comparison.analysis_window.eq("recent_2016_2023")].to_csv(paths["tables"] / "final_recent_model_comparison.csv", index=False)
    winners.loc[winners.analysis_window.eq("primary_2000_2023")].to_csv(paths["tables"] / "final_primary_winners.csv", index=False)
    winners.loc[winners.analysis_window.eq("recent_2016_2023")].to_csv(paths["tables"] / "final_recent_winners.csv", index=False)
    dm.to_csv(paths["tables"] / "final_statistical_dm_tests.csv", index=False)
    bootstrap.to_csv(paths["tables"] / "final_statistical_bootstrap.csv", index=False)
    distribution.to_csv(paths["tables"] / "final_statistical_bootstrap_distributions.csv", index=False)
    output_hashes = {str(path): _sha256(path) for path in [paths["primary_forecasts"], paths["ensemble_forecasts"], paths["fit_manifest"]]}
    if input_hashes != output_hashes:
        raise RuntimeError("A seed-level forecast artifact changed during integration repair.")
    audit = {
        "input_artifact_sha256": input_hashes,
        "output_artifact_sha256": output_hashes,
        "fit_count": int(len(pd.read_csv(root / "outputs" / "phase2c_final" / "progress" / "completed_fit_manifest.csv"))),
        "diagnostic_models_excluded_from_headline": DIAGNOSTIC,
        "cross_family_comparison_rows": int(len(comparison)),
        "primary_winner_rows": int(len(winners.loc[winners.analysis_window.eq("primary_2000_2023")])),
        "recent_winner_rows": int(len(winners.loc[winners.analysis_window.eq("recent_2016_2023")])),
    }
    (paths["tables"] / "phase2d_input_artifact_hashes.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Phase 2D.0 integration repaired: comparison_rows={len(comparison)} primary_winners=6 recent_winners=6; forecasts_unchanged={input_hashes == output_hashes}; retraining=NO")
    return audit


if __name__ == "__main__":
    run()
