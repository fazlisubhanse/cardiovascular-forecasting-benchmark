"""Nested expanding-window direct-horizon classical-ML engine."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features.scaling import scale_train_and_forecast
from .features.supervised import build_direct_dataset
from .forecasting.conformal import conformal_bounds
from .forecasting.random_forest import build_random_forest
from .forecasting.svr import build_svr
from .forecasting.xgboost_model import build_xgboost
from .tuning.forward_search import candidate_grid, complexity_key
from .tuning.inner_origins import valid_inner_origins


def _json(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def stable_seed(base: int, *parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return (base + int.from_bytes(digest[:4], "little")) % (2**32)


def _model(model: str, params: dict[str, Any], seed: int, n_jobs: int):
    if model == "SVR":
        return build_svr(params)
    if model == "RandomForest":
        return build_random_forest(params, seed, n_jobs)
    if model == "XGBoost":
        return build_xgboost(params, seed, n_jobs)
    raise ValueError(model)


@dataclass
class PredictionResult:
    prediction: float
    feature_seconds: float
    fit_seconds: float
    forecast_seconds: float
    state: dict[str, float]
    scaler: dict[str, Any]
    training_n: int


def fit_predict(data: pd.DataFrame, target: str, training_end: int, horizon: int, representation: str,
                lookback: int, model: str, params: dict[str, Any], config: dict[str, Any],
                seed_parts: tuple[Any, ...]) -> PredictionResult:
    """Fit one fold and forecast one direct horizon using no data after training_end."""

    started = time.perf_counter()
    dataset = build_direct_dataset(data[target], data["year"], training_end, horizon, lookback, representation)
    feature_seconds = time.perf_counter() - started
    minimum = int(config["inner_validation"]["minimum_supervised_samples"])
    if len(dataset.y) < minimum:
        raise ValueError(f"only {len(dataset.y)} supervised samples; {minimum} required")
    X, y, forecast_X = dataset.X, dataset.y, dataset.forecast_X
    scaler: dict[str, Any] = {}
    y_scaler = None
    if model == "SVR":
        X, y, forecast_X, x_scaler, y_scaler = scale_train_and_forecast(X, y, forecast_X)
        scaler = {
            "x_mean": x_scaler.mean_.tolist(), "x_scale": x_scaler.scale_.tolist(),
            "y_mean": float(y_scaler.mean_[0]), "y_scale": float(y_scaler.scale_[0]),
        }
    # A common prespecified seed across candidates makes stochastic-model
    # comparisons fair and permits exact reuse of nested ensemble prefixes.
    seed = int(config["runtime"]["random_seed"])
    estimator = _model(model, params, seed, int(config["runtime"]["n_jobs"]))
    fit_started = time.perf_counter(); estimator.fit(X, y); fit_seconds = time.perf_counter() - fit_started
    forecast_started = time.perf_counter(); transformed = float(estimator.predict(forecast_X)[0])
    if y_scaler is not None:
        transformed = float(y_scaler.inverse_transform([[transformed]])[0, 0])
    prediction = dataset.invert(transformed)
    forecast_seconds = time.perf_counter() - forecast_started
    return PredictionResult(prediction, feature_seconds, fit_seconds, forecast_seconds,
                            dataset.state, scaler, len(dataset.y))


class NestedMLRunner:
    """Run and audit nested selection, outer forecasts, and conformal calibration."""

    def __init__(self, data: pd.DataFrame, manifest: pd.DataFrame, config: dict[str, Any]):
        self.data, self.manifest, self.config = data, manifest, config
        self.models = list(config["models"])
        self.first_year = int(data["year"].min())
        self.lookup = data.set_index("year")
        self.cache: dict[tuple, tuple[float, str]] = {}
        self.candidates: list[dict[str, Any]] = []
        self.selected: list[dict[str, Any]] = []
        self.calibration: list[dict[str, Any]] = []
        self.trends: list[dict[str, Any]] = []
        self.scalers: list[dict[str, Any]] = []
        self.timings: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []

    def _cached_inner(self, target: str, model: str, horizon: int, representation: str,
                      lookback: int, params: dict[str, Any], inner_origin: int) -> tuple[float, str]:
        key = (target, model, horizon, representation, lookback, _json(params), inner_origin)
        if key not in self.cache:
            try:
                if model == "RandomForest":
                    self._cache_random_forest_prefixes(target, horizon, representation, lookback, params, inner_origin)
                elif model == "XGBoost":
                    self._cache_xgboost_prefixes(target, horizon, representation, lookback, params, inner_origin)
                else:
                    result = fit_predict(self.data, target, inner_origin, horizon, representation, lookback,
                                         model, params, self.config, key)
                    self.cache[key] = (result.prediction, "")
            except Exception as exc:  # candidate failure is explicit and never silently imputed
                reason = f"{type(exc).__name__}: {exc}"
                if model == "RandomForest":
                    for n in self.config["models"][model]["n_estimators"]:
                        candidate={**params,"n_estimators":n}
                        candidate_key=(target,model,horizon,representation,lookback,_json(candidate),inner_origin)
                        self.cache[candidate_key]=(np.nan,reason)
                elif model == "XGBoost":
                    for n in self.config["models"][model]["n_estimators"]:
                        candidate={**params,"n_estimators":n}
                        candidate_key=(target,model,horizon,representation,lookback,_json(candidate),inner_origin)
                        self.cache[candidate_key]=(np.nan,reason)
                else:
                    self.cache[key] = (np.nan, reason)
        return self.cache[key]

    def _cache_random_forest_prefixes(self, target: str, horizon: int, representation: str,
                                      lookback: int, params: dict[str, Any], inner_origin: int) -> None:
        """Fit one 500-tree forest and reuse its exact 200-tree estimator prefix."""
        base={**params,"n_estimators":max(self.config["models"]["RandomForest"]["n_estimators"])}
        dataset=build_direct_dataset(self.data[target],self.data["year"],inner_origin,horizon,lookback,representation)
        minimum=int(self.config["inner_validation"]["minimum_supervised_samples"])
        if len(dataset.y)<minimum: raise ValueError(f"only {len(dataset.y)} supervised samples; {minimum} required")
        estimator=build_random_forest(base,int(self.config["runtime"]["random_seed"]),1)
        estimator.fit(dataset.X,dataset.y)
        for n in self.config["models"]["RandomForest"]["n_estimators"]:
            trees=estimator.estimators_[:int(n)]
            transformed=float(np.mean([float(tree.predict(dataset.forecast_X)[0]) for tree in trees]))
            candidate={**params,"n_estimators":n}
            candidate_key=(target,"RandomForest",horizon,representation,lookback,_json(candidate),inner_origin)
            self.cache[candidate_key]=(dataset.invert(transformed),"")

    def _cache_xgboost_prefixes(self, target: str, horizon: int, representation: str,
                                lookback: int, params: dict[str, Any], inner_origin: int) -> None:
        """Fit the maximum boosting rounds once and reuse exact iteration prefixes."""
        maximum=max(self.config["models"]["XGBoost"]["n_estimators"])
        base={**params,"n_estimators":maximum}
        dataset=build_direct_dataset(self.data[target],self.data["year"],inner_origin,horizon,lookback,representation)
        minimum=int(self.config["inner_validation"]["minimum_supervised_samples"])
        if len(dataset.y)<minimum: raise ValueError(f"only {len(dataset.y)} supervised samples; {minimum} required")
        estimator=build_xgboost(base,int(self.config["runtime"]["random_seed"]),1)
        estimator.fit(dataset.X,dataset.y)
        for n in self.config["models"]["XGBoost"]["n_estimators"]:
            transformed=float(estimator.predict(dataset.forecast_X,iteration_range=(0,int(n)))[0])
            candidate={**params,"n_estimators":n}
            candidate_key=(target,"XGBoost",horizon,representation,lookback,_json(candidate),inner_origin)
            self.cache[candidate_key]=(dataset.invert(transformed),"")

    def _select(self, target: str, model: str, horizon: int, outer_origin: int,
                representations: list[str], selection_family: str) -> tuple[dict[str, Any], list[int], float]:
        started = time.perf_counter()
        inner = valid_inner_origins(
            self.first_year, outer_origin, horizon,
            int(self.config["inner_validation"]["recent_origins"]),
            int(self.config["inner_validation"]["minimum_training_years"]),
        )
        scored = []
        for representation in representations:
            for lookback in self.config["lookbacks"]:
                for params in candidate_grid(self.config["models"][model]):
                    predictions, actuals, reasons = [], [], []
                    for inner_origin in inner:
                        prediction, reason = self._cached_inner(
                            target, model, horizon, representation, int(lookback), params, inner_origin
                        )
                        if np.isfinite(prediction):
                            predictions.append(prediction)
                            actuals.append(float(self.lookup.loc[inner_origin + horizon, target]))
                        elif reason:
                            reasons.append(reason)
                    errors = np.asarray(actuals) - np.asarray(predictions)
                    rmse = float(np.sqrt(np.mean(errors**2))) if len(errors) == len(inner) else np.nan
                    mae = float(np.mean(np.abs(errors))) if len(errors) == len(inner) else np.nan
                    status = "success" if np.isfinite(rmse) else "failed"
                    full_fold_coverage = len(predictions) == len(inner)
                    row = {
                        "selection_family": selection_family, "target": target, "model": model,
                        "horizon": horizon, "outer_origin_year": outer_origin,
                        "representation": representation, "lookback": int(lookback),
                        "hyperparameters_json": _json(params), "inner_origin_years_json": json.dumps(inner),
                        "inner_expected_n": len(inner), "inner_successful_n": len(predictions),
                        "inner_rmse": rmse, "inner_mae": mae, "candidate_status": status,
                        "full_available_fold_coverage": full_fold_coverage,
                        "selection_eligible": bool(status == "success" and full_fold_coverage),
                        "failure_reason": " | ".join(sorted(set(reasons))),
                    }
                    self.candidates.append(row)
                    if row["selection_eligible"]:
                        scored.append((rmse, mae, int(lookback), complexity_key(model, params),
                                       representation, _json(params), params, row))
        if not scored:
            raise RuntimeError(f"No valid nested candidate for {target}/{model}/h{horizon}/{outer_origin}")
        winner = min(scored, key=lambda item: item[:6])
        chosen = {**winner[7], "params": winner[6]}
        selection_seconds = time.perf_counter() - started
        self.selected.append({k: v for k, v in chosen.items() if k != "params"} | {
            "selection_seconds": selection_seconds,
            "tie_break_rule": "RMSE, MAE, smaller lookback, lower complexity, lexical representation/specification",
        })
        return chosen, inner, selection_seconds

    def _outer_record(self, outer, model: str, representations: list[str], family: str,
                      fixed_specification: dict[str, Any] | None = None) -> dict[str, Any]:
        total_started = time.perf_counter()
        target, horizon, origin = outer.target, int(outer.horizon), int(outer.origin_year)
        if fixed_specification is None:
            chosen, inner, selection_seconds = self._select(target, model, horizon, origin, representations, family)
        else:
            inner = valid_inner_origins(
                self.first_year, origin, horizon, int(self.config["inner_validation"]["recent_origins"]),
                int(self.config["inner_validation"]["minimum_training_years"]),
            )
            chosen={"representation":representations[0],"lookback":int(fixed_specification["lookback"]),
                    "params":fixed_specification["params"],"inner_rmse":np.nan,"inner_mae":np.nan}
            selection_seconds=0.0
            self.selected.append({"selection_family":family,"target":target,"model":model,"horizon":horizon,
                                  "outer_origin_year":origin,"representation":representations[0],
                                  "lookback":chosen["lookback"],"hyperparameters_json":_json(chosen["params"]),
                                  "inner_expected_n":len(inner),"inner_successful_n":len(inner),
                                  "inner_rmse":np.nan,"inner_mae":np.nan,"candidate_status":"matched_ablation",
                                  "failure_reason":"","selection_seconds":0.0,
                                  "tie_break_rule":"matched primary lookback and hyperparameters; representation changed only"})
        params, representation, lookback = chosen["params"], chosen["representation"], int(chosen["lookback"])
        result = fit_predict(self.data, target, origin, horizon, representation, lookback, model,
                             params, self.config, ("outer", target, model, horizon, origin, family))
        if result.state:
            self.trends.append({"selection_family": family, "target": target, "model": model,
                                "origin_year": origin, "horizon": horizon,
                                "representation": representation, **result.state})
        if result.scaler:
            self.scalers.append({"selection_family": family, "target": target, "model": model,
                                 "origin_year": origin, "horizon": horizon, "lookback": lookback,
                                 "training_n": result.training_n,
                                 **{key: json.dumps(value) if isinstance(value, list) else value for key, value in result.scaler.items()}})
        residuals = []
        interval_started = time.perf_counter()
        for inner_origin in inner[-int(self.config["conformal"]["calibration_origins"]):]:
            prediction, reason = self._cached_inner(target, model, horizon, representation, lookback, params, inner_origin)
            actual = float(self.lookup.loc[inner_origin + horizon, target])
            residual = abs(actual - prediction) if np.isfinite(prediction) else np.nan
            if np.isfinite(residual): residuals.append(float(residual))
            self.calibration.append({
                "selection_family": family, "target": target, "model": model, "horizon": horizon,
                "outer_origin_year": origin, "calibration_origin_year": inner_origin,
                "calibration_target_year": inner_origin + horizon, "representation": representation,
                "lookback": lookback, "hyperparameters_json": _json(params), "actual": actual,
                "oof_forecast": prediction, "absolute_residual": residual,
                "fully_inside_outer_history": inner_origin + horizon <= origin,
                "status": "success" if np.isfinite(residual) else reason,
            })
        bounds = {}
        for level, suffix in ((0.80, "80"), (0.95, "95")):
            lower, upper, quantile, rank, valid = conformal_bounds(
                result.prediction, residuals, level, int(self.config["conformal"]["minimum_residual_count"])
            )
            bounds.update({f"lower_{suffix}": lower, f"upper_{suffix}": upper,
                           f"interval_{suffix}_valid": valid, f"conformal_q_{suffix}": quantile,
                           f"conformal_rank_{suffix}": rank})
        interval_seconds = time.perf_counter() - interval_started
        history = self.data.loc[self.data["year"] <= origin, target].to_numpy(dtype=float)
        actual = float(self.lookup.loc[int(outer.target_year), target])
        total_seconds = time.perf_counter() - total_started
        timing = {"selection_family": family, "target": target, "model": model, "horizon": horizon,
                  "origin_year": origin, "feature_seconds": result.feature_seconds,
                  "selection_seconds": selection_seconds, "fit_seconds": result.fit_seconds,
                  "forecast_seconds": result.forecast_seconds, "interval_seconds": interval_seconds,
                  "total_seconds": total_seconds}
        self.timings.append(timing)
        return {
            "analysis_window": "primary_2000_2023", "target": target, "model": model,
            "model_family": "classical_ml", "horizon": horizon, "origin_year": origin,
            "target_year": int(outer.target_year), "training_start_year": int(outer.training_start_year),
            "training_end_year": origin, "training_n": int(outer.training_n), "actual": actual,
            "point_forecast": result.prediction, **bounds, "fit_status": "success", "fit_warning": "",
            "selected_specification": _json({"representation": representation, "lookback": lookback, **params}),
            "representation": representation, "lookback": lookback, "hyperparameters_json": _json(params),
            "selection_criterion": "nested_expanding_inner_rmse_then_mae", "selection_score": chosen["inner_rmse"],
            "selection_seconds": selection_seconds, "fit_seconds": result.fit_seconds,
            "forecast_seconds": result.forecast_seconds, "interval_seconds": interval_seconds,
            "total_seconds": total_seconds, "seed": int(self.config["runtime"]["random_seed"]),
            "last_observed_value": float(history[-1]),
            "in_sample_naive_mae": float(np.mean(np.abs(np.diff(history)))),
            "in_sample_naive_mse": float(np.mean(np.square(np.diff(history)))),
            "calibration_n": len(residuals), "selection_family": family,
        }

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        primary = self.manifest.loc[self.manifest["analysis_window"] == "primary_2000_2023"]
        primary_rows, raw_rows = [], []
        for outer in primary.itertuples(index=False):
            for model in self.models:
                try:
                    primary_record=self._outer_record(
                        outer, model, self.config["representations"]["primary"], "primary_transformed"
                    )
                    primary_rows.append(primary_record)
                    primary_params=json.loads(primary_record["hyperparameters_json"])
                    raw_rows.append(self._outer_record(
                        outer, model, self.config["representations"]["diagnostic"], "raw_level_ablation",
                        {"lookback":primary_record["lookback"],"params":primary_params},
                    ))
                except Exception as exc:
                    self.failures.append({"target": outer.target, "model": model, "horizon": outer.horizon,
                                          "origin_year": outer.origin_year, "error": f"{type(exc).__name__}: {exc}"})
                    raise
        def with_recent(rows):
            frame = pd.DataFrame(rows)
            recent = frame.loc[frame["target_year"] >= int(self.config["backtest"]["recent_window_start_year"])].copy()
            recent["analysis_window"] = "recent_2016_2023"
            return pd.concat([frame, recent], ignore_index=True).sort_values(
                ["analysis_window", "target", "model", "horizon", "origin_year"]
            ).reset_index(drop=True)
        return with_recent(primary_rows), with_recent(raw_rows)
