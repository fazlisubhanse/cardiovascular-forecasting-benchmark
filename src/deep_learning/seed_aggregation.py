"""Seed-level metric distributions and non-interval ensemble aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.aggregation import aggregate_point_metrics


def aggregate_seed_forecasts(seed_forecasts: pd.DataFrame, recent_start: int=2016) -> pd.DataFrame:
    rows=[]; keys=["target","architecture","model_role","horizon","origin_year","target_year"]
    for key,group in seed_forecasts.loc[seed_forecasts.analysis_window.eq("primary_2000_2023")].groupby(keys,sort=True):
        success=group.loc[group.fit_status.eq("success") & np.isfinite(group.point_forecast)]
        first=group.iloc[0].to_dict(); predictions=success.point_forecast.to_numpy(float); complete=len(success)==len(group)==3
        row={name:first.get(name,np.nan) for name in first}; row.update({
            "model":key[1],"architecture":key[1],"model_family":"deep_learning_pilot","seed_success_n":len(success),
            "seed_expected_n":3,"fit_status":"success" if complete else "incomplete_seed_ensemble",
            "fit_warning":"" if complete else "one or more seed forecasts failed",
            "point_forecast":float(np.mean(predictions)) if complete else np.nan,
            "seed_mean_prediction":float(np.mean(predictions)) if len(predictions) else np.nan,
            "seed_median_prediction":float(np.median(predictions)) if len(predictions) else np.nan,
            "seed_prediction_std":float(np.std(predictions,ddof=1)) if len(predictions)>1 else np.nan,
            "seed_min_prediction":float(np.min(predictions)) if len(predictions) else np.nan,
            "seed_max_prediction":float(np.max(predictions)) if len(predictions) else np.nan,
            "lower_80":np.nan,"upper_80":np.nan,"lower_95":np.nan,"upper_95":np.nan,
            "interval_80_valid":False,"interval_95_valid":False,
            "selected_specification":first.get("configuration_json",""),"selection_criterion":"development_targets_through_1999",
            "selection_score":np.nan,"selection_seconds":np.nan,
            "fit_seconds":float(group.fit_seconds.sum()) if "fit_seconds" in group else np.nan,
            "forecast_seconds":float(group.forecast_seconds.sum()) if "forecast_seconds" in group else np.nan,
            "total_seconds":float(group.total_seconds.sum()) if "total_seconds" in group else np.nan})
        rows.append(row)
    primary=pd.DataFrame(rows); primary["analysis_window"]="primary_2000_2023"
    recent=primary.loc[primary.target_year>=recent_start].copy(); recent["analysis_window"]="recent_2016_2023"
    return pd.concat([primary,recent],ignore_index=True).sort_values(
        ["analysis_window","target","architecture","horizon","origin_year"]).reset_index(drop=True)


def seed_metric_distribution(seed_forecasts: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    metric_rows=[]
    for seed,frame in seed_forecasts.groupby("seed"):
        models=frame.copy(); models["model"]=models.architecture
        metrics, diagnostics = aggregate_point_metrics(models, manifest)
        metrics = metrics.merge(
            diagnostics[["analysis_window", "target", "model", "horizon",
                         "error_standard_deviation", "median_absolute_error",
                         "maximum_absolute_error"]],
            on=["analysis_window", "target", "model", "horizon"], how="left"
        )
        metrics["seed"] = int(seed)
        metric_rows.append(metrics)
    long=pd.concat(metric_rows,ignore_index=True)
    measures=["rmse", "mae", "mase", "rmsse", "smape_percent", "mean_error",
              "median_error", "directional_accuracy_percent", "median_absolute_error",
              "maximum_absolute_error", "error_standard_deviation"]
    rows=[]
    for keys,group in long.groupby(["analysis_window","target","model","horizon"],sort=True):
        row={"analysis_window":keys[0],"target":keys[1],"architecture":keys[2],"horizon":int(keys[3])}
        for measure in measures:
            values=group[measure].to_numpy(float); row.update({f"mean_seed_{measure}":float(np.mean(values)),
                f"std_seed_{measure}":float(np.std(values,ddof=1)),f"min_seed_{measure}":float(np.min(values)),
                f"max_seed_{measure}":float(np.max(values))})
        rows.append(row)
    return pd.DataFrame(rows)
