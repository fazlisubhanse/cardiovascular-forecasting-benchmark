"""Fourteen internal diagnostic figures for Phase 2B."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _line_metric(metrics, metric, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, target in zip(axes, ["asph", "mtc"]):
        for model, group in metrics.loc[metrics.target == target].groupby("model"):
            ax.plot(group.horizon, group[metric], marker="o", label=model)
        ax.set(title=target.upper(), xlabel="Horizon", ylabel=metric.upper()); ax.set_xticks([1,2,5]); ax.grid(alpha=.25)
    axes[1].legend(bbox_to_anchor=(1.02,1), loc="upper left", fontsize=7); fig.suptitle(title); _save(fig, path)


def generate_phase2b_figures(data: pd.DataFrame, ml: pd.DataFrame, raw: pd.DataFrame,
                             combined_metrics: pd.DataFrame, interval_metrics: pd.DataFrame,
                             selected: pd.DataFrame, timings: pd.DataFrame, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True); paths=[]
    primary_metrics = combined_metrics.loc[combined_metrics.analysis_window == "primary_2000_2023"]
    for metric, name in (("rmse","combined_rmse_by_horizon.png"),("mase","combined_mase_by_horizon.png")):
        path=output/name; _line_metric(primary_metrics, metric, path, f"Combined primary {metric.upper()}"); paths.append(path)
    for target in ["asph","mtc"]:
        fig, axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
        for ax,h in zip(axes,[1,2,5]):
            subset=ml.loc[(ml.analysis_window=="primary_2000_2023")&(ml.target==target)&(ml.horizon==h)]
            ax.plot(data.year,data[target],color="black",label="Observed")
            for model,g in subset.groupby("model"): ax.plot(g.target_year,g.point_forecast,marker=".",label=model)
            ax.set_title(f"h={h}"); ax.grid(alpha=.2)
        axes[0].legend(ncol=4,fontsize=8); path=output/f"rolling_ml_forecasts_{target}.png"; _save(fig,path); paths.append(path)
    for column,name,title in (("representation","representation_selection_frequency.png","Selected representation"),
                              ("lookback","lookback_selection_frequency.png","Selected lookback")):
        counts=selected.loc[selected.selection_family=="primary_transformed"][column].value_counts()
        fig,ax=plt.subplots(figsize=(8,5)); counts.plot.bar(ax=ax); ax.set_ylabel("Selections"); ax.set_title(title)
        path=output/name; _save(fig,path); paths.append(path)
    for level in [80,95]:
        subset=interval_metrics.loc[(interval_metrics.analysis_window=="primary_2000_2023") & np.isclose(interval_metrics.nominal_level,level/100)]
        pivot=subset.pivot_table(index="model",columns="target",values="coverage_probability")
        fig,ax=plt.subplots(figsize=(9,5)); pivot.plot.bar(ax=ax); ax.axhline(level/100,color="black",ls="--"); ax.set_ylim(0,1.05); ax.set_title(f"{level}% conformal coverage")
        path=output/f"conformal_coverage_{level}.png"; _save(fig,path); paths.append(path)
    subset=interval_metrics.loc[interval_metrics.analysis_window=="primary_2000_2023"]
    fig,ax=plt.subplots(figsize=(9,5)); subset.groupby(["model","nominal_level"]).mean_interval_width.mean().unstack().plot.bar(ax=ax); ax.set_title("Mean conformal interval width")
    path=output/"conformal_interval_width.png"; _save(fig,path); paths.append(path)
    diag=raw.loc[raw.analysis_window=="primary_2000_2023"].copy(); diag["outside"]=(diag.point_forecast<diag.training_min)|(diag.point_forecast>diag.training_max)
    fig,ax=plt.subplots(figsize=(8,5)); diag.groupby(["target","model"]).outside.mean().unstack().plot.bar(ax=ax); ax.set_ylabel("Proportion"); ax.set_title("Raw-level training-range extrapolation")
    path=output/"raw_level_extrapolation_rate.png"; _save(fig,path); paths.append(path)
    merged=ml.loc[(ml.analysis_window=="primary_2000_2023")&(ml.target=="mtc")].merge(
        raw.loc[(raw.analysis_window=="primary_2000_2023")&(raw.target=="mtc")], on=["target","model","horizon","origin_year"], suffixes=("_transformed","_raw"))
    fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(merged.point_forecast_raw,merged.point_forecast_transformed,alpha=.6); lo=min(merged.point_forecast_raw.min(),merged.point_forecast_transformed.min()); hi=max(merged.point_forecast_raw.max(),merged.point_forecast_transformed.max()); ax.plot([lo,hi],[lo,hi],ls="--",color="black"); ax.set(xlabel="Raw forecast",ylabel="Transformed forecast",title="MTC raw vs transformed forecasts")
    path=output/"mtc_raw_vs_transformed.png"; _save(fig,path); paths.append(path)
    fig,ax=plt.subplots(figsize=(8,5)); timings.groupby("model").total_seconds.sum().sort_values().plot.bar(ax=ax); ax.set_ylabel("Seconds"); ax.set_title("Total computation time")
    path=output/"computation_time_by_model.png"; _save(fig,path); paths.append(path)
    fig,ax=plt.subplots(figsize=(10,5)); timings.loc[timings.selection_family=="primary_transformed"].groupby("origin_year").selection_seconds.sum().plot(ax=ax); ax.set_ylabel("Seconds"); ax.set_title("Nested-selection time by outer origin")
    path=output/"selection_time_by_origin.png"; _save(fig,path); paths.append(path)
    errors=ml.loc[ml.analysis_window=="primary_2000_2023"].assign(error=lambda x:x.actual-x.point_forecast)
    fig,ax=plt.subplots(figsize=(9,5)); [ax.hist(g.error,alpha=.45,bins=15,label=m) for m,g in errors.groupby("model")]; ax.legend(); ax.set_title("Primary ML error distributions")
    path=output/"ml_error_distributions.png"; _save(fig,path); paths.append(path)
    return paths
