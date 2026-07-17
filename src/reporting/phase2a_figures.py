"""Internal Phase 2A diagnostic figure generation using matplotlib only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _metric_by_horizon(metrics: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, target in zip(axes, ["asph", "mtc"]):
        subset = metrics.loc[metrics["target"] == target]
        for model, group in subset.groupby("model"):
            ax.plot(group["horizon"], group[metric], marker="o", label=model)
        ax.set_title(target.upper()); ax.set_xlabel("Forecast horizon (years)"); ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 5]); ax.grid(alpha=0.25)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(title + " (internal diagnostic)")
    _save(fig, path)


def generate_phase2a_figures(
    data: pd.DataFrame,
    forecasts: pd.DataFrame,
    primary_metrics: pd.DataFrame,
    recent_metrics: pd.DataFrame,
    interval_metrics: pd.DataFrame,
    computational_cost: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate all ten requested internal baseline diagnostic figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for metric, ylabel, title, filename in (
        ("rmse", "RMSE", "Primary RMSE by horizon", "primary_rmse_by_horizon.png"),
        ("mase", "MASE", "Primary MASE by horizon", "primary_mase_by_horizon.png"),
        ("directional_accuracy_percent", "Directional accuracy (%)", "Directional accuracy by horizon", "directional_accuracy_by_horizon.png"),
    ):
        path = output_dir / filename
        _metric_by_horizon(primary_metrics, metric, ylabel, title, path); paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    models = sorted(primary_metrics["model"].unique())
    x = np.arange(len(models)); width = 0.25
    for ax, target in zip(axes, ["asph", "mtc"]):
        subset = primary_metrics.loc[primary_metrics["target"] == target]
        for index, horizon in enumerate([1, 2, 5]):
            lookup = subset.loc[subset["horizon"] == horizon].set_index("model")["mean_error"]
            ax.bar(x + (index - 1) * width, [lookup.get(model, np.nan) for model in models], width, label=f"h={horizon}")
        ax.axhline(0, color="black", linewidth=0.8); ax.set_ylabel(f"{target.upper()} mean error")
        ax.grid(axis="y", alpha=0.2); ax.legend()
    axes[-1].set_xticks(x, models, rotation=30, ha="right")
    fig.suptitle("Directional bias: positive values indicate underforecasting (internal diagnostic)")
    path = output_dir / "mean_error_by_model.png"; _save(fig, path); paths.append(path)

    for level, filename in ((0.80, "interval_coverage_80.png"), (0.95, "interval_coverage_95.png")):
        subset_level = interval_metrics.loc[
            (interval_metrics["analysis_window"] == "primary_2000_2023")
            & np.isclose(interval_metrics["nominal_level"], level)
        ]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, target in zip(axes, ["asph", "mtc"]):
            subset = subset_level.loc[subset_level["target"] == target]
            for model, group in subset.groupby("model"):
                ax.plot(group["horizon"], 100 * group["coverage_probability"], marker="o", label=model)
            ax.axhline(100 * level, color="black", linestyle="--", label="Nominal")
            ax.set_title(target.upper()); ax.set_xlabel("Forecast horizon (years)"); ax.set_ylabel("Empirical coverage (%)")
            ax.set_xticks([1, 2, 5]); ax.set_ylim(0, 105); ax.grid(alpha=0.2)
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.suptitle(f"{int(level*100)}% interval coverage (internal diagnostic)")
        path = output_dir / filename; _save(fig, path); paths.append(path)

    primary_forecasts = forecasts.loc[forecasts["analysis_window"] == "primary_2000_2023"]
    for target, filename in (("asph", "rolling_forecasts_asph.png"), ("mtc", "rolling_forecasts_mtc.png")):
        best = primary_metrics.loc[(primary_metrics["target"] == target) & (primary_metrics["horizon"] == 1)].nsmallest(3, "rmse")["model"].tolist()
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(data["year"], data[target], color="black", linewidth=1.8, label="Observed")
        subset = primary_forecasts.loc[
            (primary_forecasts["target"] == target) & (primary_forecasts["horizon"] == 1)
            & primary_forecasts["model"].isin(best) & primary_forecasts["fit_status"].eq("success")
        ]
        for model, group in subset.groupby("model"):
            ax.plot(group["target_year"], group["point_forecast"], marker="o", linestyle="--", label=f"{model} rolling-origin")
        ax.axvline(1999.5, color="firebrick", linestyle=":", label="Primary evaluation begins")
        ax.set_title(f"{target.upper()} one-year expanding-origin forecasts (internal diagnostic)")
        ax.set_xlabel("Target year"); ax.set_ylabel(target.upper()); ax.grid(alpha=0.2); ax.legend()
        path = output_dir / filename; _save(fig, path); paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    models = sorted(recent_metrics["model"].unique()); x = np.arange(len(models)); width = 0.25
    for ax, target in zip(axes, ["asph", "mtc"]):
        subset = recent_metrics.loc[recent_metrics["target"] == target]
        for index, horizon in enumerate([1, 2, 5]):
            lookup = subset.loc[subset["horizon"] == horizon].set_index("model")["rmse"]
            ax.bar(x + (index - 1) * width, [lookup.get(model, np.nan) for model in models], width, label=f"h={horizon}")
        ax.set_title(target.upper()); ax.set_ylabel("RMSE"); ax.grid(axis="y", alpha=0.2); ax.legend()
        ax.set_xticks(x, models, rotation=30, ha="right")
    fig.suptitle("Recent 2016-2023 target-year sensitivity (internal diagnostic)")
    path = output_dir / "recent_window_comparison.png"; _save(fig, path); paths.append(path)

    primary_cost = computational_cost.loc[computational_cost["analysis_window"] == "primary_2000_2023"]
    totals = primary_cost.groupby("model")["total_seconds"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(totals.index, totals.values); ax.set_ylabel("Measured total seconds"); ax.set_xlabel("Model")
    ax.set_title("Observed computation time on this run (internal diagnostic)")
    ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=0.2)
    path = output_dir / "computation_time_by_model.png"; _save(fig, path); paths.append(path)
    return paths
