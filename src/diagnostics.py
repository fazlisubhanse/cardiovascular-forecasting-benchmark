"""Internal, audit-only time-series diagnostic figure generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

LOGGER = logging.getLogger(__name__)


def _save(fig: plt.Figure, path: Path) -> None:
    """Save and always close a matplotlib figure."""

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_diagnostics(frame: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> list[Path]:
    """Generate the five required audit-only diagnostic figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    years = frame["year"]
    specs = (("asph", "Hypertension prevalence (proportion)"), ("mtc", "Mean total cholesterol (mmol/L)"))
    paths: list[Path] = []

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, (column, label) in zip(axes, specs):
        ax.plot(years, frame[column], marker="o", markersize=3, linewidth=1.4)
        ax.axvline(2015.5, color="firebrick", linestyle="--", label="2015/2016 boundary")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Year")
    axes[0].set_title("Observed annual series (audit diagnostic)")
    path = output_dir / "series_levels_audit.png"; _save(fig, path); paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    difference_labels = {"asph": "Annual ASPH change (proportion)", "mtc": "Annual MTC change (mmol/L)"}
    for ax, (column, label) in zip(axes, specs):
        ax.axhline(0, color="black", linewidth=0.8)
        ax.plot(years, frame[column].diff(), marker="o", markersize=3, linewidth=1.2)
        ax.set_ylabel(difference_labels[column])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Year")
    axes[0].set_title("First differences (audit diagnostic)")
    path = output_dir / "first_differences_audit.png"; _save(fig, path); paths.append(path)

    window = int(config["diagnostics"]["rolling_window"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, (column, label) in zip(axes, specs):
        rolling = frame[column].rolling(window=window)
        ax.plot(years, frame[column], color="0.65", label="Observed")
        ax.plot(years, rolling.mean(), label=f"{window}-year rolling mean")
        ax2 = ax.twinx()
        ax2.plot(years, rolling.std(), color="darkorange", linestyle="--", label=f"{window}-year rolling SD")
        ax.set_ylabel(label)
        ax2.set_ylabel("Rolling SD")
        handles, labels = ax.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, loc="best")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Year")
    axes[0].set_title(f"Rolling statistics ({window}-year trailing window)")
    path = output_dir / "rolling_statistics_audit.png"; _save(fig, path); paths.append(path)

    acf_lags = min(int(config["diagnostics"]["acf_lags"]), len(frame) // 2 - 1)
    pacf_lags = min(int(config["diagnostics"]["pacf_lags"]), len(frame) // 2 - 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, (column, label) in enumerate(specs):
        plot_acf(frame[column], lags=acf_lags, ax=axes[row, 0], zero=False)
        plot_pacf(frame[column], lags=pacf_lags, ax=axes[row, 1], zero=False, method="ywm")
        axes[row, 0].set_title(f"ACF: {label}")
        axes[row, 1].set_title(f"PACF: {label}")
    path = output_dir / "acf_pacf_audit.png"; _save(fig, path); paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, (column, label) in zip(axes, specs):
        train_max = frame.loc[frame["year"] <= 2015, column].max()
        exceeds = (frame["year"] >= 2016) & (frame[column] > train_max)
        ax.plot(years, frame[column], color="steelblue", label="Observed")
        ax.axhline(train_max, color="firebrick", linestyle="--", label="Training maximum through 2015")
        ax.scatter(frame.loc[exceeds, "year"], frame.loc[exceeds, column], color="darkorange", zorder=3,
                   label="2016-2023 exceeds training maximum")
        ax.axvline(2015.5, color="0.3", linestyle=":")
        ax.set_ylabel(label); ax.grid(alpha=0.2); ax.legend(loc="best")
    axes[-1].set_xlabel("Year")
    axes[0].set_title("Training-range exceedance audit")
    path = output_dir / "training_range_exceedance_audit.png"; _save(fig, path); paths.append(path)
    LOGGER.info("Generated %d audit figures", len(paths))
    return paths
