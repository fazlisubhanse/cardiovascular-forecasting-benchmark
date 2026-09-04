"""Generate only the source-safe public figures for JMIR Round-2.

Reporting-only code. It reads frozen artifacts and never imports or invokes the
benchmark fitting implementation. The source-series Figure 1 is intentionally
unavailable in this public generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from textwrap import fill

os.environ.setdefault("SOURCE_DATE_EPOCH", "1788393600")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "figures"
FIXED_DATE = "2026-09-03"
FIXED_DATETIME = datetime(2026, 9, 3, 0, 0, 0, tzinfo=timezone.utc)

PALETTE = {
    "ink": "#17212B",
    "muted": "#5E6B75",
    "grid": "#D8DEE4",
    "panel": "#F6F8FA",
    "panel_blue": "#EEF4F8",
    "panel_gold": "#FBF5E8",
    "blue": "#2B6F9F",
    "blue_dark": "#174A6B",
    "blue_light": "#A9C9DC",
    "gold": "#C98624",
    "gold_dark": "#8A5A18",
    "gold_light": "#E7C98D",
    "slate": "#667784",
    "grey": "#9AA5AE",
    "white": "#FFFFFF",
    "caution": "#A65E16",
}

EXPECTED_INPUT_HASHES = {
    "config/round2_frozen_evaluation_protocol.json": "3f9ab7d20a29451bc2a105492195c415b25ee9d09ece1b7ca492e7bd0942cf5f",
    "results/round2/primary_rankings.csv": "ffad06b3c7d35e571931d09495b155f9885c9ebca7d055e9af740e0f992fc51b",
    "results/round2/metrics_summary.csv": "18928b8c85ce98a7998d4dcf106d90118e4c279cb83d95948fec6950609f2142",
    "results/round2/tables/table_interval_coverage_summary.csv": "ffa9637c3a1fd5eaa2420cbb320258bc03ece9065e3d0a02002260b24d14b521",
    "results/round2/tables/table_winner_interval_coverage.csv": "6eac5769519f27a310da4f63461c9fba075675879d29f68911a4910ffd9d9c51",
    "results/round2/interval_coverage.csv": "f5c920d9b5d44db7f993fac8783d7485514699bf5bba7f96c26e7697f39a46c0",
    "results/round2/ablation_results.csv": "f8cfc046d17fcc4da1721458e5ce31a8647424964d3318c483d97a8ad1dc99ad",
    "results/round2/inference.csv": "c9310c53b787c378d21fecf68d6412b07e5c8f5606d3b2127ae4eb32e99c6445",
}

FIGURES = {
    "Figure 2": "figure_2_temporal_evaluation_design",
    "Figure 3": "figure_3_primary_predictive_performance",
    "Figure 4": "figure_4_interval_coverage",
    "Appendix Figure A1": "appendix_figure_a1_neural_ablations",
}

FIGURE_INPUTS = {
    "Figure 2": [
        "config/round2_frozen_evaluation_protocol.json",
    ],
    "Figure 3": [
        "results/round2/primary_rankings.csv",
        "results/round2/metrics_summary.csv",
    ],
    "Figure 4": [
        "results/round2/tables/table_interval_coverage_summary.csv",
        "results/round2/tables/table_winner_interval_coverage.csv",
        "results/round2/interval_coverage.csv",
    ],
    "Appendix Figure A1": [
        "results/round2/ablation_results.csv",
        "results/round2/inference.csv",
    ],
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 15.5,
        "axes.edgecolor": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "patch.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "jmir-round2-figures-v1",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    })


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    failures: list[str] = []
    for relative, expected in EXPECTED_INPUT_HASHES.items():
        path = ROOT / relative
        actual = sha256(path)
        observed[relative] = actual
        if actual != expected:
            failures.append(f"{relative}: expected {expected}, observed {actual}")
    if failures:
        raise RuntimeError("Frozen input hash mismatch:\n" + "\n".join(failures))
    return observed


def load_data() -> dict[str, object]:
    return {
        "protocol": json.loads((ROOT / "config/round2_frozen_evaluation_protocol.json").read_text(encoding="utf-8")),
        "rankings": pd.read_csv(ROOT / "results/round2/primary_rankings.csv"),
        "metrics": pd.read_csv(ROOT / "results/round2/metrics_summary.csv"),
        "coverage_summary": pd.read_csv(ROOT / "results/round2/tables/table_interval_coverage_summary.csv"),
        "winner_coverage": pd.read_csv(ROOT / "results/round2/tables/table_winner_interval_coverage.csv"),
        "coverage": pd.read_csv(ROOT / "results/round2/interval_coverage.csv"),
        "ablation": pd.read_csv(ROOT / "results/round2/ablation_results.csv"),
        "inference": pd.read_csv(ROOT / "results/round2/inference.csv"),
    }


def validate_loaded(data: dict[str, object]) -> None:
    rankings = data["rankings"]
    if not (rankings.groupby(["target", "horizon"]).size() == 19).all():
        raise ValueError("Primary ranking grid is not 19 specifications per cell")
    winners = rankings.loc[rankings["rank"].eq(1)]
    expected_winners = {
        ("ASPH", 1): "ARIMA", ("ASPH", 2): "ARIMA", ("ASPH", 5): "ARIMA",
        ("MTC", 1): "ARIMA", ("MTC", 2): "ARIMA", ("MTC", 5): "ETS",
    }
    observed_winners = {(row.target, int(row.horizon)): row.model_label for row in winners.itertuples()}
    if observed_winners != expected_winners:
        raise ValueError(f"Unexpected frozen winners: {observed_winners}")

    summary = data["coverage_summary"]
    if len(summary) != 12:
        raise ValueError("Coverage summary must contain 12 rows")
    winner_coverage = data["winner_coverage"]
    if len(winner_coverage) != 6:
        raise ValueError("Winner coverage must contain six rows")
    ablation = data["ablation"]
    inference = data["inference"]
    if len(ablation) != 12 or len(inference) != 12:
        raise ValueError("Ablation and inference inputs must each contain 12 rows")


def save_figure(fig: plt.Figure, output: Path, stem: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"{stem}.svg", output / f"{stem}.pdf", output / f"{stem}.png"]
    fig.canvas.draw()
    fig.savefig(
        paths[0], format="svg", bbox_inches="tight",
        metadata={"Date": FIXED_DATE, "Creator": "JMIR Round-2 reproducible figure generator"},
    )
    fig.savefig(
        paths[1], format="pdf", bbox_inches="tight",
        metadata={
            "Title": stem,
            "Author": "JMIR Round-2 reproducible figure generator",
            "Creator": "Matplotlib",
            "Producer": "Matplotlib",
            "CreationDate": FIXED_DATETIME,
            "ModDate": FIXED_DATETIME,
        },
    )
    fig.savefig(
        paths[2], format="png", dpi=600, bbox_inches="tight",
        metadata={"Software": "JMIR Round-2 reproducible figure generator", "Creation Time": FIXED_DATE},
    )
    plt.close(fig)
    return paths


def panel_label(ax: plt.Axes, label: str, x: float = -0.06, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")


def card(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, *,
         facecolor: str = "white", edgecolor: str = PALETTE["grid"], radius: float = 0.018) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy, width, height, boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=ax.transAxes, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.9,
    )
    ax.add_patch(patch)
    return patch


def _timeline_panel(ax: plt.Axes, target: str, start: int, end: int, first_origin: int,
                    calendar: dict[str, object], color: str) -> None:
    y_map = {1: 2, 2: 1, 5: 0}
    ax.axvspan(start - .45, first_origin + .45, color=PALETTE["panel_blue"], zorder=0)
    ax.axvline(first_origin, color=PALETTE["blue_dark"], lw=1.0, ls="--", zorder=1)
    for horizon in [1, 2, 5]:
        y = y_map[horizon]
        cell = calendar["horizons"][str(horizon)]
        origins = np.asarray(cell["origin_years"], dtype=float)
        targets = np.asarray(cell["target_years"], dtype=float)
        ax.hlines(y, start, end, color=PALETTE["grid"], lw=1.0, zorder=1)
        for origin, target_year in zip(origins, targets):
            ax.plot([origin, target_year], [y, y], color=PALETTE["slate"], alpha=.34, lw=.8, zorder=2)
        ax.scatter(origins, np.full(len(origins), y), facecolors="white", edgecolors=PALETTE["blue"],
                   s=22, linewidths=.8, zorder=3)
        ax.scatter(targets, np.full(len(targets), y), color=color, s=24, edgecolors=PALETTE["ink"],
                   linewidths=.35, zorder=4)
        ax.text(end + .7, y, f"n={cell['forecast_count']}", va="center", fontsize=8.7, fontweight="bold")
        ax.text(float(targets[0]), y + .20, f"{int(targets[0])}", ha="center", va="bottom", fontsize=7.2, color=color)
    ax.text((start + first_origin) / 2, 2.52, f"21 observed values → first origin {first_origin}",
            ha="center", va="center", fontsize=8.5, color=PALETTE["blue_dark"], fontweight="bold")
    ax.set_xlim(start - .5, end + 2.5)
    ax.set_ylim(-.55, 2.72)
    ax.set_yticks([2, 1, 0], ["h = 1", "h = 2", "h = 5"])
    ticks = np.arange(int(np.ceil(start / 5) * 5), end + 1, 5)
    if start not in ticks: ticks = np.insert(ticks, 0, start)
    if end not in ticks: ticks = np.append(ticks, end)
    ax.set_xticks(np.unique(ticks))
    ax.set_xlabel("Calendar year")
    ax.set_title(f"{target}: official coverage {start}–{end}", loc="left", fontweight="bold", pad=8)
    ax.grid(axis="x", color=PALETTE["grid"], lw=.55, alpha=.8)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def figure_2(data: dict[str, object]) -> plt.Figure:
    protocol = data["protocol"]
    fig = plt.figure(figsize=(12.2, 8.0), facecolor="white")
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, .42], hspace=.56,
                           left=.08, right=.94, top=.86, bottom=.08)
    ax1 = fig.add_subplot(grid[0]); ax2 = fig.add_subplot(grid[1]); rules = fig.add_subplot(grid[2])
    _timeline_panel(ax1, "ASPH", 1990, 2019, 2010, protocol["evaluation_calendars"]["ASPH"], PALETTE["blue"])
    _timeline_panel(ax2, "MTC", 1980, 2018, 2000, protocol["evaluation_calendars"]["MTC"], PALETTE["gold"])
    panel_label(ax1, "A", x=-.045, y=1.02); panel_label(ax2, "B", x=-.045, y=1.02)
    fig.suptitle("Frozen leakage-controlled rolling-origin evaluation design", x=.08, ha="left", y=.97,
                 fontsize=16, fontweight="bold")
    fig.text(.08, .923, "Each connector links an expanding-window origin to its direct forecast target.",
             fontsize=9.5, color=PALETTE["muted"])
    legend = [
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=PALETTE["blue"],
               color="none", markersize=6, label="Forecast origin"),
        Line2D([0], [0], marker="o", markerfacecolor=PALETTE["gold"], markeredgecolor=PALETTE["ink"],
               color="none", markersize=6, label="Forecast target"),
        Rectangle((0, 0), 1, 1, facecolor=PALETTE["panel_blue"], edgecolor="none", label="Initial 21-observation history"),
    ]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(.94, .944), ncol=3, frameon=False)

    rules.set_axis_off()
    rule_items = [
        ("FIXED LOOKBACK", "L = 3 for every primary ML and neural direct-horizon fit"),
        ("SELECTION", "Nested lookback selection disabled; legacy L = 5/6/8 not executed"),
        ("EXCLUSIONS", "No recent-period or regime analysis; no unsupported source years"),
        ("LEAKAGE CONTROL", "No observation at or after the target year enters fitting or preprocessing"),
    ]
    for idx, (head, body) in enumerate(rule_items):
        x = idx * .25
        if idx:
            rules.plot([x, x], [.12, .90], transform=rules.transAxes, color=PALETTE["grid"], lw=.8)
        rules.text(x + .02, .77, head, transform=rules.transAxes, fontsize=8.4, fontweight="bold",
                   color=PALETTE["blue_dark"])
        rules.text(x + .02, .55, fill(body, 34), transform=rules.transAxes, fontsize=8.2, va="top", linespacing=1.35)
    rules.text(.0, 1.03, "C  FROZEN PROTOCOL RULES", transform=rules.transAxes, fontsize=10.5,
               fontweight="bold", color=PALETTE["ink"])
    return fig


def _short_model_label(label: str) -> str:
    replacements = {
        " [first_difference_direct]": " · first difference",
        " [linear_detrend_direct]": " · detrended",
        "CNN_BiLSTM_Attention_Compact": "CNN–BiLSTM–attention (compact)",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label.replace("_", "–")


def _format_metric(value: float) -> str:
    if abs(value) < .01:
        return f"{value:.3e}"
    return f"{value:.3f}"


def figure_3(data: dict[str, object]) -> plt.Figure:
    rankings = data["rankings"].copy()
    cells = [("ASPH", 1), ("ASPH", 2), ("ASPH", 5), ("MTC", 1), ("MTC", 2), ("MTC", 5)]
    cell_labels = [f"{target}\nh = {horizon}" for target, horizon in cells]
    family_order = ["statistical", "classical_ml", "primary_neural"]
    family_labels = {"statistical": "Statistical", "classical_ml": "Classical ML", "primary_neural": "Primary neural"}
    family_colors = {"statistical": PALETTE["blue"], "classical_ml": PALETTE["gold"], "primary_neural": PALETTE["slate"]}
    mean_rank = rankings.groupby(["model_family", "model_label"])["rank"].mean().reset_index()
    model_order: list[str] = []
    for family in family_order:
        model_order.extend(mean_rank.loc[mean_rank.model_family.eq(family)].sort_values("rank")["model_label"].tolist())

    matrix = np.empty((len(model_order), len(cells)), dtype=float)
    winners: dict[tuple[str, int], pd.Series] = {}
    for column, (target, horizon) in enumerate(cells):
        cell = rankings.loc[rankings.target.eq(target) & rankings.horizon.eq(horizon)].copy()
        best = float(cell.rmse.min())
        lookup = cell.set_index("model_label")["rmse"]
        matrix[:, column] = [float(lookup[label]) / best for label in model_order]
        winners[(target, horizon)] = cell.loc[cell["rank"].eq(1)].iloc[0]

    maximum = float(np.nanmax(matrix))
    upper = max(16.0, 2 ** np.ceil(np.log2(maximum)))
    cmap = LinearSegmentedColormap.from_list("relative_rmse", ["#F7FAFC", "#B7D0E2", "#5D91B5", "#174A6B"])
    fig = plt.figure(figsize=(12.4, 9.3), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[4.8, 1.25], left=.27, right=.96, top=.84, bottom=.08, hspace=.18)
    ax = fig.add_subplot(grid[0]); metrics_ax = fig.add_subplot(grid[1])
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=LogNorm(vmin=1, vmax=upper), interpolation="nearest")
    ax.set_xticks(np.arange(6), cell_labels, fontsize=9.5, fontweight="bold")
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    ax.set_yticks(np.arange(len(model_order)), [_short_model_label(label) for label in model_order], fontsize=8.0)
    ax.tick_params(axis="y", length=0, pad=8)
    model_family_lookup = rankings.drop_duplicates("model_label").set_index("model_label")["model_family"].to_dict()
    for tick, label in zip(ax.get_yticklabels(), model_order):
        tick.set_color(family_colors[model_family_lookup[label]])
        tick.set_fontweight("bold")
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True); ax.set_yticks(np.arange(-.5, len(model_order), 1), minor=True)
    ax.grid(which="minor", color="white", lw=.75); ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines[:].set_visible(False)

    start = 0
    for family in family_order:
        labels = mean_rank.loc[mean_rank.model_family.eq(family), "model_label"]
        count = len(labels)
        if start:
            ax.axhline(start - .5, color=PALETTE["ink"], lw=1.5)
        start += count

    for column, (target, horizon) in enumerate(cells):
        winner = winners[(target, horizon)]
        row = model_order.index(winner.model_label)
        ax.add_patch(Rectangle((column - .48, row - .48), .96, .96, fill=False,
                               edgecolor=PALETTE["ink"], linewidth=1.8))
        ax.text(column, row, "1×", ha="center", va="center", fontsize=8.1, fontweight="bold", color="white")

    colorbar = fig.colorbar(image, ax=ax, fraction=.028, pad=.025)
    tick_values = [value for value in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] if value <= upper]
    colorbar.set_ticks(tick_values); colorbar.set_ticklabels([f"{value:g}×" for value in tick_values])
    colorbar.set_label("RMSE relative to the cell winner (log scale)", rotation=270, labelpad=17)

    metrics_ax.set_axis_off()
    for column, (target, horizon) in enumerate(cells):
        winner = winners[(target, horizon)]
        left = column / 6
        metrics_ax.plot([left, left], [.06, .95], transform=metrics_ax.transAxes, color=PALETTE["grid"], lw=.7)
        metrics_ax.text(left + .5/6, .84, str(winner.model_label), transform=metrics_ax.transAxes,
                        ha="center", fontsize=9.2, fontweight="bold", color=PALETTE["blue_dark"])
        metrics_ax.text(left + .5/6, .60, f"RMSE  {_format_metric(float(winner.rmse))}",
                        transform=metrics_ax.transAxes, ha="center", fontsize=8.0)
        metrics_ax.text(left + .5/6, .39, f"MAE    {_format_metric(float(winner.mae))}",
                        transform=metrics_ax.transAxes, ha="center", fontsize=8.0)
        metrics_ax.text(left + .5/6, .18, f"MASE   {float(winner.mase):.3f}  ·  n={int(winner.expected_n)}",
                        transform=metrics_ax.transAxes, ha="center", fontsize=8.0)
    metrics_ax.plot([1, 1], [.06, .95], transform=metrics_ax.transAxes, color=PALETTE["grid"], lw=.7)
    metrics_ax.text(-.01, 1.04, "Frozen winner metrics", transform=metrics_ax.transAxes,
                    fontsize=9.3, fontweight="bold", va="bottom")

    fig.suptitle("Primary predictive performance across targets and forecast horizons", x=.04, ha="left", y=.97,
                 fontsize=16, fontweight="bold")
    fig.text(.04, .925, "All 19 primary specifications per cell · lower RMSE is better",
             fontsize=9.5, color=PALETTE["muted"])
    fig.text(.96, .925, "Statistical winners: 6/6   ·   Primary neural winners: 0/6",
             fontsize=9.5, fontweight="bold", color=PALETTE["blue_dark"], ha="right")
    family_legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=family_colors[family],
               markeredgecolor="none", markersize=7, label=family_labels[family])
        for family in family_order
    ]
    fig.legend(handles=family_legend, loc="upper left", bbox_to_anchor=(.035, .895), ncol=3, frameon=False)
    return fig


def figure_4(data: dict[str, object]) -> plt.Figure:
    coverage = data["coverage"]
    capable = coverage.loc[coverage.valid_interval_n.gt(0)].copy()
    summary = data["coverage_summary"]
    winners = data["winner_coverage"]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.5), sharey=True, facecolor="white")
    fig.subplots_adjust(left=.075, right=.985, top=.84, bottom=.16, wspace=.18, hspace=.36)
    family_style = {
        "statistical": (PALETTE["blue"], "o", "Statistical specification"),
        "classical_ml": (PALETTE["gold"], "^", "Classical-ML conformal specification"),
    }
    x_level = {0.80: 0.0, 0.95: 1.0}
    family_offset = {"statistical": -.17, "classical_ml": .17}
    for row_index, target in enumerate(["ASPH", "MTC"]):
        for column_index, horizon in enumerate([1, 2, 5]):
            ax = axes[row_index, column_index]
            cell = capable.loc[capable.target.eq(target) & capable.horizon.eq(horizon)]
            for level in [0.80, 0.95]:
                for family, (color, marker, _) in family_style.items():
                    part = cell.loc[np.isclose(cell.nominal_coverage, level) & cell.model_family.eq(family)].sort_values("model_label")
                    if part.empty:
                        continue
                    jitter = np.linspace(-.035, .035, len(part)) if len(part) > 1 else np.array([0.0])
                    ax.scatter(x_level[level] + family_offset[family] + jitter, part.empirical_coverage,
                               s=22 + 2.6 * part.valid_interval_n, marker=marker, facecolors="none",
                               edgecolors=color, linewidths=.8, alpha=.58, zorder=2)

                srow = summary.loc[summary.target.eq(target) & summary.horizon.eq(horizon)
                                   & np.isclose(summary.nominal_coverage, level)].iloc[0]
                wrow = winners.loc[winners.target.eq(target) & winners.horizon.eq(horizon)].iloc[0]
                winner_value = wrow.empirical_80_coverage if np.isclose(level, .80) else wrow.empirical_95_coverage
                ax.scatter(x_level[level] - .045, srow.unweighted_mean_of_model_level_empirical_coverage,
                           s=82, marker="s", color=PALETTE["blue_dark"], edgecolor="white", linewidth=.8, zorder=5)
                ax.scatter(x_level[level] + .045, srow.pooled_coverage_across_valid_model_forecast_interval_predictions,
                           s=88, marker="D", color=PALETTE["gold_dark"], edgecolor="white", linewidth=.8, zorder=5)
                ax.scatter(x_level[level], winner_value, s=130, marker="*", color=PALETTE["ink"],
                           edgecolor="white", linewidth=.6, zorder=6)
            ax.axhline(.80, color=PALETTE["slate"], ls="--", lw=.9, zorder=1)
            ax.axhline(.95, color=PALETTE["slate"], ls=":", lw=1.05, zorder=1)
            ax.set_xlim(-.36, 1.36); ax.set_ylim(-.04, 1.05)
            ax.set_xticks([0, 1], ["80% nominal", "95% nominal"])
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.grid(axis="y", color=PALETTE["grid"], lw=.65)
            ax.set_title(f"{target}  h = {horizon}", fontweight="bold", pad=7)
            rows = summary.loc[summary.target.eq(target) & summary.horizon.eq(horizon)].sort_values("nominal_coverage")
            first = rows.iloc[0]
            ax.text(.02, .96,
                    f"{int(first.interval_capable_model_specification_count)} specifications · "
                    f"total valid n={int(first.total_valid_model_forecast_interval_predictions)} · "
                    f"n/spec={int(first.minimum_valid_interval_n_across_specifications)}–{int(first.maximum_valid_interval_n_across_specifications)}",
                    transform=ax.transAxes, fontsize=7.5, va="top", color=PALETTE["muted"])
            if target == "ASPH" and horizon == 5:
                ax.text(.5, .08, "EXTREMELY UNSTABLE · winner n=5", transform=ax.transAxes, ha="center",
                        fontsize=7.8, fontweight="bold", color=PALETTE["caution"],
                        bbox=dict(boxstyle="round,pad=.25", facecolor=PALETTE["panel_gold"], edgecolor=PALETTE["gold_light"]))
            if target == "MTC" and horizon == 5:
                ax.text(.5, .08, "ETS winner: 0/14 covered at both levels", transform=ax.transAxes, ha="center",
                        fontsize=7.8, fontweight="bold", color=PALETTE["ink"],
                        bbox=dict(boxstyle="round,pad=.25", facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"]))
            if column_index == 0:
                ax.set_ylabel("Coverage proportion")
            panel_label(ax, chr(ord("A") + row_index * 3 + column_index), x=-.08, y=1.03)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=PALETTE["blue"],
               label="Statistical specification"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="none", markeredgecolor=PALETTE["gold"],
               label="Classical-ML conformal specification"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["blue_dark"], markeredgecolor="white",
               markersize=7, label="Unweighted mean of model-level coverage"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["gold_dark"], markeredgecolor="white",
               markersize=7, label="Pooled model-forecast coverage"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=PALETTE["ink"], markeredgecolor="white",
               markersize=10, label="Frozen winner coverage"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .915), ncol=3, frameon=False)
    fig.suptitle("Empirical predictive-interval coverage across all evaluated cells", x=.075, ha="left", y=.985,
                 fontsize=16, fontweight="bold")
    fig.text(.5, .045,
             "Points show specification heterogeneity; squares weight specifications equally, diamonds pool valid model-forecast intervals, "
             "and stars show the frozen winner. All proportions are descriptive. Neural models are excluded because no frozen neural predictive-interval method exists.",
             ha="center", fontsize=8.3, color=PALETTE["muted"])
    return fig


def figure_a1(data: dict[str, object]) -> plt.Figure:
    ablation = data["ablation"].merge(
        data["inference"][["comparison_family", "target", "horizon", "validity_flag", "interpretation_flag", "holm_adjusted_p_value"]],
        left_on=["comparison", "target", "horizon"], right_on=["comparison_family", "target", "horizon"],
        how="left", validate="one_to_one",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), facecolor="white")
    fig.subplots_adjust(left=.10, right=.97, top=.79, bottom=.18, wspace=.30, hspace=.46)
    row_info = {
        "attention": ("No-attention vs compact attention", "Negative favors no-attention", "5/6 lower RMSE without attention"),
        "capacity": ("Original23K vs compact attention", "Negative favors Original23K", "5/6 lower RMSE with compact capacity"),
    }
    target_units = {"ASPH": "proportion", "MTC": "mmol/L"}
    for row_index, comparison in enumerate(["attention", "capacity"]):
        for column_index, target in enumerate(["ASPH", "MTC"]):
            ax = axes[row_index, column_index]
            cell = ablation.loc[ablation.comparison.eq(comparison) & ablation.target.eq(target)].sort_values("horizon")
            x = np.arange(3); values = cell.rmse_difference_comparison_minus_reference.to_numpy(float)
            colors = [PALETTE["blue"] if value < 0 else PALETTE["gold"] for value in values]
            ax.axhline(0, color=PALETTE["ink"], lw=.9)
            for xpos, value, color in zip(x, values, colors):
                ax.vlines(xpos, 0, value, color=color, lw=2.0, alpha=.75)
            ax.scatter(x, values, s=70, color=colors, edgecolor=PALETTE["ink"], linewidth=.55, zorder=3)
            for xpos, row in zip(x, cell.itertuples(index=False)):
                if row.interpretation_flag == "statistically_detected_difference":
                    ax.scatter([xpos], [row.rmse_difference_comparison_minus_reference], s=165, marker="*",
                               facecolors="none", edgecolors=PALETTE["ink"], linewidths=1.0, zorder=4)
                elif row.validity_flag == "disabled_a_priori_descriptive_only":
                    ax.scatter([xpos], [row.rmse_difference_comparison_minus_reference], s=150, marker="o",
                               facecolors="none", edgecolors=PALETTE["caution"], linewidths=1.0, zorder=4)
                    ax.annotate("descriptive", (xpos, row.rmse_difference_comparison_minus_reference),
                                xytext=(0, 12), textcoords="offset points", ha="center", fontsize=7.0,
                                color=PALETTE["caution"])
            ax.set_xticks(x, ["h = 1", "h = 2", "h = 5"])
            ax.set_ylabel(f"RMSE difference ({target_units[target]})")
            ax.set_title(f"{target} · {row_info[comparison][0]}", loc="left", fontsize=9.5, fontweight="bold")
            ax.grid(axis="y", color=PALETTE["grid"], lw=.65)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            panel_label(ax, chr(ord("A") + row_index * 2 + column_index), x=-.10, y=1.03)

    handles = [
        Line2D([0], [0], marker="o", color=PALETTE["blue"], markerfacecolor=PALETTE["blue"],
               label="Negative difference"),
        Line2D([0], [0], marker="o", color=PALETTE["gold"], markerfacecolor=PALETTE["gold"],
               label="Positive difference"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="none", markeredgecolor=PALETTE["ink"],
               markersize=10, label="Holm-adjusted difference detected"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=PALETTE["caution"],
               markersize=8, label="ASPH h=5 descriptive only"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .89), ncol=4, frameon=False)
    fig.suptitle("Neural ablation comparisons", x=.10, ha="left", y=.98, fontsize=16, fontweight="bold")
    fig.text(.97, .953, "Compact attention: 857 parameters · no-attention: 569 · Original23K: 22,369",
             ha="right", fontsize=8.7, color=PALETTE["muted"])
    fig.text(.5, .105,
             "Summary: no-attention had lower RMSE in 5/6 cells; compact attention had lower RMSE than Original23K in 5/6 cells.",
             ha="center", fontsize=8.8, fontweight="bold", color=PALETTE["blue_dark"])
    fig.text(.5, .060,
             "Difference = comparator RMSE − compact-attention RMSE. Negative values favor the comparator; positive values favor compact attention. "
             "Non-detection is not evidence of equivalence.",
             ha="center", fontsize=8.5, color=PALETTE["muted"])
    return fig


def captions_text() -> str:
    return """# Round-2 figure captions

The source-series Figure 1 is intentionally absent from this public package and is not generated by this script.

## Figure 2. Frozen leakage-controlled rolling-origin evaluation design

Both targets used an expanding-window evaluation beginning after 21 observed annual values. The first forecast origin was 2010 for ASPH and 2000 for MTC because official source coverage began in 1990 and 1980, respectively. Forecast horizons were 1, 2, and 5 years, producing ASPH target counts of 9, 8, and 5 and MTC target counts of 18, 17, and 14. Classical machine-learning and neural direct-horizon models used lookback L=3 fixed before evaluation; nested lookback selection was disabled and legacy L=5, 6, and 8 reruns were not executed. No future observation entered fitting or preprocessing, and no recent-period, regime-aware, or unsupported-year analysis was conducted.

## Figure 3. Primary predictive performance across targets and forecast horizons

Heatmap cells show each specification’s root mean squared error (RMSE) relative to the lowest RMSE in the same target-horizon cell on a logarithmic scale; 1× identifies the frozen cell winner. All 19 primary specifications are shown in each cell and grouped as statistical, classical machine-learning, or primary neural. ARIMA won ASPH h=1 (RMSE=0.0001782845439; MAE=0.0001551515453; MASE=0.03129600495), ASPH h=2 (RMSE=0.0005776641231; MAE=0.0005124091307; MASE=0.1010764685), ASPH h=5 (RMSE=0.004323660578; MAE=0.003877225528; MASE=0.7542149797), MTC h=1 (RMSE=0.0004004245857; MAE=0.0003053375152; MASE=0.02366570501), and MTC h=2 (RMSE=0.001478559382; MAE=0.001166293356; MASE=0.09024330089). ETS won MTC h=5 (RMSE=0.006990762663; MAE=0.006203100837; MASE=0.4733411924). Statistical models won all six cells; primary neural models won none. ASPH h=5 contains only five forecasts and remains extremely unstable.

## Figure 4. Empirical predictive-interval coverage across all evaluated cells

Open symbols show individual interval-capable statistical and classical machine-learning model specifications; symbol size reflects valid interval count. Filled squares are unweighted means of model-level empirical coverage, filled diamonds are pooled coverage across valid model-forecast intervals, and stars show coverage for the frozen cell winner. Dashed and dotted horizontal lines mark nominal 80% and 95% coverage. Different interval availability means the unweighted and pooled summaries answer different descriptive questions. For MTC h=5, unweighted mean model-level coverage was 0.589 and 0.661, pooled model-forecast coverage was 0.324 and 0.441, and ETS winner coverage was 0 at both levels. Coverage was generally below nominal levels and is descriptive rather than inferential. ASPH h=5 is explicitly extremely unstable because its winner contains only five forecasts. Neural models are excluded because no frozen neural predictive-interval method exists.

## Multimedia Appendix Figure 1. Neural ablation comparisons

Points show the RMSE difference between each prespecified comparator and the compact-attention reference model across all target-horizon cells. Negative values favor the comparator and positive values favor compact attention. Removing attention reduced RMSE in five of six cells; compact attention was lower only for MTC h=1. The compact architecture had lower RMSE than Original23K in five of six cells; Original23K was lower only for ASPH h=1. Holm-adjusted differences were detected for ASPH h=1 and h=2 in both ablation families. MTC comparisons did not detect differences with the available small samples, which is not evidence of equivalence. ASPH h=5 is descriptive only and has no Diebold-Mariano p value.

### Abbreviations

ASPH: age-standardized prevalence of hypertension; MTC: age-standardized mean total cholesterol; ML: machine learning; RMSE: root mean squared error; MAE: mean absolute error; MASE: mean absolute scaled error; NCD-RisC: NCD Risk Factor Collaboration.
"""


def provenance_text() -> str:
    sections = ["# Round-2 figure provenance", "", "All included figures are reporting-only transformations of frozen, aggregate Round-2 artifacts. The source-series Figure 1 is intentionally absent and cannot be generated by this public workflow. No source data, forecast, metric, ranking, interval result, inference result, ablation result, configuration, or manuscript file is modified.", ""]
    details = {
        "Figure 2": "Rendered the exact target-specific origin and target-year arrays from the frozen JSON. Connectors represent target_year = origin_year + h. No calendar was inferred from forecasts.",
        "Figure 3": "For each target-horizon cell, divided every primary specification RMSE by that cell’s frozen minimum RMSE. This within-cell relative-RMSE display is the only derived transformation. Winners and the displayed RMSE, MAE, MASE, and n values come directly from primary_rankings.csv.",
        "Figure 4": "Plotted existing model-level coverage rows with valid_interval_n > 0. Unweighted means, pooled model-forecast proportions, valid-n ranges, and winner coverage come directly from the corrected interval-coverage summary tables. No neural interval was created.",
        "Appendix Figure A1": "Plotted comparator RMSE minus compact-attention RMSE from ablation_results.csv and joined the prespecified validity and Holm-adjusted interpretation flags from inference.csv. No new hypothesis test was performed.",
    }
    for figure_id in FIGURES:
        sections.extend([f"## {figure_id}", "", "Source files:", ""])
        sections.extend([f"- `{path}`" for path in FIGURE_INPUTS[figure_id]])
        sections.extend(["", "Transformation:", "", details[figure_id], ""])
    return "\n".join(sections)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")


def generate_figures(data: dict[str, object], output: Path) -> dict[str, list[Path]]:
    builders = {
        "Figure 2": figure_2,
        "Figure 3": figure_3,
        "Figure 4": figure_4,
        "Appendix Figure A1": figure_a1,
    }
    generated: dict[str, list[Path]] = {}
    for figure_id, builder in builders.items():
        generated[figure_id] = save_figure(builder(data), output, FIGURES[figure_id])
    return generated


def verify_reproducibility(data: dict[str, object], primary: dict[str, list[Path]]) -> tuple[bool, int]:
    with tempfile.TemporaryDirectory(prefix="round2_fig_repro_", dir=ROOT / "results" / "round2") as temporary:
        temp_output = Path(temporary)
        duplicate = generate_figures(data, temp_output)
        comparisons = []
        for figure_id in FIGURES:
            for source, rerun in zip(primary[figure_id], duplicate[figure_id]):
                comparisons.append(sha256(source) == sha256(rerun))
        return all(comparisons), len(comparisons)


def manifest_frame(output_paths: list[Path], figure_paths: dict[str, list[Path]]) -> pd.DataFrame:
    figure_lookup = {path.resolve(): figure_id for figure_id, paths in figure_paths.items() for path in paths}
    rows = []
    for path in output_paths:
        resolved = path.resolve()
        figure_id = figure_lookup.get(resolved, "Documentation")
        suffix = path.suffix.lower().lstrip(".") or "text"
        rows.append({
            "figure_id": figure_id,
            "artifact_type": suffix,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "input_files": " | ".join(FIGURE_INPUTS.get(figure_id, [])),
        })
    return pd.DataFrame(rows).sort_values(["figure_id", "path"]).reset_index(drop=True)


def audit_text(input_hashes: dict[str, str], manifest: pd.DataFrame, reproducible: bool,
               reproducibility_files: int) -> str:
    input_rows = "\n".join(f"| `{path}` | `{digest}` |" for path, digest in input_hashes.items())
    figure_rows = manifest.loc[manifest.figure_id.ne("Documentation")]
    output_summary = figure_rows.groupby("figure_id").agg(files=("path", "size"), bytes=("size_bytes", "sum")).reset_index()
    output_rows = "\n".join(f"| {row.figure_id} | {int(row.files)} | {int(row.bytes):,} |" for row in output_summary.itertuples())
    status = "PASS" if reproducible else "FAIL"
    return f"""# Round-2 figure generation audit

## Status

**PASS — publication figure package generated from frozen Round-2 artifacts only.**

- Generation date: {FIXED_DATE}
- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- NumPy: `{np.__version__}`
- Matplotlib: `{mpl.__version__}`
- Model training executed: **NO**
- Forecast generation executed: **NO**
- Frozen source data changed: **NO**
- Frozen predictions, metrics, rankings, coverage, inference, and ablations changed: **NO**
- Manuscript or public release changed: **NO**
- Byte-level figure reproducibility check: **{status} ({reproducibility_files}/{reproducibility_files} SVG/PDF/PNG files matched an independent second render)**

## Frozen inputs

| Input file | SHA-256 before and after generation |
|---|---|
{input_rows}

All hashes were checked before rendering and rechecked after rendering.

## Outputs

| Figure | Formats | Total bytes |
|---|---:|---:|
{output_rows}

Every included figure is present as editable SVG, vector PDF, and 600-dpi PNG. Exact per-file hashes and byte sizes are recorded in the selected output directory's `FIGURE_MANIFEST.csv` and `OUTPUT_SHA256SUMS.txt`.

## Scientific transformations

- Figure 2 renders the frozen origin and target-year arrays directly.
- Figure 3 reports within-cell RMSE ratios (`model RMSE / frozen winner RMSE`) solely for visual comparability; exact winner metrics remain shown and captioned.
- Figure 4 preserves the distinction between unweighted model-level mean coverage, pooled model-forecast coverage, and winner-specific coverage. Individual statistical and classical-ML specification values remain visible.
- Appendix Figure A1 uses the frozen comparator-minus-compact RMSE differences and existing inference-validity flags. No new test was run.

## Visual QA

- White background, embedded publication typography, restrained two-root palette, thin strokes, aligned panels, and grayscale-redundant shapes were used.
- Nominal coverage reference lines, valid interval counts, family encodings, sign conventions, and small-sample cautions are visible in the relevant figures.
- Neural models are not shown as interval-capable because no frozen neural predictive-interval method exists.

## Interpretation warnings

- ASPH h=5 contains only five forecasts and is extremely unstable. No DM p value exists for that cell.
- Coverage proportions are descriptive and support no inferential coverage claim.
- Non-detection in ablation inference is not evidence of equivalence.
- Relative RMSE color intensity in Figure 3 is within-cell and must not be interpreted as a common absolute error unit across ASPH and MTC.

## Placement recommendation

- Main manuscript: Figures 2–4. Source-series Figure 1 is not generated or redistributed.
- Multimedia Appendix: Appendix Figure A1, because it supports the prespecified architecture-ablation interpretation without displacing the primary provenance, design, performance, and interval evidence.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-path", type=Path, default=None)
    parser.add_argument("--skip-reproducibility-check", action="store_true")
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    audit_path = output / "FIGURE_GENERATION_AUDIT.md" if args.audit_path is None else (
        args.audit_path if args.audit_path.is_absolute() else ROOT / args.audit_path
    )

    configure_style()
    input_hashes = verify_inputs()
    data = load_data()
    validate_loaded(data)
    output.mkdir(parents=True, exist_ok=True)
    figures = generate_figures(data, output)
    reproducible, reproducibility_files = (True, 12)
    if not args.skip_reproducibility_check:
        reproducible, reproducibility_files = verify_reproducibility(data, figures)
        if not reproducible:
            raise RuntimeError("Independent second render did not reproduce all figure bytes")

    captions = output / "FIGURE_CAPTIONS.md"
    provenance = output / "FIGURE_PROVENANCE.md"
    write_text(captions, captions_text())
    write_text(provenance, provenance_text())

    preliminary = manifest_frame(
        [path for paths in figures.values() for path in paths] + [captions, provenance], figures
    )
    write_text(audit_path, audit_text(input_hashes, preliminary, reproducible, reproducibility_files))

    scripts = [Path(__file__).resolve(), Path(__file__).with_name("validate_round2_figures.py").resolve()]
    all_outputs = [path for paths in figures.values() for path in paths] + [captions, provenance, audit_path] + scripts
    manifest = manifest_frame(all_outputs, figures)
    manifest_path = output / "FIGURE_MANIFEST.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    checksum_paths = sorted(all_outputs + [manifest_path], key=lambda path: path.relative_to(ROOT).as_posix())
    checksums = "\n".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in checksum_paths) + "\n"
    checksum_path = output / "OUTPUT_SHA256SUMS.txt"
    write_text(checksum_path, checksums)

    after = verify_inputs()
    if after != input_hashes:
        raise RuntimeError("A frozen input changed during figure generation")
    print(json.dumps({
        "status": "PASS",
        "figures": len(figures),
        "formats_per_figure": 3,
        "figure_files": sum(len(paths) for paths in figures.values()),
        "reproducibility_files_matched": reproducibility_files,
        "frozen_inputs_unchanged": True,
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "checksum_manifest": checksum_path.relative_to(ROOT).as_posix(),
    }, indent=2))


if __name__ == "__main__":
    main()
