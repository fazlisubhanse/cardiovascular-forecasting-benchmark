"""Phase 2B.1 reporting-only correction for frozen interval coverage outputs.

This script reads already-frozen Phase 2B artifacts. It performs no fitting and
does not generate forecasts, predictions, metrics, rankings, or inference.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "round2"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

PROTECTED_HASHES = {
    "data/processed/china_male_official_cardiovascular_series.csv":
        "4575b29feee41e3832c9c2e51c9c0f8f0e312d5e7f4df9b87ce75e13e705c3e2",
    "config/round2_frozen_evaluation_protocol.json":
        "3f9ab7d20a29451bc2a105492195c415b25ee9d09ece1b7ca492e7bd0942cf5f",
    "results/round2/predictions_long.csv":
        "37ccdeaf018fbf9296c2c2b8f4bcdb2cdfea27ef5a89b3786582c1954b305f59",
    "results/round2/primary_rankings.csv":
        "ffad06b3c7d35e571931d09495b155f9885c9ebca7d055e9af740e0f992fc51b",
    "results/round2/inference.csv":
        "c9310c53b787c378d21fecf68d6412b07e5c8f5606d3b2127ae4eb32e99c6445",
    "results/round2/ablation_results.csv":
        "f8cfc046d17fcc4da1721458e5ce31a8647424964d3318c483d97a8ad1dc99ad",
    "results/round2/interval_coverage.csv":
        "f5c920d9b5d44db7f993fac8783d7485514699bf5bba7f96c26e7697f39a46c0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protected_files() -> None:
    failures: list[str] = []
    for relative, expected in PROTECTED_HASHES.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            failures.append(f"{relative}: expected {expected}, observed {actual}")
    if failures:
        raise RuntimeError("Protected frozen artifact hash mismatch:\n" + "\n".join(failures))


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".phase2b1.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def load_interval_capable_rows() -> pd.DataFrame:
    coverage = pd.read_csv(RESULTS / "interval_coverage.csv")
    required = {
        "target", "horizon", "model_family", "model_label", "nominal_coverage",
        "forecast_n", "valid_interval_n", "empirical_coverage", "interval_method",
        "mean_interval_width", "small_sample_flag",
    }
    missing = sorted(required.difference(coverage.columns))
    if missing:
        raise ValueError(f"Missing interval-coverage columns: {missing}")

    keys = ["target", "horizon", "model_family", "model_label", "nominal_coverage"]
    if coverage.duplicated(keys).any():
        raise ValueError("Duplicate model-specification/nominal-level coverage rows")
    if (coverage["valid_interval_n"] < 0).any():
        raise ValueError("Negative valid interval count")
    if (coverage["valid_interval_n"] > coverage["forecast_n"]).any():
        raise ValueError("Valid interval count exceeds forecast count")

    capable = coverage.loc[coverage["valid_interval_n"].gt(0)].copy()
    if not capable["model_family"].isin(["statistical", "classical_ml"]).all():
        raise ValueError("Unexpected interval-capable model family")
    if capable["empirical_coverage"].isna().any():
        raise ValueError("Interval-capable row has missing coverage")
    if not capable["empirical_coverage"].between(0, 1).all():
        raise ValueError("Coverage outside [0, 1]")

    covered = capable["empirical_coverage"] * capable["valid_interval_n"]
    if not np.allclose(covered, np.rint(covered), atol=1e-10, rtol=0):
        raise ValueError("Coverage proportion does not reconstruct an integer covered count")
    capable["covered_model_forecast_intervals"] = np.rint(covered).astype(int)
    return capable


def build_coverage_summary(capable: pd.DataFrame) -> pd.DataFrame:
    group = ["target", "horizon", "nominal_coverage"]
    summary = capable.groupby(group, sort=False, observed=True).agg(
        interval_capable_model_specification_count=("model_label", "size"),
        total_valid_model_forecast_interval_predictions=("valid_interval_n", "sum"),
        minimum_valid_interval_n_across_specifications=("valid_interval_n", "min"),
        maximum_valid_interval_n_across_specifications=("valid_interval_n", "max"),
        unweighted_mean_of_model_level_empirical_coverage=("empirical_coverage", "mean"),
        covered_model_forecast_interval_predictions=("covered_model_forecast_intervals", "sum"),
        minimum_model_level_empirical_coverage=("empirical_coverage", "min"),
        maximum_model_level_empirical_coverage=("empirical_coverage", "max"),
    ).reset_index()
    summary["pooled_coverage_across_valid_model_forecast_interval_predictions"] = (
        summary["covered_model_forecast_interval_predictions"]
        / summary["total_valid_model_forecast_interval_predictions"]
    )
    summary = summary.drop(columns="covered_model_forecast_interval_predictions")
    summary = summary[[
        "target",
        "horizon",
        "nominal_coverage",
        "interval_capable_model_specification_count",
        "total_valid_model_forecast_interval_predictions",
        "minimum_valid_interval_n_across_specifications",
        "maximum_valid_interval_n_across_specifications",
        "unweighted_mean_of_model_level_empirical_coverage",
        "pooled_coverage_across_valid_model_forecast_interval_predictions",
        "minimum_model_level_empirical_coverage",
        "maximum_model_level_empirical_coverage",
    ]]
    target_order = pd.Categorical(summary["target"], ["ASPH", "MTC"], ordered=True)
    summary = summary.assign(_target_order=target_order).sort_values(
        ["_target_order", "horizon", "nominal_coverage"]
    ).drop(columns="_target_order").reset_index(drop=True)
    if len(summary) != 12:
        raise ValueError(f"Expected 12 target-horizon-level summaries, observed {len(summary)}")
    return summary


def build_winner_coverage(capable: pd.DataFrame) -> pd.DataFrame:
    rankings = pd.read_csv(RESULTS / "primary_rankings.csv")
    winners = rankings.loc[rankings["rank"].eq(1), ["target", "horizon", "model_label"]].copy()
    if len(winners) != 6 or winners.duplicated(["target", "horizon"]).any():
        raise ValueError("Frozen rankings do not contain exactly one winner in each of six cells")

    rows: list[dict[str, object]] = []
    for winner in winners.itertuples(index=False):
        cell = capable.loc[
            capable["target"].eq(winner.target)
            & capable["horizon"].eq(winner.horizon)
            & capable["model_label"].eq(winner.model_label)
        ].sort_values("nominal_coverage")
        if len(cell) != 2 or not np.allclose(cell["nominal_coverage"], [0.80, 0.95]):
            raise ValueError(f"Winner coverage is incomplete for {winner.target} h={winner.horizon}")
        if cell["forecast_n"].nunique() != 1 or cell["valid_interval_n"].nunique() != 1:
            raise ValueError(f"Winner interval counts differ by nominal level for {winner.target} h={winner.horizon}")
        if cell["interval_method"].nunique() != 1:
            raise ValueError(f"Winner interval methods differ by nominal level for {winner.target} h={winner.horizon}")
        forecast_n = int(cell["forecast_n"].iloc[0])
        if winner.target == "ASPH" and int(winner.horizon) == 5:
            caution = "EXTREME instability: only 5 forecast observations; descriptive coverage only."
        else:
            caution = f"Small evaluation window (n={forecast_n}); descriptive coverage only."
        rows.append({
            "target": winner.target,
            "horizon": int(winner.horizon),
            "winner": winner.model_label,
            "forecast_n": forecast_n,
            "valid_interval_n": int(cell["valid_interval_n"].iloc[0]),
            "nominal_80_coverage": 0.80,
            "empirical_80_coverage": float(cell.iloc[0]["empirical_coverage"]),
            "nominal_95_coverage": 0.95,
            "empirical_95_coverage": float(cell.iloc[1]["empirical_coverage"]),
            "interval_method": cell["interval_method"].iloc[0],
            "mean_interval_width_80": float(cell.iloc[0]["mean_interval_width"]),
            "mean_interval_width_95": float(cell.iloc[1]["mean_interval_width"]),
            "small_sample_caution": caution,
        })
    result = pd.DataFrame(rows)
    target_order = pd.Categorical(result["target"], ["ASPH", "MTC"], ordered=True)
    return result.assign(_target_order=target_order).sort_values(
        ["_target_order", "horizon"]
    ).drop(columns="_target_order").reset_index(drop=True)


def make_coverage_figure(capable: pd.DataFrame, summary: pd.DataFrame) -> Path:
    colors = {"statistical": "#276FBF", "classical_ml": "#D97706"}
    markers = {"statistical": "o", "classical_ml": "^"}
    labels = {"statistical": "Statistical", "classical_ml": "Classical ML conformal"}
    x_positions = {0.80: 0.0, 0.95: 1.0}
    family_offsets = {"statistical": -0.08, "classical_ml": 0.08}

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.86, wspace=0.18, hspace=0.38)
    for row_index, target in enumerate(["ASPH", "MTC"]):
        for column_index, horizon in enumerate([1, 2, 5]):
            ax = axes[row_index, column_index]
            cell = capable.loc[capable["target"].eq(target) & capable["horizon"].eq(horizon)]
            for level in [0.80, 0.95]:
                for family in ["statistical", "classical_ml"]:
                    part = cell.loc[
                        np.isclose(cell["nominal_coverage"], level)
                        & cell["model_family"].eq(family)
                    ].sort_values("model_label")
                    if part.empty:
                        continue
                    jitter = np.linspace(-0.035, 0.035, len(part)) if len(part) > 1 else np.array([0.0])
                    x = x_positions[level] + family_offsets[family] + jitter
                    sizes = 28 + 4.5 * part["valid_interval_n"].to_numpy(float)
                    ax.scatter(
                        x,
                        part["empirical_coverage"],
                        s=sizes,
                        marker=markers[family],
                        c=colors[family],
                        edgecolors="#1F2937",
                        linewidths=0.55,
                        alpha=0.82,
                        zorder=3,
                    )
            ax.axhline(0.80, color="#374151", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)
            ax.axhline(0.95, color="#374151", linestyle=":", linewidth=1.2, alpha=0.9, zorder=1)
            ax.set_xlim(-0.28, 1.28)
            ax.set_ylim(-0.04, 1.05)
            ax.set_xticks([0, 1], ["80% nominal", "95% nominal"])
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.65)
            ax.set_title(f"{target} h={horizon}", fontsize=11, weight="bold")
            note_rows = summary.loc[
                summary["target"].eq(target) & summary["horizon"].eq(horizon)
            ].sort_values("nominal_coverage")
            notes = []
            for note in note_rows.itertuples(index=False):
                notes.append(
                    f"{int(note.nominal_coverage * 100)}%: "
                    f"{note.interval_capable_model_specification_count} specs, "
                    f"total valid n={note.total_valid_model_forecast_interval_predictions}, "
                    f"n/spec={note.minimum_valid_interval_n_across_specifications}-"
                    f"{note.maximum_valid_interval_n_across_specifications}"
                )
            ax.text(
                0.5, -0.20, "\n".join(notes), transform=ax.transAxes, ha="center", va="top",
                fontsize=7.8, color="#374151",
            )
            if column_index == 0:
                ax.set_ylabel("Model-specification coverage proportion")

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["statistical"],
               markeredgecolor="#1F2937", markersize=8, label=labels["statistical"]),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=colors["classical_ml"],
               markeredgecolor="#1F2937", markersize=8, label=labels["classical_ml"]),
        Line2D([0], [0], color="#374151", linestyle="--", linewidth=1.2, label="80% reference"),
        Line2D([0], [0], color="#374151", linestyle=":", linewidth=1.4, label="95% reference"),
    ]
    fig.suptitle("Model-specification interval coverage by target and horizon", fontsize=15, weight="bold", y=0.975)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False)
    fig.text(
        0.5, 0.025,
        "Each point is one interval-capable model specification; marker size scales with its valid interval n. "
        "Neural models are excluded because no frozen neural predictive-interval method exists. "
        "Coverage proportions are descriptive, not inferential.",
        ha="center", va="bottom", fontsize=9, color="#374151",
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "empirical_interval_coverage_all_cells.png"
    temporary = path.with_name(path.stem + ".phase2b1.tmp.png")
    fig.savefig(temporary, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    temporary.replace(path)
    return path


def main() -> None:
    verify_protected_files()
    capable = load_interval_capable_rows()
    summary = build_coverage_summary(capable)
    winners = build_winner_coverage(capable)

    summary_path = TABLES / "table_interval_coverage_summary.csv"
    winner_path = TABLES / "table_winner_interval_coverage.csv"
    atomic_csv(summary, summary_path)
    atomic_csv(winners, winner_path)
    figure_path = make_coverage_figure(capable, summary)
    verify_protected_files()

    for path in [summary_path, winner_path, figure_path]:
        print(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
