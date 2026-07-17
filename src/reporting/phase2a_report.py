"""Cautious Markdown method, result, and failure reports for Phase 2A."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_method_validation_report(
    path: Path, manifest: pd.DataFrame, config: dict[str, Any], validated_hash: str
) -> None:
    """Document the exact evaluation design and mathematical definitions."""

    counts = manifest.groupby(["analysis_window", "target", "horizon"]).size()
    lines = [
        "# Phase 2A Method Validation", "", "## Integrity and design", "",
        f"Validated-data SHA-256: `{validated_hash}`. The validated dataset and configurations are protected read-only inputs to this phase, except the explicitly generated environment snapshots.", "",
        "The primary analysis uses an expanding window ending initially in 1999. At every origin `t`, fitting, model selection, residual construction, and interval estimation use only years `<=t`; the observed target is year `t+h`. Models are refitted at every distinct origin. The recent sensitivity view filters the same correctly generated rolling forecasts by target years 2016-2023; it is not a fixed-end-2015 forecast.", "",
        "## Forecast-origin counts", "", "| Window | Target | Horizon | Origins | Target years |", "|---|---|---:|---:|---|",
    ]
    for (window, target, horizon), count in counts.items():
        subset = manifest.loc[(manifest["analysis_window"] == window) & (manifest["target"] == target) & (manifest["horizon"] == horizon)]
        lines.append(f"| {window} | {target} | {horizon} | {count} | {subset['target_year'].min()}-{subset['target_year'].max()} |")
    lines.extend([
        "", "Every horizon therefore has multiple forecast origins. A single fixed-origin error at H=2 or H=5 is an absolute error, not RMSE; RMSE requires the square root of the mean squared errors across multiple paired targets. Phase 2A stores each forecast first and only then aggregates across origins.", "",
        "## Model selection", "",
        "- Naive: last training observation at every horizon.",
        "- Drift: last observation plus `h` times the average first-to-last annual change.",
        "- Linear trend: OLS on the actual calendar year, with regression observation prediction intervals.",
        "- ETS: nonseasonal level, additive-trend, and additive-damped-trend candidates; finite AICc selects the current-origin specification. Failed candidates remain recorded.",
        "- ARIMA: bounded `(p,d,q)` grid with admissible no-trend, constant, linear-trend, constant-plus-trend, and first-difference drift choices under statsmodels constraints. Finite fitted parameters, AICc, and forecast production are required. Selection never sees the future target.",
        "- Theta: statsmodels `ThetaModel(period=1, deseasonalize=False, use_test=False)`; no substitute is relabeled Theta after failure.", "",
        "## Prediction intervals", "",
        "ARIMA, linear trend, and Theta use statsmodels model-native 80% and 95% intervals. Naive and drift use centered empirical historical h-step error quantiles calculated solely within the current training prefix. ETS uses centered fitted-residual quantiles from the selected current-origin fit because statsmodels Holt-Winters does not expose reliable native prediction intervals. At least eight residuals are required. Missing or invalid intervals are not counted as covered.", "",
        "## Point metrics", "",
        "Errors are `actual - forecast`, so positive Mean Error indicates underforecasting. RMSE and MAE are original-scale averages across origins. SMAPE uses `100*mean(2*|a-f|/(|a|+|f|))`, with a zero contribution when both values are zero. MASE is the mean of each absolute error divided by that origin's in-sample one-step naive MAE. RMSSE is the square root of the mean of each squared error divided by that origin's in-sample one-step naive MSE. Directional accuracy compares `sign(forecast-last observed)` with `sign(actual-last observed)`; zero is a separate direction and is correct only when both changes are exactly zero. R-squared is not used.", "",
        "## Statistical comparisons", "",
        f"Within each window, target, and horizon, complete models are compared with the lowest-RMSE model using squared-error loss. The loss autocovariance truncation lag is `h-1`; the Harvey-Leybourne-Newbold correction and two-sided Student-t reference are applied. Holm adjustment controls the within-target/horizon family. Tests with nonpositive or invalid long-run variance are undefined. The exact paired count is reported. Non-rejection is not treated as equivalence.", "",
        f"Paired bootstrap intervals use {config['statistics']['bootstrap_repetitions']} repetitions and seed {config['statistics']['bootstrap_seed']}. Horizon 1 uses iid paired resampling; horizons 2 and 5 use circular moving blocks with length at least the forecast horizon to address overlap.", "",
        "## Leakage and failure controls", "",
        "Record checks enforce `training_end_year=origin_year`, `target_year=origin_year+h`, and `target_year>training_end_year`. No target enters selection, fitting, scaling, or interval residuals. Every expected model/origin/horizon record is retained. Failed ARIMA, ETS, or Theta forecasts are not replaced with another model, and only 100%-complete models are ranking-eligible.", "",
        "## Limitations", "",
        "Annual samples remain small, especially in the recent sensitivity window. Interval calibration and DM power are limited; overlapping horizons reduce effective information. Timings are specific to the recorded hardware and workload. Phase 2A establishes trusted classical baselines on this dataset and does not establish general model superiority.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_failure_report(
    path: Path, failures: pd.DataFrame, arima_candidates: pd.DataFrame, ets_candidates: pd.DataFrame,
    interval_violations: pd.DataFrame,
) -> None:
    """Write a complete fit/candidate/interval failure summary."""

    arima_failed = int((~arima_candidates["fit_success"].astype(bool)).sum()) if not arima_candidates.empty else 0
    ets_failed = int((~ets_candidates["fit_success"].astype(bool)).sum()) if not ets_candidates.empty else 0
    lines = [
        "# Phase 2A Failure Log", "", "## Summary", "",
        f"- Failed stored forecast records: {len(failures)}",
        f"- Rejected/failed ARIMA candidate rows: {arima_failed}",
        f"- Rejected/failed ETS candidate rows: {ets_failed}",
        f"- Missing/invalid interval records: {len(interval_violations)}", "",
        "Candidate failures are expected during bounded statistical-model searches and remain visible in the candidate-search CSV files. They are not counted as failed forecasts when another valid candidate is selected and successfully refit.", "",
    ]
    if failures.empty:
        lines.extend(["## Forecast-level failures", "", "No forecast-level fit failures occurred.", ""])
    else:
        lines.extend(["## Forecast-level failures", ""])
        for _, row in failures.iterrows():
            lines.append(f"- {row['analysis_window']} / {row['target']} / {row['model']} / h={row['horizon']} / origin={row['origin_year']}: {row['fit_warning']}")
        lines.append("")
    if not interval_violations.empty:
        lines.extend(["## Interval violations", ""])
        for _, row in interval_violations.head(100).iterrows():
            lines.append(f"- {row['analysis_window']} / {row['target']} / {row['model']} / h={row['horizon']} / origin={row['origin_year']} / level={row['nominal_level']}: {row['reason']}")
        if len(interval_violations) > 100:
            lines.append("- Additional violations are retained in the interval metrics/forecast records.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_results_report(
    path: Path, primary: pd.DataFrame, recent: pd.DataFrame, intervals: pd.DataFrame,
    dm_tests: pd.DataFrame, bootstrap: pd.DataFrame, costs: pd.DataFrame, failures: pd.DataFrame,
) -> None:
    """Write cautious dataset-specific Phase 2A results without equivalence claims."""

    lines = [
        "# Phase 2A Baseline Results", "", "## Executive summary", "",
        "This report evaluates six classical baselines using leakage-free expanding origins. Rankings describe only the supplied two annual series. Data provenance remains unresolved, including country/population, original sources, processing history, recent-year status, and licensing; this remains a critical blocker for manuscript interpretation.", "",
        f"All stored forecast-level failures remain visible; failed records in this run: {len(failures)}. Raw rankings, interval uncertainty, statistical tests, and interpretation are separated below.", "",
        "## Primary 2000-2023 raw rankings", "", "| Target | Horizon | Best RMSE model | RMSE | Best MASE model | MASE | Origins |", "|---|---:|---|---:|---|---:|---:|",
    ]
    for (target, horizon), group in primary.groupby(["target", "horizon"], sort=True):
        eligible = group.loc[group["ranking_eligible"].astype(bool)]
        rmse = eligible.nsmallest(1, "rmse").iloc[0]
        mase = eligible.nsmallest(1, "mase").iloc[0]
        lines.append(f"| {target} | {horizon} | {rmse['model']} | {rmse['rmse']:.8g} | {mase['model']} | {mase['mase']:.6g} | {int(rmse['successful_n'])} |")
    lines.extend(["", "Positive Mean Error denotes underforecasting; model/horizon bias and directional accuracy are in the primary metrics CSV and diagnostic figures.", "",
                  "## Recent 2016-2023 sensitivity", "", "| Target | Horizon | Best RMSE model | RMSE | Best MASE model | MASE | Origins |", "|---|---:|---|---:|---|---:|---:|"])
    for (target, horizon), group in recent.groupby(["target", "horizon"], sort=True):
        eligible = group.loc[group["ranking_eligible"].astype(bool)]
        rmse = eligible.nsmallest(1, "rmse").iloc[0]; mase = eligible.nsmallest(1, "mase").iloc[0]
        lines.append(f"| {target} | {horizon} | {rmse['model']} | {rmse['rmse']:.8g} | {mase['model']} | {mase['mase']:.6g} | {int(rmse['successful_n'])} |")
    lines.extend(["", "The recent analysis contains eight target years per horizon and therefore has materially greater uncertainty than the primary analysis. It uses expanding history before every target rather than one fixed anchor.", "",
                  "## Interval quality", ""])
    primary_intervals = intervals.loc[intervals["analysis_window"] == "primary_2000_2023"]
    for level in (0.80, 0.95):
        subset = primary_intervals.loc[np.isclose(primary_intervals["nominal_level"], level)]
        valid = subset.loc[subset["valid_interval_n"] > 0]
        lines.append(f"- {int(level*100)}% intervals: mean empirical coverage across model/target/horizon cells={valid['coverage_probability'].mean():.3f}; valid intervals={int(valid['valid_interval_n'].sum())}; missing/invalid={int(subset['missing_or_invalid_interval_n'].sum())}.")
    lines.extend(["", "Coverage, width, and Winkler scores must be considered together; nominal coverage alone does not reward sharp intervals.", "",
                  "## Statistical comparisons and uncertainty", ""])
    defined = dm_tests.loc[dm_tests["interpretation"] != "test_undefined"] if not dm_tests.empty else dm_tests
    detected = int(defined["statistically_detected_difference"].sum()) if not defined.empty else 0
    undefined = int((dm_tests["interpretation"] == "test_undefined").sum()) if not dm_tests.empty else 0
    lines.append(f"HLN/DM comparisons with a defined test: {len(defined)}; Holm-detected differences: {detected}; undefined tests: {undefined}.")
    lines.append("A non-detected difference is reported only as limited evidence from a small paired sample. Bootstrap RMSE/MAE difference intervals provide effect-size uncertainty and use moving blocks for overlapping horizons.")
    lines.extend(["", "## Computational cost", ""])
    primary_cost = costs.loc[costs["analysis_window"] == "primary_2000_2023"].groupby("model")["total_seconds"].sum().sort_values(ascending=False)
    for model, seconds in primary_cost.items():
        lines.append(f"- {model}: {seconds:.3f} measured seconds across recorded primary forecast work.")
    lines.extend(["", "These wall-clock timings are not hardware-independent and must be interpreted with `runtime_environment.csv`.", "",
                  "## Failed fits", "", f"Forecast-level failed records: {len(failures)}. Candidate-level rejections are documented separately and do not imply a missing forecast when another candidate succeeds.", "",
                  "## Implications for Phase 2B", "",
                  "Phase 2B models should consume the same origin manifest, record schema, origin-specific scaling denominators, interval validation, and completeness rules. Any machine-learning transformation or tuning must occur inside each training prefix. Phase 2A alone does not justify publication-level superiority or generalization beyond these supplied series.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
