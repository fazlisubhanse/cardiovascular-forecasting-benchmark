"""Phase 2B Markdown reporting."""

from __future__ import annotations
from pathlib import Path
import pandas as pd


def _write(path: Path, lines: list[str]): path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def _table(frame: pd.DataFrame) -> str:
    """Render dependency-free CSV in a fenced Markdown block."""
    return "```csv\n" + frame.to_csv(index=False, lineterminator="\n").rstrip() + "\n```"


def write_phase2b_reports(report_dir: Path, metrics: pd.DataFrame, intervals: pd.DataFrame,
                          selected: pd.DataFrame, raw_diag: pd.DataFrame, failures: pd.DataFrame,
                          integrity: pd.DataFrame, versions: dict[str,str], candidates: pd.DataFrame,
                          interval_violations: pd.DataFrame, dm_fixed: pd.DataFrame,
                          dm_overall: pd.DataFrame, timings: pd.DataFrame) -> None:
    primary=metrics.loc[metrics.analysis_window=="primary_2000_2023"]
    winners=primary.loc[primary.groupby(["target","horizon"]).rmse.idxmin(),["target","horizon","model","rmse","mase"]]
    recent=metrics.loc[metrics.analysis_window=="recent_2016_2023"]
    recent_winners=recent.loc[recent.groupby(["target","horizon"]).rmse.idxmin(),["target","horizon","model","rmse","mase"]]
    detection=pd.DataFrame([
        {"family":"ML versus best Phase 2A baseline","comparisons":len(dm_fixed),"detected":int(dm_fixed.statistically_detected_difference.sum()),"undefined":int(dm_fixed.interpretation.eq("test_undefined").sum())},
        {"family":"All nonwinners versus best overall","comparisons":len(dm_overall),"detected":int(dm_overall.statistically_detected_difference.sum()),"undefined":int(dm_overall.interpretation.eq("test_undefined").sum())},
    ])
    lines=["# Phase 2B Classical-ML Results","","## Scope","","Nested expanding-window SVR, Random Forest, and XGBoost direct-horizon forecasts were evaluated alongside the frozen Phase 2A baselines. Raw-level models are diagnostic only and are excluded from rankings.","","## Primary RMSE winners","",_table(winners),"","## Recent-window RMSE winners","",_table(recent_winners),"","## Interval audit","",_table(intervals.loc[intervals.analysis_window=="primary_2000_2023"]),"","With eight calibration residuals, the finite-sample ranks for both 80% and 95% intervals cap at the eighth order statistic. Their bounds therefore coincide by construction; this is reported transparently rather than altered post hoc.","","## Paired statistical comparisons","",_table(detection),"","Statistical non-rejection is not evidence of equal performance; the paired annual samples have limited power."]
    _write(report_dir/"PHASE2B_ML_RESULTS.md",lines)
    freq=selected.loc[selected.selection_family=="primary_transformed"].groupby(["model","representation","lookback"]).size().reset_index(name="n")
    runtime=timings.groupby(["selection_family","model"]).total_seconds.sum().reset_index()
    _write(report_dir/"PHASE2B_METHOD_VALIDATION.md",["# Phase 2B Method Validation","","## Leakage controls","","Every outer model uses observations through its origin only. Inner validation targets satisfy `pseudo_origin + horizon <= outer_origin`; detrending and SVR scaling are refitted inside each fold. Direct-horizon labels and finite-sample conformal residuals remain on the original reporting scale after inversion.","","The six independent target-horizon cells were scheduled across three processes. Every estimator retained the prespecified `n_jobs=1`; cells used identical fixed seeds and were merged in deterministic target-horizon order.","","For exact computational reuse, a 500-tree Random Forest fit supplies its identical first-200-tree prefix at the same depth/leaf/feature settings, and a 300-round XGBoost fit supplies its identical first-100-round prediction prefix. Different depth and other structural candidates are always independently fitted.","","Raw-level ablations reuse each primary model's selected lookback and hyperparameters and change only the representation, isolating representation effects.","","## Environment","",* [f"- {k}: `{v}`" for k,v in versions.items()],"","## Candidate and selection audit","",f"- Prespecified primary candidate evaluations: **{len(candidates):,}**.",f"- Successful evaluations: **{int(candidates.candidate_status.eq('success').sum()):,}**.",f"- Minimum-sample feasibility rejections: **{int(candidates.candidate_status.eq('failed').sum()):,}**.","",_table(freq),"","## Computation","",_table(runtime),"","## Integrity","",_table(integrity)])
    summary=raw_diag.groupby(["target","model","horizon"])[["actual_exceeds_training_max","forecast_exceeds_training_max","forecast_below_training_min"]].mean().reset_index()
    _write(report_dir/"PHASE2B_RAW_LEVEL_ABLATION.md",["# Phase 2B Raw-Level Ablation","","Raw-level direct models are a diagnostic ablation and never enter primary rankings. Exceedance fields quantify whether observed or forecast targets lie outside each outer training range.","",_table(summary)])
    candidate_failures=candidates.loc[candidates.candidate_status.eq("failed")]
    reasons=candidate_failures.groupby("failure_reason").size().reset_index(name="candidate_n").sort_values("candidate_n",ascending=False)
    _write(report_dir/"PHASE2B_FAILURE_LOG.md",["# Phase 2B Failure Log","","## Outer forecasts","",f"Recorded outer-run failures: **{len(failures)}**.","",_table(failures) if len(failures) else "No outer forecast failed.","","## Candidate feasibility","",f"Rejected candidate evaluations: **{len(candidate_failures):,}**. These rows are explicit prespecified minimum-sample rejections, not estimator crashes or imputed forecasts.","",_table(reasons),"","## Interval availability","",f"Record-level missing/invalid interval flags: **{len(interval_violations)}**. They occur where fewer than eight historical OOF residuals were available; they are excluded from coverage denominators and never counted as covered.","",_table(interval_violations)])
