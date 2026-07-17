"""Phase 2C.0 method, results, architecture, failure, and runtime reports."""

from __future__ import annotations
from pathlib import Path
import pandas as pd


def _table(frame): return "```csv\n"+frame.to_csv(index=False,lineterminator="\n").rstrip()+"\n```"
def _write(path,lines): path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def write_phase2c_reports(report_dir:Path,config:dict,candidates:pd.DataFrame,selected:pd.DataFrame,
    metrics:pd.DataFrame,seed_distribution:pd.DataFrame,comparison:pd.DataFrame,parameters:pd.DataFrame,
    ratios:pd.DataFrame,ablations:pd.DataFrame,costs:pd.DataFrame,projection:pd.DataFrame,
    failures:pd.DataFrame,environment:pd.DataFrame)->None:
    _write(report_dir/"PHASE2C_PILOT_METHOD_VALIDATION.md",["# Phase 2C.0 Pilot Method Validation","",
        "All architecture and training choices use expanding historical folds whose target years are no later than 1999. The 2000–2023 rolling evaluation never contributes to selection.","",
        "The pilot uses a staged factorial development search: structural representation/lookback/size screening followed by optimizer/dropout/weight-decay refinement. This reduces pilot runtime but does not estimate all factor interactions and is a limitation.","",
        "First differences predict cumulative original-scale change. Linear detrending is refitted inside every fold. Feature and target scalers are fitted only to supervised training rows.","",
        "Development folds use early stopping. Final outer models use the median development best epoch, bounded by configured limits, train without a validation callback, and use every available supervised sample.","",
        "Three deterministic initialization seeds quantify seed initialization variability only; seed ranges are not prediction intervals.","","## Environment","",_table(environment),"",
        f"Development candidates: **{len(candidates)}**; selected configurations: **{len(selected)}**."])
    primary=metrics.loc[metrics.analysis_window.eq("primary_2000_2023")]
    _write(report_dir/"PHASE2C_PILOT_RESULTS.md",["# Phase 2C.0 Pilot Results","",
        "This three-seed pilot is a methodology and runtime validation, not the final 30-seed analysis. No superiority claim is made from pilot results.","","## Neural metrics","",_table(primary),"","## Neural and non-neural comparison","",_table(comparison),"","## Seed variability","",_table(seed_distribution),"","## Computational cost","",_table(costs)])
    _write(report_dir/"PHASE2C_ARCHITECTURE_AUDIT.md",["# Phase 2C Architecture Audit","",
        "Configured dropout is applied immediately before each scalar output (after the dense layer in attention hybrids). Causal convolutions use left padding only. The compact attention block uses a residual attention connection and layer normalization.","",
        "The reconstructed original-capacity model has 22,369 trainable parameters, close to but not exactly the manuscript's approximate 23,000. The discrepancy is reported rather than forced.","","## Parameter counts","",_table(parameters),"","## Sample/parameter ratios","",_table(ratios),"","## Ablations","",_table(ablations),"",
        "Combining established components is not by itself evidence of architectural novelty. High parameter count is treated as a capacity risk, not a benefit."])
    _write(report_dir/"PHASE2C_PILOT_FAILURE_LOG.md",["# Phase 2C.0 Pilot Failure Log","",f"Visible failure rows: **{len(failures)}**.","",_table(failures) if len(failures) else "No development seed or outer seed failed."])
    recommendation="CONDITIONAL GO" if len(failures)==0 else "NO-GO"
    _write(report_dir/"PHASE2C_FINAL_RUNTIME_PROJECTION.md",["# Phase 2C Final Runtime Projection","",f"## Recommendation: {recommendation}","",
        "The final 30-seed experiment was not run. Projection multiplies measured three-seed final-fit time by ten and applies a conservative overhead multiplier.","",_table(projection),"",
        "Data provenance remains unresolved and continues to block country/provider-specific manuscript claims."])
