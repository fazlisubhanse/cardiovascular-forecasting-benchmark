"""Fifteen matplotlib-only internal Phase 2C.0 diagnostics."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig,path): fig.tight_layout(); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def generate_phase2c_figures(data:pd.DataFrame, metrics:pd.DataFrame, comparison:pd.DataFrame,
    ensemble:pd.DataFrame, parameters:pd.DataFrame, ratios:pd.DataFrame, ablations:pd.DataFrame,
    costs:pd.DataFrame, output:Path)->list[Path]:
    output.mkdir(parents=True,exist_ok=True); paths=[]; primary=metrics.loc[metrics.analysis_window.eq("primary_2000_2023")]
    for target in ("asph","mtc"):
      for measure in ("rmse","mase"):
        fig,ax=plt.subplots(figsize=(10,5)); subset=primary.loc[primary.target.eq(target)]
        for model,g in subset.groupby("model"): ax.plot(g.horizon,g[measure],marker="o",label=model)
        ax.set(xlabel="Horizon",ylabel=measure.upper(),title=f"Neural {measure.upper()} — {target.upper()}"); ax.set_xticks([1,2,5]); ax.grid(alpha=.25); ax.legend(fontsize=6,ncol=3)
        path=output/f"dl_{measure}_by_horizon_{target}.png"; _save(fig,path); paths.append(path)
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(comparison)); ax.bar(x-.2,comparison.best_neural_rmse,.4,label="Best neural"); ax.bar(x+.2,comparison.best_nonneural_rmse,.4,label="Best non-neural"); ax.set_xticks(x,[f"{t.upper()} h={h}" for t,h in zip(comparison.target,comparison.horizon)],rotation=30); ax.set_ylabel("RMSE"); ax.legend(); path=output/"neural_vs_nonneural_rmse.png"; _save(fig,path); paths.append(path)
    fig,ax=plt.subplots(figsize=(10,5)); ensemble.loc[ensemble.analysis_window.eq("primary_2000_2023")].groupby("architecture").seed_prediction_std.mean().sort_values().plot.bar(ax=ax); ax.set_ylabel("Mean seed prediction SD"); path=output/"seed_variability_by_architecture.png"; _save(fig,path); paths.append(path)
    fig,ax=plt.subplots(figsize=(10,5)); parameters.groupby("architecture").trainable_parameter_count.max().sort_values().plot.bar(ax=ax,logy=True); ax.set_ylabel("Trainable parameters (log scale)"); path=output/"parameter_count_by_architecture.png"; _save(fig,path); paths.append(path)
    fig,ax=plt.subplots(figsize=(10,5)); ratios.groupby("architecture").samples_to_parameter_ratio.median().sort_values().plot.bar(ax=ax); ax.set_ylabel("Median samples / parameter"); path=output/"samples_per_parameter.png"; _save(fig,path); paths.append(path)
    for label,name in (("attention_operation","compact_attention_ablation.png"),("capacity","capacity_ablation.png")):
        sub=ablations.loc[ablations.ablation.eq(label)]; fig,ax=plt.subplots(figsize=(9,5)); ax.bar(np.arange(len(sub)),sub.rmse_comparison_minus_reference); ax.axhline(0,color="black"); ax.set_xticks(np.arange(len(sub)),[f"{t.upper()} h={h}" for t,h in zip(sub.target,sub.horizon)],rotation=30); ax.set_ylabel("RMSE comparison - compact attention"); path=output/name; _save(fig,path); paths.append(path)
    for measure,name,ylabel in (("mean_error","dl_mean_error.png","Mean error"),("directional_accuracy_percent","dl_directional_accuracy.png","Directional accuracy (%)")):
        fig,ax=plt.subplots(figsize=(10,5)); primary.groupby("model")[measure].mean().sort_values().plot.bar(ax=ax); ax.set_ylabel(ylabel); path=output/name; _save(fig,path); paths.append(path)
    fig,ax=plt.subplots(figsize=(10,5)); costs.set_index("architecture").total_seed_level_seconds.sort_values().plot.bar(ax=ax); ax.set_ylabel("Seed-level seconds"); path=output/"dl_computation_time.png"; _save(fig,path); paths.append(path)
    for target in ("asph","mtc"):
        best=comparison.loc[comparison.target.eq(target)].set_index("horizon").best_neural_model.to_dict(); fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
        for ax,h in zip(axes,[1,2,5]):
            sub=ensemble.loc[(ensemble.analysis_window=="primary_2000_2023")&(ensemble.target==target)&(ensemble.horizon==h)&(ensemble.architecture==best[h])]
            ax.plot(data.year,data[target],color="black",label="Observed"); ax.plot(sub.target_year,sub.point_forecast,marker="o",label=best[h]); ax.set_title(f"h={h}"); ax.grid(alpha=.2); ax.legend()
        path=output/f"rolling_best_dl_{target}.png"; _save(fig,path); paths.append(path)
    return paths
