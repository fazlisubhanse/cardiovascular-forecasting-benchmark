# Regime- and Horizon-Dependent Forecasting of Hypertension Prevalence and Total Cholesterol

Reproducibility code and frozen final evidence for **“Regime- and Horizon-Dependent Forecasting of Hypertension Prevalence and Total Cholesterol: A Leakage-Controlled Benchmark of Statistical, Machine-Learning, and Neural Models.”**

Public repository: **https://github.com/fazlisubhanse/cardiovascular-forecasting-benchmark**. Frozen reproducibility release: **v1.0.0**. No DOI has been assigned.

## Study objective

This study benchmarks statistical, classical machine-learning, and neural methods for forecasting two annual cardiovascular indicators—age-standardized prevalence of hypertension (ASPH) and mean total cholesterol (MTC)—at 1-, 2-, and 5-year horizons. It tests whether model rankings persist across targets, horizons, and the recent 2016-2023 period, and whether attention or substantially larger neural capacity consistently improves performance.

## Dataset scope and limitations

The analysis uses a supplied 54-row annual series covering 1970-2023 for men aged 30-79 years. The complete dataset is **not included** because its exact upstream provenance and redistribution rights have not been established. Country/population identity, original provider release, processing lineage, and the observational or projected status of recent values remain unresolved. See [Data access and provenance](docs/DATA_ACCESS.md) before attempting reproduction.

## Forecasting framework

Evaluation uses expanding-window origins for target years 2000-2023, with horizons of 1, 2, and 5 years. Model and representation selection is confined to development targets ending by 1999. Recent-period results reuse the frozen fits and evaluate target years 2016-2023 without retraining. Primary metrics are RMSE, MAE, and MASE; the workflow also reports paired bootstrap intervals, HLN-corrected Diebold-Mariano tests with Holm adjustment, forecast-interval behavior, seed variability, architecture ablations, and computational cost.

## Model families

- Statistical: Naive, Drift, linear trend, ETS, ARIMA, and Theta.
- Classical machine learning: SVR, RandomForest, and XGBoost with fold-local transformed representations.
- Neural: LSTM, GRU, BiLSTM, CausalCNN, CNN-LSTM, CNN-GRU, and compact CNN-BiLSTM-attention, plus no-attention and approximately 23K-parameter diagnostic ablations.

## Leakage controls

All outer forecasts use only observations available at the forecast origin. Scaling and transformations are fit within each fold. Hyperparameter, representation, lookback, and epoch selection occurs only in the development period. The ten-seed neural evaluation reuses frozen pilot selections; recent-period sensitivity does not retune models. Synthetic unit tests cover target alignment, future-value mutation resistance, scaler boundaries, development cutoffs, and deterministic configuration controls.

## Repository structure

```text
configs/       Frozen analysis and data-validation configurations
docs/          Data-access and reproducibility guidance
environment/   Exact observed runtime and dependency lock
figures/       Final Phase 2D.1 figures in PNG, PDF, and SVG
results/       Final manuscript tables, figure-source tables, and frozen manifest
scripts/       Dataset and release validation entry points
src/           Statistical, ML, neural, comparison, and reporting code
tests/         Fast non-training scientific and release checks
```

Large intermediate forecasts, fit-progress files, checkpoints, development reports, manuscripts, correspondence, and third-party candidate data are intentionally excluded.

## Quick verification — no training

Create the verified Python environment, then run the non-training checks:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```powershell
python scripts/verify_release.py
python -m pytest -q
```

These checks do not require the study dataset and do not fit models.

## Full reproduction

Full reproduction requires an authorized dataset copy whose SHA-256 matches the frozen configuration. The required sequence is dataset validation, statistical baselines, classical ML, neural pilot, final ten-seed neural evaluation, non-training cross-family integration, and final table/figure generation. The neural steps are extremely long on CPU and generate large intermediate artifacts.

See [Reproducibility guide](docs/REPRODUCIBILITY.md) for exact commands, prerequisites, and runtime labels.

Expected generated outputs are phase-specific long-form forecasts, selected-configuration and metric tables, statistical comparisons, runtime/provenance records, and figures under `outputs/`, followed by the final Phase 2D.1 evidence package. The complete output tree is large; only final inspection artifacts are staged here.

## Frozen final results

`results/tables/` contains the five final manuscript result tables and ten figure-source tables. `figures/` contains ten final figures in three formats. `results/FROZEN_RESULTS_MANIFEST.json` is the Phase 2D.1 source-artifact manifest. These files are a read-only final evidence snapshot; intermediate model outputs are not included.

The primary overall winners were RandomForest for ASPH at horizons 1 and 2, Naive for ASPH at horizon 5, CausalCNN for MTC at horizon 1, XGBoost for MTC at horizon 2, and ARIMA for MTC at horizon 5. No family was universally best.

## Interpretation cautions

The benchmark contains only 54 annual observations and two supplied series. Rankings are target-, horizon-, and period-dependent. Non-rejection in a statistical test is not proof of equivalence. Neural ablations are descriptive, and greater capacity did not confer a general advantage. The unresolved provenance and licensing limitations constrain external interpretation and data redistribution.

## Computational cost

The frozen final neural run comprised 12,060 successful model fits across ten prespecified seeds and retained 540 representative checkpoint records. The supplied code can reproduce that experiment, but model binaries/checkpoints are not distributed. Consult `results/tables/table_computational_cost.csv` for measured phase-level and model-level costs.

## Software environment

The verified neural environment used Python 3.11.9, PyTorch 2.13.0+cpu, deterministic algorithms, one PyTorch thread per fit, and CPU execution on Windows-10-10.0.26200-SP0. `requirements.txt` and `environment/requirements-lock.txt` contain the exact captured package versions; `environment/runtime_environment.csv` records the directly observed runtime. No claim of bitwise equivalence on other hardware or operating systems is made.

## Citation

Citation metadata are supplied in `CITATION.cff` with software version `1.0.0`. No repository URL, DOI, or ORCID has been added because none has been author-verified for this release. Add repository or archival identifiers only after they actually exist.

## License

- **Code and software:** MIT License. See `LICENSE`.
- **Original documentation and author-created figures/tables:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Data and third-party materials:** excluded unless separately and explicitly stated.

The complete study dataset, candidate NCD-RisC files, third-party downloads, and uncertain-provenance candidate materials are not distributed and are outside both repository license scopes. See `LICENSE-CONTENT.md` for the precise file-level scope.

## Contact

Fazli Subhan, corresponding author

School of Information and Artificial Intelligence

Yangzhou University

Yangzhou, Jiangsu 225127, China

Email: fazlisubhanse@outlook.com
