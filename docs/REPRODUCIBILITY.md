# Reproducibility guide

All commands below are run from the repository root. Runtime labels are qualitative and based on the verified CPU workflow; hardware and package-source availability will affect elapsed time.

## 1. Environment setup — short

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The verified interpreter was Python 3.11.9. Exact package versions and the observed CPU environment are in `environment/`.

## 2. Quick release verification — short, no training

```powershell
python scripts/verify_release.py
python -m pytest -q
```

These commands validate the staging structure, frozen evidence metadata, leakage-control utilities, metrics, configurations, and architecture parameter counting. They do not fit forecasting models.

## 3. Dataset validation — short, no model fitting

After obtaining an authorized copy through the route in `DATA_ACCESS.md`:

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
# Place the authorized CSV at data/raw/input_series.csv
python scripts/validate_dataset.py --input data/raw/input_series.csv
```

The command enforces the frozen schema, row/year constraints, finiteness checks, and supplied-file SHA-256 before writing a lossless validated copy to `data/validated/`.

## 4. Statistical baselines — moderate to long

```powershell
python -m src.phase2a_baselines
```

This runs leakage-controlled expanding-window Naive, Drift, linear-trend, ETS, ARIMA, and Theta evaluations, interval calculations, paired tests, tables, and figures.

## 5. Classical machine learning — long

```powershell
python -m src.phase2b_ml
python -m src.phase2b1_verification
```

These commands run nested development-only selection for SVR, RandomForest, and XGBoost; fold-local transformations/scaling; conformal intervals; and the Phase 2B.1 non-neural verification analyses. Phase 2A outputs must already exist.

## 6. Neural pilot and final ten-seed experiment — extremely long on CPU

```powershell
python -m src.phase2c_pilot
python -m src.phase2c_final_10seed
```

The pilot performs development-only selection and three-seed outer evaluation. The final command uses the frozen pilot choices and ten prespecified seeds, producing 12,060 fits and representative checkpoint metadata. These commands can require many CPU-hours and substantial storage. They are provided for complete reproduction, not quick verification. No 30-seed experiment is part of the frozen evidence package.

## 7. Cross-family integration — short, no training

```powershell
python -m src.phase2d_integration_repair
```

This reads completed Phase 2A-2C artifacts and rebuilds cross-family comparison/statistical tables without fitting models.

## 8. Final table generation — short, no training

```powershell
python -m src.final_evidence_package
```

This requires the completed Phase 2A-2D.0 output tree. The actual entry point creates both final tables and figures in one non-training pass; there is no separate table-only command.

## 9. Final figure generation — short, no training

```powershell
python -m src.final_evidence_package
```

This is the same combined evidence-package entry point used in section 8. Run it once, not twice. It recreates the final publication tables and figures but never invokes model builders or training code. The repository includes a read-only snapshot of those final tables, figure-source tables, figures, and the Phase 2D.1 frozen-source manifest under `results/` and `figures/`; large intermediate forecasts and checkpoints are intentionally absent.

## 10. Complete validation — short, no training

```powershell
python -m compileall -q src scripts tests
python scripts/verify_release.py
python -m pytest -q
```

This checks syntax, staging integrity, scientific unit tests, and release-snapshot assertions. It does not recreate model forecasts.

## Complete end-to-end order

Run sections 1, 3, 4, 5, 6, 7, and then section 8 (which also completes section 9) in order. Only proceed when you hold an authorized, hash-matching input dataset. For inspection of the published evidence, sections 2 and 10 are sufficient and perform no training.
