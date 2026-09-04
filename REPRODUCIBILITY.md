# Reproducibility workflow

This publication-facing workflow reconstructs the source-dependent files locally. It never asks users to type annual source values.

## 1. Clone the repository

```shell
git clone https://github.com/fazlisubhanse/cardiovascular-forecasting-benchmark.git
cd cardiovascular-forecasting-benchmark
git checkout v2.0.0-round2
```

## 2. Create the Python environment

The authoritative run used CPython 3.11.9 on Windows with CPU-only PyTorch. Create an isolated environment:

```shell
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-phase2b-lock.txt
```

Linux or macOS:

```shell
source .venv/bin/activate
python -m pip install -r requirements-phase2b-lock.txt
```

Install a CPU build of PyTorch 2.13.0 compatible with the local platform using the official PyTorch installation instructions. The generic `requirements.txt` records PyTorch as a required dependency; the supplied lock records the remaining benchmark environment. Do not silently substitute a GPU implementation when attempting strict replication of the recorded CPU run.

Recorded package and execution details are in `environment/ENVIRONMENT.md`.

## 3. Download the official source files

Follow [DATA_ACCESS.md](DATA_ACCESS.md). Download the two NCD-RisC CSVs and the WHO cross-check CSV directly from their official pages. Place them at the documented `data/raw/` paths without renaming them.

## 4. Verify source hashes

The reconstruction script refuses files whose SHA-256 does not match the value-free public manifest. The files can also be verified independently.

Windows PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 data\raw\*.csv
```

Linux or macOS:

```shell
sha256sum data/raw/*.csv
```

Compare the output with `provenance/source_manifest_public.csv`.

## 5. Reconstruct and validate the processed dataset

```shell
python scripts/reconstruct_round2_official_data.py
```

The script verifies the three source hashes, applies only the prespecified source filters, retains the central estimates and uncertainty bounds without numerical alteration, checks the independent WHO rounding relationship, and verifies:

- 30 ASPH observations for 1990–2019;
- 39 MTC observations for 1980–2018;
- 69 unique target-year rows;
- WHO rounding agreement of 30/30; and
- processed SHA-256 `4575b29feee41e3832c9c2e51c9c0f8f0e312d5e7f4df9b87ce75e13e705c3e2`.

No interpolation, extrapolation, imputation, smoothing, splicing, or synthetic extension is performed.

Run the focused unit tests:

```shell
pytest
```

## 6. Review the frozen protocol and configurations

The authoritative design is `config/round2_frozen_evaluation_protocol.json`. Key settings are H=21, horizons 1/2/5, primary lookback L=3, disabled nested lookback selection, and 10 fixed neural seeds.

The two value-free configuration maps in `config/` preserve the exact development-selected classical-ML hyperparameters and neural specifications used by the benchmark. They contain configuration metadata and no annual source observations.

The legacy-protocol object in the frozen JSON is historical context only; it is not an active analysis specification.

## 7. Verify and execute the benchmark entry point

First run the non-fitting preflight:

```shell
python -m src.round2_phase2b --workers 4 --preflight-only
```

The preflight validates data and protocol identities, frozen configuration coverage, temporal calendars, fit identifiers, and the planned total of 7,242 fits. It does not fit a model or generate a forecast.

Execute the complete frozen benchmark only when the full computational run is intended:

```shell
python -m src.round2_phase2b --workers 4
```

The worker count controls parallel scheduling, not the frozen scientific specification. The executable is resumable through per-fit checkpoints. Keep the resulting prediction ledgers and `results/round2/progress/` private unless redistribution permission is established, because they contain or permit reconstruction of source observations.

## 8. Reproduce aggregate reporting outputs

The benchmark writes the complete frozen result set locally. After it completes, apply the reporting-only interval summary correction:

```shell
python scripts/round2_phase2b1_coverage_reporting.py
```

This correction consumes frozen outputs and performs no fitting or forecasting. The inspected aggregate tables distributed with this release are under `results/round2/`.

## 9. Generate only the permitted public figures

```shell
python scripts/round2_figures/generate_round2_figures.py
python scripts/round2_figures/validate_round2_figures.py
```

The public generator reads only the included frozen protocol and aggregate result tables. It generates exactly Figures 2–4 and Appendix Figure A1 in SVG, PDF, and 600-dpi PNG formats. It neither reads the reconstructed annual dataset nor generates source-series Figure 1.

## 10. Verify release integrity

Linux or macOS:

```shell
sha256sum --check MODE_A_SHA256SUMS.txt
```

Windows PowerShell:

```powershell
Get-Content MODE_A_SHA256SUMS.txt | ForEach-Object {
    $expected, $relative = $_ -split '  ', 2
    $observed = (Get-FileHash -Algorithm SHA256 -LiteralPath $relative).Hash.ToLowerInvariant()
    if ($observed -ne $expected) { throw "SHA-256 mismatch: $relative" }
}
```

`MODE_A_RELEASE_MANIFEST.csv` records the redistribution decision for every included artifact. The checksum file deliberately omits itself and the release manifest to avoid recursive hashing; their externally observed release hashes are reported with the published release notes.

## Repository safeguards

The included `.gitignore` prevents accidental versioning of downloaded datasets, reconstructed values, prediction ledgers, checkpoints, model binaries, and source-series graphics. This preventive safeguard does not grant or change any right to redistribute third-party data.

The included `.gitattributes` normalizes ordinary authored text to LF and preserves byte identity for frozen CSV, JSON, SVG, checksum, manifest, and binary artifacts.

## Determinism and interpretation

Fixed seeds control the stochastic neural fits. CPU hardware, process scheduling, and library builds may still affect timing and potentially last-bit numerical behavior. The frozen aggregate result files in this package are the authoritative outputs from the recorded run.

