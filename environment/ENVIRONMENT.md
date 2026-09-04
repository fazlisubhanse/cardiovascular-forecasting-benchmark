# Recorded benchmark environment

The authoritative Round-2 run recorded the following environment before training:

- Python: CPython 3.11.9
- Operating system: Windows 10 build family
- Execution device: CPU only; no GPU and no CUDA
- CPU: Intel Core Ultra 9 185H, 16 physical and 22 logical cores
- RAM: 33,733,398,528 bytes
- Configured process workers: 18
- Statistical, classical-ML, and reporting workers: 1 each
- Neural workers: 18, with one Torch/BLAS thread per process

Core libraries:

| Package | Version |
|---|---|
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| SciPy | 1.17.1 |
| statsmodels | 0.14.6 |
| scikit-learn | 1.9.0 |
| XGBoost | 3.2.0 |
| PyTorch | 2.13.0+cpu |
| Matplotlib | 3.11.0 |
| PyYAML | 6.0.3 |

Deterministic PyTorch algorithms were enabled, Torch used one thread per worker, and the effective `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` values were all 1 during final execution.

The benchmark used statistical/classical seed 20260810, bootstrap base seed 20260810, and neural seeds 3101 through 3110. Package installation or upgrade was not performed during the authoritative execution.

The original environment record contained a machine-specific interpreter path. This public version intentionally omits that local absolute path while preserving the reproducibility-relevant version and execution settings.
