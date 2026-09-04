# Round-2 public figure provenance

The included figures are reporting-only transformations of the frozen aggregate artifacts. They do not reproduce the annual NCD-RisC source series.

## Figure 2

Source: `config/round2_frozen_evaluation_protocol.json`. The diagram renders the target-specific rolling-origin calendar and frozen design metadata.

## Figure 3

Sources: `results/round2/primary_rankings.csv` and `results/round2/metrics_summary.csv`. Each primary specification's RMSE is divided by the frozen minimum RMSE in its target-horizon cell for display.

## Figure 4

Sources: `results/round2/interval_coverage.csv`, `results/round2/tables/table_interval_coverage_summary.csv`, and `results/round2/tables/table_winner_interval_coverage.csv`. The figure displays model-level coverage, pooled and unweighted summaries, valid interval counts, and winner coverage. It performs no inference.

## Multimedia Appendix Figure 1

Sources: `results/round2/ablation_results.csv` and `results/round2/inference.csv`. The figure plots prespecified comparator-minus-reference RMSE differences and existing validity/adjustment labels; it performs no new test.

## Deliberate exclusion

Source-series Figure 1 is not part of this public package. Visual and code inspection confirmed that it contains mini-series of all 30 ASPH and 39 MTC central estimates with their source uncertainty bands. The separate intermediate source-series PNG is also excluded.
