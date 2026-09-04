# Round-2 public figure captions

Figure 1 is intentionally absent because it plots the annual source estimates and uncertainty bounds.

## Figure 2. Frozen leakage-controlled rolling-origin evaluation design

Both targets used an expanding-window evaluation beginning after 21 observed annual values. The first forecast origin was 2010 for ASPH and 2000 for MTC because official source coverage began in 1990 and 1980, respectively. Forecast horizons were 1, 2, and 5 years, producing ASPH target counts of 9, 8, and 5 and MTC target counts of 18, 17, and 14. Classical machine-learning and neural direct-horizon models used lookback L=3 fixed before evaluation; nested lookback selection was disabled and legacy L=5, 6, and 8 reruns were not executed. No future observation entered fitting or preprocessing, and no recent-period analysis was conducted.

## Figure 3. Primary predictive performance across targets and forecast horizons

Heatmap cells show each specification's root mean squared error relative to the lowest RMSE in the same target-horizon cell on a logarithmic scale; 1× identifies the frozen cell winner. All 19 primary specifications are shown in each cell and grouped as statistical, classical machine-learning, or primary neural. ARIMA won five cells and ETS won MTC h=5. Statistical models won all six cells; primary neural models won none. ASPH h=5 contains only five forecasts and remains extremely unstable.

## Figure 4. Empirical predictive-interval coverage across all evaluated cells

Open symbols show individual interval-capable statistical and classical machine-learning model specifications; symbol size reflects valid interval count. Filled squares are unweighted means of model-level empirical coverage, filled diamonds are pooled coverage across valid model-forecast intervals, and stars show coverage for the frozen cell winner. Dashed and dotted horizontal lines mark nominal 80% and 95% coverage. Different interval availability means the unweighted and pooled summaries answer different descriptive questions. Coverage was generally below nominal levels and is descriptive rather than inferential. ASPH h=5 is explicitly extremely unstable because its winner contains only five forecasts. Neural models are excluded because no frozen neural predictive-interval method exists.

## Multimedia Appendix Figure 1. Neural ablation comparisons

Points show the RMSE difference between each prespecified comparator and the compact-attention reference model across all target-horizon cells. Negative values favor the comparator and positive values favor compact attention. Holm-adjusted differences were detected for ASPH h=1 and h=2 in both ablation families. MTC comparisons did not detect differences with the available small samples, which is not evidence of equivalence. ASPH h=5 is descriptive only and has no Diebold-Mariano p value.

ASPH: age-standardized prevalence of hypertension; MTC: age-standardized mean total cholesterol; ML: machine learning; RMSE: root mean squared error.
