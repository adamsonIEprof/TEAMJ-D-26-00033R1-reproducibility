# Methods and Outputs

## Synthetic data generation

`scripts/00_generate_synthetic_data.py` uses random seed 42 and explicit generation rules to create a calendar-complete synthetic master table, eligibility log, feature panel, modeling panel, feature catalog, and external-context series.

The structural validation targets are:

- 731 calendar rows;
- 724 nonmissing primary-target rows;
- 726 nonmissing secondary-target rows;
- 725 valid operating-day rows;
- 727 raw descriptive rows;
- 118 synthetic quality-flag rows;
- 537 eligible modeling rows, consisting of 266 training and 271 holdout rows; and
- 164 engineered features, consisting of 127 main and 37 sensitivity-only features.

## Forecast timing and leakage controls

Each modeling row represents a next-day forecast. `forecast_origin_date` equals `target_date` minus one calendar day. Noncalendar predictors are lagged, previous-day, forecast-origin, carried-forward, or otherwise historical transformations. Model parameters, feature caps, scalers, and imputers are estimated without primary holdout information.

## Validation design

The 266 eligible 2024 observations are evaluated through five expanding-window folds:

| Fold | Training rows | Validation rows |
|---:|---:|---:|
| 1 | 46 | 44 |
| 2 | 90 | 44 |
| 3 | 134 | 44 |
| 4 | 178 | 44 |
| 5 | 222 | 44 |

After model selection, the selected specifications are fitted on eligible 2024 observations and evaluated on 271 eligible 2025 holdout observations.

## LSTM implementation

The LSTM uses the S2 feature block and sequences of 28 consecutive eligible modeling-panel rows. Input scaling is fitted on the applicable training rows, target scaling is fitted on the applicable training endpoints, and the internal validation segment is the final 20 percent of training sequences in chronological order.

## Operational and statistical analyses

Forecast cutoffs are calibrated from 2024 out-of-fold predictions. The workflow evaluates trigger metrics, hypothetical counterfactual penalty scenarios, break-even penalties, condition-specific errors, and asymmetric decision metrics.

Paired absolute-error comparisons use a seven-observation circular moving-block bootstrap with 2,000 resamples. Holm adjustment controls multiplicity. Paired Wilcoxon results are retained as supplementary diagnostics.

Missingness sensitivity applies additional masking rates of 5, 10, and 20 percent to noncalendar autoregressive predictors, with training-sample median imputation. Runtime benchmarks record wall-clock fit and batch-inference times. Retraining-cadence diagnostics compare no-refit, quarterly expanding-refit, and monthly expanding-refit policies.

## Output locations

| Location | Contents |
|---|---|
| `data/synthetic/` | Synthetic source and modeling files |
| `data/metadata/` | Data dictionary and generation metadata |
| `outputs/tables/` | Synthetic analytical tables and diagnostics |
| `outputs/predictions/` | Synthetic out-of-fold and holdout predictions |
| `outputs/metadata/` | Model registries, configurations, validation results, and runtime metadata |
| `outputs/figures/` | Synthetic figures in PNG and PDF formats |

All outputs in this repository are generated from synthetic data.
