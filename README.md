# Reproducibility Package

## Forecast Accuracy versus Operational Trigger Value in Next-Day Truck Utilization Forecasting

This repository contains the executable analysis workflow, fixed model configuration, deterministic synthetic data, validation tests, and representative outputs associated with the study.

The package supports inspection and execution of the following procedures:

- synthetic daily-panel construction and feature engineering;
- chronological training, expanding-window validation, and holdout evaluation;
- baseline, linear, random-forest, XGBoost, LightGBM, and LSTM models;
- alternative-target and data-quality sensitivity analyses;
- operational threshold calibration and counterfactual cost analysis;
- paired forecast comparisons with serial-dependence-aware resampling;
- missingness, runtime, retraining-cadence, and regime diagnostics; and
- automated structural, temporal, and privacy validation.

## Reproducibility scope

The package contains independently generated synthetic data only. It does not contain confidential company records, customer-level information, private-data-derived predictions, or private-data-derived empirical outputs.

Confidentiality applies to the underlying company data, not to the analysis code. The synthetic records reproduce the computational schemas, temporal structure, model interfaces, and output formats needed to execute and inspect the workflow. Synthetic numerical results are demonstrations of the code path and are not the empirical findings reported in the study.

Additional scope information is provided in [docs/REPRODUCIBILITY_SCOPE.md](docs/REPRODUCIBILITY_SCOPE.md).

## Repository contents

```text
config/          Fixed analysis and model configuration
data/             Synthetic input data and data documentation
docs/             Reproducibility scope and methods notes
outputs/          Representative synthetic tables, figures, predictions, and validation metadata
scripts/          Ordered analysis scripts
src/              Shared modeling and utility modules
tests/            Automated integrity and privacy tests
environment.yml   Conda environment specification
requirements.txt  Pinned Python dependencies
run_all.py        End-to-end workflow runner
checksums.sha256  SHA-256 hashes for package files
```

## Requirements

- Python 3.13
- 64-bit operating system
- Internet access for initial dependency installation
- CPU execution is supported; a GPU is not required

Exact dependency versions are provided in `requirements.txt` and `environment.yml`. The included reference outputs were generated and validated with Python 3.13.5. A CPU build of PyTorch may report the local version suffix `+cpu` while corresponding to the pinned `torch==2.10.0` release.

## Quick start

Run commands from the repository root.

### Conda

```bash
conda env create -f environment.yml
conda activate predictive-logistics-repro
python run_all.py
```

### Python virtual environment on Windows

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_all.py
```

### Python virtual environment on macOS or Linux

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_all.py
```

The complete run includes repeated LSTM fits for alternative-target checks. A shorter inspection run is available through:

```bash
python run_all.py --skip-alternative-lstm
```

A successful complete run ends with:

```text
Synthetic reproduction workflow completed and validated.
```

## Workflow

`run_all.py` executes the following scripts in order:

1. `scripts/00_generate_synthetic_data.py`
2. `scripts/01_descriptive_targets_backlog.py`
3. `scripts/02_core_models.py`
4. `scripts/03_advanced_models.py`
5. `scripts/04_robustness_and_alternative_targets.py`
6. `scripts/05_threshold_calibration_and_costs.py`
7. `scripts/06_significance_runtime_missingness.py`
8. `scripts/07_make_figures.py`
9. `scripts/08_verify_outputs.py`

Generated files are written under `outputs/`. Structural and privacy checks are summarized in:

- `outputs/metadata/validation_report.csv`
- `outputs/metadata/privacy_scan.csv`
- `outputs/metadata/run_manifest.json`

## Temporal validation design

The synthetic panel preserves the analytical workflow's temporal structure:

- 266 eligible 2024 observations form the training and model-selection sample;
- five expanding-window folds use training sizes of 46, 90, 134, 178, and 222 observations, with 44 validation observations per fold;
- 271 eligible 2025 observations form the primary holdout; and
- operational thresholds are calibrated from 2024 out-of-fold predictions only.

The primary 2025 holdout is excluded from model fitting, scaling, imputation, hyperparameter selection, threshold calibration, and model selection.

The LSTM uses sequences of 28 consecutive eligible modeling-panel rows. These sequences are not necessarily 28 consecutive calendar days when ineligible dates occur.

## Verification

The package includes automated tests for data structure, temporal alignment, model-output coverage, cost arithmetic, and privacy safeguards. They can be run independently using:

```bash
python -m unittest discover -s tests -v
```

The file `checksums.sha256` provides integrity hashes for the public package.
