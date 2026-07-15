# Reproducibility Scope

## Included materials

The repository includes:

- deterministic synthetic operational records;
- a data dictionary and feature catalog;
- target, feature, and row-eligibility construction;
- temporal split and expanding-window validation logic;
- baseline, linear, tree-based, boosting, and LSTM model implementations;
- operational threshold and counterfactual cost procedures;
- robustness, significance, missingness, runtime, retraining, and regime analyses;
- table and figure generation; and
- automated validation and privacy checks.

## Excluded materials

The repository excludes:

- original company workbooks and operational databases;
- customer-, order-, route-, employee-, or transaction-level identifiers;
- confidential processed panels;
- private-data-derived predictions, residuals, tables, and figures; and
- access credentials or storage locations for confidential data.

## Interpretation of the synthetic data

The synthetic dataset is independently generated and is not a masked copy, sample, perturbation, or reconstruction of the confidential company data. It preserves the computational schema, temporal dimensions, feature interfaces, validation design, and output structure required to execute the workflow.

Accordingly, the repository supports workflow reproducibility and code inspection. It does not provide numerical replication of the confidential empirical estimates. Model rankings, statistical results, cost comparisons, runtimes, and figures produced from the synthetic data have no evidentiary role in the study.

## Confidentiality boundary

Confidentiality restrictions apply to the underlying company data. The analysis code and independently generated synthetic workflow are publicly shareable and contain no company-sensitive records.
