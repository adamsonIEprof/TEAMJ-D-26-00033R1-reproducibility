from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
TABLES = ROOT / "outputs" / "tables"
PREDICTIONS = ROOT / "outputs" / "predictions"
FIGURES = ROOT / "outputs" / "figures"
OUTPUT_METADATA = ROOT / "outputs" / "metadata"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, check: str, details: str, rows: list[dict]) -> None:
    status = "PASS" if condition else "FAIL"
    rows.append({"check": check, "status": status, "details": details})
    if not condition:
        raise AssertionError(f"{check}: {details}")


def expected_outputs() -> list[Path]:
    tables = [
        "table01_target_definitions.csv",
        "table02_feature_timing.csv",
        "table03_expanding_window_folds.csv",
        "table04_backlog_signal_definitions.csv",
        "table05_backlog_attribution_rules.csv",
        "table06_annual_operational_profile.csv",
        "table07_panel_diagnostics.csv",
        "table08_log_ratio_decomposition.csv",
        "table09_backlog_attribution_counts.csv",
        "table10_backlog_attribution_profile.csv",
        "table11_target_year_comparison.csv",
        "table12_target_specific_model_results.csv",
        "table13_core_baseline_results.csv",
        "table13_core_family_spec_results.csv",
        "table14_s2_advanced_benchmark.csv",
        "table15_random_forest_importance.csv",
        "table16_base_threshold_trigger_metrics.csv",
        "table17_counterfactual_cost_scenarios.csv",
        "table18_break_even_penalties.csv",
        "table19_cleaned_training_random_forest.csv",
        "table20_regime_operational_profile.csv",
        "table21_holdout_performance_by_regime.csv",
        "table22_fold_stability_summary.csv",
        "conditional_error_diagnostics.csv",
        "supplementary_table_s1_raw_descriptives.csv",
        "paired_forecast_significance_tests.csv",
        "missingness_sensitivity.csv",
        "runtime_benchmarks.csv",
        "retraining_cadence_evaluation.csv",
    ]
    figures = [f"figure{number:02d}{suffix}" for number, suffix in [
        (1, "_utilization_timeseries"),
        (2, "_utilization_by_weekday"),
        (3, "_scheduled_served"),
        (4, "_trucks_drivers"),
        (5, "_deferred_carryover"),
        (6, "_monthly_operational_profile"),
        (7, "_log_ratio_decomposition"),
        (8, "_backlog_attribution"),
        (9, "_monthly_targets"),
        (10, "_holdout_trajectories"),
        (11, "_random_forest_importance"),
        (12, "_stress_precision_recall"),
        (13, "_critical_risk_cost"),
        (14, "_h1_h2_bias"),
        (15, "_fold_mae_stability"),
    ]]
    metadata_files = [
        OUTPUT_METADATA / "fold_level_model_results.csv",
        OUTPUT_METADATA / "model_hyperparameters.csv",
        OUTPUT_METADATA / "threshold_calibration_grid.csv",
        OUTPUT_METADATA / "precision_recall_curves.csv",
    ]
    return [TABLES / name for name in tables] + metadata_files + [
        FIGURES / f"{stem}.{extension}" for stem in figures for extension in ["png", "pdf"]
    ]


def privacy_scan() -> list[dict]:
    patterns = {
        "absolute Windows user path": re.compile(r"[A-Za-z]:\\Users\\", re.I),
        "raw fleet workbook name": re.compile("BT" + r"\s+Fleet\s+Deliveries", re.I),
        "row-level business identifier": re.compile("customer" + r"[_ -]?(name|id|code)", re.I),
        "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    }
    findings = []
    text_extensions = {".py", ".md", ".txt", ".log", ".csv", ".json", ".yaml", ".yml", ".cff"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_extensions:
            continue
        if path == OUTPUT_METADATA / "privacy_scan.csv":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in patterns.items():
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "file": path.relative_to(ROOT).as_posix(),
                        "finding": label,
                        "matched_text": match.group(0),
                    }
                )
    return findings


def main() -> None:
    OUTPUT_METADATA.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    missing = [path.relative_to(ROOT).as_posix() for path in expected_outputs() if not path.exists()]
    require(not missing, "Expected outputs exist", f"missing={missing}", checks)

    master = pd.read_csv(DATA / "daily_master_locked.csv", parse_dates=["date"])
    panel = pd.read_csv(DATA / "chunk3_main_modeling_panel.csv", parse_dates=["target_date", "forecast_origin_date"])
    catalog = pd.read_csv(DATA / "chunk3_feature_catalog.csv")
    row_log = pd.read_csv(DATA / "row_eligibility_log.csv", parse_dates=["target_date"])
    require(len(master) == 731, "Calendar row count", f"observed={len(master)}, expected=731", checks)
    require(
        master["target_utilization_intensity"].notna().sum() == 724,
        "Primary target row count",
        f"observed={master['target_utilization_intensity'].notna().sum()}, expected=724",
        checks,
    )
    require(
        master["secondary_trip_capacity_utilization_rate"].notna().sum() == 726,
        "Secondary target row count",
        f"observed={master['secondary_trip_capacity_utilization_rate'].notna().sum()}, expected=726",
        checks,
    )
    require(
        master["structural_nonoperational_day"].eq(0).sum() == 725,
        "Valid operating-day count",
        f"observed={master['structural_nonoperational_day'].eq(0).sum()}, expected=725",
        checks,
    )
    require(
        master["all_core_missing"].eq(0).sum() == 727,
        "Raw descriptive row count",
        f"observed={master['all_core_missing'].eq(0).sum()}, expected=727",
        checks,
    )
    require(
        master["any_qa_flag"].eq(1).sum() == 118,
        "Synthetic QA-flag row count",
        f"observed={master['any_qa_flag'].eq(1).sum()}, expected=118",
        checks,
    )
    truck_components = master[
        [
            "available_trucks_30kl",
            "available_trucks_18kl",
            "available_trucks_36kl",
            "available_trucks_20kl",
        ]
    ]
    truck_component_sum = truck_components.sum(axis=1, min_count=1)
    truck_arithmetic_ok = truck_component_sum.eq(master["available_trucks_total"]) | (
        truck_components.isna().all(axis=1) & master["available_trucks_total"].isna()
    )
    require(
        truck_arithmetic_ok.all(),
        "Synthetic truck-class arithmetic",
        "daily truck-class components sum exactly to available_trucks_total",
        checks,
    )
    require(len(panel) == 537, "Modeling row count", f"observed={len(panel)}, expected=537", checks)
    train_rows = panel["split"].eq("Train 2024").sum()
    holdout_rows = panel["split"].eq("Test 2025").sum()
    require(
        train_rows == 266 and holdout_rows == 271,
        "Temporal split counts",
        f"train={train_rows}, holdout={holdout_rows}, expected=266/271",
        checks,
    )
    require(
        panel["target_date"].min() == pd.Timestamp("2024-02-06")
        and panel["target_date"].max() == pd.Timestamp("2025-12-24"),
        "Modeling date boundaries",
        f"first={panel['target_date'].min().date()}, last={panel['target_date'].max().date()}",
        checks,
    )
    require(len(catalog) == 164, "Engineered feature count", f"observed={len(catalog)}, expected=164", checks)
    main_count = int(catalog["retained_main"].eq(1).sum())
    sensitivity_count = int(
        (catalog["retained_main"].eq(0) & catalog["retained_sensitivity"].eq(1)).sum()
    )
    require(
        main_count == 127 and sensitivity_count == 37,
        "Main and sensitivity feature counts",
        f"main={main_count}, sensitivity_only={sensitivity_count}, expected=127/37",
        checks,
    )
    require(
        row_log["included_main"].astype(bool).sum() == 537,
        "Row eligibility inclusion count",
        f"observed={row_log['included_main'].astype(bool).sum()}, expected=537",
        checks,
    )
    non_calendar = catalog.loc[~catalog["block"].eq("Calendar and seasonality")]
    timing_ok = non_calendar["transform"].str.contains(
        r"lag|previous|excluding target|carried|forecast-origin", case=False, regex=True
    ).all()
    require(bool(timing_ok), "Anti-leakage transformation audit", "all non-calendar features are historical", checks)
    require(
        (panel["forecast_origin_date"] == panel["target_date"] - pd.Timedelta(days=1)).all(),
        "Forecast origin alignment",
        "forecast_origin_date equals target_date minus one calendar day",
        checks,
    )

    folds = pd.read_csv(TABLES / "table03_expanding_window_folds.csv")
    require(
        folds["train_rows"].tolist() == [46, 90, 134, 178, 222]
        and folds["validation_rows"].tolist() == [44, 44, 44, 44, 44],
        "Expanding-window fold sizes",
        f"train={folds['train_rows'].tolist()}, validation={folds['validation_rows'].tolist()}",
        checks,
    )
    table20 = pd.read_csv(TABLES / "table20_regime_operational_profile.csv")
    require(
        table20["regime"].tolist()
        == ["2024 training", "2025 H1 holdout", "2025 H2 holdout"]
        and int(table20["eligible_days"].sum()) == 537,
        "Regime profile coverage",
        f"regimes={table20['regime'].tolist()}, eligible_days={table20['eligible_days'].sum()}",
        checks,
    )
    table21 = pd.read_csv(TABLES / "table21_holdout_performance_by_regime.csv")
    regime_model_totals = table21.groupby("model")["holdout_days"].sum().to_dict()
    require(
        len(table21) == 10 and set(regime_model_totals.values()) == {271},
        "Holdout regime model coverage",
        f"rows={len(table21)}, model_totals={regime_model_totals}",
        checks,
    )
    table22 = pd.read_csv(TABLES / "table22_fold_stability_summary.csv")
    require(
        len(table22) == 5 and table22["n_folds"].eq(5).all(),
        "Fold stability summary coverage",
        f"models={len(table22)}, n_folds={table22['n_folds'].tolist()}",
        checks,
    )
    holdout = pd.read_csv(PREDICTIONS / "holdout_predictions.csv")
    model_counts = holdout.groupby("model").size().to_dict()
    require(
        set(model_counts.values()) == {271} and len(model_counts) == 5,
        "Advanced holdout prediction coverage",
        f"model_counts={model_counts}",
        checks,
    )
    require(
        holdout[["actual", "prediction"]].notna().all().all(),
        "Prediction completeness",
        "no missing actuals or predictions",
        checks,
    )
    importance = pd.read_csv(TABLES / "table15_random_forest_importance.csv")
    require(
        importance["holdout_permutation_mae_increase_mean"].is_monotonic_decreasing
        and importance["holdout_permutation_rank"].tolist() == list(range(1, len(importance) + 1)),
        "Permutation-importance ranking",
        "Table 15 is ranked by post-hoc holdout permutation MAE increase",
        checks,
    )
    significance = pd.read_csv(TABLES / "paired_forecast_significance_tests.csv")
    require(
        significance["moving_block_length_eligible_rows"].eq(7).all()
        and significance["bootstrap_resamples"].eq(2000).all()
        and significance["holm_adjusted_p_value"].between(0, 1).all(),
        "Serial-dependence-aware significance analysis",
        "seven-eligible-observation circular moving-block bootstrap with 2,000 resamples and valid adjusted p-values",
        checks,
    )
    costs = pd.read_csv(TABLES / "table17_counterfactual_cost_scenarios.csv")
    combined = costs.loc[costs["cost_variant"].eq("Combined stress and backlog")]
    combined_expected = (
        combined["alert_days"]
        + combined["missed_stress_penalty"] * combined["missed_stress_days"]
        + combined["missed_backlog_penalty"] * combined["missed_backlog_days"]
    )
    require(
        (combined_expected - combined["cost_units"]).abs().max() < 1e-12,
        "Counterfactual combined-cost arithmetic",
        "C = alerts + lambda_stress times missed stress + lambda_backlog times missed backlog",
        checks,
    )

    forbidden_extensions = {".xlsx", ".xls", ".parquet", ".pkl", ".joblib"}
    forbidden_files = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_extensions
    ]
    require(
        not forbidden_files,
        "No confidential-style binary data files",
        f"forbidden_files={forbidden_files}",
        checks,
    )
    findings = privacy_scan()
    pd.DataFrame(findings, columns=["file", "finding", "matched_text"]).to_csv(
        OUTPUT_METADATA / "privacy_scan.csv", index=False
    )
    require(not findings, "Privacy text scan", f"findings={findings}", checks)

    validation = pd.DataFrame(checks)
    validation.to_csv(OUTPUT_METADATA / "validation_report.csv", index=False)
    versions = {}
    for package in [
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "torch",
        "matplotlib",
        "PyYAML",
        "openpyxl",
        "scipy",
    ]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    manifest = {
        "package_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "data_mode": "synthetic_public",
        "contains_company_data": False,
        "validation_status": "PASS",
        "validation_checks": len(validation),
        "file_count_before_checksums": sum(1 for path in ROOT.rglob("*") if path.is_file()),
        "package_versions": versions,
    }
    (OUTPUT_METADATA / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    checksum_rows = []
    for path in sorted(ROOT.rglob("*")):
        if (
            path.is_file()
            and path.name != "checksums.sha256"
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ):
            checksum_rows.append(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}")
    (ROOT / "checksums.sha256").write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")
    print(validation.to_string(index=False))
    print(f"Validation PASS: {len(validation)} checks; {len(checksum_rows)} files checksummed.")


if __name__ == "__main__":
    main()
