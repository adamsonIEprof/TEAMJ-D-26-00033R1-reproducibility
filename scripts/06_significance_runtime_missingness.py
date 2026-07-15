from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lstm_model import fit_predict_lstm
from repro_common import (
    backlog_context,
    build_specs,
    load_config,
    load_inputs,
    metric_dict,
    set_global_seed,
)


TABLES = ROOT / "outputs" / "tables"
PREDICTIONS = ROOT / "outputs" / "predictions"
METADATA = ROOT / "outputs" / "metadata"


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = (m - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def circular_moving_block_means(
    values: np.ndarray,
    block_length: int,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample a serial sequence with circular moving blocks and return means."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.array([], dtype=float)
    block_length = max(1, min(int(block_length), n))
    blocks_per_draw = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    means = np.empty(int(resamples), dtype=float)
    for draw in range(int(resamples)):
        starts = rng.integers(0, n, size=blocks_per_draw)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        means[draw] = values[indices].mean()
    return means


def paired_significance(holdout: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    alpha = float(cfg["validation"]["paired_significance_alpha"])
    block_length = int(cfg["validation"]["block_bootstrap_length"])
    resamples = int(cfg["validation"]["block_bootstrap_resamples"])
    seed = int(cfg["project"]["random_seed"])
    pivot_prediction = holdout.pivot_table(
        index="target_date", columns="model", values="prediction", aggfunc="first"
    ).sort_index()
    actual = (
        holdout[["target_date", "actual"]]
        .drop_duplicates()
        .set_index("target_date")["actual"]
        .sort_index()
    )
    rows = []
    models = list(pivot_prediction.columns)
    pair_number = 0
    for left_index, left in enumerate(models):
        for right in models[left_index + 1 :]:
            pair_number += 1
            common = pivot_prediction[[left, right]].dropna().index.intersection(actual.dropna().index)
            loss_left = np.abs(actual.loc[common] - pivot_prediction.loc[common, left])
            loss_right = np.abs(actual.loc[common] - pivot_prediction.loc[common, right])
            difference = loss_left.to_numpy(float) - loss_right.to_numpy(float)
            observed_mean = float(difference.mean())
            rng = np.random.default_rng(seed + pair_number)
            bootstrap_means = circular_moving_block_means(
                difference, block_length, resamples, rng
            )
            centered_means = circular_moving_block_means(
                difference - observed_mean, block_length, resamples, rng
            )
            ci_lower, ci_upper = np.quantile(bootstrap_means, [alpha / 2, 1 - alpha / 2])
            bootstrap_p = float(
                (1 + np.count_nonzero(np.abs(centered_means) >= abs(observed_mean)))
                / (resamples + 1)
            )
            if np.allclose(difference, 0):
                statistic, wilcoxon_p = 0.0, 1.0
                bootstrap_p = 1.0
                ci_lower = ci_upper = 0.0
            else:
                result = wilcoxon(difference, zero_method="wilcox", correction=False, alternative="two-sided")
                statistic, wilcoxon_p = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "paired_days": len(common),
                    "mean_absolute_error_a": float(loss_left.mean()),
                    "mean_absolute_error_b": float(loss_right.mean()),
                    "mean_loss_difference_a_minus_b": observed_mean,
                    "moving_block_length_eligible_rows": block_length,
                    "bootstrap_resamples": resamples,
                    "block_bootstrap_ci_lower": float(ci_lower),
                    "block_bootstrap_ci_upper": float(ci_upper),
                    "block_bootstrap_raw_p_value": bootstrap_p,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_raw_p_value_supplementary": wilcoxon_p,
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p_value"] = holm_adjust(
        frame["block_bootstrap_raw_p_value"].tolist()
    )
    frame["significant_at_alpha"] = frame["holm_adjusted_p_value"].lt(alpha)
    frame["alpha"] = alpha
    frame["interpretation_limit"] = (
        "Exploratory paired absolute-error comparison using circular moving blocks and Holm correction."
    )
    return frame


def missingness_sensitivity(panel, catalog, cfg) -> pd.DataFrame:
    specs = build_specs(catalog, cfg["feature_blocks"]["order"])
    features = specs["S2"]
    catalog_indexed = catalog.set_index("feature_name")
    maskable_features = [
        feature
        for feature in features
        if catalog_indexed.loc[feature, "block"] != "Calendar and seasonality"
    ]
    target = cfg["project"]["target_primary"]
    train = panel.loc[panel["split"].eq(cfg["project"]["train_label"])].reset_index(drop=True)
    test = panel.loc[panel["split"].eq(cfg["project"]["test_label"])].reset_index(drop=True)
    rows = []
    seed = int(cfg["project"]["random_seed"])
    rates = cfg["missingness_sensitivity"]["synthetic_mask_rates"]
    repeats = int(cfg["missingness_sensitivity"]["repeats"])
    rf_config = cfg["models"]["random_forest_s2"]
    for rate in [0.0] + list(rates):
        for repeat in range(repeats if rate > 0 else 1):
            rng = np.random.default_rng(seed + repeat + int(rate * 1000))
            x_train = train[features].copy()
            x_test = test[features].copy()
            if rate > 0:
                train_mask = rng.random((len(x_train), len(maskable_features))) < float(rate)
                test_mask = rng.random((len(x_test), len(maskable_features))) < float(rate)
                x_train.loc[:, maskable_features] = x_train[maskable_features].mask(train_mask)
                x_test.loc[:, maskable_features] = x_test[maskable_features].mask(test_mask)
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(**rf_config)),
                ]
            )
            pipeline.fit(x_train, train[target])
            prediction = np.clip(pipeline.predict(x_test), 0, None)
            rows.append(
                {
                    "synthetic_mask_rate": rate,
                    "repeat": repeat + 1,
                    "masking_scope": "Lagged autoregressive features only; calendar fields remain observed",
                    "maskable_feature_count": len(maskable_features),
                    "imputation": "Training-sample median fitted without holdout information",
                    "masked_training_cells": int(x_train.isna().sum().sum()),
                    "masked_holdout_cells": int(x_test.isna().sum().sum()),
                    **metric_dict(test[target], prediction),
                }
            )
    return pd.DataFrame(rows)


def runtime_benchmarks(panel, catalog, cfg) -> pd.DataFrame:
    specs = build_specs(catalog, cfg["feature_blocks"]["order"])
    features = specs["S2"]
    target = cfg["project"]["target_primary"]
    train = panel.loc[panel["split"].eq(cfg["project"]["train_label"])].reset_index(drop=True)
    test = panel.loc[panel["split"].eq(cfg["project"]["test_label"])].reset_index(drop=True)
    builders = {
        "14-day moving average": None,
        "Random forest (S2)": lambda: RandomForestRegressor(
            **cfg["models"]["random_forest_s2"]
        ),
        "XGBoost (S2)": lambda: XGBRegressor(**cfg["models"]["xgboost_s2"]),
        "LightGBM (S2)": lambda: LGBMRegressor(**cfg["models"]["lightgbm_s2"]),
    }
    repeats = int(cfg["runtime_benchmark"]["repeats"])
    rows = []
    for model_name, builder in builders.items():
        for repeat in range(repeats):
            if builder is None:
                start = time.perf_counter()
                prediction = test["roll14_mean_target_utilization_intensity"].to_numpy(float)
                fit_seconds = 0.0
                inference_seconds = time.perf_counter() - start
            else:
                model = builder()
                start = time.perf_counter()
                model.fit(train[features], train[target])
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                prediction = model.predict(test[features])
                inference_seconds = time.perf_counter() - start
            rows.append(
                {
                    "model": model_name,
                    "repeat": repeat + 1,
                    "timing_scope": "Separate model fit and batch prediction wall-clock times",
                    "training_rows": len(train),
                    "inference_rows": len(test),
                    "fit_seconds": fit_seconds,
                    "batch_inference_seconds": inference_seconds,
                    "milliseconds_per_prediction": 1000 * inference_seconds / len(test),
                    "MAE": metric_dict(test[target], prediction)["MAE"],
                }
            )

    # LSTM is timed once because its deterministic early-stopping path is substantially slower.
    combined = pd.concat([train, test], ignore_index=True)
    start = time.perf_counter()
    prediction, kept, log = fit_predict_lstm(
        combined[features].to_numpy(float),
        combined[target].to_numpy(float),
        np.arange(len(train)),
        np.arange(len(train), len(combined)),
        cfg["models"]["lstm_28"],
    )
    elapsed = time.perf_counter() - start
    fit_seconds = float(log["preprocessing_seconds"] + log["training_seconds"])
    inference_seconds = float(log["inference_seconds"])
    rows.append(
        {
            "model": "LSTM 28-step (S2)",
            "repeat": 1,
            "timing_scope": (
                "Fit includes scaling, eligible-row sequence construction, and training; "
                "inference is timed separately"
            ),
            "training_rows": len(train),
            "inference_rows": len(kept),
            "fit_seconds": fit_seconds,
            "batch_inference_seconds": inference_seconds,
            "milliseconds_per_prediction": 1000 * inference_seconds / len(kept),
            "total_function_seconds": elapsed,
            "MAE": metric_dict(combined.iloc[kept][target], prediction)["MAE"],
            "epochs_run": log["epochs_run"],
        }
    )
    return pd.DataFrame(rows)


def retraining_cadence(panel, catalog, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = build_specs(catalog, cfg["feature_blocks"]["order"])
    features = specs["S2"]
    target = cfg["project"]["target_primary"]
    train = panel.loc[panel["split"].eq(cfg["project"]["train_label"])].reset_index(drop=True)
    test = panel.loc[panel["split"].eq(cfg["project"]["test_label"])].reset_index(drop=True)
    cadences = {
        "No 2025 refit": lambda month: month == 1,
        "Quarterly expanding refit": lambda month: month in {1, 4, 7, 10},
        "Monthly expanding refit": lambda month: True,
    }
    predictions = []
    for cadence, should_refit in cadences.items():
        fitted_model = None
        latest_refit_size = None
        latest_refit_month = None
        for period, month_rows in test.groupby(test["target_date"].dt.to_period("M"), sort=True):
            first_index = int(month_rows.index.min())
            if fitted_model is None or should_refit(period.month):
                history = pd.concat([train, test.iloc[:first_index]], ignore_index=True)
                fitted_model = RandomForestRegressor(**cfg["models"]["random_forest_s2"])
                fitted_model.fit(history[features], history[target])
                latest_refit_size = len(history)
                latest_refit_month = str(period)
            prediction = np.clip(fitted_model.predict(month_rows[features]), 0, None)
            predictions.append(
                pd.DataFrame(
                    {
                        "target_date": month_rows["target_date"],
                        "actual": month_rows[target],
                        "prediction": prediction,
                        "cadence": cadence,
                        "forecast_month": str(period),
                        "latest_refit_month": latest_refit_month,
                        "training_rows_at_latest_refit": latest_refit_size,
                    }
                )
            )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    rows = []
    for cadence, group in prediction_frame.groupby("cadence", sort=False):
        rows.append({"cadence": cadence, "holdout_rows": len(group), **metric_dict(group["actual"], group["prediction"])})
    return pd.DataFrame(rows), prediction_frame


def assign_regime(dates: pd.Series, training_mask: pd.Series | None = None) -> pd.Series:
    dates = pd.to_datetime(dates)
    regime = pd.Series(np.where(dates.dt.month.le(6), "2025 H1 holdout", "2025 H2 holdout"), index=dates.index)
    if training_mask is not None:
        regime.loc[training_mask.to_numpy(bool)] = "2024 training"
    return regime


def regime_operational_profile(
    panel: pd.DataFrame,
    master: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    threshold = float(cfg["threshold_calibration"]["actual_stress_threshold"])
    context = backlog_context(master).rename(columns={"date": "target_date"})
    eligible = panel[["target_date", "split"]].merge(
        context,
        on="target_date",
        how="left",
        validate="one_to_one",
    )
    eligible["regime"] = assign_regime(
        eligible["target_date"],
        eligible["split"].eq(cfg["project"]["train_label"]),
    )
    order = ["2024 training", "2025 H1 holdout", "2025 H2 holdout"]
    rows = []
    for regime in order:
        group = eligible.loc[eligible["regime"].eq(regime)]
        qa = group["any_qa_flag"].fillna(False).astype(bool)
        backlog = group["backlog_risk_day"].fillna(False).astype(bool)
        stress = group["target_utilization_intensity"].ge(threshold)
        rows.append(
            {
                "regime": regime,
                "eligible_days": len(group),
                "start_date": group["target_date"].min().date().isoformat(),
                "end_date": group["target_date"].max().date().isoformat(),
                "mean_utilization": group["target_utilization_intensity"].mean(),
                "mean_scheduled_deliveries": group["total_scheduled_deliveries"].mean(),
                "mean_served_deliveries": group["total_served"].mean(),
                "mean_available_trucks": group["available_trucks_total"].mean(),
                "mean_available_drivers": group["available_drivers"].mean(),
                "qa_flagged_days": int(qa.sum()),
                "qa_flag_share": float(qa.mean()),
                "backlog_risk_days": int(backlog.sum()),
                "backlog_risk_share": float(backlog.mean()),
                "stress_days": int(stress.sum()),
                "stress_share": float(stress.mean()),
                "data_scope": "Synthetic demonstration data",
            }
        )
    return pd.DataFrame(rows)


def outcome_recall(actual_positive: pd.Series, alert: pd.Series) -> float:
    actual = actual_positive.fillna(False).astype(bool).to_numpy()
    predicted = alert.fillna(False).astype(bool).to_numpy()
    positives = int(actual.sum())
    return float((actual & predicted).sum() / positives) if positives else np.nan


def holdout_performance_by_regime(
    holdout_context: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    threshold = float(cfg["threshold_calibration"]["actual_stress_threshold"])
    frame = holdout_context.copy()
    frame["regime"] = assign_regime(frame["target_date"])
    rows = []
    for (model, regime), group in frame.groupby(["model", "regime"], sort=False):
        error = group["prediction"].to_numpy(float) - group["actual"].to_numpy(float)
        alert = group["prediction"].ge(threshold)
        stress = group["actual"].ge(threshold)
        backlog = group["backlog_risk_day"].fillna(False).astype(bool)
        critical = stress | backlog
        rows.append(
            {
                "model": model,
                "regime": regime,
                "holdout_days": len(group),
                "MAE": float(np.mean(np.abs(error))),
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "bias_prediction_minus_actual": float(np.mean(error)),
                "underprediction_rate": float(np.mean(error < 0)),
                "forecast_alert_threshold": threshold,
                "alert_days": int(alert.sum()),
                "alert_burden": float(alert.mean()),
                "stress_days": int(stress.sum()),
                "stress_recall": outcome_recall(stress, alert),
                "backlog_risk_days": int(backlog.sum()),
                "backlog_recall": outcome_recall(backlog, alert),
                "critical_risk_days": int(critical.sum()),
                "critical_recall": outcome_recall(critical, alert),
                "data_scope": "Synthetic demonstration data",
            }
        )
    return pd.DataFrame(rows)


def fold_stability_summary(
    fold_results: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    threshold = float(cfg["threshold_calibration"]["actual_stress_threshold"])
    recall_rows = []
    for (model, fold), group in oof_predictions.groupby(["model", "fold"], sort=False):
        actual = group["actual"].ge(threshold)
        alert = group["prediction"].ge(threshold)
        recall_rows.append(
            {
                "model": model,
                "fold": fold,
                "fold_stress_days": int(actual.sum()),
                "fold_stress_recall": outcome_recall(actual, alert),
            }
        )
    recall_frame = pd.DataFrame(recall_rows)
    raw = fold_results.merge(recall_frame, on=["model", "fold"], how="left", validate="one_to_one")
    rows = []
    for model, group in raw.groupby("model", sort=False):
        rows.append(
            {
                "model": model,
                "n_folds": group["fold"].nunique(),
                "cv_MAE_mean": group["MAE"].mean(),
                "cv_MAE_sd": group["MAE"].std(ddof=1),
                "cv_MAE_min": group["MAE"].min(),
                "cv_MAE_max": group["MAE"].max(),
                "cv_MAE_range": group["MAE"].max() - group["MAE"].min(),
                "cv_RMSE_mean": group["RMSE"].mean(),
                "fold_stress_days_total": int(group["fold_stress_days"].sum()),
                "fold_stress_recall_mean": group["fold_stress_recall"].mean(),
                "fold_stress_recall_sd": group["fold_stress_recall"].std(ddof=1),
                "data_scope": "Synthetic demonstration data",
            }
        )
    return pd.DataFrame(rows).sort_values("cv_MAE_mean", ignore_index=True)


def main() -> None:
    cfg = load_config()
    seed = int(cfg["project"]["random_seed"])
    set_global_seed(seed)
    panel, catalog, _, master = load_inputs()
    holdout = pd.read_csv(PREDICTIONS / "holdout_predictions.csv", parse_dates=["target_date"])
    holdout_context = pd.read_csv(
        PREDICTIONS / "holdout_predictions_with_context.csv", parse_dates=["target_date"]
    )
    oof = pd.read_csv(PREDICTIONS / "oof_train_predictions.csv", parse_dates=["target_date"])
    fold_results = pd.read_csv(METADATA / "fold_level_model_results.csv")
    paired_significance(holdout, cfg).to_csv(
        TABLES / "paired_forecast_significance_tests.csv", index=False
    )
    missingness_sensitivity(panel, catalog, cfg).to_csv(
        TABLES / "missingness_sensitivity.csv", index=False
    )
    runtime = runtime_benchmarks(panel, catalog, cfg)
    runtime.to_csv(TABLES / "runtime_benchmarks.csv", index=False)
    cadence_summary, cadence_predictions = retraining_cadence(panel, catalog, cfg)
    cadence_summary.to_csv(TABLES / "retraining_cadence_evaluation.csv", index=False)
    cadence_predictions.to_csv(PREDICTIONS / "retraining_cadence_predictions.csv", index=False)
    regime_operational_profile(panel, master, cfg).to_csv(
        TABLES / "table20_regime_operational_profile.csv", index=False
    )
    holdout_performance_by_regime(holdout_context, cfg).to_csv(
        TABLES / "table21_holdout_performance_by_regime.csv", index=False
    )
    fold_stability_summary(fold_results, oof, cfg).to_csv(
        TABLES / "table22_fold_stability_summary.csv", index=False
    )
    runtime_metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "timing_method": "time.perf_counter wall-clock seconds",
        "interpretation": "Synthetic small-sample benchmark; production times depend on hardware and data size.",
    }
    (METADATA / "runtime_environment.json").write_text(
        json.dumps(runtime_metadata, indent=2), encoding="utf-8"
    )
    print(runtime.groupby("model", as_index=False)[["fit_seconds", "batch_inference_seconds"]].median().to_string(index=False))
    print(cadence_summary[["cadence", "MAE", "RMSE"]].to_string(index=False))


if __name__ == "__main__":
    main()
