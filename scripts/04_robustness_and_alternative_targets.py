from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lstm_model import fit_predict_lstm
from repro_common import (
    add_qa_flags,
    build_specs,
    expanding_splits,
    load_config,
    load_inputs,
    metric_dict,
    model_cv_and_holdout,
    set_global_seed,
)


TABLES = ROOT / "outputs" / "tables"
PREDICTIONS = ROOT / "outputs" / "predictions"
METADATA = ROOT / "outputs" / "metadata"


def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    return {
        "true_positive_days": tp,
        "false_positive_days": fp,
        "false_negative_days": fn,
        "true_negative_days": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def model_builders(cfg):
    return {
        "Random forest (S2)": lambda: RandomForestRegressor(
            **cfg["models"]["random_forest_s2"]
        ),
        "XGBoost (S2)": lambda: XGBRegressor(**cfg["models"]["xgboost_s2"]),
        "LightGBM (S2)": lambda: LGBMRegressor(**cfg["models"]["lightgbm_s2"]),
    }


def cleaned_training_check(cfg, panel, catalog, master):
    master = add_qa_flags(master)
    specs = build_specs(catalog, cfg["feature_blocks"]["order"])
    s2 = specs["S2"]
    train = panel.loc[panel["split"].eq(cfg["project"]["train_label"])].reset_index(drop=True)
    test = panel.loc[panel["split"].eq(cfg["project"]["test_label"])].reset_index(drop=True)
    target = cfg["project"]["target_primary"]
    train_flagged = train.merge(
        master[["date", "any_qa_flag", "structural_nonoperational_day"]],
        left_on="target_date",
        right_on="date",
        how="left",
    )
    test_flagged = test.merge(
        master[["date", "any_qa_flag", "structural_nonoperational_day"]],
        left_on="target_date",
        right_on="date",
        how="left",
    )
    clean_train = (
        ~train_flagged["any_qa_flag"].fillna(False).astype(bool)
        & train_flagged["structural_nonoperational_day"].fillna(0).eq(0)
    )
    clean_test = (
        ~test_flagged["any_qa_flag"].fillna(False).astype(bool)
        & test_flagged["structural_nonoperational_day"].fillna(0).eq(0)
    )

    baseline = test["roll14_mean_target_utilization_intensity"].to_numpy(float)
    rf_all = RandomForestRegressor(**cfg["models"]["random_forest_s2"])
    rf_all.fit(train[s2], train[target])
    prediction_all = np.clip(rf_all.predict(test[s2]), 0, None)
    rf_clean = RandomForestRegressor(**cfg["models"]["random_forest_s2"])
    rf_clean.fit(train_flagged.loc[clean_train, s2], train_flagged.loc[clean_train, target])
    prediction_clean = np.clip(rf_clean.predict(test[s2]), 0, None)
    rows = []
    scenarios = [
        ("14-day moving average", "No training", np.ones(len(test), dtype=bool), baseline),
        ("Random forest (S2)", "All synthetic 2024 training rows", np.ones(len(test), dtype=bool), prediction_all),
        ("Random forest (S2)", "QA-clean synthetic 2024 training rows", np.ones(len(test), dtype=bool), prediction_clean),
        ("14-day moving average", "No training", clean_test.to_numpy(bool), baseline),
        ("Random forest (S2)", "All synthetic 2024 training rows", clean_test.to_numpy(bool), prediction_all),
        ("Random forest (S2)", "QA-clean synthetic 2024 training rows", clean_test.to_numpy(bool), prediction_clean),
    ]
    for model, training_variant, subset, prediction in scenarios:
        rows.append(
            {
                "model": model,
                "training_variant": training_variant,
                "test_subset": "QA-clean synthetic holdout" if not subset.all() else "All synthetic holdout",
                "training_rows": int(clean_train.sum())
                if "QA-clean" in training_variant
                else (len(train) if model != "14-day moving average" else 0),
                "test_rows": int(subset.sum()),
                **metric_dict(test.loc[subset, target], prediction[subset]),
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "table19_cleaned_training_random_forest.csv", index=False)
    pd.DataFrame(
        {
            "target_date": test["target_date"],
            "actual": test[target],
            "baseline_14day": baseline,
            "rf_all_training": prediction_all,
            "rf_cleaned_training": prediction_clean,
            "qa_clean_test_day": clean_test.to_numpy(bool),
        }
    ).to_csv(PREDICTIONS / "primary_robustness_predictions.csv", index=False)


def add_calendar_baselines(panel: pd.DataFrame, full_panel: pd.DataFrame, targets: dict[str, str]):
    full = full_panel.sort_values("target_date").copy()
    baseline_columns = {}
    for target_column in targets:
        baseline_name = f"baseline14_{target_column}"
        full[baseline_name] = full[target_column].shift(1).rolling(14, min_periods=7).mean()
        baseline_columns[target_column] = baseline_name
    baseline_frame = full[["target_date"] + list(baseline_columns.values())]
    return panel.merge(baseline_frame, on="target_date", how="left"), baseline_columns


def evaluate_target(
    target_column: str,
    target_label: str,
    baseline_column: str,
    panel: pd.DataFrame,
    features: list[str],
    cfg: dict,
    run_lstm: bool,
):
    train = panel.loc[panel["split"].eq(cfg["project"]["train_label"])].reset_index(drop=True)
    test = panel.loc[panel["split"].eq(cfg["project"]["test_label"])].reset_index(drop=True)
    splits = expanding_splits(train, int(cfg["validation"]["n_splits"]))
    fold_frames = []
    oof_frames = []
    hold_frames = []
    for model_name, builder in model_builders(cfg).items():
        folds, oof, hold = model_cv_and_holdout(
            model_name, builder, train, test, features, target_column, splits
        )
        fold_frames.append(folds)
        oof_frames.append(oof)
        hold_frames.append(hold)

    baseline_name = "14-day moving average"
    y_train = train[target_column].to_numpy(float)
    baseline_fold_rows = []
    baseline_oof = []
    for fold, (_, validation_index) in enumerate(splits, start=1):
        prediction = train.iloc[validation_index][baseline_column].to_numpy(float)
        baseline_fold_rows.append(
            {"model": baseline_name, "fold": fold, **metric_dict(y_train[validation_index], prediction)}
        )
        baseline_oof.append(
            pd.DataFrame(
                {
                    "target_date": train.iloc[validation_index]["target_date"],
                    "actual": y_train[validation_index],
                    "prediction": prediction,
                    "model": baseline_name,
                    "fold": fold,
                }
            )
        )
    baseline_hold = pd.DataFrame(
        {
            "target_date": test["target_date"],
            "actual": test[target_column],
            "prediction": test[baseline_column].clip(lower=0),
            "model": baseline_name,
        }
    )
    fold_frames.append(pd.DataFrame(baseline_fold_rows))
    oof_frames.append(pd.concat(baseline_oof, ignore_index=True))
    hold_frames.append(baseline_hold)

    if run_lstm:
        lstm_cfg = cfg["models"]["lstm_28"]
        x_train = train[features].to_numpy(float)
        y_train = train[target_column].to_numpy(float)
        lstm_fold_rows = []
        lstm_oof = []
        for fold, (train_index, validation_index) in enumerate(splits, start=1):
            prediction, kept, _ = fit_predict_lstm(
                x_train, y_train, train_index, validation_index, lstm_cfg
            )
            lstm_fold_rows.append(
                {"model": "LSTM 28-step (S2)", "fold": fold, **metric_dict(y_train[kept], prediction)}
            )
            lstm_oof.append(
                pd.DataFrame(
                    {
                        "target_date": train.iloc[kept]["target_date"],
                        "actual": y_train[kept],
                        "prediction": prediction,
                        "model": "LSTM 28-step (S2)",
                        "fold": fold,
                    }
                )
            )
        combined = pd.concat([train, test], ignore_index=True)
        x_all = combined[features].to_numpy(float)
        y_all = combined[target_column].to_numpy(float)
        prediction, kept, _ = fit_predict_lstm(
            x_all,
            y_all,
            np.arange(len(train)),
            np.arange(len(train), len(combined)),
            lstm_cfg,
        )
        lstm_hold = pd.DataFrame(
            {
                "target_date": combined.iloc[kept]["target_date"],
                "actual": y_all[kept],
                "prediction": prediction,
                "model": "LSTM 28-step (S2)",
            }
        )
        fold_frames.append(pd.DataFrame(lstm_fold_rows))
        oof_frames.append(pd.concat(lstm_oof, ignore_index=True))
        hold_frames.append(lstm_hold)

    folds = pd.concat(fold_frames, ignore_index=True)
    oof = pd.concat(oof_frames, ignore_index=True)
    hold = pd.concat(hold_frames, ignore_index=True)
    high_pressure_threshold = float(train[target_column].quantile(0.75))
    summary_rows = []
    for model_name, group in hold.groupby("model", sort=False):
        fold_group = folds.loc[folds["model"].eq(model_name)]
        class_metrics = classification_metrics(
            group["actual"].to_numpy(float) >= high_pressure_threshold,
            group["prediction"].to_numpy(float) >= high_pressure_threshold,
        )
        summary_rows.append(
            {
                "target": target_label,
                "target_column": target_column,
                "model": model_name,
                "training_high_pressure_threshold_q75": high_pressure_threshold,
                "cv_MAE_mean": fold_group["MAE"].mean(),
                "cv_MAE_sd": fold_group["MAE"].std(ddof=1),
                "holdout_n": len(group),
                **{f"holdout_{key}": value for key, value in metric_dict(group["actual"], group["prediction"]).items()},
                **{f"high_pressure_{key}": value for key, value in class_metrics.items()},
            }
        )
    summary = pd.DataFrame(summary_rows)
    for frame in [folds, oof, hold]:
        frame.insert(0, "target", target_label)
        frame.insert(1, "target_column", target_column)
    return summary, folds, oof, hold


def main() -> None:
    cfg = load_config()
    set_global_seed(int(cfg["project"]["random_seed"]))
    panel, catalog, full_panel, master = load_inputs()
    cleaned_training_check(cfg, panel, catalog, master)
    specs = build_specs(catalog, cfg["feature_blocks"]["order"])
    s2 = specs["S2"]
    targets = {
        "y_utilization_nextday": "Realized served-truck utilization",
        "y_scheduled_demand_pressure": "Scheduled demand pressure",
        "y_open_order_pressure": "Open-order pressure",
        "y_served_plus_deferred_pressure": "Served plus deferred pressure",
        "y_expanded_demand_pressure_index": "Expanded demand-pressure index",
        "y_secondary_capacity_rate": "Secondary trip-capacity utilization",
    }
    panel, baseline_columns = add_calendar_baselines(panel, full_panel, targets)
    run_lstm = os.environ.get("REPRO_SKIP_ALTERNATIVE_LSTM", "0") != "1"
    summaries = []
    folds = []
    oof = []
    holdout = []
    for target_column, target_label in targets.items():
        summary, fold_frame, oof_frame, hold_frame = evaluate_target(
            target_column,
            target_label,
            baseline_columns[target_column],
            panel,
            s2,
            cfg,
            run_lstm,
        )
        summaries.append(summary)
        folds.append(fold_frame)
        oof.append(oof_frame)
        holdout.append(hold_frame)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(TABLES / "table12_target_specific_model_results.csv", index=False)
    winners_mae = (
        summary.sort_values(["target", "holdout_MAE", "cv_MAE_mean"])
        .groupby("target", as_index=False)
        .first()
    )
    winners_recall = (
        summary.sort_values(
            ["target", "high_pressure_recall", "high_pressure_precision"],
            ascending=[True, False, False],
        )
        .groupby("target", as_index=False)
        .first()
    )
    winners_mae.to_csv(TABLES / "table12_best_mae_by_target.csv", index=False)
    winners_recall.to_csv(TABLES / "table12_best_recall_by_target.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(
        METADATA / "alternative_target_fold_results.csv", index=False
    )
    pd.concat(oof, ignore_index=True).to_csv(
        PREDICTIONS / "alternative_target_oof_predictions.csv", index=False
    )
    pd.concat(holdout, ignore_index=True).to_csv(
        PREDICTIONS / "alternative_target_holdout_predictions.csv", index=False
    )
    print(winners_mae[["target", "model", "holdout_MAE"]].to_string(index=False))
    print(winners_recall[["target", "model", "high_pressure_recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
