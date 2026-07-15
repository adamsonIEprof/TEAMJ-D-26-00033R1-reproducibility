from __future__ import annotations

import importlib.metadata as md
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lstm_model import fit_predict_lstm
from repro_common import (
    build_specs,
    decision_metric_dict,
    expanding_splits,
    fold_definition_table,
    load_config,
    load_inputs,
    metric_dict,
    model_cv_and_holdout,
    save_json,
    set_global_seed,
)


def main() -> None:
    cfg = load_config()
    seed = int(cfg["project"]["random_seed"])
    set_global_seed(seed)
    panel, catalog, _, _ = load_inputs()
    specs = build_specs(catalog, cfg["feature_blocks"]["order"])
    s2 = specs["S2"]
    train = panel.loc[panel["split"].eq(cfg["project"]["train_label"])].reset_index(drop=True)
    test = panel.loc[panel["split"].eq(cfg["project"]["test_label"])].reset_index(drop=True)
    splits = expanding_splits(train, int(cfg["validation"]["n_splits"]))
    target = cfg["project"]["target_primary"]

    fold_defs = fold_definition_table(train, splits)
    fold_defs.to_csv(ROOT / "outputs" / "tables" / "table03_expanding_window_folds.csv", index=False)

    rf_cfg = cfg["models"]["random_forest_s2"]
    xgb_cfg = cfg["models"]["xgboost_s2"]
    lgb_cfg = cfg["models"]["lightgbm_s2"]

    builders = {
        "Random forest (S2)": lambda: RandomForestRegressor(**rf_cfg),
        "XGBoost (S2)": lambda: XGBRegressor(**xgb_cfg),
        "LightGBM (S2)": lambda: LGBMRegressor(**lgb_cfg),
    }

    fold_frames, oof_frames, hold_frames = [], [], []
    for name, builder in builders.items():
        folds, oof, hold = model_cv_and_holdout(
            name, builder, train, test, s2, target, splits
        )
        fold_frames.append(folds)
        oof_frames.append(oof)
        hold_frames.append(hold)

    # Short-memory baseline: the forecast is the 14-day moving mean feature.
    baseline_name = "14-day moving average"
    baseline_col = "roll14_mean_target_utilization_intensity"
    baseline_fold_rows = []
    baseline_oof_rows = []
    y_train = train[target].to_numpy(float)
    for fold, (_, va) in enumerate(splits, start=1):
        pred = train.iloc[va][baseline_col].to_numpy(float)
        baseline_fold_rows.append(
            {"model": baseline_name, "fold": fold, **metric_dict(y_train[va], pred)}
        )
        baseline_oof_rows.append(
            pd.DataFrame(
                {
                    "target_date": train.iloc[va]["target_date"].to_numpy(),
                    "actual": y_train[va],
                    "prediction": pred,
                    "model": baseline_name,
                    "fold": fold,
                }
            )
        )
    baseline_hold = pd.DataFrame(
        {
            "target_date": test["target_date"],
            "actual": test[target],
            "prediction": test[baseline_col].clip(lower=0),
            "model": baseline_name,
        }
    )
    fold_frames.append(pd.DataFrame(baseline_fold_rows))
    oof_frames.append(pd.concat(baseline_oof_rows, ignore_index=True))
    hold_frames.append(baseline_hold)

    # 28-step LSTM on the same S2 feature universe. Sequences follow eligible-panel row order.
    lstm_cfg = cfg["models"]["lstm_28"]
    X_train = train[s2].to_numpy(float)
    y_train = train[target].to_numpy(float)
    lstm_fold_rows = []
    lstm_oof_rows = []
    lstm_training_log = []
    for fold, (tr, va) in enumerate(splits, start=1):
        pred, kept, log = fit_predict_lstm(X_train, y_train, tr, va, lstm_cfg)
        lstm_fold_rows.append(
            {"model": "LSTM 28-step (S2)", "fold": fold, **metric_dict(y_train[kept], pred)}
        )
        lstm_oof_rows.append(
            pd.DataFrame(
                {
                    "target_date": train.iloc[kept]["target_date"].to_numpy(),
                    "actual": y_train[kept],
                    "prediction": pred,
                    "model": "LSTM 28-step (S2)",
                    "fold": fold,
                }
            )
        )
        lstm_training_log.append({"fold": fold, **log})

    combined = pd.concat([train, test], ignore_index=True)
    X_all = combined[s2].to_numpy(float)
    y_all = combined[target].to_numpy(float)
    train_endpoints = np.arange(len(train), dtype=int)
    test_endpoints = np.arange(len(train), len(combined), dtype=int)
    pred_lstm, kept_test, log = fit_predict_lstm(
        X_all, y_all, train_endpoints, test_endpoints, lstm_cfg
    )
    lstm_hold = pd.DataFrame(
        {
            "target_date": combined.iloc[kept_test]["target_date"].to_numpy(),
            "actual": y_all[kept_test],
            "prediction": pred_lstm,
            "model": "LSTM 28-step (S2)",
        }
    )
    lstm_training_log.append({"fold": "full_2024_to_2025", **log})
    fold_frames.append(pd.DataFrame(lstm_fold_rows))
    oof_frames.append(pd.concat(lstm_oof_rows, ignore_index=True))
    hold_frames.append(lstm_hold)

    fold_results = pd.concat(fold_frames, ignore_index=True)
    oof_predictions = pd.concat(oof_frames, ignore_index=True)
    holdout_predictions = pd.concat(hold_frames, ignore_index=True)

    summary_rows = []
    for model, group in holdout_predictions.groupby("model", sort=False):
        m = metric_dict(group["actual"], group["prediction"])
        d = decision_metric_dict(group["actual"], group["prediction"])
        cv = fold_results.loc[fold_results["model"].eq(model)]
        summary_rows.append(
            {
                "model": model,
                "specification": "S2: calendar + autoregressive" if model != baseline_name else "14-day short-memory baseline",
                "n_features": len(s2) if model != baseline_name else 1,
                "cv_MAE_mean": cv["MAE"].mean(),
                "cv_MAE_sd": cv["MAE"].std(ddof=1),
                "cv_RMSE_mean": cv["RMSE"].mean(),
                "holdout_n": len(group),
                **{f"holdout_{k}": v for k, v in m.items()},
                **d,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["holdout_MAE", "cv_MAE_mean"], ignore_index=True
    )

    # Hyperparameter and preprocessing registry.
    hp_rows = [
        {
            "model": "Random forest (S2)",
            "preprocessing": "No scaling; nonnegative prediction clipping",
            "hyperparameters_json": json.dumps(rf_cfg, sort_keys=True),
            "random_seed": rf_cfg["random_state"],
        },
        {
            "model": "XGBoost (S2)",
            "preprocessing": "No scaling; histogram tree method; nonnegative prediction clipping",
            "hyperparameters_json": json.dumps(xgb_cfg, sort_keys=True),
            "random_seed": xgb_cfg["random_state"],
        },
        {
            "model": "LightGBM (S2)",
            "preprocessing": "No scaling; deterministic/force_col_wise; nonnegative clipping",
            "hyperparameters_json": json.dumps(lgb_cfg, sort_keys=True),
            "random_seed": lgb_cfg["random_state"],
        },
        {
            "model": "LSTM 28-step (S2)",
            "preprocessing": "Fold-specific StandardScaler for inputs and target; 28 eligible-row sequences; temporal internal validation; nonnegative clipping",
            "hyperparameters_json": json.dumps(lstm_cfg, sort_keys=True),
            "random_seed": lstm_cfg["random_seed"],
        },
        {
            "model": baseline_name,
            "preprocessing": "Direct use of roll14_mean_target_utilization_intensity",
            "hyperparameters_json": "{}",
            "random_seed": None,
        },
    ]
    hp = pd.DataFrame(hp_rows)

    feature_registry = catalog.loc[catalog["feature_name"].isin(s2), [
        "feature_name", "block", "source_variable", "transform", "lookback_days", "notes"
    ]].copy()
    feature_registry["s2_order"] = feature_registry["feature_name"].map({f: i + 1 for i, f in enumerate(s2)})
    feature_registry = feature_registry.sort_values("s2_order")

    fold_results.to_csv(ROOT / "outputs" / "metadata" / "fold_level_model_results.csv", index=False)
    summary.to_csv(ROOT / "outputs" / "tables" / "table14_s2_advanced_benchmark.csv", index=False)
    hp.to_csv(ROOT / "outputs" / "metadata" / "model_hyperparameters.csv", index=False)
    feature_registry.to_csv(ROOT / "outputs" / "metadata" / "s2_feature_block_assignments.csv", index=False)
    oof_predictions.to_csv(ROOT / "outputs" / "predictions" / "oof_train_predictions.csv", index=False)
    holdout_predictions.to_csv(ROOT / "outputs" / "predictions" / "holdout_predictions.csv", index=False)
    pd.DataFrame(lstm_training_log).to_csv(ROOT / "outputs" / "metadata" / "lstm_training_log.csv", index=False)

    versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "torch", "matplotlib", "PyYAML", "openpyxl", "scipy"]:
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = "not installed"
    versions["python"] = sys.version
    save_json(versions, ROOT / "outputs" / "metadata" / "package_versions.json")
    print(summary[["model", "cv_MAE_mean", "holdout_MAE", "holdout_RMSE"]].to_string(index=False))


if __name__ == "__main__":
    main()
