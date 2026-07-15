from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from repro_common import (
    build_specs,
    expanding_splits,
    fold_definition_table,
    load_config,
    load_inputs,
    metric_dict,
    set_global_seed,
)


TABLES = ROOT / "outputs" / "tables"
PREDICTIONS = ROOT / "outputs" / "predictions"
METADATA = ROOT / "outputs" / "metadata"


def make_models(seed: int):
    return {
        "OLS": lambda: Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Ridge": lambda: Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=20.0))]
        ),
        "LASSO": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.01, max_iter=20000, selection="cyclic")),
            ]
        ),
        "Random forest": lambda: RandomForestRegressor(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        ),
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    seed = int(cfg["project"]["random_seed"])
    set_global_seed(seed)
    panel, catalog, _, _ = load_inputs()
    specs = build_specs(catalog, cfg["feature_blocks"]["order"])
    train = panel.loc[panel["split"].eq(cfg["project"]["train_label"])].reset_index(drop=True)
    test = panel.loc[panel["split"].eq(cfg["project"]["test_label"])].reset_index(drop=True)
    target = cfg["project"]["target_primary"]
    splits = expanding_splits(train, int(cfg["validation"]["n_splits"]))
    fold_definition_table(train, splits).to_csv(TABLES / "table03_expanding_window_folds.csv", index=False)

    required = sorted(set(feature for features in specs.values() for feature in features))
    if train[required + [target]].isna().any().any() or test[required + [target]].isna().any().any():
        raise ValueError("The modeling panel contains missing values in retained main features or target.")

    baseline_defs = {
        "Naive lag-1": "lag1_target_utilization_intensity",
        "Seasonal naive lag-7": "lag7_target_utilization_intensity",
        "7-day moving average": "roll7_mean_target_utilization_intensity",
        "14-day moving average": "roll14_mean_target_utilization_intensity",
    }
    y_train = train[target].to_numpy(float)
    y_test = test[target].to_numpy(float)
    baseline_fold_rows = []
    baseline_holdout_rows = []
    baseline_predictions = []
    for model_name, column in baseline_defs.items():
        for fold, (_, valid_index) in enumerate(splits, start=1):
            prediction = train.iloc[valid_index][column].to_numpy(float)
            baseline_fold_rows.append(
                {
                    "model": model_name,
                    "specification": "Baseline",
                    "fold": fold,
                    **metric_dict(y_train[valid_index], prediction),
                }
            )
        prediction = np.clip(test[column].to_numpy(float), 0, None)
        baseline_holdout_rows.append(
            {"model": model_name, "specification": "Baseline", **metric_dict(y_test, prediction)}
        )
        baseline_predictions.append(
            pd.DataFrame(
                {
                    "target_date": test["target_date"],
                    "actual": y_test,
                    "prediction": prediction,
                    "model": model_name,
                    "specification": "Baseline",
                }
            )
        )
    baseline_folds = pd.DataFrame(baseline_fold_rows)
    baseline_holdout = pd.DataFrame(baseline_holdout_rows).sort_values(["MAE", "RMSE"])
    baseline_folds.to_csv(METADATA / "core_baseline_fold_results.csv", index=False)
    baseline_holdout.to_csv(TABLES / "table13_core_baseline_results.csv", index=False)
    pd.concat(baseline_predictions, ignore_index=True).to_csv(
        PREDICTIONS / "core_baseline_holdout_predictions.csv", index=False
    )

    family_rows = []
    fold_rows = []
    holdout_predictions = []
    models = make_models(seed)
    for family, builder in models.items():
        for specification, features in specs.items():
            fold_metrics = []
            for fold, (train_index, valid_index) in enumerate(splits, start=1):
                model = builder()
                model.fit(train.iloc[train_index][features], y_train[train_index])
                prediction = np.clip(model.predict(train.iloc[valid_index][features]), 0, None)
                metrics = metric_dict(y_train[valid_index], prediction)
                fold_metrics.append(metrics)
                fold_rows.append(
                    {
                        "model_family": family,
                        "specification": specification,
                        "fold": fold,
                        "n_features": len(features),
                        **metrics,
                    }
                )
            fold_frame = pd.DataFrame(fold_metrics)
            model = builder()
            model.fit(train[features], y_train)
            holdout_prediction = np.clip(model.predict(test[features]), 0, None)
            holdout_metrics = metric_dict(y_test, holdout_prediction)
            family_rows.append(
                {
                    "model_family": family,
                    "specification": specification,
                    "n_features": len(features),
                    "cv_MAE_mean": fold_frame["MAE"].mean(),
                    "cv_MAE_sd": fold_frame["MAE"].std(ddof=1),
                    "cv_RMSE_mean": fold_frame["RMSE"].mean(),
                    **{f"holdout_{key}": value for key, value in holdout_metrics.items()},
                }
            )
            holdout_predictions.append(
                pd.DataFrame(
                    {
                        "target_date": test["target_date"],
                        "actual": y_test,
                        "prediction": holdout_prediction,
                        "model": family,
                        "specification": specification,
                    }
                )
            )

    family_results = pd.DataFrame(family_rows)
    fold_results = pd.DataFrame(fold_rows)
    selected = (
        family_results.sort_values(
            ["model_family", "cv_MAE_mean", "cv_RMSE_mean", "specification"]
        )
        .groupby("model_family", as_index=False)
        .first()
        .sort_values(["holdout_MAE", "cv_MAE_mean"])
    )
    family_results.to_csv(TABLES / "table13_core_family_spec_results.csv", index=False)
    selected.to_csv(TABLES / "table13_core_selected_models.csv", index=False)
    fold_results.to_csv(METADATA / "core_family_fold_results.csv", index=False)
    predictions = pd.concat(holdout_predictions, ignore_index=True)
    predictions.to_csv(PREDICTIONS / "core_learned_holdout_predictions.csv", index=False)

    # Interpretability for the fixed S2 random forest used in direct comparisons.
    s2 = specs["S2"]
    rf = models["Random forest"]()
    rf.fit(train[s2], y_train)
    rf_prediction = np.clip(rf.predict(test[s2]), 0, None)
    permutation = permutation_importance(
        rf,
        test[s2],
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=seed,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": s2,
            "impurity_importance": rf.feature_importances_,
            "holdout_permutation_mae_increase_mean": permutation.importances_mean,
            "holdout_permutation_mae_increase_sd": permutation.importances_std,
        }
    ).sort_values("holdout_permutation_mae_increase_mean", ascending=False, ignore_index=True)
    importance.insert(0, "holdout_permutation_rank", np.arange(1, len(importance) + 1))
    importance["interpretation_scope"] = (
        "Post-hoc 2025 synthetic-holdout permutation importance; not used for model selection"
    )
    importance.to_csv(TABLES / "table15_random_forest_importance.csv", index=False)

    linear_rows = []
    for family in ["OLS", "Ridge", "LASSO"]:
        model = models[family]()
        model.fit(train[s2], y_train)
        coefficients = model.named_steps["model"].coef_
        for feature, coefficient in zip(s2, coefficients):
            linear_rows.append(
                {"model": family, "feature": feature, "standardized_coefficient": coefficient}
            )
    pd.DataFrame(linear_rows).to_csv(METADATA / "linear_s2_coefficients.csv", index=False)

    pd_rows = []
    for feature in importance.head(5)["feature"]:
        feature_index = s2.index(feature)
        result = partial_dependence(
            rf, test[s2], features=[feature_index], kind="average", grid_resolution=20
        )
        grid = result.get("grid_values", result.get("values"))[0]
        average = result["average"][0]
        for value, prediction in zip(grid, average):
            pd_rows.append(
                {"feature": feature, "feature_value": value, "partial_dependence": prediction}
            )
    pd.DataFrame(pd_rows).to_csv(METADATA / "random_forest_partial_dependence.csv", index=False)

    registry = {
        "target": target,
        "selection_rule": "Minimum mean 2024 expanding-window MAE, then RMSE, then specification ID",
        "holdout_used_for_selection": False,
        "random_seed": seed,
        "feature_counts": {key: len(value) for key, value in specs.items()},
        "model_hyperparameters": {
            "OLS": {},
            "Ridge": {"alpha": 20.0},
            "LASSO": {"alpha": 0.01, "max_iter": 20000},
            "Random forest": {
                "n_estimators": 160,
                "max_depth": 6,
                "min_samples_leaf": 4,
                "max_features": "sqrt",
                "random_state": seed,
                "n_jobs": 1,
            },
        },
    }
    (METADATA / "core_model_registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )
    print(baseline_holdout[["model", "MAE", "RMSE"]].to_string(index=False))
    print(selected[["model_family", "specification", "cv_MAE_mean", "holdout_MAE"]].to_string(index=False))


if __name__ == "__main__":
    main()
