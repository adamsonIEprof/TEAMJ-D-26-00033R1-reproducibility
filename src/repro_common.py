from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "model_config.yaml"
DATA_DIR = ROOT / "data" / "synthetic"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(
        DATA_DIR / "chunk3_main_modeling_panel.csv",
        parse_dates=["target_date", "forecast_origin_date"],
    )
    catalog = pd.read_csv(DATA_DIR / "chunk3_feature_catalog.csv")
    full_panel = pd.read_csv(DATA_DIR / "chunk3_full_feature_panel_with_flags.csv")
    for column in ["date", "target_date", "forecast_origin_date"]:
        if column in full_panel:
            full_panel[column] = pd.to_datetime(full_panel[column])
    master = pd.read_csv(DATA_DIR / "daily_master_locked.csv", parse_dates=["date"])
    return panel, catalog, full_panel, master


def build_specs(catalog: pd.DataFrame, block_order: list[str]) -> dict[str, list[str]]:
    cat = catalog.copy()
    retained = cat["retained_main"].astype(str).str.lower().isin(["1", "true", "yes"])
    cat = cat.loc[retained & cat["feature_name"].ne("target_day_of_week_name")].copy()
    specs: dict[str, list[str]] = {}
    used: list[str] = []
    for i, block in enumerate(block_order, start=1):
        used.extend(cat.loc[cat["block"].eq(block), "feature_name"].tolist())
        specs[f"S{i}"] = list(dict.fromkeys(used))
    return specs


def expanding_splits(train: pd.DataFrame, n_splits: int = 5):
    return list(TimeSeriesSplit(n_splits=n_splits).split(train))


def fold_definition_table(train: pd.DataFrame, splits) -> pd.DataFrame:
    rows = []
    for fold, (tr, va) in enumerate(splits, start=1):
        rows.append(
            {
                "fold": fold,
                "train_rows": len(tr),
                "validation_rows": len(va),
                "train_start": train.iloc[tr[0]]["target_date"].date().isoformat(),
                "train_end": train.iloc[tr[-1]]["target_date"].date().isoformat(),
                "validation_start": train.iloc[va[0]]["target_date"].date().isoformat(),
                "validation_end": train.iloc[va[-1]]["target_date"].date().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def clip_predictions(pred: Iterable[float], floor: float = 0.0) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=float), floor, None)


def metric_dict(y_true, y_pred) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = clip_predictions(y_pred)
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAPE": float(np.mean(np.abs((y - p) / np.clip(np.abs(y), 1e-6, None))) * 100.0),
        "R2": float(r2_score(y, p)),
        "MeanBias": float(np.mean(p - y)),
        "MedianAE": float(np.median(np.abs(y - p))),
    }


def decision_metric_dict(
    y_true,
    y_pred,
    stress_threshold: float = 1.20,
    underprediction_weights: tuple[int, ...] = (2, 5),
    stress_day_weight: float = 3.0,
    pinball_quantiles: tuple[float, ...] = (0.75, 0.90),
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = clip_predictions(y_pred)
    err = p - y
    out: dict[str, float] = {}
    for w in underprediction_weights:
        losses = np.where(err < 0, w * np.abs(err), np.abs(err))
        out[f"AsymAbsLoss_under_w{w}"] = float(np.mean(losses))
    weights = np.where(y >= stress_threshold, stress_day_weight, 1.0)
    out[f"WeightedMAE_stress_w{stress_day_weight:g}"] = float(
        np.average(np.abs(err), weights=weights)
    )
    for q in pinball_quantiles:
        out[f"Pinball_q{q:.2f}"] = float(mean_pinball_loss(y, p, alpha=q))
    labels = (y >= stress_threshold).astype(int)
    out["Stress_average_precision"] = float(average_precision_score(labels, p))
    return out


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def model_cv_and_holdout(
    model_name: str,
    builder: Callable[[], object],
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target_col: str,
    splits,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_train = train[target_col].to_numpy(float)
    y_test = test[target_col].to_numpy(float)
    fold_rows = []
    oof_rows = []
    for fold, (tr, va) in enumerate(splits, start=1):
        model = builder()
        model.fit(train.iloc[tr][features], y_train[tr])
        pred = clip_predictions(model.predict(train.iloc[va][features]))
        fold_rows.append({"model": model_name, "fold": fold, **metric_dict(y_train[va], pred)})
        oof_rows.append(
            pd.DataFrame(
                {
                    "target_date": train.iloc[va]["target_date"].to_numpy(),
                    "actual": y_train[va],
                    "prediction": pred,
                    "model": model_name,
                    "fold": fold,
                }
            )
        )
    final_model = builder()
    final_model.fit(train[features], y_train)
    test_pred = clip_predictions(final_model.predict(test[features]))
    holdout = pd.DataFrame(
        {
            "target_date": test["target_date"].to_numpy(),
            "actual": y_test,
            "prediction": test_pred,
            "model": model_name,
        }
    )
    return pd.DataFrame(fold_rows), pd.concat(oof_rows, ignore_index=True), holdout


def add_qa_flags(master: pd.DataFrame) -> pd.DataFrame:
    out = master.copy()
    qa_cols = [
        "qa_scheduled_parts_mismatch",
        "qa_served_segment_mismatch",
        "qa_committed_total_mismatch",
        "qa_zero_trucks_positive_served",
        "qa_negative_numeric",
    ]
    out["any_qa_flag"] = out[qa_cols].fillna(0).astype(int).any(axis=1)
    return out


def backlog_context(master: pd.DataFrame) -> pd.DataFrame:
    m = add_qa_flags(master)
    m["unmet_scheduled_orders"] = np.maximum(
        m["total_scheduled_deliveries"] - m["total_served"], 0
    )
    scheduled_source = (
        m["scheduled_from_open_orders"]
        if "scheduled_from_open_orders" in m
        else m["total_scheduled_deliveries"]
    )
    m["unconverted_open_orders"] = np.maximum(m["open_orders"] - scheduled_source, 0)
    if "backload_order_count" not in m:
        m["backload_order_count"] = m.get("backload_records", 0)
    m["backlog_risk_day"] = (
        m["carry_over"].fillna(0).gt(0)
        | m["deferred_rescheduled"].fillna(0).gt(0)
        | m["unmet_scheduled_orders"].fillna(0).gt(0)
        | m["unconverted_open_orders"].fillna(0).gt(0)
        | m["backload_order_count"].fillna(0).gt(0)
    )
    m["stress_day"] = m["target_utilization_intensity"].ge(1.20)
    m["critical_risk_day"] = m["stress_day"] | m["backlog_risk_day"]
    return m
