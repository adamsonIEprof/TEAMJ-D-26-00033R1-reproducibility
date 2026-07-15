from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from repro_common import backlog_context, decision_metric_dict, load_config, load_inputs


TABLES = ROOT / "outputs" / "tables"
PREDICTIONS = ROOT / "outputs" / "predictions"
METADATA = ROOT / "outputs" / "metadata"


def classification_metrics(actual_positive, predicted_positive):
    actual = np.asarray(actual_positive, dtype=bool)
    predicted = np.asarray(predicted_positive, dtype=bool)
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


def calibrate_thresholds(oof: pd.DataFrame, cfg: dict):
    stress_threshold = float(cfg["threshold_calibration"]["actual_stress_threshold"])
    start = float(cfg["threshold_calibration"]["forecast_cutoff_grid_start"])
    stop = float(cfg["threshold_calibration"]["forecast_cutoff_grid_stop"])
    step = float(cfg["threshold_calibration"]["forecast_cutoff_grid_step"])
    false_positive_cost = float(cfg["threshold_calibration"]["false_positive_cost"])
    ratios = cfg["threshold_calibration"]["false_negative_cost_ratios"]
    grid = np.round(np.arange(start, stop + step / 2, step), 10)
    all_rows = []
    selected_rows = []
    for model, group in oof.groupby("model", sort=False):
        actual = group["actual"].to_numpy(float) >= stress_threshold
        scores = group["prediction"].to_numpy(float)
        for ratio in ratios:
            candidate_rows = []
            for cutoff in grid:
                metrics = classification_metrics(actual, scores >= cutoff)
                cost = false_positive_cost * metrics["false_positive_days"] + float(ratio) * metrics[
                    "false_negative_days"
                ]
                row = {
                    "model": model,
                    "false_negative_to_false_positive_cost_ratio": float(ratio),
                    "forecast_alert_cutoff": cutoff,
                    "calibration_days": len(group),
                    "cost_units": cost,
                    "normalized_cost": cost / len(group),
                    **metrics,
                }
                candidate_rows.append(row)
                all_rows.append(row)
            selected = pd.DataFrame(candidate_rows).sort_values(
                ["cost_units", "f1", "forecast_alert_cutoff"],
                ascending=[True, False, True],
            )
            selected_rows.append(selected.iloc[0].to_dict())
    return pd.DataFrame(all_rows), pd.DataFrame(selected_rows)


def context_frame(master: pd.DataFrame) -> pd.DataFrame:
    context = backlog_context(master).copy()
    keep = [
        "date",
        "backlog_risk_day",
        "critical_risk_day",
        "any_qa_flag",
        "total_scheduled_deliveries",
        "total_served",
        "available_trucks_total",
        "carry_over",
        "deferred_rescheduled",
    ]
    return context[keep].rename(columns={"date": "target_date"})


def base_trigger_metrics(holdout_context: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for model, group in holdout_context.groupby("model", sort=False):
        prediction_high = group["prediction"].ge(threshold)
        stress = group["actual"].ge(threshold)
        backlog = group["backlog_risk_day"].fillna(False).astype(bool)
        critical = stress | backlog
        stress_units = np.maximum(group["actual"].to_numpy(float) - threshold, 0)
        captured_units = float(stress_units[prediction_high.to_numpy(bool)].sum())
        for label, actual in [
            ("Stress day", stress),
            ("Backlog-risk day", backlog),
            ("Critical-risk day", critical),
        ]:
            metrics = classification_metrics(actual, prediction_high)
            rows.append(
                {
                    "model": model,
                    "outcome": label,
                    "forecast_high_action_threshold": threshold,
                    "holdout_days": len(group),
                    "alert_days": int(prediction_high.sum()),
                    "alert_burden": float(prediction_high.mean()),
                    "actual_positive_days": int(actual.sum()),
                    "stress_unit_capture_rate": captured_units / stress_units.sum()
                    if label == "Stress day" and stress_units.sum() > 0
                    else np.nan,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def scenario_costs(holdout_context: pd.DataFrame, cfg: dict, threshold: float) -> pd.DataFrame:
    rows = []
    for model, group in holdout_context.groupby("model", sort=False):
        alert = group["prediction"].ge(threshold)
        stress = group["actual"].ge(threshold)
        backlog = group["backlog_risk_day"].fillna(False).astype(bool)
        critical = stress | backlog
        alert_days = int(alert.sum())
        missed_stress = int((stress & ~alert).sum())
        missed_backlog = int((backlog & ~alert).sum())
        missed_critical = int((critical & ~alert).sum())
        for scenario in cfg["cost_scenarios"]:
            lambda_stress = float(scenario["missed_stress_penalty"])
            lambda_backlog = float(scenario["missed_backlog_penalty"])
            combined_cost = alert_days + lambda_stress * missed_stress + lambda_backlog * missed_backlog
            stress_only_cost = alert_days + lambda_stress * missed_stress
            critical_penalty = max(lambda_stress, lambda_backlog)
            critical_cost = alert_days + critical_penalty * missed_critical
            rows.extend(
                [
                    {
                        "scenario": scenario["name"],
                        "cost_variant": "Combined stress and backlog",
                        "model": model,
                        "alert_threshold": threshold,
                        "alert_days": alert_days,
                        "missed_stress_days": missed_stress,
                        "missed_backlog_days": missed_backlog,
                        "missed_critical_days": missed_critical,
                        "missed_stress_penalty": lambda_stress,
                        "missed_backlog_penalty": lambda_backlog,
                        "cost_units": combined_cost,
                        "normalized_cost": combined_cost / len(group),
                    },
                    {
                        "scenario": scenario["name"],
                        "cost_variant": "Stress only",
                        "model": model,
                        "alert_threshold": threshold,
                        "alert_days": alert_days,
                        "missed_stress_days": missed_stress,
                        "missed_backlog_days": missed_backlog,
                        "missed_critical_days": missed_critical,
                        "missed_stress_penalty": lambda_stress,
                        "missed_backlog_penalty": 0.0,
                        "cost_units": stress_only_cost,
                        "normalized_cost": stress_only_cost / len(group),
                    },
                    {
                        "scenario": scenario["name"],
                        "cost_variant": "Critical risk",
                        "model": model,
                        "alert_threshold": threshold,
                        "alert_days": alert_days,
                        "missed_stress_days": missed_stress,
                        "missed_backlog_days": missed_backlog,
                        "missed_critical_days": missed_critical,
                        "missed_stress_penalty": critical_penalty,
                        "missed_backlog_penalty": critical_penalty,
                        "cost_units": critical_cost,
                        "normalized_cost": critical_cost / len(group),
                    },
                ]
            )
    return pd.DataFrame(rows)


def break_even_penalties(holdout_context: pd.DataFrame, threshold: float) -> pd.DataFrame:
    baseline_name = "14-day moving average"
    rows = []
    summary = {}
    for model, group in holdout_context.groupby("model", sort=False):
        alert = group["prediction"].ge(threshold)
        stress = group["actual"].ge(threshold)
        backlog = group["backlog_risk_day"].fillna(False).astype(bool)
        critical = stress | backlog
        summary[model] = {
            "alerts": int(alert.sum()),
            "missed_stress": int((stress & ~alert).sum()),
            "missed_critical": int((critical & ~alert).sum()),
        }
    base = summary[baseline_name]
    for model, values in summary.items():
        if model == baseline_name:
            continue
        for outcome, missed_key in [
            ("Stress day", "missed_stress"),
            ("Critical-risk day", "missed_critical"),
        ]:
            numerator = values["alerts"] - base["alerts"]
            denominator = base[missed_key] - values[missed_key]
            break_even = numerator / denominator if denominator > 0 else np.nan
            rows.append(
                {
                    "comparison_model": model,
                    "baseline_model": baseline_name,
                    "outcome": outcome,
                    "additional_alert_days": numerator,
                    "missed_days_avoided": denominator,
                    "break_even_missed_day_penalty_per_alert": break_even,
                }
            )
    return pd.DataFrame(rows)


def condition_metrics(holdout_context: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = holdout_context.copy()
    frame["H1_2025"] = frame["target_date"].dt.month.le(6)
    conditions = {
        "Stress day (>=1.20)": frame["actual"].ge(threshold),
        "Non-stress day": frame["actual"].lt(threshold),
        "Slack day (<0.80)": frame["actual"].lt(0.80),
        "Backlog-risk day": frame["backlog_risk_day"].fillna(False),
        "No backlog-risk day": ~frame["backlog_risk_day"].fillna(False),
        "QA-flagged day": frame["any_qa_flag"].fillna(False),
        "QA-clean day": ~frame["any_qa_flag"].fillna(False),
        "H1 2025": frame["H1_2025"],
        "H2 2025": ~frame["H1_2025"],
    }
    rows = []
    for condition, all_mask in conditions.items():
        for model, group in frame.groupby("model", sort=False):
            mask = all_mask.loc[group.index].to_numpy(bool)
            subset = group.loc[mask]
            if subset.empty:
                continue
            error = subset["prediction"].to_numpy(float) - subset["actual"].to_numpy(float)
            rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "n_days": len(subset),
                    "MAE": float(np.mean(np.abs(error))),
                    "RMSE": float(np.sqrt(np.mean(error**2))),
                    "bias_prediction_minus_actual": float(np.mean(error)),
                    "underprediction_rate": float(np.mean(error < 0)),
                    "overprediction_rate": float(np.mean(error > 0)),
                }
            )
    return pd.DataFrame(rows)


def decision_metrics(holdout_context: pd.DataFrame, cfg: dict, threshold: float) -> pd.DataFrame:
    underprediction_weights = tuple(cfg["asymmetric_metrics"]["underprediction_weights"])
    stress_day_weight = float(cfg["asymmetric_metrics"]["stress_day_weight"])
    pinball_quantiles = tuple(cfg["asymmetric_metrics"]["pinball_quantiles"])
    rows = []
    for model, group in holdout_context.groupby("model", sort=False):
        metrics = decision_metric_dict(
            group["actual"],
            group["prediction"],
            stress_threshold=threshold,
            underprediction_weights=underprediction_weights,
            stress_day_weight=stress_day_weight,
            pinball_quantiles=pinball_quantiles,
        )
        error = np.abs(group["prediction"].to_numpy(float) - group["actual"].to_numpy(float))
        backlog_weights = np.where(group["backlog_risk_day"].fillna(False), 3.0, 1.0)
        stress_mask = group["actual"].ge(threshold).to_numpy(bool)
        under_mask = stress_mask & group["prediction"].lt(group["actual"]).to_numpy(bool)
        rows.append(
            {
                "model": model,
                "holdout_n": len(group),
                "stress_day_MAE": float(error[stress_mask].mean()) if stress_mask.any() else np.nan,
                "stress_underprediction_MAE": float(error[under_mask].mean()) if under_mask.any() else np.nan,
                "backlog_weighted_MAE_w3": float(np.average(error, weights=backlog_weights)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def precision_recall_coordinates(holdout_context: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for model, group in holdout_context.groupby("model", sort=False):
        labels = group["actual"].ge(threshold).astype(int).to_numpy()
        scores = group["prediction"].to_numpy(float)
        precision, recall, cutoffs = precision_recall_curve(labels, scores)
        average_precision = average_precision_score(labels, scores)
        for index in range(len(precision)):
            rows.append(
                {
                    "model": model,
                    "precision": precision[index],
                    "recall": recall[index],
                    "threshold": cutoffs[index] if index < len(cutoffs) else np.nan,
                    "average_precision": average_precision,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    _, _, _, master = load_inputs()
    context = context_frame(master)
    oof = pd.read_csv(PREDICTIONS / "oof_train_predictions.csv", parse_dates=["target_date"])
    holdout = pd.read_csv(PREDICTIONS / "holdout_predictions.csv", parse_dates=["target_date"])
    holdout_context = holdout.merge(context, on="target_date", how="left")
    holdout_context.to_csv(PREDICTIONS / "holdout_predictions_with_context.csv", index=False)

    grid, selected = calibrate_thresholds(oof, cfg)
    grid.to_csv(METADATA / "threshold_calibration_grid.csv", index=False)
    selected.to_csv(TABLES / "threshold_calibration_selected.csv", index=False)
    threshold = float(cfg["threshold_calibration"]["actual_stress_threshold"])
    base_trigger_metrics(holdout_context, threshold).to_csv(
        TABLES / "table16_base_threshold_trigger_metrics.csv", index=False
    )
    costs = scenario_costs(holdout_context, cfg, threshold)
    costs.to_csv(TABLES / "table17_counterfactual_cost_scenarios.csv", index=False)
    break_even_penalties(holdout_context, threshold).to_csv(
        TABLES / "table18_break_even_penalties.csv", index=False
    )
    condition_metrics(holdout_context, threshold).to_csv(
        TABLES / "conditional_error_diagnostics.csv", index=False
    )
    decision_metrics(holdout_context, cfg, threshold).to_csv(
        TABLES / "decision_consistent_metrics.csv", index=False
    )
    precision_recall_coordinates(holdout_context, threshold).to_csv(
        METADATA / "precision_recall_curves.csv", index=False
    )
    print(costs.loc[costs["cost_variant"].eq("Combined stress and backlog")].to_string(index=False))


if __name__ == "__main__":
    main()
