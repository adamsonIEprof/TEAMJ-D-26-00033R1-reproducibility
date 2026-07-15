from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from repro_common import load_inputs


FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
PREDICTIONS = ROOT / "outputs" / "predictions"
METADATA = ROOT / "outputs" / "metadata"


def save_figure(fig, filename: str) -> None:
    fig.savefig(FIGURES / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def synthetic_title(title: str) -> str:
    return f"{title}\nSynthetic demonstration data"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "figure.dpi": 140,
        }
    )
    _, _, _, master = load_inputs()
    master["year"] = master["date"].dt.year
    master["month_period"] = master["date"].dt.to_period("M").astype(str)

    # Figure 1: utilization time series with synthetic QA flags.
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(master["date"], master["target_utilization_intensity"], linewidth=0.8, alpha=0.55)
    rolling = master.set_index("date")["target_utilization_intensity"].rolling(14, min_periods=7).mean()
    ax.plot(rolling.index, rolling, linewidth=1.8, label="14-day moving mean")
    flagged = master.loc[master["any_qa_flag"].eq(1)]
    ax.scatter(flagged["date"], flagged["target_utilization_intensity"], s=8, alpha=0.35, label="Synthetic QA flag")
    ax.axvline(pd.Timestamp("2025-01-01"), color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Realized utilization")
    ax.set_title(synthetic_title("Realized served-truck utilization and data-quality flags"))
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "figure01_utilization_timeseries")

    # Figure 2: weekday utilization profile.
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = master.assign(day=master["date"].dt.day_name()).groupby("day")["target_utilization_intensity"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(weekday.index, weekday.values)
    ax.set_ylabel("Mean realized utilization")
    ax.set_title(synthetic_title("Mean utilization by day of week"))
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    save_figure(fig, "figure02_utilization_by_weekday")

    # Figure 3: scheduled and served.
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.plot(master["date"], master["total_scheduled_deliveries"].rolling(7, min_periods=1).mean(), label="Scheduled")
    ax.plot(master["date"], master["total_served"].rolling(7, min_periods=1).mean(), label="Served")
    ax.set_ylabel("Seven-day mean deliveries")
    ax.set_title(synthetic_title("Scheduled and served deliveries"))
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "figure03_scheduled_served")

    # Figure 4: trucks and drivers.
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.plot(master["date"], master["available_trucks_total"].rolling(7, min_periods=1).mean(), label="Available trucks")
    ax.plot(master["date"], master["available_drivers"].rolling(7, min_periods=1).mean(), label="Available drivers")
    ax.set_ylabel("Seven-day mean count")
    ax.set_title(synthetic_title("Fleet capacity and driver readiness"))
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "figure04_trucks_drivers")

    # Figure 5: deferred and carry-over.
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.plot(master["date"], master["deferred_rescheduled"].rolling(7, min_periods=1).mean(), label="Deferred/rescheduled")
    ax.plot(master["date"], master["carry_over"].rolling(7, min_periods=1).mean(), label="Carry-over")
    ax.set_ylabel("Seven-day mean orders")
    ax.set_title(synthetic_title("Deferred and carry-over workload"))
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "figure05_deferred_carryover")

    # Figure 6: indexed monthly profile.
    monthly = master.groupby("month_period")[[
        "open_orders", "total_scheduled_deliveries", "total_served", "available_trucks_total"
    ]].mean()
    indexed = 100 * monthly / monthly.iloc[:12].mean()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for column in indexed.columns:
        ax.plot(indexed.index, indexed[column], marker="o", markersize=3, label=column)
    ax.axhline(100, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Index, synthetic 2024 mean = 100")
    ax.set_title(synthetic_title("Monthly operational profile"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "figure06_monthly_operational_profile")

    # Figure 7: decomposition.
    decomposition = pd.read_csv(TABLES / "table08_log_ratio_decomposition.csv")
    positions = np.arange(len(decomposition))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(positions - width / 2, decomposition["served_volume_component"], width, label="Served-volume component")
    ax.bar(positions + width / 2, decomposition["capacity_denominator_component"], width, label="Capacity-denominator component")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions, decomposition["comparison"])
    ax.set_ylabel("Log-point contribution")
    ax.set_title(synthetic_title("Log-ratio decomposition of utilization change"))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    save_figure(fig, "figure07_log_ratio_decomposition")

    # Figure 8: backlog attribution by regime.
    counts = pd.read_csv(TABLES / "table09_backlog_attribution_counts.csv")
    pivot_counts = counts.pivot(index="year", columns="backlog_attribution", values="days").fillna(0)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    pivot_counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Synthetic days")
    ax.set_title(synthetic_title("Backlog attribution by operating year"))
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, "figure08_backlog_attribution")

    # Figure 9: monthly target constructs.
    targets = pd.read_csv(PREDICTIONS / "synthetic_daily_targets.csv", parse_dates=["date"])
    targets["month"] = targets["date"].dt.to_period("M").astype(str)
    target_columns = [c for c in targets.columns if c not in {"date", "year", "month"}]
    monthly_targets = targets.groupby("month")[target_columns].mean()
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for column in monthly_targets.columns:
        ax.plot(monthly_targets.index, monthly_targets[column], linewidth=1.4, label=column)
    ax.set_ylabel("Mean target value")
    ax.set_title(synthetic_title("Monthly target constructs"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "figure09_monthly_targets")

    # Figure 10: holdout trajectories.
    holdout = pd.read_csv(PREDICTIONS / "holdout_predictions.csv", parse_dates=["target_date"])
    pivot = holdout.pivot_table(index="target_date", columns="model", values="prediction", aggfunc="first").sort_index()
    actual = holdout[["target_date", "actual"]].drop_duplicates().set_index("target_date").sort_index()
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.plot(actual.index, actual["actual"], linewidth=0.8, alpha=0.50, label="Actual")
    for model in pivot.columns:
        ax.plot(pivot.index, pivot[model].rolling(7, min_periods=1).mean(), linewidth=1.3, label=model)
    ax.set_ylabel("Primary target")
    ax.set_title(synthetic_title("Actual and predicted holdout trajectories"))
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "figure10_holdout_trajectories")

    # Figure 11: RF importance.
    importance = (
        pd.read_csv(TABLES / "table15_random_forest_importance.csv")
        .head(12)
        .sort_values("holdout_permutation_mae_increase_mean")
    )
    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    ax.barh(importance["feature"], importance["holdout_permutation_mae_increase_mean"])
    ax.set_xlabel("Increase in holdout MAE after permutation")
    ax.set_title(synthetic_title("Post-hoc S2 random-forest permutation importance"))
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    fig.tight_layout()
    save_figure(fig, "figure11_random_forest_importance")

    # Figure 12: precision-recall curves.
    pr = pd.read_csv(METADATA / "precision_recall_curves.csv")
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for model, group in pr.groupby("model", sort=False):
        ap = group["average_precision"].iloc[0]
        ax.plot(group["recall"], group["precision"], linewidth=1.5, label=f"{model} (AP={ap:.3f})")
    prevalence = float((actual["actual"] >= 1.20).mean())
    ax.axhline(prevalence, color="black", linestyle="--", linewidth=0.8, label=f"Prevalence={prevalence:.3f}")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(synthetic_title("Stress-day precision-recall curves"))
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, "figure12_stress_precision_recall")

    # Figure 13: critical-risk cost lines.
    costs = pd.read_csv(TABLES / "table17_counterfactual_cost_scenarios.csv")
    cost_plot = costs.loc[costs["cost_variant"].eq("Critical risk")].copy()
    scenario_order = list(cost_plot["scenario"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for model, group in cost_plot.groupby("model", sort=False):
        group = group.set_index("scenario").reindex(scenario_order)
        ax.plot(scenario_order, group["normalized_cost"], marker="o", linewidth=1.5, label=model)
    ax.set_ylabel("Normalized cost per holdout day")
    ax.set_title(synthetic_title("Critical-risk cost scenarios"))
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "figure13_critical_risk_cost")

    # Figure 14: H1/H2 bias shift.
    regime = pd.read_csv(TABLES / "table21_holdout_performance_by_regime.csv")
    bias_pivot = regime.pivot(
        index="model", columns="regime", values="bias_prediction_minus_actual"
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(bias_pivot))
    ax.bar(x - 0.18, bias_pivot.get("2025 H1 holdout"), 0.36, label="H1 2025")
    ax.bar(x + 0.18, bias_pivot.get("2025 H2 holdout"), 0.36, label="H2 2025")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, bias_pivot.index, rotation=25, ha="right")
    ax.set_ylabel("Mean bias, prediction minus actual")
    ax.set_title(synthetic_title("Prediction-bias shift by holdout regime"))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    save_figure(fig, "figure14_h1_h2_bias")

    # Figure 15: fold MAE stability.
    folds = pd.read_csv(METADATA / "fold_level_model_results.csv")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for model, group in folds.groupby("model", sort=False):
        group = group.sort_values("fold")
        ax.plot(group["fold"], group["MAE"], marker="o", linewidth=1.5, label=model)
    ax.set_xticks(sorted(folds["fold"].unique()))
    ax.set_xlabel("Expanding-window fold")
    ax.set_ylabel("Validation MAE")
    ax.set_title(synthetic_title("Fold-level MAE stability"))
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "figure15_fold_mae_stability")
    print("Created 15 PNG and 15 PDF figures in outputs/figures")


if __name__ == "__main__":
    main()
