from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from repro_common import backlog_context, load_inputs


TABLES = ROOT / "outputs" / "tables"
METADATA = ROOT / "outputs" / "metadata"
PREDICTIONS = ROOT / "outputs" / "predictions"


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def target_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": "Realized served-truck utilization",
                "formula": "total_served / available_trucks_total",
                "role": "Primary target",
                "interpretation": "Constrained realized utilization, not unconstrained demand",
            },
            {
                "target": "Scheduled demand pressure",
                "formula": "total_scheduled_deliveries / available_trucks_total",
                "role": "Alternative target",
                "interpretation": "Scheduled workload relative to available fleet",
            },
            {
                "target": "Open-order pressure",
                "formula": "open_orders / available_trucks_total",
                "role": "Alternative target",
                "interpretation": "Demand pipeline relative to available fleet",
            },
            {
                "target": "Served plus deferred pressure",
                "formula": "(total_served + deferred_rescheduled) / available_trucks_total",
                "role": "Alternative target",
                "interpretation": "Completed and explicitly deferred workload",
            },
            {
                "target": "Expanded demand-pressure index",
                "formula": "(max(total_served,total_scheduled_deliveries) + backlog_open_minus_scheduled + backload_order_count) / available_trucks_total",
                "role": "Alternative target",
                "interpretation": "Synthetic operational pressure index; components may overlap",
            },
            {
                "target": "Secondary trip-capacity utilization",
                "formula": "total_served / committed_trips_total_clean",
                "role": "Denominator robustness target",
                "interpretation": "Served deliveries relative to cleaned committed-trip capacity",
            },
        ]
    )


def feature_timing(catalog: pd.DataFrame) -> pd.DataFrame:
    out = (
        catalog.groupby("block", as_index=False)
        .agg(
            engineered_features=("feature_name", "size"),
            main_features=("retained_main", "sum"),
            sensitivity_eligible=("retained_sensitivity", "sum"),
            all_known_before_target=(
                "known_by_target_date_start",
                lambda values: bool(pd.Series(values).astype(str).str.lower().eq("yes").all()),
            ),
        )
    )
    out["timing_rule"] = np.where(
        out["block"].eq("Calendar and seasonality"),
        "Target-date calendar fields known in advance",
        "Lagged, shifted, rolling, previous-day, or carried-forward history only",
    )
    return out


def backlog_attribution(master: pd.DataFrame) -> pd.DataFrame:
    m = backlog_context(master)
    m["drivers_per_truck"] = safe_ratio(m["available_drivers"], m["available_trucks_total"])
    m["scheduled_pressure"] = safe_ratio(
        m["total_scheduled_deliveries"], m["available_trucks_total"]
    )
    m["open_order_pressure"] = safe_ratio(m["open_orders"], m["available_trucks_total"])
    valid = m.loc[m["structural_nonoperational_day"].eq(0)].copy()
    demand_thresholds = {
        "open_orders": float(valid["open_orders"].quantile(0.75)),
        "total_scheduled_deliveries": float(valid["total_scheduled_deliveries"].quantile(0.75)),
        "scheduled_pressure": float(valid["scheduled_pressure"].quantile(0.75)),
        "open_order_pressure": float(valid["open_order_pressure"].quantile(0.75)),
    }
    capacity_thresholds = {
        "available_trucks_total": float(valid["available_trucks_total"].quantile(0.25)),
        "available_drivers": float(valid["available_drivers"].quantile(0.25)),
        "drivers_per_truck": float(valid["drivers_per_truck"].quantile(0.25)),
    }
    m["high_demand_condition"] = (
        m["open_orders"].ge(demand_thresholds["open_orders"])
        | m["total_scheduled_deliveries"].ge(demand_thresholds["total_scheduled_deliveries"])
        | m["scheduled_pressure"].ge(demand_thresholds["scheduled_pressure"])
        | m["open_order_pressure"].ge(demand_thresholds["open_order_pressure"])
    )
    m["low_capacity_readiness_condition"] = (
        m["available_trucks_total"].le(capacity_thresholds["available_trucks_total"])
        | m["available_drivers"].le(capacity_thresholds["available_drivers"])
        | m["drivers_per_truck"].le(capacity_thresholds["drivers_per_truck"])
    )
    m["backlog_attribution"] = np.select(
        [
            m["backlog_risk_day"] & m["high_demand_condition"] & m["low_capacity_readiness_condition"],
            m["backlog_risk_day"] & m["high_demand_condition"],
            m["backlog_risk_day"] & m["low_capacity_readiness_condition"],
            m["backlog_risk_day"] & m["backload_order_count"].fillna(0).gt(0),
        ],
        [
            "Mixed demand and capacity/readiness",
            "Demand-side dominant",
            "Capacity/readiness dominant",
            "Backload-specific administrative/unspecified",
        ],
        default="Normal-range or unattributed",
    )
    thresholds = pd.DataFrame(
        [
            {"condition": "High demand", "variable": key, "synthetic_threshold": value, "quantile": 0.75}
            for key, value in demand_thresholds.items()
        ]
        + [
            {"condition": "Low capacity/readiness", "variable": key, "synthetic_threshold": value, "quantile": 0.25}
            for key, value in capacity_thresholds.items()
        ]
    )
    thresholds.to_csv(METADATA / "synthetic_backlog_attribution_thresholds.csv", index=False)
    return m


def annual_profile(master: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "open_orders",
        "total_scheduled_deliveries",
        "total_served",
        "available_trucks_total",
        "available_drivers",
        "carry_over",
        "deferred_rescheduled",
        "backload_order_count",
        "target_utilization_intensity",
    ]
    rows = []
    for variable in variables:
        means = master.groupby(master["date"].dt.year)[variable].mean()
        value_2024 = float(means.get(2024, np.nan))
        value_2025 = float(means.get(2025, np.nan))
        rows.append(
            {
                "variable": variable,
                "mean_2024": value_2024,
                "mean_2025": value_2025,
                "percent_change_2025_vs_2024": 100 * (value_2025 / value_2024 - 1)
                if value_2024
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def log_ratio_decomposition(master: pd.DataFrame) -> pd.DataFrame:
    m = master.loc[
        master["total_served"].gt(0) & master["available_trucks_total"].gt(0)
    ].copy()
    periods = {
        "2024": m["date"].dt.year.eq(2024),
        "2025": m["date"].dt.year.eq(2025),
        "2025-H1": m["date"].between("2025-01-01", "2025-06-30"),
        "2025-H2": m["date"].between("2025-07-01", "2025-12-31"),
    }
    base = m.loc[periods["2024"]]
    base_log_served = np.log(base["total_served"]).mean()
    base_log_trucks = np.log(base["available_trucks_total"]).mean()
    rows = []
    for name in ["2025", "2025-H1", "2025-H2"]:
        group = m.loc[periods[name]]
        served_component = np.log(group["total_served"]).mean() - base_log_served
        capacity_component = -(
            np.log(group["available_trucks_total"]).mean() - base_log_trucks
        )
        rows.append(
            {
                "comparison": f"{name} versus 2024",
                "served_volume_component": served_component,
                "capacity_denominator_component": capacity_component,
                "net_log_utilization_change": served_component + capacity_component,
                "approximate_percent_change": 100 * (np.exp(served_component + capacity_component) - 1),
            }
        )
    return pd.DataFrame(rows)


def raw_descriptives(master: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "open_orders",
        "total_scheduled_deliveries",
        "total_served",
        "available_trucks_total",
        "available_drivers",
        "carry_over",
        "deferred_rescheduled",
        "backload_order_count",
        "target_utilization_intensity",
    ]
    rows = []
    for variable in variables:
        values = master[variable]
        rows.append(
            {
                "variable": variable,
                "valid_n": int(values.notna().sum()),
                "missing_n": int(values.isna().sum()),
                "qa_flagged_n": int((master["any_qa_flag"].eq(1) & values.notna()).sum()),
                "mean": values.mean(),
                "sd": values.std(),
                "min": values.min(),
                "q1": values.quantile(0.25),
                "median": values.median(),
                "q3": values.quantile(0.75),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    panel, catalog, _, master = load_inputs()
    master = backlog_context(master)
    master["backload_order_count"] = master.get("backload_order_count", master["backload_records"])

    target_definitions().to_csv(TABLES / "table01_target_definitions.csv", index=False)
    feature_timing(catalog).to_csv(TABLES / "table02_feature_timing.csv", index=False)
    pd.DataFrame(
        [
            {"signal": "Carry-over", "rule": "carry_over > 0"},
            {"signal": "Deferred/rescheduled", "rule": "deferred_rescheduled > 0"},
            {"signal": "Unmet scheduled", "rule": "max(total_scheduled_deliveries - total_served, 0) > 0"},
            {"signal": "Unconverted open orders", "rule": "max(open_orders - scheduled_from_open_orders, 0) > 0"},
            {"signal": "Backload record", "rule": "backload_order_count > 0"},
        ]
    ).to_csv(TABLES / "table04_backlog_signal_definitions.csv", index=False)
    pd.DataFrame(
        [
            {"attribution": "Demand-side dominant", "rule": "Backlog signal and high-demand condition only"},
            {"attribution": "Capacity/readiness dominant", "rule": "Backlog signal and low-capacity/readiness condition only"},
            {"attribution": "Mixed demand and capacity/readiness", "rule": "Both high-demand and low-capacity/readiness conditions"},
            {"attribution": "Backload-specific administrative/unspecified", "rule": "Backload signal without demand or capacity threshold"},
            {"attribution": "Normal-range or unattributed", "rule": "No preceding attribution rule"},
        ]
    ).to_csv(TABLES / "table05_backlog_attribution_rules.csv", index=False)

    annual_profile(master).to_csv(TABLES / "table06_annual_operational_profile.csv", index=False)
    diagnostics = pd.DataFrame(
        [
            {"diagnostic": "Calendar rows", "value": len(master)},
            {"diagnostic": "Valid primary target rows", "value": master["target_utilization_intensity"].notna().sum()},
            {"diagnostic": "Synthetic QA-flagged rows", "value": master["any_qa_flag"].sum()},
            {"diagnostic": "Primary target ACF(1)", "value": master["target_utilization_intensity"].autocorr(1)},
            {"diagnostic": "Primary target ACF(7)", "value": master["target_utilization_intensity"].autocorr(7)},
            {"diagnostic": "Modeling rows", "value": len(panel)},
        ]
    )
    diagnostics.to_csv(TABLES / "table07_panel_diagnostics.csv", index=False)
    log_ratio_decomposition(master).to_csv(TABLES / "table08_log_ratio_decomposition.csv", index=False)

    attributed = backlog_attribution(master)
    attributed.to_csv(PREDICTIONS / "synthetic_daily_backlog_attribution.csv", index=False)
    counts = (
        attributed.assign(year=attributed["date"].dt.year)
        .groupby(["year", "backlog_attribution"], as_index=False)
        .size()
        .rename(columns={"size": "days"})
    )
    counts.to_csv(TABLES / "table09_backlog_attribution_counts.csv", index=False)
    profile = (
        attributed.groupby("backlog_attribution", as_index=False)
        .agg(
            days=("date", "size"),
            mean_open_orders=("open_orders", "mean"),
            mean_scheduled=("total_scheduled_deliveries", "mean"),
            mean_served=("total_served", "mean"),
            mean_trucks=("available_trucks_total", "mean"),
            mean_drivers=("available_drivers", "mean"),
            mean_utilization=("target_utilization_intensity", "mean"),
        )
    )
    profile.to_csv(TABLES / "table10_backlog_attribution_profile.csv", index=False)

    target_frame = pd.DataFrame(
        {
            "date": master["date"],
            "year": master["date"].dt.year,
            "realized_utilization": master["target_utilization_intensity"],
            "scheduled_pressure": safe_ratio(master["total_scheduled_deliveries"], master["available_trucks_total"]),
            "open_order_pressure": safe_ratio(master["open_orders"], master["available_trucks_total"]),
            "served_plus_deferred_pressure": safe_ratio(
                master["total_served"] + master["deferred_rescheduled"], master["available_trucks_total"]
            ),
            "expanded_pressure_index": safe_ratio(
                np.maximum(master["total_served"], master["total_scheduled_deliveries"])
                + master["unconverted_open_orders"]
                + master["backload_order_count"],
                master["available_trucks_total"],
            ),
        }
    )
    target_frame.to_csv(PREDICTIONS / "synthetic_daily_targets.csv", index=False)
    target_long = target_frame.melt(id_vars=["date", "year"], var_name="target", value_name="value")
    target_summary = target_long.groupby(["year", "target"], as_index=False).agg(
        valid_n=("value", "count"), mean=("value", "mean"), sd=("value", "std"),
        stress_days=("value", lambda x: int((x >= 1.20).sum()))
    )
    target_summary.to_csv(TABLES / "table11_target_year_comparison.csv", index=False)
    raw_descriptives(master).to_csv(TABLES / "supplementary_table_s1_raw_descriptives.csv", index=False)
    print("Created descriptive, target, and backlog outputs in outputs/tables")


if __name__ == "__main__":
    main()
