from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SEED = 42
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
METADATA_DIR = ROOT / "data" / "metadata"


def _select_evenly(dates: pd.DatetimeIndex, n: int) -> list[pd.Timestamp]:
    """Select n ordered dates, always including the first and last."""
    if n > len(dates):
        raise ValueError(f"Cannot select {n} dates from {len(dates)} candidates.")
    positions = np.linspace(0, len(dates) - 1, n)
    selected = np.unique(np.round(positions).astype(int))
    if len(selected) != n:
        remaining = [i for i in range(len(dates)) if i not in set(selected)]
        selected = np.sort(np.r_[selected, remaining[: n - len(selected)]])
    return list(dates[selected])


def synthetic_eligible_dates() -> list[pd.Timestamp]:
    """Create the configured 266/271 synthetic demonstration split."""
    train_blocks = [
        ("2024-02-06", "2024-03-22", 46),
        ("2024-03-23", "2024-06-03", 44),
        ("2024-06-04", "2024-08-14", 44),
        ("2024-08-15", "2024-09-27", 44),
        ("2024-09-28", "2024-11-10", 44),
        ("2024-11-11", "2024-12-24", 44),
    ]
    train_dates: list[pd.Timestamp] = []
    for start, end, count in train_blocks:
        train_dates.extend(_select_evenly(pd.date_range(start, end, freq="D"), count))
    test_dates = _select_evenly(pd.date_range("2025-01-01", "2025-12-24", freq="D"), 271)
    selected = train_dates + test_dates
    if len(selected) != 537 or len(set(selected)) != 537:
        raise AssertionError("Synthetic eligibility construction did not produce 537 unique dates.")
    return selected


def simulate_daily_master(seed: int = SEED) -> pd.DataFrame:
    """Simulate a daily operational panel with no company-derived values."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    n = len(dates)
    day_num = dates.dayofweek.to_numpy()
    year = dates.year.to_numpy()
    regime = np.where(year == 2025, 0.84, 1.00)
    weekly = np.array([1.08, 1.12, 1.10, 1.04, 1.03, 0.89, 0.72])[day_num]
    annual = 1.0 + 0.08 * np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 30) / 365.25)

    latent = np.zeros(n, dtype=float)
    shocks = rng.normal(0, 0.33, n)
    for i in range(1, n):
        latent[i] = 0.62 * latent[i - 1] + shocks[i]

    demand_mean = np.clip(39.0 * regime * weekly * annual * np.exp(0.17 * latent), 8, 80)
    open_orders = rng.poisson(demand_mean).astype(float)
    unscheduled = rng.poisson(np.clip(1.8 + 0.8 * np.maximum(latent, 0), 0.2, None)).astype(float)
    conversion_rate = np.where(
        rng.random(n) < 0.65,
        1.0,
        rng.uniform(0.82, 0.98, n),
    )
    scheduled_from_open = np.maximum(0, np.rint(open_orders * conversion_rate)).astype(float)
    accommodated_after_cutoff = rng.poisson(
        np.clip(1.2 + 0.5 * np.maximum(latent, 0), 0.1, None)
    ).astype(float)
    carry_over = rng.poisson(np.clip(0.5 + 1.3 * np.maximum(latent - 0.15, 0), 0.05, None)).astype(float)
    total_scheduled = scheduled_from_open + accommodated_after_cutoff + carry_over + unscheduled

    available_trucks_total = np.clip(
        np.rint(37.0 + np.where(year == 2025, -1.5, 0.0) + rng.normal(0, 3.2, n)),
        25,
        48,
    ).astype(float)
    available_drivers = np.clip(
        np.rint(available_trucks_total + rng.normal(2.2, 3.0, n)), 22, 55
    ).astype(float)

    pressure = total_scheduled / np.maximum(available_trucks_total, 1)
    deferred = rng.poisson(np.clip((pressure - 0.95) * 4.0, 0.05, 8.0)).astype(float)
    cancelled = rng.binomial(np.maximum(total_scheduled.astype(int), 0), 0.012).astype(float)
    best_effort = rng.poisson(np.clip(0.8 + 0.4 * np.maximum(latent, 0), 0.1, None)).astype(float)
    execution_noise = rng.normal(0, 2.5, n)
    total_served = np.maximum(
        0,
        np.rint(total_scheduled - deferred - cancelled + best_effort + execution_noise),
    ).astype(float)
    retail_share = np.clip(rng.normal(0.70, 0.07, n), 0.45, 0.90)
    retail_served = np.rint(total_served * retail_share).astype(float)
    b2b_served = (total_served - retail_served).astype(float)

    # A multinomial allocation preserves the arithmetic identity between the
    # four synthetic truck classes and the reported daily total.
    truck_allocations = np.vstack(
        [
            rng.multinomial(int(total), [0.18, 0.05, 0.08, 0.69])
            for total in available_trucks_total
        ]
    ).astype(float)
    truck30 = truck_allocations[:, 0]
    truck18 = truck_allocations[:, 1]
    truck36 = truck_allocations[:, 2]
    truck20 = truck_allocations[:, 3]

    committed_retail = np.full(n, 35.0)
    committed_b2b = np.full(n, 15.0)
    committed_total = committed_retail + committed_b2b

    fuel_update_day = (day_num == 1).astype(int)
    weekly_diesel = np.zeros(n)
    weekly_gasoline = np.zeros(n)
    weekly_kerosene = np.zeros(n)
    current = np.zeros(3)
    for i in range(n):
        if fuel_update_day[i]:
            current = rng.normal([0.02, 0.03, 0.01], [0.70, 0.65, 0.60])
        weekly_diesel[i], weekly_gasoline[i], weekly_kerosene[i] = current

    excess = np.maximum(total_scheduled - total_served, 0)
    backload_records = rng.poisson(np.clip(0.25 + 0.15 * excess, 0.05, 4.0)).astype(float)
    backload_total_kl = np.where(
        backload_records > 0,
        np.maximum(0.5, backload_records * rng.uniform(6.0, 22.0, n)),
        np.nan,
    )

    df = pd.DataFrame(
        {
            "date": dates,
            "open_orders": open_orders,
            "unscheduled": unscheduled,
            "scheduled_from_open_orders": scheduled_from_open,
            "accommodated_after_cutoff": accommodated_after_cutoff,
            "carry_over": carry_over,
            "total_scheduled_deliveries": total_scheduled,
            "deferred_rescheduled": deferred,
            "cancelled": cancelled,
            "best_effort": best_effort,
            "changed_mot_to_pk": 0.0,
            "total_served": total_served,
            "available_trucks_30kl": truck30,
            "available_trucks_18kl": truck18,
            "available_trucks_36kl": truck36,
            "available_trucks_20kl": truck20,
            "available_drivers": available_drivers,
            "committed_trips_total": committed_total,
            "retail_served_orders": retail_served,
            "retail_committed_trips": committed_retail,
            "b2b_served_orders": b2b_served,
            "b2b_committed_trips": committed_b2b,
            "source_sheet": "SYNTHETIC_DAILY_PANEL",
            "source_year_x": year,
            "observed_in_source": 1,
            "inserted_calendar_row": 0,
            "year": year,
            "month": dates.month,
            "month_name": dates.month_name(),
            "day_of_week": dates.day_name(),
            "day_of_year": dates.dayofyear,
            "available_trucks_total": available_trucks_total,
            "retail_committed_trips_clean": committed_retail,
            "b2b_committed_trips_clean": committed_b2b,
            "committed_trips_total_clean": committed_total,
            "scheduled_total_from_parts": total_scheduled,
            "served_total_from_segments": retail_served + b2b_served,
            "all_core_missing": 0,
            "zero_trucks_zero_served": 0,
            "structural_nonoperational_day": 0,
            "qa_scheduled_parts_mismatch": 0,
            "qa_served_segment_mismatch": 0,
            "qa_committed_total_mismatch": 0,
            "qa_zero_trucks_positive_served": 0,
            "qa_negative_numeric": 0,
            "backload_records": backload_records,
            "backload_total_kl": backload_total_kl,
            "backload_reason_nonnull_records": backload_records,
            "backload_retail_records": np.rint(backload_records * 0.7),
            "backload_b2b_records": backload_records - np.rint(backload_records * 0.7),
            "source_year_y": year,
            "weekly_change_diesel": weekly_diesel,
            "weekly_change_gasoline": weekly_gasoline,
            "weekly_change_kerosene": weekly_kerosene,
            "fuel_update_day": fuel_update_day,
        }
    )

    # Match documented row populations with synthetic-only missingness patterns.
    raw_missing_dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    structural_dates = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2025-12-25", "2025-12-26"]
    )
    primary_extra_invalid = pd.Timestamp("2025-12-27")
    df.loc[df["date"].isin(raw_missing_dates), "all_core_missing"] = 1
    df.loc[df["date"].isin(raw_missing_dates), "observed_in_source"] = 0
    df.loc[df["date"].isin(structural_dates), "structural_nonoperational_day"] = 1
    df.loc[df["date"].eq(pd.Timestamp("2025-12-25")), "inserted_calendar_row"] = 1

    operational_cols = [
        "open_orders",
        "unscheduled",
        "scheduled_from_open_orders",
        "accommodated_after_cutoff",
        "carry_over",
        "total_scheduled_deliveries",
        "deferred_rescheduled",
        "cancelled",
        "best_effort",
        "total_served",
        "available_trucks_total",
        "available_trucks_30kl",
        "available_trucks_18kl",
        "available_trucks_36kl",
        "available_trucks_20kl",
        "available_drivers",
    ]
    df.loc[df["date"].isin(raw_missing_dates), operational_cols] = np.nan
    zero_truck_dates = [pd.Timestamp("2025-12-25"), pd.Timestamp("2025-12-26"), primary_extra_invalid]
    zero_truck_columns = [
        "available_trucks_total",
        "available_trucks_30kl",
        "available_trucks_18kl",
        "available_trucks_36kl",
        "available_trucks_20kl",
        "total_served",
    ]
    df.loc[df["date"].isin(zero_truck_dates), zero_truck_columns] = 0.0
    df.loc[df["date"].eq(pd.Timestamp("2025-12-25")), "committed_trips_total_clean"] = np.nan

    # Exactly 118 dates carry at least one synthetic QA flag.
    candidates = df.index[~df["date"].isin(raw_missing_dates)].to_numpy()
    flagged = np.sort(rng.choice(candidates, size=118, replace=False))
    flag_columns = [
        "qa_scheduled_parts_mismatch",
        "qa_served_segment_mismatch",
        "qa_committed_total_mismatch",
        "qa_zero_trucks_positive_served",
        "qa_negative_numeric",
    ]
    assignments = rng.integers(0, len(flag_columns), size=len(flagged))
    for row, choice in zip(flagged, assignments):
        df.loc[row, flag_columns[int(choice)]] = 1

    df["target_utilization_intensity"] = np.where(
        (df["available_trucks_total"] > 0) & df["total_served"].notna(),
        df["total_served"] / df["available_trucks_total"],
        np.nan,
    )
    df["secondary_trip_capacity_utilization_rate"] = np.where(
        df["total_served"].notna() & (df["committed_trips_total_clean"] > 0),
        df["total_served"] / df["committed_trips_total_clean"],
        np.nan,
    )
    df["completion_rate"] = np.where(
        df["total_scheduled_deliveries"] > 0,
        df["total_served"] / df["total_scheduled_deliveries"],
        np.nan,
    )
    df["service_gap"] = np.maximum(
        df["total_scheduled_deliveries"] - df["total_served"], 0
    )
    df["any_qa_flag"] = df[flag_columns].fillna(0).astype(int).any(axis=1).astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["date"].dt.dayofweek.ge(5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["backlog_open_minus_scheduled"] = np.maximum(
        df["open_orders"] - df["scheduled_from_open_orders"], 0
    )
    df["backload_order_count"] = df["backload_records"].fillna(0)
    return df.sort_values("date").reset_index(drop=True)


def build_feature_sets(df: pd.DataFrame):
    """Port of the study feature-generation rules, applied to synthetic records."""
    d = df.copy().sort_values("date").reset_index(drop=True)
    training_mask = d["date"] < pd.Timestamp("2025-01-01")
    completion_cap = float(d.loc[training_mask, "completion_rate"].quantile(0.99))
    backload_cap = float(d.loc[training_mask, "backload_total_kl"].quantile(0.99))
    d["completion_rate_capped"] = d["completion_rate"].clip(upper=completion_cap)
    d["backload_record_flag"] = d["backload_total_kl"].notna().astype(int)
    d["backload_total_kl_zero"] = d["backload_total_kl"].fillna(0)
    d["backload_total_kl_capped"] = d["backload_total_kl_zero"].clip(upper=backload_cap)
    d["backload_log1p_capped"] = np.log1p(d["backload_total_kl_capped"])

    days_since = []
    counter = np.nan
    for value in d["fuel_update_day"].fillna(False).astype(int):
        if value == 1:
            counter = 0
        else:
            counter = np.nan if pd.isna(counter) else counter + 1
        days_since.append(counter)
    d["days_since_fuel_update"] = days_since

    model = pd.DataFrame(
        {
            "target_date": d["date"],
            "forecast_origin_date": d["date"] - pd.Timedelta(days=1),
            "y_utilization_nextday": d["target_utilization_intensity"],
            "y_secondary_capacity_rate": d["secondary_trip_capacity_utilization_rate"],
            "y_total_served": d["total_served"],
            "y_scheduled_demand_pressure": d["total_scheduled_deliveries"]
            / d["available_trucks_total"].replace(0, np.nan),
            "y_open_order_pressure": d["open_orders"]
            / d["available_trucks_total"].replace(0, np.nan),
            "y_served_plus_deferred_pressure": (
                d["total_served"] + d["deferred_rescheduled"]
            )
            / d["available_trucks_total"].replace(0, np.nan),
            "y_expanded_demand_pressure_index": (
                np.maximum(d["total_served"], d["total_scheduled_deliveries"])
                + d["backlog_open_minus_scheduled"]
                + d["backload_order_count"]
            )
            / d["available_trucks_total"].replace(0, np.nan),
        }
    )

    dow_num = d["date"].dt.dayofweek + 1
    calendar_features = {
        "target_day_of_week_name": d["date"].dt.day_name(),
        "target_day_of_week_num": dow_num,
        "target_is_weekend": d["date"].dt.dayofweek.ge(5).astype(int),
        "target_month": d["date"].dt.month,
        "target_quarter": d["date"].dt.quarter,
        "target_week": d["date"].dt.isocalendar().week.astype(int),
        "target_is_month_start": d["date"].dt.is_month_start.astype(int),
        "target_is_month_end": d["date"].dt.is_month_end.astype(int),
        "target_dow_sin": np.sin(2 * np.pi * dow_num / 7),
        "target_dow_cos": np.cos(2 * np.pi * dow_num / 7),
        "target_month_sin": np.sin(2 * np.pi * d["date"].dt.month / 12),
        "target_month_cos": np.cos(2 * np.pi * d["date"].dt.month / 12),
        "target_week_sin": np.sin(2 * np.pi * d["date"].dt.isocalendar().week.astype(int) / 52),
        "target_week_cos": np.cos(2 * np.pi * d["date"].dt.isocalendar().week.astype(int) / 52),
    }
    for name, values in calendar_features.items():
        model[name] = values

    metadata: list[dict] = []
    feature_series: dict[str, pd.Series] = {}

    def register(
        name,
        series,
        block,
        source_variable,
        transform,
        lookback_days,
        retained_main=True,
        retained_sensitivity=True,
        notes="",
    ):
        feature_series[name] = pd.Series(series).reset_index(drop=True)
        metadata.append(
            {
                "feature_name": name,
                "block": block,
                "source_variable": source_variable,
                "transform": transform,
                "lookback_days": lookback_days,
                "known_by_target_date_start": "Yes",
                "retained_main": int(retained_main),
                "retained_sensitivity": int(retained_sensitivity),
                "notes": notes,
            }
        )

    calendar_map = {
        "target_day_of_week_num": "day-of-week numeric encoding",
        "target_is_weekend": "weekend indicator",
        "target_month": "month number",
        "target_quarter": "quarter number",
        "target_week": "ISO week number",
        "target_is_month_start": "month-start indicator",
        "target_is_month_end": "month-end indicator",
        "target_dow_sin": "cyclical day-of-week sine",
        "target_dow_cos": "cyclical day-of-week cosine",
        "target_month_sin": "cyclical month sine",
        "target_month_cos": "cyclical month cosine",
        "target_week_sin": "cyclical week sine",
        "target_week_cos": "cyclical week cosine",
    }
    for name, description in calendar_map.items():
        register(
            name,
            model[name],
            "Calendar and seasonality",
            "target_date",
            description,
            "target-date calendar",
            notes="Known in advance without operational leakage.",
        )

    def add_lags(base_col, block, lags=(1, 7, 14, 28), main_lags=None, notes=""):
        if main_lags is None:
            main_lags = set(lags)
        for lag in lags:
            register(
                f"lag{lag}_{base_col}",
                d[base_col].shift(lag),
                block,
                base_col,
                f"lag {lag} day(s)",
                lag,
                retained_main=lag in main_lags,
                notes=notes,
            )

    def add_rolls(base_col, block, windows=(3, 7, 14, 28), stats=("mean",), notes=""):
        shifted = d[base_col].shift(1)
        for window in windows:
            for stat in stats:
                rolled = shifted.rolling(window, min_periods=window)
                series = rolled.mean() if stat == "mean" else rolled.std()
                register(
                    f"roll{window}_{stat}_{base_col}",
                    series,
                    block,
                    base_col,
                    f"{stat} of previous {window} days excluding target date",
                    window,
                    notes=notes,
                )

    add_lags(
        "target_utilization_intensity",
        "Autoregressive utilization",
        lags=(1, 7, 14, 28),
        notes="Primary autoregressive signal.",
    )
    add_rolls(
        "target_utilization_intensity",
        "Autoregressive utilization",
        windows=(3, 7, 14, 28),
        stats=("mean", "std"),
        notes="Rolling target history.",
    )
    add_lags(
        "secondary_trip_capacity_utilization_rate",
        "Autoregressive utilization",
        lags=(1, 7, 14),
        main_lags={1, 7},
        notes="Secondary utilization proxy.",
    )
    add_rolls(
        "secondary_trip_capacity_utilization_rate",
        "Autoregressive utilization",
        windows=(7, 14),
        notes="Secondary proxy rolling history.",
    )

    workload_cols = [
        "open_orders",
        "total_scheduled_deliveries",
        "total_served",
        "carry_over",
        "unscheduled",
        "accommodated_after_cutoff",
        "deferred_rescheduled",
        "cancelled",
        "best_effort",
        "service_gap",
        "completion_rate",
        "completion_rate_capped",
        "retail_served_orders",
        "b2b_served_orders",
    ]
    for col in workload_cols:
        add_lags(
            col,
            "Workload and execution history",
            lags=(1, 7, 14),
            main_lags={1, 7},
            notes="Operational workload and service execution history.",
        )
    for col in [
        "open_orders",
        "total_scheduled_deliveries",
        "total_served",
        "service_gap",
        "completion_rate_capped",
    ]:
        stats = ("mean", "std") if col in {"total_served", "service_gap"} else ("mean",)
        add_rolls(
            col,
            "Workload and execution history",
            windows=(7, 14, 28),
            stats=stats,
            notes="Rolling workload summary.",
        )

    capacity_cols = [
        "available_trucks_total",
        "available_drivers",
        "available_trucks_30kl",
        "available_trucks_18kl",
        "available_trucks_36kl",
        "available_trucks_20kl",
    ]
    for col in capacity_cols:
        add_lags(
            col,
            "Capacity and readiness history",
            lags=(1, 7, 14),
            main_lags={1, 7},
            notes="Fleet and staffing readiness history.",
        )
    for col in ["available_trucks_total", "available_drivers"]:
        add_rolls(
            col,
            "Capacity and readiness history",
            windows=(7, 14, 28),
            stats=("mean", "std"),
            notes="Rolling capacity history.",
        )

    for col in [
        "weekly_change_diesel",
        "weekly_change_gasoline",
        "weekly_change_kerosene",
        "days_since_fuel_update",
    ]:
        add_lags(
            col,
            "External fuel-price context",
            lags=(1, 7, 14),
            main_lags={1, 7},
            notes="Fuel context known by forecast origin.",
        )
    for col in ["weekly_change_diesel", "weekly_change_gasoline", "weekly_change_kerosene"]:
        add_rolls(
            col,
            "External fuel-price context",
            windows=(7, 14, 28),
            notes="Rolling fuel-change context.",
        )
    register(
        "lag1_fuel_update_day",
        d["fuel_update_day"].astype(float).shift(1),
        "External fuel-price context",
        "fuel_update_day",
        "lag 1 day indicator",
        1,
        notes="Forecast-origin update indicator.",
    )

    for col in [
        "backload_record_flag",
        "backload_total_kl_zero",
        "backload_total_kl_capped",
        "backload_log1p_capped",
    ]:
        add_lags(
            col,
            "Sparse backload and event context",
            lags=(1, 7),
            main_lags={1},
            notes="Sparse backload representation.",
        )
    add_rolls(
        "backload_log1p_capped",
        "Sparse backload and event context",
        windows=(7, 28),
        notes="Rolling backload context.",
    )
    for col in ["any_qa_flag", "structural_nonoperational_day"]:
        add_lags(
            col,
            "Sparse backload and event context",
            lags=(1, 7),
            main_lags={1},
            notes="Robustness context flag.",
        )

    lag1_served = d["total_served"].shift(1)
    lag1_trucks = d["available_trucks_total"].shift(1)
    lag1_drivers = d["available_drivers"].shift(1)
    lag1_open = d["open_orders"].shift(1)
    lag1_gap = d["service_gap"].shift(1)
    lag1_retail = d["retail_served_orders"].shift(1)
    lag1_b2b = d["b2b_served_orders"].shift(1)
    register(
        "lag1_served_per_driver",
        lag1_served / lag1_drivers.replace(0, np.nan),
        "Capacity and readiness history",
        "total_served, available_drivers",
        "lag1 ratio served orders / available drivers",
        1,
        notes="Forecast-origin productivity signal.",
    )
    register(
        "lag1_open_orders_per_truck",
        lag1_open / lag1_trucks.replace(0, np.nan),
        "Workload and execution history",
        "open_orders, available_trucks_total",
        "lag1 ratio open orders / available trucks",
        1,
        notes="Forecast-origin demand pressure.",
    )
    register(
        "lag1_service_gap_per_truck",
        lag1_gap / lag1_trucks.replace(0, np.nan),
        "Workload and execution history",
        "service_gap, available_trucks_total",
        "lag1 ratio service gap / available trucks",
        1,
        notes="Forecast-origin shortfall pressure.",
    )
    register(
        "lag1_driver_truck_gap",
        lag1_drivers - lag1_trucks,
        "Capacity and readiness history",
        "available_drivers, available_trucks_total",
        "lag1 difference available drivers - available trucks",
        1,
        notes="Forecast-origin staffing slack.",
    )
    register(
        "lag1_retail_share_served",
        lag1_retail / (lag1_retail + lag1_b2b).replace(0, np.nan),
        "Workload and execution history",
        "retail_served_orders, b2b_served_orders",
        "lag1 retail served share",
        1,
        notes="Forecast-origin customer-mix proxy.",
    )

    feature_df = pd.DataFrame(feature_series)
    model = pd.concat([model.reset_index(drop=True), feature_df], axis=1)
    model = model.loc[:, ~model.columns.duplicated()].copy()
    catalog = pd.DataFrame(metadata)
    catalog.loc[
        catalog["feature_name"].str.contains(r"completion_rate(?!_capped)", regex=True),
        "retained_main",
    ] = 0
    catalog.loc[
        catalog["feature_name"].str.contains(r"backload_total_kl_zero|backload_total_kl_capped"),
        "retained_main",
    ] = 0
    catalog.loc[
        catalog["feature_name"].str.contains(
            r"lag\d+_any_qa_flag|lag\d+_structural_nonoperational_day", regex=True
        ),
        "retained_main",
    ] = 0

    main_features = catalog.loc[catalog["retained_main"].eq(1), "feature_name"].tolist()
    eligible_dates = set(synthetic_eligible_dates())
    row_log = pd.DataFrame(
        {
            "target_date": model["target_date"],
            "forecast_origin_date": model["forecast_origin_date"],
            "target_missing": model["y_utilization_nextday"].isna(),
            "insufficient_history_main": model[main_features].isna().any(axis=1),
        }
    )
    row_log["included_main"] = row_log["target_date"].isin(eligible_dates)
    row_log["split"] = np.where(
        row_log["target_date"] < pd.Timestamp("2025-01-01"), "Train 2024", "Test 2025"
    )
    row_log["primary_exclusion_reason"] = np.select(
        [
            row_log["included_main"],
            row_log["target_missing"],
            row_log["insufficient_history_main"],
        ],
        [
            "Included in synthetic demonstration panel",
            "Synthetic target unavailable",
            "Synthetic burn-in history unavailable",
        ],
        default="Synthetic demonstration sampling rule",
    )
    main_panel = model.loc[row_log["included_main"]].copy().reset_index(drop=True)
    main_panel["split"] = np.where(
        main_panel["target_date"] < pd.Timestamp("2025-01-01"), "Train 2024", "Test 2025"
    )
    ordered = ["target_date", "forecast_origin_date", "split"] + [
        c for c in main_panel.columns if c not in {"target_date", "forecast_origin_date", "split"}
    ]
    main_panel = main_panel[ordered]

    catalog["full_panel_nonmissing_rows"] = catalog["feature_name"].map(
        model[catalog["feature_name"].tolist()].notna().sum()
    )
    catalog["full_panel_missing_rows"] = catalog["feature_name"].map(
        model[catalog["feature_name"].tolist()].isna().sum()
    )
    catalog["full_panel_unique_nonmissing"] = catalog["feature_name"].map(
        model[catalog["feature_name"].tolist()].nunique(dropna=True)
    )
    catalog["included_main_nonmissing_rows"] = catalog["feature_name"].map(
        main_panel[catalog["feature_name"].tolist()].notna().sum()
    )
    catalog = catalog.sort_values(
        ["block", "retained_main", "feature_name"], ascending=[True, False, True]
    ).reset_index(drop=True)
    return d, model, main_panel, catalog, row_log, completion_cap, backload_cap


def build_data_dictionary(master: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        "date": "Synthetic calendar date.",
        "target_date": "Date being forecast.",
        "forecast_origin_date": "Close-of-day forecast origin, one day before target_date.",
        "total_served": "Synthetic completed delivery count.",
        "available_trucks_total": "Synthetic available-truck count.",
        "target_utilization_intensity": "total_served / available_trucks_total.",
        "y_utilization_nextday": "Primary next-day realized-utilization target.",
        "y_secondary_capacity_rate": "total_served / committed_trips_total_clean.",
        "y_scheduled_demand_pressure": "Scheduled deliveries per available truck.",
        "y_open_order_pressure": "Open orders per available truck.",
        "y_served_plus_deferred_pressure": "Served plus deferred work per available truck.",
        "y_expanded_demand_pressure_index": "Expanded synthetic pressure index per available truck.",
        "any_qa_flag": "One or more synthetic data-quality flags.",
        "structural_nonoperational_day": "Synthetic structural non-operational indicator.",
    }
    rows = []
    for dataset, frame in [("daily_master_locked.csv", master), ("chunk3_main_modeling_panel.csv", model)]:
        for column in frame.columns:
            rows.append(
                {
                    "dataset": dataset,
                    "variable": column,
                    "dtype": str(frame[column].dtype),
                    "description": descriptions.get(
                        column,
                        "Synthetic operational field or derived feature; see feature catalog when applicable.",
                    ),
                    "contains_real_company_data": "No",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    master = simulate_daily_master(SEED)
    d, full_model, main_panel, catalog, row_log, completion_cap, backload_cap = build_feature_sets(master)

    master.to_csv(SYNTHETIC_DIR / "daily_master_locked.csv", index=False)
    full_model.to_csv(SYNTHETIC_DIR / "chunk3_full_feature_panel_with_flags.csv", index=False)
    main_panel.to_csv(SYNTHETIC_DIR / "chunk3_main_modeling_panel.csv", index=False)
    catalog.to_csv(SYNTHETIC_DIR / "chunk3_feature_catalog.csv", index=False)
    row_log.to_csv(SYNTHETIC_DIR / "row_eligibility_log.csv", index=False)
    build_data_dictionary(master, main_panel).to_csv(
        METADATA_DIR / "data_dictionary.csv", index=False
    )

    fuel_weekly = d.loc[d["fuel_update_day"].eq(1), [
        "date", "weekly_change_diesel", "weekly_change_gasoline", "weekly_change_kerosene"
    ]]
    fuel_weekly.to_csv(SYNTHETIC_DIR / "synthetic_fuel_price_weekly.csv", index=False)

    summary = {
        "synthetic": True,
        "contains_company_data": False,
        "random_seed": SEED,
        "calendar_rows": int(len(master)),
        "raw_descriptive_rows": int((master["all_core_missing"].eq(0)).sum()),
        "valid_operating_rows": int((master["structural_nonoperational_day"].eq(0)).sum()),
        "primary_target_rows": int(master["target_utilization_intensity"].notna().sum()),
        "secondary_target_rows": int(master["secondary_trip_capacity_utilization_rate"].notna().sum()),
        "qa_flagged_rows": int(master["any_qa_flag"].sum()),
        "modeling_rows": int(len(main_panel)),
        "training_rows": int(main_panel["split"].eq("Train 2024").sum()),
        "holdout_rows": int(main_panel["split"].eq("Test 2025").sum()),
        "engineered_features": int(len(catalog)),
        "main_features": int(catalog["retained_main"].eq(1).sum()),
        "sensitivity_only_features": int(
            (catalog["retained_main"].eq(0) & catalog["retained_sensitivity"].eq(1)).sum()
        ),
        "completion_rate_cap_synthetic": completion_cap,
        "backload_kl_cap_synthetic": backload_cap,
        "first_modeling_date": str(main_panel["target_date"].min().date()),
        "last_modeling_date": str(main_panel["target_date"].max().date()),
    }
    (METADATA_DIR / "synthetic_generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
