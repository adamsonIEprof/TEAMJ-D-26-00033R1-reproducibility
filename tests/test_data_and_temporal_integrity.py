from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
METADATA = ROOT / "data" / "metadata"
TABLES = ROOT / "outputs" / "tables"


class SyntheticDataAndTemporalIntegrityTests(unittest.TestCase):
    """Validate the public synthetic panel and its time-indexed transforms."""

    @classmethod
    def setUpClass(cls) -> None:
        required = [
            DATA / "daily_master_locked.csv",
            DATA / "chunk3_main_modeling_panel.csv",
            DATA / "chunk3_feature_catalog.csv",
            DATA / "row_eligibility_log.csv",
            METADATA / "synthetic_generation_summary.json",
            ROOT / "config" / "model_config.yaml",
            TABLES / "table03_expanding_window_folds.csv",
        ]
        missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        if missing:
            raise AssertionError(
                "Run `python run_all.py` before the tests. Missing: " + ", ".join(missing)
            )

        cls.master = pd.read_csv(DATA / "daily_master_locked.csv", parse_dates=["date"])
        cls.panel = pd.read_csv(
            DATA / "chunk3_main_modeling_panel.csv",
            parse_dates=["target_date", "forecast_origin_date"],
        )
        cls.catalog = pd.read_csv(DATA / "chunk3_feature_catalog.csv")
        cls.row_log = pd.read_csv(
            DATA / "row_eligibility_log.csv",
            parse_dates=["target_date", "forecast_origin_date"],
        )
        cls.summary = json.loads(
            (METADATA / "synthetic_generation_summary.json").read_text(encoding="utf-8")
        )
        cls.config = yaml.safe_load(
            (ROOT / "config" / "model_config.yaml").read_text(encoding="utf-8")
        )

    def test_synthetic_declaration_and_documented_populations(self) -> None:
        self.assertIs(self.summary["synthetic"], True)
        self.assertIs(self.summary["contains_company_data"], False)
        self.assertEqual(self.summary["random_seed"], self.config["project"]["random_seed"])
        self.assertEqual(self.config["project"]["data_mode"], "synthetic_public")

        computed = {
            "calendar_rows": len(self.master),
            "raw_descriptive_rows": int(self.master["all_core_missing"].eq(0).sum()),
            "valid_operating_rows": int(
                self.master["structural_nonoperational_day"].eq(0).sum()
            ),
            "primary_target_rows": int(
                self.master["target_utilization_intensity"].notna().sum()
            ),
            "secondary_target_rows": int(
                self.master["secondary_trip_capacity_utilization_rate"].notna().sum()
            ),
            "qa_flagged_rows": int(self.master["any_qa_flag"].sum()),
            "modeling_rows": len(self.panel),
            "training_rows": int(self.panel["split"].eq("Train 2024").sum()),
            "holdout_rows": int(self.panel["split"].eq("Test 2025").sum()),
            "engineered_features": len(self.catalog),
            "main_features": int(self.catalog["retained_main"].eq(1).sum()),
            "sensitivity_only_features": int(
                (
                    self.catalog["retained_main"].eq(0)
                    & self.catalog["retained_sensitivity"].eq(1)
                ).sum()
            ),
        }
        documented = {
            "calendar_rows": 731,
            "raw_descriptive_rows": 727,
            "valid_operating_rows": 725,
            "primary_target_rows": 724,
            "secondary_target_rows": 726,
            "qa_flagged_rows": 118,
            "modeling_rows": 537,
            "training_rows": 266,
            "holdout_rows": 271,
            "engineered_features": 164,
            "main_features": 127,
            "sensitivity_only_features": 37,
        }
        self.assertEqual(computed, documented)
        for key, value in computed.items():
            self.assertEqual(self.summary[key], value, key)

    def test_calendar_keys_are_complete_unique_and_ordered(self) -> None:
        expected_calendar = pd.date_range("2024-01-01", "2025-12-31", freq="D")
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(self.master["date"]), expected_calendar, check_names=False
        )
        self.assertTrue(self.master["date"].is_unique)
        self.assertTrue(self.panel["target_date"].is_unique)
        self.assertTrue(self.panel["target_date"].is_monotonic_increasing)
        self.assertEqual(self.panel["target_date"].min(), pd.Timestamp("2024-02-06"))
        self.assertEqual(self.panel["target_date"].max(), pd.Timestamp("2025-12-24"))

    def test_target_definitions_are_arithmetically_exact(self) -> None:
        primary_valid = (
            self.master["total_served"].notna()
            & self.master["available_trucks_total"].gt(0)
        )
        primary_expected = (
            self.master.loc[primary_valid, "total_served"]
            / self.master.loc[primary_valid, "available_trucks_total"]
        )
        np.testing.assert_allclose(
            self.master.loc[primary_valid, "target_utilization_intensity"],
            primary_expected,
            rtol=0,
            atol=1e-12,
        )

        secondary_valid = (
            self.master["total_served"].notna()
            & self.master["committed_trips_total_clean"].gt(0)
        )
        secondary_expected = (
            self.master.loc[secondary_valid, "total_served"]
            / self.master.loc[secondary_valid, "committed_trips_total_clean"]
        )
        np.testing.assert_allclose(
            self.master.loc[secondary_valid, "secondary_trip_capacity_utilization_rate"],
            secondary_expected,
            rtol=0,
            atol=1e-12,
        )

        target_by_date = self.master.set_index("date")["target_utilization_intensity"]
        expected_panel_target = target_by_date.reindex(self.panel["target_date"]).to_numpy()
        np.testing.assert_allclose(
            self.panel["y_utilization_nextday"], expected_panel_target, rtol=0, atol=1e-12
        )

    def test_modeling_panel_has_complete_retained_features(self) -> None:
        main_features = self.catalog.loc[
            self.catalog["retained_main"].eq(1), "feature_name"
        ].tolist()
        absent = sorted(set(main_features) - set(self.panel.columns))
        self.assertEqual(absent, [])
        missing_counts = self.panel[main_features].isna().sum()
        self.assertEqual(
            missing_counts[missing_counts.gt(0)].to_dict(),
            {},
            "Retained main features must be complete on every included modeling row.",
        )
        included = self.row_log["included_main"].astype(str).str.lower().eq("true")
        self.assertEqual(int(included.sum()), len(self.panel))
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(self.row_log.loc[included, "target_date"]),
            pd.DatetimeIndex(self.panel["target_date"]),
            check_names=False,
        )

    def test_forecast_origin_and_split_alignment(self) -> None:
        expected_origin = self.panel["target_date"] - pd.Timedelta(days=1)
        pd.testing.assert_series_equal(
            self.panel["forecast_origin_date"].reset_index(drop=True),
            expected_origin.reset_index(drop=True),
            check_names=False,
        )
        row_log_origin = self.row_log["target_date"] - pd.Timedelta(days=1)
        pd.testing.assert_series_equal(
            self.row_log["forecast_origin_date"].reset_index(drop=True),
            row_log_origin.reset_index(drop=True),
            check_names=False,
        )

        holdout_start = pd.Timestamp(self.config["validation"]["holdout_start"])
        expected_split = np.where(
            self.panel["target_date"].lt(holdout_start), "Train 2024", "Test 2025"
        )
        np.testing.assert_array_equal(self.panel["split"].to_numpy(), expected_split)
        self.assertLess(
            self.panel.loc[self.panel["split"].eq("Train 2024"), "target_date"].max(),
            holdout_start,
        )
        self.assertGreaterEqual(
            self.panel.loc[self.panel["split"].eq("Test 2025"), "target_date"].min(),
            holdout_start,
        )

    def test_expanding_window_fold_table_matches_timeseriessplit(self) -> None:
        train = self.panel.loc[self.panel["split"].eq("Train 2024")].reset_index(drop=True)
        n_splits = int(self.config["validation"]["n_splits"])
        splits = list(TimeSeriesSplit(n_splits=n_splits).split(train))
        recorded = pd.read_csv(TABLES / "table03_expanding_window_folds.csv")
        self.assertEqual(len(recorded), n_splits)

        validation_date_sets = []
        for fold_number, ((train_idx, valid_idx), row) in enumerate(
            zip(splits, recorded.itertuples(index=False)), start=1
        ):
            self.assertEqual(int(row.fold), fold_number)
            self.assertEqual(int(row.train_rows), len(train_idx))
            self.assertEqual(int(row.validation_rows), len(valid_idx))
            self.assertLess(train_idx[-1], valid_idx[0])
            self.assertEqual(str(row.train_start), train.iloc[train_idx[0]]["target_date"].date().isoformat())
            self.assertEqual(str(row.train_end), train.iloc[train_idx[-1]]["target_date"].date().isoformat())
            self.assertEqual(
                str(row.validation_start),
                train.iloc[valid_idx[0]]["target_date"].date().isoformat(),
            )
            self.assertEqual(
                str(row.validation_end),
                train.iloc[valid_idx[-1]]["target_date"].date().isoformat(),
            )
            validation_date_sets.append(set(train.iloc[valid_idx]["target_date"]))

        for i, left in enumerate(validation_date_sets):
            for right in validation_date_sets[i + 1 :]:
                self.assertTrue(left.isdisjoint(right))

    def test_lags_and_rolls_use_only_pre_target_records(self) -> None:
        master = self.master.set_index("date")
        target_dates = pd.DatetimeIndex(self.panel["target_date"])

        comparisons = {
            "lag1_target_utilization_intensity": master[
                "target_utilization_intensity"
            ].shift(1),
            "lag7_target_utilization_intensity": master[
                "target_utilization_intensity"
            ].shift(7),
            "lag1_available_trucks_total": master["available_trucks_total"].shift(1),
            "roll7_mean_total_served": master["total_served"].shift(1).rolling(7).mean(),
            "roll14_mean_open_orders": master["open_orders"].shift(1).rolling(14).mean(),
        }
        for feature, full_expected in comparisons.items():
            expected = full_expected.reindex(target_dates).to_numpy()
            np.testing.assert_allclose(
                self.panel[feature],
                expected,
                rtol=0,
                atol=1e-12,
                err_msg=f"Temporal alignment failed for {feature}",
            )

    def test_catalog_rules_and_training_only_caps_prevent_leakage(self) -> None:
        known = self.catalog["known_by_target_date_start"].astype(str).str.lower()
        self.assertTrue(known.eq("yes").all())
        self.assertFalse(
            self.catalog["feature_name"].isin(
                {
                    "y_utilization_nextday",
                    "target_utilization_intensity",
                    "y_total_served",
                    "total_served",
                }
            ).any()
        )

        operational = self.catalog.loc[
            self.catalog["block"].ne("Calendar and seasonality")
        ]
        for row in operational.itertuples(index=False):
            self.assertRegex(row.feature_name, r"^(lag\d+_|roll\d+_)")
            self.assertGreaterEqual(int(row.lookback_days), 1)
            transform = str(row.transform).lower()
            if row.feature_name.startswith("roll"):
                self.assertIn("excluding target date", transform)
                self.assertIn("previous", transform)
            else:
                self.assertIn("lag", transform)

        train_master = self.master.loc[self.master["date"].lt("2025-01-01")]
        expected_completion_cap = float(train_master["completion_rate"].quantile(0.99))
        expected_backload_cap = float(train_master["backload_total_kl"].quantile(0.99))
        self.assertAlmostEqual(
            float(self.summary["completion_rate_cap_synthetic"]),
            expected_completion_cap,
            places=12,
        )
        self.assertAlmostEqual(
            float(self.summary["backload_kl_cap_synthetic"]),
            expected_backload_cap,
            places=12,
        )

        master = self.master.set_index("date")
        target_dates = pd.DatetimeIndex(self.panel["target_date"])
        expected_completion = (
            master["completion_rate"].clip(upper=expected_completion_cap).shift(1)
        ).reindex(target_dates)
        expected_backload = (
            np.log1p(master["backload_total_kl"].fillna(0).clip(upper=expected_backload_cap))
            .shift(1)
            .reindex(target_dates)
        )
        np.testing.assert_allclose(
            self.panel["lag1_completion_rate_capped"],
            expected_completion,
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            self.panel["lag1_backload_log1p_capped"],
            expected_backload,
            rtol=0,
            atol=1e-12,
        )

        # Guard against a future catalog edit that silently introduces a zero-day
        # operational lookback despite retaining a leakage-safe feature name.
        zero_day_operational = operational.loc[
            operational["lookback_days"].astype(str).str.fullmatch(r"0(?:\.0+)?")
        ]
        self.assertTrue(zero_day_operational.empty)


if __name__ == "__main__":
    unittest.main()
