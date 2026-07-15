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
PREDICTIONS = ROOT / "outputs" / "predictions"
TABLES = ROOT / "outputs" / "tables"


class OutputCoverageCostAndPrivacyTests(unittest.TestCase):
    """Validate prediction coverage, cost calculations, and public-package hygiene."""

    EXPECTED_ADVANCED_MODELS = {
        "Random forest (S2)",
        "XGBoost (S2)",
        "LightGBM (S2)",
        "14-day moving average",
        "LSTM 28-step (S2)",
    }

    @classmethod
    def setUpClass(cls) -> None:
        required = [
            DATA / "chunk3_main_modeling_panel.csv",
            PREDICTIONS / "holdout_predictions.csv",
            PREDICTIONS / "oof_train_predictions.csv",
            TABLES / "threshold_calibration_selected.csv",
            TABLES / "table17_counterfactual_cost_scenarios.csv",
            METADATA / "synthetic_generation_summary.json",
            METADATA / "data_dictionary.csv",
            ROOT / "config" / "model_config.yaml",
        ]
        missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        if missing:
            raise AssertionError(
                "Run `python run_all.py` before the tests. Missing: " + ", ".join(missing)
            )

        cls.panel = pd.read_csv(
            DATA / "chunk3_main_modeling_panel.csv",
            parse_dates=["target_date", "forecast_origin_date"],
        )
        cls.holdout = pd.read_csv(
            PREDICTIONS / "holdout_predictions.csv", parse_dates=["target_date"]
        )
        cls.oof = pd.read_csv(
            PREDICTIONS / "oof_train_predictions.csv", parse_dates=["target_date"]
        )
        cls.config = yaml.safe_load(
            (ROOT / "config" / "model_config.yaml").read_text(encoding="utf-8")
        )

    def _assert_actuals_match_panel(self, predictions: pd.DataFrame) -> None:
        target_map = self.panel.set_index("target_date")["y_utilization_nextday"]
        expected = target_map.reindex(predictions["target_date"]).to_numpy()
        np.testing.assert_allclose(predictions["actual"], expected, rtol=0, atol=1e-12)

    def test_holdout_predictions_cover_every_model_date_once(self) -> None:
        expected_dates = pd.DatetimeIndex(
            self.panel.loc[self.panel["split"].eq("Test 2025"), "target_date"]
        )
        self.assertEqual(set(self.holdout["model"]), self.EXPECTED_ADVANCED_MODELS)
        self.assertEqual(len(self.holdout), len(expected_dates) * len(self.EXPECTED_ADVANCED_MODELS))
        self.assertFalse(self.holdout.duplicated(["model", "target_date"]).any())
        self.assertFalse(self.holdout[["actual", "prediction"]].isna().any().any())
        self.assertTrue(np.isfinite(self.holdout[["actual", "prediction"]].to_numpy()).all())
        floor = float(self.config["project"]["prediction_floor"])
        self.assertTrue(self.holdout["prediction"].ge(floor).all())

        for model, group in self.holdout.groupby("model"):
            pd.testing.assert_index_equal(
                pd.DatetimeIndex(group.sort_values("target_date")["target_date"]),
                expected_dates,
                check_names=False,
                obj=model,
            )
        self._assert_actuals_match_panel(self.holdout)

    def test_oof_predictions_cover_all_validation_rows_and_folds(self) -> None:
        train = self.panel.loc[self.panel["split"].eq("Train 2024")].reset_index(drop=True)
        splits = list(
            TimeSeriesSplit(n_splits=int(self.config["validation"]["n_splits"])).split(train)
        )
        expected_fold_by_date = {}
        for fold, (_, valid_idx) in enumerate(splits, start=1):
            for date in train.iloc[valid_idx]["target_date"]:
                expected_fold_by_date[pd.Timestamp(date)] = fold

        expected_dates = pd.DatetimeIndex(sorted(expected_fold_by_date))
        self.assertEqual(set(self.oof["model"]), self.EXPECTED_ADVANCED_MODELS)
        self.assertEqual(len(self.oof), len(expected_dates) * len(self.EXPECTED_ADVANCED_MODELS))
        self.assertFalse(self.oof.duplicated(["model", "target_date"]).any())
        self.assertFalse(self.oof[["actual", "prediction", "fold"]].isna().any().any())
        self.assertTrue(np.isfinite(self.oof[["actual", "prediction"]].to_numpy()).all())

        for model, group in self.oof.groupby("model"):
            ordered = group.sort_values("target_date")
            pd.testing.assert_index_equal(
                pd.DatetimeIndex(ordered["target_date"]),
                expected_dates,
                check_names=False,
                obj=model,
            )
            recorded_folds = ordered["target_date"].map(expected_fold_by_date).to_numpy()
            np.testing.assert_array_equal(ordered["fold"].astype(int), recorded_folds)
        self._assert_actuals_match_panel(self.oof)

    def test_threshold_calibration_cost_equals_fp_plus_ratio_times_fn(self) -> None:
        calibration = pd.read_csv(TABLES / "threshold_calibration_selected.csv")
        self.assertEqual(set(calibration["model"]), self.EXPECTED_ADVANCED_MODELS)
        expected_days = len(
            self.panel.loc[self.panel["split"].eq("Train 2024")]
        ) - 46
        self.assertTrue(calibration["calibration_days"].eq(expected_days).all())

        confusion_total = (
            calibration["true_positive_days"]
            + calibration["false_positive_days"]
            + calibration["false_negative_days"]
            + calibration["true_negative_days"]
        )
        np.testing.assert_array_equal(confusion_total, calibration["calibration_days"])
        expected_cost = (
            calibration["false_positive_days"]
            + calibration["false_negative_to_false_positive_cost_ratio"]
            * calibration["false_negative_days"]
        )
        np.testing.assert_allclose(calibration["cost_units"], expected_cost, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            calibration["normalized_cost"],
            expected_cost / calibration["calibration_days"],
            rtol=0,
            atol=1e-12,
        )

    def test_counterfactual_cost_formula_and_configured_penalties(self) -> None:
        costs = pd.read_csv(TABLES / "table17_counterfactual_cost_scenarios.csv")
        self.assertEqual(set(costs["model"]), self.EXPECTED_ADVANCED_MODELS)
        self.assertEqual(
            set(costs["cost_variant"]),
            {"Combined stress and backlog", "Stress only", "Critical risk"},
        )
        holdout_days = int(self.panel["split"].eq("Test 2025").sum())
        scenario_config = {
            row["name"]: (
                float(row["missed_stress_penalty"]),
                float(row["missed_backlog_penalty"]),
            )
            for row in self.config["cost_scenarios"]
        }
        self.assertEqual(set(costs["scenario"]), set(scenario_config))

        for row in costs.itertuples(index=False):
            configured_stress, configured_backlog = scenario_config[row.scenario]
            if row.cost_variant == "Combined stress and backlog":
                self.assertEqual(float(row.missed_stress_penalty), configured_stress)
                self.assertEqual(float(row.missed_backlog_penalty), configured_backlog)
                expected = (
                    row.alert_days
                    + configured_stress * row.missed_stress_days
                    + configured_backlog * row.missed_backlog_days
                )
            elif row.cost_variant == "Stress only":
                self.assertEqual(float(row.missed_stress_penalty), configured_stress)
                self.assertEqual(float(row.missed_backlog_penalty), 0.0)
                expected = row.alert_days + configured_stress * row.missed_stress_days
            else:
                critical_penalty = max(configured_stress, configured_backlog)
                self.assertEqual(float(row.missed_stress_penalty), critical_penalty)
                self.assertEqual(float(row.missed_backlog_penalty), critical_penalty)
                expected = row.alert_days + critical_penalty * row.missed_critical_days

            self.assertAlmostEqual(float(row.cost_units), float(expected), places=12)
            self.assertAlmostEqual(
                float(row.normalized_cost), float(expected) / holdout_days, places=12
            )

    def test_no_private_or_binary_input_artifacts_are_packaged(self) -> None:
        forbidden_input_suffixes = {
            ".xls",
            ".xlsx",
            ".xlsm",
            ".xlsb",
            ".parquet",
            ".feather",
            ".pkl",
            ".pickle",
            ".joblib",
            ".sav",
            ".dta",
            ".sas7bdat",
            ".sqlite",
            ".sqlite3",
            ".db",
            ".zip",
            ".7z",
            ".rar",
        }
        input_roots = [ROOT / "data", ROOT / "config", ROOT / "scripts", ROOT / "src"]
        prohibited = []
        symlinks = []
        for input_root in input_roots:
            for path in input_root.rglob("*"):
                if path.is_symlink():
                    symlinks.append(str(path.relative_to(ROOT)))
                if path.is_file() and path.suffix.lower() in forbidden_input_suffixes:
                    prohibited.append(str(path.relative_to(ROOT)))
        self.assertEqual(prohibited, [], "Private/binary input candidates were packaged.")
        self.assertEqual(symlinks, [], "Input paths must not point outside the package.")

        data_files = [path for path in (ROOT / "data").rglob("*") if path.is_file()]
        self.assertTrue(data_files)
        self.assertTrue(
            all(path.suffix.lower() in {".csv", ".json"} for path in data_files),
            "Public input data must remain transparent text files.",
        )

        absolute_path_pattern = re.compile(r"(?:[A-Za-z]:\\Users\\|/" + "mnt" + "/data/)", re.I)
        email_pattern = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
        text_findings = []
        for path in data_files:
            text = path.read_text(encoding="utf-8")
            if absolute_path_pattern.search(text) or email_pattern.search(text):
                text_findings.append(str(path.relative_to(ROOT)))
        self.assertEqual(text_findings, [], "Public data contain a path or email identifier.")

        summary = json.loads(
            (METADATA / "synthetic_generation_summary.json").read_text(encoding="utf-8")
        )
        self.assertIs(summary["contains_company_data"], False)
        master_sources = pd.read_csv(DATA / "daily_master_locked.csv", usecols=["source_sheet"])
        self.assertEqual(set(master_sources["source_sheet"]), {"SYNTHETIC_DAILY_PANEL"})
        dictionary = pd.read_csv(METADATA / "data_dictionary.csv")
        self.assertTrue(
            dictionary["contains_real_company_data"].astype(str).str.lower().eq("no").all()
        )


if __name__ == "__main__":
    unittest.main()
