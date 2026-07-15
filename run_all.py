from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "outputs" / "logs"

STEPS = [
    "scripts/00_generate_synthetic_data.py",
    "scripts/01_descriptive_targets_backlog.py",
    "scripts/02_core_models.py",
    "scripts/03_advanced_models.py",
    "scripts/04_robustness_and_alternative_targets.py",
    "scripts/05_threshold_calibration_and_costs.py",
    "scripts/06_significance_runtime_missingness.py",
    "scripts/07_make_figures.py",
    "scripts/08_verify_outputs.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the public synthetic-data reproduction workflow end to end."
    )
    parser.add_argument(
        "--skip-alternative-lstm",
        action="store_true",
        help="Skip the repeated LSTM fits for alternative targets. The primary LSTM still runs.",
    )
    args = parser.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for prior_log in LOG_DIR.glob("*.log"):
        prior_log.unlink()
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "42"
    if args.skip_alternative_lstm:
        environment["REPRO_SKIP_ALTERNATIVE_LSTM"] = "1"

    for index, relative in enumerate(STEPS, start=1):
        script = ROOT / relative
        print(f"[{index}/{len(STEPS)}] {relative}", flush=True)
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        log_path = LOG_DIR / f"{Path(relative).stem}.log"
        log_path.write_text(
            result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else ""),
            encoding="utf-8",
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr)
            print(f"Workflow stopped at {relative}. See {log_path}.", file=sys.stderr)
            return result.returncode
    print("Synthetic reproduction workflow completed and validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
