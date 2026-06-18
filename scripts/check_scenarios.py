"""Validate B15 demo scenario log files.

The checker is intentionally small and boring: it only proves that each
scenario log CSV is internally consistent. Vision scores still have to be
measured by running the real vision pipeline.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


REQUIRED_COLUMNS = [
    "unit_id",
    "timestamp",
    "station",
    "measure_name",
    "measure_value",
    "spec_low",
    "spec_high",
    "pass_fail",
]


def check_logs(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != REQUIRED_COLUMNS:
                return [
                    "header must exactly match: " + ",".join(REQUIRED_COLUMNS),
                ]

            fail_count = 0
            row_count = 0
            for line_number, row in enumerate(reader, start=2):
                row_count += 1
                errors.extend(_check_row(row, line_number))
                if row.get("pass_fail") == "FAIL":
                    fail_count += 1
    except OSError as exc:
        return [f"could not read file: {exc}"]
    except csv.Error as exc:
        return [f"malformed CSV: {exc}"]

    if row_count == 0:
        errors.append("logs.csv must contain at least one data row")
    if fail_count == 0:
        errors.append("logs.csv must contain at least one FAIL row")
    return errors


def _check_row(row: dict[str, str], line_number: int) -> list[str]:
    errors: list[str] = []
    for column in REQUIRED_COLUMNS:
        if row.get(column, "").strip() == "":
            errors.append(f"line {line_number}: {column} must not be blank")

    pass_fail = row.get("pass_fail", "").strip()
    if pass_fail not in {"PASS", "FAIL"}:
        errors.append(f"line {line_number}: pass_fail must be PASS or FAIL")

    value = _parse_float(row.get("measure_value", ""), "measure_value", line_number, errors)
    spec_low = _parse_float(row.get("spec_low", ""), "spec_low", line_number, errors)
    spec_high = _parse_float(row.get("spec_high", ""), "spec_high", line_number, errors)
    if value is None or spec_low is None or spec_high is None or pass_fail not in {"PASS", "FAIL"}:
        return errors

    if spec_low > spec_high:
        errors.append(f"line {line_number}: spec_low must be <= spec_high")
        return errors

    inside_spec = spec_low <= value <= spec_high
    expected = "PASS" if inside_spec else "FAIL"
    if pass_fail != expected:
        errors.append(
            f"line {line_number}: pass_fail={pass_fail} but value/spec imply {expected}"
        )
    return errors


def _parse_float(
    raw_value: str,
    field_name: str,
    line_number: int,
    errors: list[str],
) -> float | None:
    try:
        value = float(raw_value)
    except ValueError:
        errors.append(f"line {line_number}: {field_name} must be a finite float")
        return None
    if not math.isfinite(value):
        errors.append(f"line {line_number}: {field_name} must be a finite float")
        return None
    return value


def iter_log_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob("*/logs.csv"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate B15 scenario logs.")
    parser.add_argument(
        "root",
        nargs="?",
        default="assets/demo/scenarios",
        help="Scenario root folder, or one logs.csv file.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    log_files = iter_log_files(root)
    if not log_files:
        print(f"FAIL {root}: no logs.csv files found")
        return 1

    exit_code = 0
    for log_file in log_files:
        errors = check_logs(log_file)
        if errors:
            exit_code = 1
            print(f"FAIL {log_file}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {log_file}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
