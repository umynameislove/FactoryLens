"""Generate deterministic sample test logs for the hazelnut demo."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


FIELDNAMES = [
    "unit_id",
    "timestamp",
    "station",
    "measure_name",
    "measure_value",
    "spec_low",
    "spec_high",
    "pass_fail",
]


@dataclass(frozen=True)
class MeasureSpec:
    name: str
    station: str
    nominal: float
    spec_low: float
    spec_high: float
    noise: float


MEASURES = (
    MeasureSpec("diameter_mm", "sizing_station_01", 20.0, 18.0, 22.0, 0.35),
    MeasureSpec("surface_defect_area", "vision_station_01", 0.6, 0.0, 2.5, 0.25),
    MeasureSpec("weight_g", "weigh_station_01", 2.8, 2.2, 3.4, 0.12),
    MeasureSpec("color_index", "color_station_01", 50.0, 42.0, 58.0, 2.0),
)


FAILURE_PLAN = {
    "crack": ("surface_defect_area", "diameter_mm"),
    "cut": ("surface_defect_area",),
    "hole": ("surface_defect_area", "weight_g"),
    "print": ("color_index",),
}


def generate_sample_logs(
    manifest_csv: str,
    out_csv: str,
    max_units: int = 30,
    seed: int = 13,
) -> int:
    """Generate sample test logs from a MVTec manifest.

    The output follows the MVP contract exactly:
    ``unit_id,timestamp,station,measure_name,measure_value,spec_low,spec_high,pass_fail``.
    Defect units get one or two out-of-spec measurements; good units stay in spec.
    """

    manifest_rows = _read_manifest(manifest_csv)
    selected_rows = _select_manifest_rows(manifest_rows, max_units=max_units)
    rng = random.Random(seed)
    base_time = datetime(2026, 6, 13, 8, 0, 0)

    log_rows: list[dict[str, str]] = []
    for unit_index, manifest_row in enumerate(selected_rows, start=1):
        label = manifest_row["label"]
        unit_id = _unit_id(manifest_row, unit_index)
        timestamp = (base_time + timedelta(minutes=unit_index * 3)).isoformat()
        failing_measures = _failing_measures(label, unit_index)

        for measure in MEASURES:
            value = _measure_value(measure, label, failing_measures, rng)
            pass_fail = (
                "FAIL" if value < measure.spec_low or value > measure.spec_high else "PASS"
            )
            log_rows.append(
                {
                    "unit_id": unit_id,
                    "timestamp": timestamp,
                    "station": measure.station,
                    "measure_name": measure.name,
                    "measure_value": _format_number(value),
                    "spec_low": _format_number(measure.spec_low),
                    "spec_high": _format_number(measure.spec_high),
                    "pass_fail": pass_fail,
                }
            )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(log_rows)

    return len(log_rows)


def _read_manifest(manifest_csv: str) -> list[dict[str, str]]:
    with Path(manifest_csv).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_manifest_rows(
    manifest_rows: list[dict[str, str]],
    max_units: int,
) -> list[dict[str, str]]:
    if max_units <= 0:
        return []

    good_rows = [
        row
        for row in manifest_rows
        if row["label"] == "good" and row.get("split") == "test_good"
    ]
    if not good_rows:
        good_rows = [row for row in manifest_rows if row["label"] == "good"]

    defect_rows_by_label = {
        label: [row for row in manifest_rows if row["label"] == label]
        for label in FAILURE_PLAN
    }

    good_target = min(len(good_rows), max_units // 2)
    selected = good_rows[:good_target]
    defect_target = max_units - len(selected)
    selected.extend(_round_robin_defects(defect_rows_by_label, defect_target))

    if len(selected) < max_units:
        used_paths = {row["image_path"] for row in selected}
        fillers = [row for row in manifest_rows if row["image_path"] not in used_paths]
        selected.extend(fillers[: max_units - len(selected)])

    return selected[:max_units]


def _round_robin_defects(
    defect_rows_by_label: dict[str, list[dict[str, str]]],
    max_rows: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    indexes = {label: 0 for label in defect_rows_by_label}

    while len(selected) < max_rows:
        made_progress = False
        for label in sorted(defect_rows_by_label):
            rows = defect_rows_by_label[label]
            index = indexes[label]
            if index >= len(rows):
                continue
            selected.append(rows[index])
            indexes[label] += 1
            made_progress = True
            if len(selected) == max_rows:
                break
        if not made_progress:
            break

    return selected


def _unit_id(manifest_row: dict[str, str], unit_index: int) -> str:
    label = manifest_row["label"].upper()
    stem = Path(manifest_row["image_path"]).stem.upper()
    return f"HZ-{label}-{stem}-{unit_index:03d}"


def _failing_measures(label: str, unit_index: int) -> tuple[str, ...]:
    planned = FAILURE_PLAN.get(label, ())
    if len(planned) <= 1:
        return planned

    return planned if unit_index % 2 == 0 else planned[:1]


def _measure_value(
    measure: MeasureSpec,
    label: str,
    failing_measures: tuple[str, ...],
    rng: random.Random,
) -> float:
    if label != "good" and measure.name in failing_measures:
        return _out_of_spec_value(measure, label, rng)

    value = measure.nominal + rng.uniform(-measure.noise, measure.noise)
    return min(max(value, measure.spec_low + 0.05), measure.spec_high - 0.05)


def _out_of_spec_value(measure: MeasureSpec, label: str, rng: random.Random) -> float:
    width = measure.spec_high - measure.spec_low
    margin = max(width * rng.uniform(0.08, 0.16), 0.1)

    if label == "hole" and measure.name == "weight_g":
        return measure.spec_low - margin
    if label == "crack" and measure.name == "diameter_mm":
        return measure.spec_low - margin

    return measure.spec_high + margin


def _format_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FactoryLens sample logs.")
    parser.add_argument(
        "--manifest",
        default="assets/mvtec_hazelnut_manifest.csv",
        help="Input MVTec manifest CSV.",
    )
    parser.add_argument(
        "--out",
        default="assets/sample_logs/hazelnut_test_logs.csv",
        help="Output sample logs CSV.",
    )
    parser.add_argument("--max-units", type=int, default=30)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = generate_sample_logs(
        manifest_csv=args.manifest,
        out_csv=args.out,
        max_units=args.max_units,
        seed=args.seed,
    )
    print(rows)


if __name__ == "__main__":
    main()
