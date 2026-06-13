from __future__ import annotations

import csv

from factorylens.data.log_generator import FIELDNAMES, generate_sample_logs


def test_generate_sample_logs_writes_contract_csv(tmp_path):
    manifest_csv = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_csv,
        [
            {
                "image_path": "data/mvtec/hazelnut/test/good/000.png",
                "label": "good",
                "split": "test_good",
                "mask_path": "",
            },
            {
                "image_path": "data/mvtec/hazelnut/test/crack/000.png",
                "label": "crack",
                "split": "test_defect",
                "mask_path": "data/mvtec/hazelnut/ground_truth/crack/000_mask.png",
            },
            {
                "image_path": "data/mvtec/hazelnut/test/print/000.png",
                "label": "print",
                "split": "test_defect",
                "mask_path": "data/mvtec/hazelnut/ground_truth/print/000_mask.png",
            },
        ],
    )

    out_csv = tmp_path / "sample_logs.csv"
    rows_written = generate_sample_logs(
        manifest_csv=str(manifest_csv),
        out_csv=str(out_csv),
        max_units=3,
        seed=1,
    )

    with out_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == FIELDNAMES
    assert rows_written == 12
    assert len(rows) == 12
    assert {row["pass_fail"] for row in rows} == {"PASS", "FAIL"}
    assert len({row["unit_id"] for row in rows}) == 3


def test_failures_are_outside_spec_and_passes_are_inside(tmp_path):
    manifest_csv = tmp_path / "manifest.csv"
    _write_manifest(
        manifest_csv,
        [
            {
                "image_path": "data/mvtec/hazelnut/test/good/000.png",
                "label": "good",
                "split": "test_good",
                "mask_path": "",
            },
            {
                "image_path": "data/mvtec/hazelnut/test/hole/000.png",
                "label": "hole",
                "split": "test_defect",
                "mask_path": "data/mvtec/hazelnut/ground_truth/hole/000_mask.png",
            },
        ],
    )

    out_csv = tmp_path / "sample_logs.csv"
    generate_sample_logs(
        manifest_csv=str(manifest_csv),
        out_csv=str(out_csv),
        max_units=2,
        seed=2,
    )

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    good_rows = [row for row in rows if "GOOD" in row["unit_id"]]
    defect_rows = [row for row in rows if "HOLE" in row["unit_id"]]
    assert good_rows
    assert defect_rows
    assert all(row["pass_fail"] == "PASS" for row in good_rows)
    assert any(row["pass_fail"] == "FAIL" for row in defect_rows)

    for row in rows:
        value = float(row["measure_value"])
        spec_low = float(row["spec_low"])
        spec_high = float(row["spec_high"])
        if row["pass_fail"] == "FAIL":
            assert value < spec_low or value > spec_high
        else:
            assert spec_low <= value <= spec_high


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "split", "mask_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
