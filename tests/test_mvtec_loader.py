from __future__ import annotations

import csv

import pytest

from factorylens.data.mvtec_loader import build_manifest, list_split


def test_list_split_returns_expected_items(tmp_path):
    root = tmp_path / "mvtec"
    category = root / "hazelnut"
    _touch(category / "train" / "good" / "000.png")
    _touch(category / "test" / "good" / "001.png")
    _touch(category / "test" / "crack" / "002.png")
    _touch(category / "ground_truth" / "crack" / "002_mask.png")

    train_items = list_split(str(root), "hazelnut", "train_good")
    good_items = list_split(str(root), "hazelnut", "test_good")
    defect_items = list_split(str(root), "hazelnut", "test_defect")

    assert len(train_items) == 1
    assert train_items[0].label == "good"
    assert train_items[0].mask_path is None
    assert len(good_items) == 1
    assert good_items[0].label == "good"
    assert len(defect_items) == 1
    assert defect_items[0].label == "crack"
    assert defect_items[0].mask_path is not None
    assert defect_items[0].mask_path.endswith("002_mask.png")


def test_build_manifest_writes_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "mvtec"
    category = root / "hazelnut"
    _touch(category / "train" / "good" / "000.png")
    _touch(category / "test" / "good" / "001.png")
    _touch(category / "test" / "cut" / "002.png")
    _touch(category / "ground_truth" / "cut" / "002_mask.png")

    out_csv = tmp_path / "assets" / "manifest.csv"
    rows_written = build_manifest("data/mvtec", "hazelnut", str(out_csv))

    assert rows_written == 3
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0] == {
        "image_path": "data/mvtec/hazelnut/train/good/000.png",
        "label": "good",
        "split": "train_good",
        "mask_path": "",
    }
    assert rows[2]["image_path"] == "data/mvtec/hazelnut/test/cut/002.png"
    assert rows[2]["mask_path"] == "data/mvtec/hazelnut/ground_truth/cut/002_mask.png"
    assert all(not row["image_path"].startswith("/") for row in rows)


def test_list_split_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="split must be one of"):
        list_split(str(tmp_path), "hazelnut", "validation")


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
