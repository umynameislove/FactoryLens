from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_per_type.py"


def load_script() -> ModuleType:
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    spec = importlib.util.spec_from_file_location("eval_per_type_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_scores(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "image_path,label,is_defect,score",
                "good-1.png,good,0,0.10",
                "good-2.png,good,0,0.20",
                "crack.png,crack,1,0.80",
                "cut.png,cut,1,0.70",
                "hole.png,hole,1,0.75",
                "print.png,print,1,0.90",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_scores_and_metrics(tmp_path: Path) -> None:
    script = load_script()
    scores_path = tmp_path / "scores.csv"
    write_scores(scores_path)

    samples = script.load_scores_csv(scores_path)
    metrics = script.binary_metrics(samples, 0.5)
    sweep = script.threshold_sweep(samples, start=0.1, end=0.9, step=0.1)
    recommended = script.choose_balanced_threshold(
        sweep,
        current_threshold=0.5,
    )

    assert len(samples) == 6
    assert metrics.accuracy == pytest.approx(1.0)
    assert recommended.f1 == pytest.approx(1.0)


def test_write_outputs_creates_report_and_small_plots(tmp_path: Path) -> None:
    script = load_script()
    scores_path = tmp_path / "scores.csv"
    write_scores(scores_path)
    source_report = tmp_path / "RESULTS.md"
    source_report.write_text("# measured B10 result\n", encoding="utf-8")
    samples = script.load_scores_csv(scores_path)

    result = script.write_outputs(
        samples,
        out_dir=tmp_path / "out",
        scores_csv=scores_path,
        source_report=source_report,
        source_root=None,
        current_threshold=0.5,
        sweep_start=0.1,
        sweep_end=0.9,
        sweep_step=0.1,
    )

    report = Path(result["report_path"])
    assert report.is_file()
    report_text = report.read_text(encoding="utf-8")
    assert "Per-Defect Results" in report_text
    assert "Input SHA-256" in report_text
    assert "does not rerun model inference" in report_text
    for plot in result["plots"]:
        assert Path(plot).is_file()
        assert Path(plot).stat().st_size < 150 * 1024


def test_load_scores_rejects_label_flag_conflict(tmp_path: Path) -> None:
    script = load_script()
    bad_path = tmp_path / "bad.csv"
    bad_path.write_text(
        "image_path,label,is_defect,score\nbad.png,good,1,0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicts"):
        script.load_scores_csv(bad_path)


def test_threshold_sweep_rejects_invalid_range() -> None:
    script = load_script()

    with pytest.raises(ValueError, match="range"):
        script.threshold_sweep([], start=0.8, end=0.2, step=0.1)


def test_exact_search_finds_better_boundary_than_coarse_grid() -> None:
    script = load_script()
    samples = [
        script.SampleScore("good-1.png", "good", False, 0.36),
        script.SampleScore("good-2.png", "good", False, 0.375),
        script.SampleScore("crack.png", "crack", True, 0.378),
        script.SampleScore("cut.png", "cut", True, 0.39),
    ]
    coarse = script.choose_balanced_threshold(
        script.threshold_sweep(samples, start=0.30, end=0.50, step=0.01),
        current_threshold=0.3884,
    )
    exact = script.choose_balanced_threshold(
        script.exact_threshold_search(
            samples,
            start=0.30,
            end=0.50,
            current_threshold=0.3884,
        ),
        current_threshold=0.3884,
    )

    assert exact.f1 == pytest.approx(1.0)
    assert exact.f1 > coarse.f1
    assert exact.threshold == pytest.approx(0.3765)


def test_real_score_regression_keeps_stronger_threshold() -> None:
    script = load_script()
    samples = script.load_scores_csv(
        SCRIPT_PATH.parents[1] / "assets/eval_samples/eval_scores.csv"
    )
    best = script.choose_balanced_threshold(
        script.exact_threshold_search(
            samples,
            start=0.30,
            end=0.50,
            current_threshold=0.3884,
        ),
        current_threshold=0.3884,
    )

    assert best.threshold == pytest.approx(0.313266)
    assert best.accuracy == pytest.approx(0.9363636364)
    assert best.f1 == pytest.approx(0.9503546099)


def test_roc_starts_at_origin_when_score_is_one() -> None:
    script = load_script()
    samples = [
        script.SampleScore("good.png", "good", False, 0.2),
        script.SampleScore("defect.png", "crack", True, 1.0),
    ]

    assert script.roc_points(samples) == [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ]


def test_cli_defaults_work_outside_repository(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "recommended_threshold=0.3133" in result.stdout
    assert (tmp_path / "out/RESULTS_PER_TYPE.md").is_file()
