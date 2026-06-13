from __future__ import annotations

import pytest

from factorylens.vision.eval_baseline import (
    SampleScore,
    choose_threshold,
    compute_auroc,
)


def test_compute_auroc_perfect_separation():
    samples = [
        SampleScore("good-1.png", "good", False, 0.1),
        SampleScore("good-2.png", "good", False, 0.2),
        SampleScore("bad-1.png", "crack", True, 0.8),
        SampleScore("bad-2.png", "cut", True, 0.9),
    ]

    assert compute_auroc(samples) == pytest.approx(1.0)


def test_compute_auroc_handles_ties():
    samples = [
        SampleScore("good-1.png", "good", False, 0.2),
        SampleScore("good-2.png", "good", False, 0.5),
        SampleScore("bad-1.png", "crack", True, 0.5),
        SampleScore("bad-2.png", "cut", True, 0.8),
    ]

    assert compute_auroc(samples) == pytest.approx(0.875)


def test_choose_threshold_reports_confusion_counts():
    samples = [
        SampleScore("good-1.png", "good", False, 0.1),
        SampleScore("good-2.png", "good", False, 0.2),
        SampleScore("bad-1.png", "crack", True, 0.6),
        SampleScore("bad-2.png", "cut", True, 0.7),
    ]

    metrics = choose_threshold(samples)

    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.true_positive == 2
    assert metrics.true_negative == 2
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert 0.2 < metrics.threshold < 0.6
