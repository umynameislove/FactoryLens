from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_retrieval_baseline.py"
GOLD_PATH = Path("assets/eval/retrieval_gold.json")
ISSUES_DIR = Path("assets/known_issues")


def load_script():
    spec = importlib.util.spec_from_file_location("eval_retrieval_baseline_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gold_set_contract_and_size():
    script = load_script()

    gold = script.load_gold(GOLD_PATH)

    assert len(gold) >= 20
    assert all(item.query for item in gold)
    assert all(item.relevant for item in gold)
    assert all(item.note for item in gold)
    hard_negative_count = sum(
        1 for item in gold if "hard negative" in item.note.lower()
    )
    assert hard_negative_count >= 5


def test_all_relevant_ids_exist():
    script = load_script()
    issues = script.load_known_issues(ISSUES_DIR)
    issue_ids = {issue.issue_id for issue in issues}
    gold = script.load_gold(GOLD_PATH)

    missing = sorted(
        {
            relevant_id
            for item in gold
            for relevant_id in item.relevant
            if relevant_id not in issue_ids
        }
    )
    assert missing == []


def test_tfidf_metrics_are_valid():
    pytest.importorskip("sklearn")
    script = load_script()

    metrics, results = script.run_baseline(
        issues_dir=ISSUES_DIR,
        gold_path=GOLD_PATH,
        top_k=5,
    )

    assert int(metrics["query_count"]) == len(results)
    for key in ["recall_at_1", "recall_at_3", "mrr"]:
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["recall_at_3"] >= metrics["recall_at_1"]
    assert all(result.ranked_ids for result in results)
