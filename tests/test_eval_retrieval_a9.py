"""Pure-logic tests for the A9 retrieval-eval helpers (no DB / no torch)."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_retrieval.py"
_spec = importlib.util.spec_from_file_location("eval_retrieval_script", _SCRIPT)
assert _spec is not None and _spec.loader is not None
eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_mod)


def test_ki_id_mapping() -> None:
    assert eval_mod._ki_id("KI-001-crack-drying-stress.md") == "KI-001"
    assert eval_mod._ki_id("KI-012-print-packaging-smudge.md") == "KI-012"


def test_recall_at() -> None:
    assert eval_mod._recall_at(["KI-002"], {"KI-002"}) == 1.0
    assert eval_mod._recall_at(["KI-001", "KI-002"], {"KI-002"}) == 1.0
    assert eval_mod._recall_at(["KI-009"], {"KI-002"}) == 0.0


def test_reciprocal_rank() -> None:
    assert eval_mod._reciprocal_rank(["KI-002"], {"KI-002"}) == 1.0
    assert eval_mod._reciprocal_rank(["KI-009", "KI-002"], {"KI-002"}) == 0.5
    assert eval_mod._reciprocal_rank(["KI-009"], {"KI-002"}) == 0.0
