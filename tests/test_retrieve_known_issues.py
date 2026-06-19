import os

import pytest
from pydantic import ValidationError

from factorylens.schemas import KnownIssuesInput
from factorylens.tools.retrieve_known_issues import (
    _rows_to_matches,
    retrieve_known_issues,
)


class _Row:
    """Giả 1 dòng SQL trả về (có .title/.snippet/.source/.similarity)."""

    def __init__(self, title, snippet, source, similarity):
        self.title = title
        self.snippet = snippet
        self.source = source
        self.similarity = similarity


# ---------- TẦNG A: thuần logic, không cần DB ----------

def test_rows_to_matches_maps_fields() -> None:
    rows = [_Row("Crack", "shell cracked", "KI-001.md", 0.83)]
    matches = _rows_to_matches(rows)

    assert len(matches) == 1
    assert matches[0].title == "Crack"
    assert matches[0].snippet == "shell cracked"
    assert matches[0].source == "KI-001.md"
    assert abs(matches[0].similarity - 0.83) < 1e-6


def test_rows_to_matches_clamps_similarity() -> None:
    rows = [_Row("A", "s", "a.md", -0.05), _Row("B", "s", "b.md", 1.2)]
    matches = _rows_to_matches(rows)

    assert matches[0].similarity == 0.0   # âm -> 0
    assert matches[1].similarity == 1.0   # >1 -> 1


def test_rows_to_matches_empty() -> None:
    assert _rows_to_matches([]) == []


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        KnownIssuesInput(query="x", top_k=0)


def test_empty_query_rejected() -> None:
    with pytest.raises(ValidationError):
        KnownIssuesInput(query="")


# ---------- TẦNG B: tích hợp, skip nếu không có Postgres ----------

@pytest.mark.skipif(
    "postgresql" not in os.getenv("DATABASE_URL", ""),
    reason="pgvector cần Postgres; bỏ qua khi không có DB (vd CI).",
)
def test_retrieve_returns_relevant_issue() -> None:
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    from factorylens.db.session import SessionLocal
    from factorylens.tools.embedder import get_default_embedder

    # Nếu Postgres không chạy -> skip (không fail).
    try:
        with SessionLocal() as probe:
            probe.execute(text("SELECT 1"))
    except OperationalError:
        pytest.skip("Postgres không chạy; bỏ qua test tích hợp.")

    embedder = get_default_embedder()
    with SessionLocal() as db:
        result = retrieve_known_issues(
            KnownIssuesInput(query="shell has a crack near the split", top_k=3),
            db,
            embedder=embedder,
        )

    assert result.matches, "phải có >=1 kết quả — đã ingest chưa?"
    assert result.matches[0].similarity > 0.2
    assert len(result.matches) <= 3