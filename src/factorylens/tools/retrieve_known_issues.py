"""Retrieve top-k known-issues bằng pgvector cosine search (read-only)."""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from factorylens.schemas import KnownIssueMatch, KnownIssuesInput, KnownIssuesResult
from factorylens.tools.embedder import Embedder

logger = logging.getLogger(__name__)

_TABLE = "known_issue_vectors"
_MAX_TOP_K = 20


def _vector_literal(vec: list[float]) -> str:
    """[0.1, 0.2] -> '[0.10000000,0.20000000]' cho pgvector (số float, an toàn)."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _rows_to_matches(rows: list) -> list[KnownIssueMatch]:
    """Đổi rows SQL -> KnownIssueMatch, clamp similarity về [0,1]."""
    matches: list[KnownIssueMatch] = []
    for row in rows:
        sim = row.similarity
        if sim < 0:
            sim = 0
        if sim > 1.0:
            sim = 1
        matches.append(
            KnownIssueMatch(
                title=row.title, snippet=row.snippet, source=row.source, similarity=sim
            )
        )
    return matches


def retrieve_known_issues(
    inp: KnownIssuesInput,
    db: Session,
    *,
    embedder: Embedder,
) -> KnownIssuesResult:
    """Embed query -> cosine top-k trên pgvector -> KnownIssuesResult.

    Fail-safe: mọi lỗi (chưa ingest, không có DB, embed lỗi) -> trả result rỗng
    + warning, KHÔNG raise, KHÔNG lộ chi tiết DB. (Giống tinh thần A8.)
    """
    safe_k = min(max(inp.top_k, 1), _MAX_TOP_K)

    try:
        query_vec = embedder.embed([inp.query])[0]
    except Exception:  # noqa: BLE001 - embedder lib có thể ném nhiều loại lỗi
        logger.warning("Embedding query failed.")
        return KnownIssuesResult(
            query=inp.query,
            matches=[],
            warnings=["Embedding model unavailable; no retrieval performed."],
        )

    q_literal = _vector_literal(query_vec)

    sql = f"""
        SELECT title, snippet, source,
        1 - (embedding <=> CAST(:q AS vector)) AS similarity
        FROM {_TABLE}
    """
    params = {"q": q_literal, "k": safe_k}
    if inp.defect_type is not None:
        sql += " WHERE defect_type = :dt"
        params["dt"] = inp.defect_type

    sql += " ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k"

    try:
        rows = db.execute(text(sql), params).all()
    except SQLAlchemyError as exc:
        logger.warning("Known-issue retrieval query failed: %s", exc)
        return KnownIssuesResult(
            query=inp.query,
            matches=[],
            warnings=["Known-issue store unavailable; run ingest first."],
        )
    matches = _rows_to_matches(rows)
    warnings = [] if matches else ["No similar known-issue found."]
    return KnownIssuesResult(query=inp.query, matches=matches, warnings=warnings)
