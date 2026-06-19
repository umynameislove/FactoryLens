from __future__ import annotations

from sqlalchemy import text

from factorylens.data.known_issues import load_known_issues
from factorylens.db.session import get_engine
from factorylens.tools.embedder import EMBED_DIM, get_default_embedder

KNOWN_ISSUES_DIR = "assets/known_issues"
TABLE = "known_issue_vectors"


def _vector_literal(vec: list[float]) -> str:
    """Đổi [0.1, 0.2, ...] -> '[0.1,0.2,...]' đúng cú pháp pgvector.
    An toàn injection: toàn số float, không phải chuỗi người dùng."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _create_table(conn) -> None:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id          SERIAL PRIMARY KEY,
                source      TEXT UNIQUE NOT NULL,
                title       TEXT NOT NULL,
                defect_type TEXT NOT NULL,
                snippet     TEXT NOT NULL,
                content     TEXT NOT NULL,
                embedding   vector({EMBED_DIM}) NOT NULL
            );
            """
        )
    )


def main() -> None:
    docs = load_known_issues(KNOWN_ISSUES_DIR)
    embedder = get_default_embedder()

    # embed TOÀN BỘ text mỗi doc (chunking = cả file, theo quyết định Bài 1)
    vectors = embedder.embed([d.text for d in docs])

    engine = get_engine()
    with engine.begin() as conn:
        _create_table(conn)
        conn.execute(text(f"TRUNCATE {TABLE};"))   # idempotent: nạp lại sạch
        for doc, vec in zip(docs, vectors, strict=True):
            snippet = doc.symptom[:200]            # snippet = Symptom, cắt 200
            conn.execute(
                text(
                    f"""
                    INSERT INTO {TABLE}
                        (source, title, defect_type, snippet, content, embedding)
                    VALUES (:source, :title, :dt, :snippet, :content, CAST(:emb AS vector))
                    """
                ),
                {
                    "source": doc.source,
                    "title": doc.title,
                    "dt": doc.defect_type,
                    "snippet": snippet,
                    "content": doc.text,
                    "emb": _vector_literal(vec),
                },
            )
    print(f"Ingested {len(docs)} known-issue docs into {TABLE}.")


if __name__ == "__main__":
    main()