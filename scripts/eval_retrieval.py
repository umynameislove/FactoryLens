

from __future__ import annotations

import json
import re
from pathlib import Path

from factorylens.db.session import SessionLocal
from factorylens.schemas import KnownIssuesInput
from factorylens.tools.embedder import get_default_embedder
from factorylens.tools.retrieve_known_issues import retrieve_known_issues

GOLD_PATH = "assets/eval/retrieval_gold.json"
CORPUS_SIZE = 12                 # xếp hạng cả corpus để tính MRR
_ID_RE = re.compile(r"(KI-\d+)")

BASELINE = {"recall_at_1": 0.833, "recall_at_3": 0.875, "mrr": 0.880}


def _ki_id(source: str) -> str:
    """'KI-001-crack-drying-stress.md' -> 'KI-001'. (MÌNH CHO SẴN)"""
    match = _ID_RE.match(source)
    return match.group(1) if match else source


def _recall_at(ranked_ids: list[str], relevant: set[str]) -> float:
    return 1.0 if any(i in relevant for i in ranked_ids) else 0.0


def _reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    for rank, i in enumerate(ranked_ids, start=1):
        if i in relevant:
            return 1/rank
    return 0.0


def main() -> None:
    gold = json.loads(Path(GOLD_PATH).read_text(encoding="utf-8"))
    embedder = get_default_embedder()

    sum_r1 = sum_r3 = sum_rr = 0.0
    table_rows = []

    for item in gold:
        relevant = set(item["relevant"])
        with SessionLocal() as db:                       # <-- mỗi query 1 session
            result = retrieve_known_issues(
                KnownIssuesInput(query=item["query"], top_k=CORPUS_SIZE),
                db,
                embedder=embedder,
            )
        ranked = [_ki_id(m.source) for m in result.matches]

        r1 = _recall_at(ranked[:1], relevant)
        r3 = _recall_at(ranked[:3], relevant)
        rr = _reciprocal_rank(ranked, relevant)
        sum_r1 += r1
        sum_r3 += r3
        sum_rr += rr
        table_rows.append((sorted(relevant), ranked[:3], rr, item["query"]))

    n = len(gold)
    print(f"A9 (pgvector):  recall@1={sum_r1/n:.3f}  recall@3={sum_r3/n:.3f}  MRR={sum_rr/n:.3f}")
    print(
        f"B16 baseline :  recall@1={BASELINE['recall_at_1']:.3f}  "
        f"recall@3={BASELINE['recall_at_3']:.3f}  MRR={BASELINE['mrr']:.3f}"
    )
    print()
    print("| # | relevant | A9 top3 | rr | query |")
    print("|---:|---|---|---:|---|")
    for i, (rel, top3, rr, query) in enumerate(table_rows, start=1):
        print(f"| {i} | {', '.join(rel)} | {', '.join(top3)} | {rr:.3f} | {query} |")




    n = len(gold)
    print(f"A9 (pgvector):  recall@1={sum_r1/n:.3f}  recall@3={sum_r3/n:.3f}  MRR={sum_rr/n:.3f}")
    print(
        f"B16 baseline :  recall@1={BASELINE['recall_at_1']:.3f}  "
        f"recall@3={BASELINE['recall_at_3']:.3f}  MRR={BASELINE['mrr']:.3f}"
    )
    print()
    print("| # | relevant | A9 top3 | rr | query |")
    print("|---:|---|---|---:|---|")
    for i, (rel, top3, rr, query) in enumerate(table_rows, start=1):
        print(f"| {i} | {', '.join(rel)} | {', '.join(top3)} | {rr:.3f} | {query} |")


if __name__ == "__main__":
    main()