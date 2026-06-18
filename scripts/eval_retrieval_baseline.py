"""TF-IDF retrieval baseline for the B16 known-issue gold set."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ISSUE_ID_RE = re.compile(r"^(KI-\d{3})")


@dataclass(frozen=True)
class KnownIssue:
    issue_id: str
    title: str
    source: str
    text: str


@dataclass(frozen=True)
class GoldQuery:
    query: str
    relevant: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class QueryResult:
    query: str
    relevant: tuple[str, ...]
    ranked_ids: tuple[str, ...]
    top_scores: tuple[float, ...]
    reciprocal_rank: float
    recall_at_1: float
    recall_at_3: float


def load_known_issues(directory: str | Path) -> list[KnownIssue]:
    """Load known-issue Markdown files, excluding the index."""

    issues: list[KnownIssue] = []
    for path in sorted(Path(directory).glob("KI-*.md")):
        if path.name == "INDEX.md":
            continue
        match = ISSUE_ID_RE.match(path.stem)
        if match is None:
            continue
        text = path.read_text(encoding="utf-8").strip()
        title = _title_from_markdown(text, fallback=path.stem)
        issues.append(
            KnownIssue(
                issue_id=match.group(1),
                title=title,
                source=path.as_posix(),
                text=text,
            )
        )
    if not issues:
        raise ValueError(f"No known-issue Markdown files found in {directory}")
    return issues


def load_gold(path: str | Path) -> list[GoldQuery]:
    """Load and validate the B16 gold query JSON contract."""

    raw_items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw_items, list):
        raise ValueError("Gold set must be a JSON list.")

    queries: list[GoldQuery] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Gold item {index} must be an object.")
        query = _required_string(item, "query", index)
        note = _required_string(item, "note", index)
        relevant_raw = item.get("relevant")
        if (
            not isinstance(relevant_raw, list)
            or not relevant_raw
            or not all(isinstance(value, str) and value for value in relevant_raw)
        ):
            raise ValueError(f"Gold item {index} must have non-empty relevant IDs.")
        queries.append(
            GoldQuery(
                query=query,
                relevant=tuple(relevant_raw),
                note=note,
            )
        )
    return queries


def run_baseline(
    *,
    issues_dir: str | Path = "assets/known_issues",
    gold_path: str | Path = "assets/eval/retrieval_gold.json",
    top_k: int = 5,
) -> tuple[dict[str, float], list[QueryResult]]:
    """Run TF-IDF cosine retrieval and return aggregate metrics plus rows."""

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    issues = load_known_issues(issues_dir)
    gold = load_gold(gold_path)
    issue_ids = [issue.issue_id for issue in issues]
    issue_texts = [issue.text for issue in issues]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    issue_matrix = vectorizer.fit_transform(issue_texts)
    query_matrix = vectorizer.transform([item.query for item in gold])
    similarities = cosine_similarity(query_matrix, issue_matrix)

    results: list[QueryResult] = []
    for item, scores in zip(gold, similarities, strict=True):
        ranked_indexes = sorted(
            range(len(issue_ids)),
            key=lambda index: (-float(scores[index]), issue_ids[index]),
        )
        ranked_ids = tuple(issue_ids[index] for index in ranked_indexes[:top_k])
        top_scores = tuple(float(scores[index]) for index in ranked_indexes[:top_k])
        relevant = set(item.relevant)
        first_rank = next(
            (
                rank
                for rank, issue_id in enumerate(
                    (issue_ids[index] for index in ranked_indexes),
                    start=1,
                )
                if issue_id in relevant
            ),
            None,
        )
        reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank
        results.append(
            QueryResult(
                query=item.query,
                relevant=item.relevant,
                ranked_ids=ranked_ids,
                top_scores=top_scores,
                reciprocal_rank=reciprocal_rank,
                recall_at_1=_recall_at(ranked_ids[:1], relevant),
                recall_at_3=_recall_at(ranked_ids[:3], relevant),
            )
        )

    metrics = {
        "query_count": float(len(results)),
        "recall_at_1": _mean(result.recall_at_1 for result in results),
        "recall_at_3": _mean(result.recall_at_3 for result in results),
        "mrr": _mean(result.reciprocal_rank for result in results),
    }
    return metrics, results


def format_results(metrics: dict[str, float], results: list[QueryResult]) -> str:
    """Format metrics and per-query top hits for CLI output."""

    lines = [
        "TF-IDF retrieval baseline",
        f"queries={int(metrics['query_count'])}",
        f"recall@1={metrics['recall_at_1']:.3f}",
        f"recall@3={metrics['recall_at_3']:.3f}",
        f"MRR={metrics['mrr']:.3f}",
        "",
        "| # | relevant | top3 | rr | query |",
        "|---:|---|---|---:|---|",
    ]
    for index, result in enumerate(results, start=1):
        relevant = ", ".join(result.relevant)
        top3 = ", ".join(result.ranked_ids[:3])
        query = result.query.replace("|", "/")
        lines.append(
            f"| {index} | {relevant} | {top3} | {result.reciprocal_rank:.3f} | {query} |"
        )
    return "\n".join(lines)


def _title_from_markdown(text: str, fallback: str) -> str:
    first_line = text.splitlines()[0] if text else ""
    return first_line.removeprefix("# ").strip() if first_line.startswith("# ") else fallback


def _required_string(item: dict[str, Any], field_name: str, index: int) -> str:
    value = item.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Gold item {index} must have a non-empty {field_name}.")
    return value.strip()


def _recall_at(ranked_ids: tuple[str, ...], relevant: set[str]) -> float:
    return 1.0 if any(issue_id in relevant for issue_id in ranked_ids) else 0.0


def _mean(values: Any) -> float:
    numbers = list(values)
    return sum(numbers) / max(len(numbers), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF known-issue retrieval.")
    parser.add_argument("--issues-dir", default="assets/known_issues")
    parser.add_argument("--gold", default="assets/eval/retrieval_gold.json")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics, results = run_baseline(
        issues_dir=args.issues_dir,
        gold_path=args.gold,
        top_k=args.top_k,
    )
    print(format_results(metrics, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
