# Retrieval Baseline Eval

This folder defines the B16 retrieval evaluation set for known-issue search.
It is meant to be reused by Bao's A9 pgvector retrieval work, so the gold-set
shape should stay stable:

```json
{
  "query": "engineer-style symptom question",
  "relevant": ["KI-002"],
  "note": "short labeling rationale"
}
```

## Gold Set

- File: `retrieval_gold.json`
- Query count: 24
- Relevant IDs: real IDs from `assets/known_issues/*.md`
- Hard negatives: 13 queries are marked in `note` with `Hard negative`
- Leakage control: queries are paraphrased as engineer observations rather than
  copied from known-issue text.

The hard negatives intentionally include misleading lexical cues, for example a
query that says "knife station" while the true evidence is transfer contact, or
a query that says "belt residue" while the true cause is packaging label contact.

## Baseline Method

`scripts/eval_retrieval_baseline.py` is a standalone lexical baseline:

1. Read `assets/known_issues/*.md` except `INDEX.md`.
2. Read `assets/eval/retrieval_gold.json`.
3. Fit scikit-learn `TfidfVectorizer` on the known-issue documents.
4. Transform each query and rank documents by cosine similarity.
5. Report recall@1, recall@3, and MRR.

It does not import Lane A tools, pgvector, database code, or LangChain.

## Metrics

Measured locally with scikit-learn 1.9.0:

| Metric | Value |
|---|---:|
| Queries | 24 |
| Recall@1 | 0.833 |
| Recall@3 | 0.875 |
| MRR | 0.880 |

A9 (pgvector) của Bao cần ĐÁNH BẠI các số này.

## Run

```bash
python scripts/eval_retrieval_baseline.py
pytest tests/test_eval_retrieval_baseline.py -q
```

If `scikit-learn` is missing in the local virtual environment, install it before
running the metric script:

```bash
python -m pip install scikit-learn
```
