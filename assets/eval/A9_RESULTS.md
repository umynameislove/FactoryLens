# A9 Retrieval Evaluation (pgvector vs TF-IDF baseline)

Same gold set as B16 (`retrieval_gold.json`, 24 queries), same metric definitions
(recall@k = at least one relevant in top-k; MRR over the full ranking).

| Metric | B16 baseline (TF-IDF) | A9 (pgvector, MiniLM) |
|---|---:|---:|
| recall@1 | 0.833 | 0.708 |
| recall@3 | 0.875 | **1.000** |
| MRR      | 0.880 | 0.840 |

## Reading
- **A9 wins where it matters for RAG (recall@3 = 1.000).** The correct known-issue is
  always within the top-3 the agent feeds to the LLM; the lexical baseline misses it
  12.5% of the time.
- **A9's lower recall@1 is semantic conflation within a defect family.** Closely
  related issues (three crack mechanisms KI-001/002/009; two hole causes KI-005/011)
  embed very close together, so MiniLM sometimes ranks a sibling first. TF-IDF keys on
  distinctive tokens and occasionally pins the exact one at rank 1.
- **Shallow vs deep misses.** A9's misses stay at rank 2–3 (still in top-3); the
  baseline buries some answers deep — e.g. query 6 at rank 8, query 22 at rank 4 —
  which then fall out of the top-3 the LLM sees.
- **A9 rescues the lexically-misleading hard negatives:** query 6 (rank 8 → 1),
  query 22 (rank 4 → 1), query 23 (rank 4 → top-3).
- **Implication for FactoryLens.** The pipeline retrieves top-k (k = 3–4) and passes
  it to the LLM for root-cause reasoning, so recall@3 is the operating metric and A9
  is the better retriever. Distinguishing the exact mechanism within a family (where
  A9's rank-1 dips) is deferred to the LLM using log + image evidence.

Reproduce: `python scripts/eval_retrieval.py` (Postgres on localhost, after ingest).
