# FactoryLens AI — MVP Spec (1 page)

- Version: `0.1.0` · Status: draft · Date: 2026-06-12
- Owner: Tran Quoc Bao · Scope: Phase 1–3 local MVP (one end-to-end demo)
- Positioning: deployed AI copilot for industrial visual defect analysis,
  root-cause investigation, and engineering report generation.

## Objective

One sentence: given a product image (+ optional test logs and a question),
return a defect/anomaly assessment, related known issues, a root-cause
hypothesis, next actions, and a structured engineering report — orchestrated by
a LangChain agent over 5 bounded tools.

## Scope (MVP) / Non-goals

**In scope (Phase 1–3, local):** FastAPI backend; upload image + logs; SQLite
DB; MVTec subset loader; anomaly score + simple heatmap/bbox; SQL over test
logs; known-issue vector retrieval; LangChain agent calling 5 tools with
Pydantic structured output; Markdown report; 1 demo video.

**Non-goals (do NOT build for MVP):** SOTA detector / heavy training; Azure
deploy, React UI, GitHub Actions, full test suite (these are Phase 4 polish, do
not gate the demo); multi-tenant auth; real-time streaming. Agent is bounded —
no free-form tool creation.

## Input schema (`AnalysisRequest`)

| field | type | notes |
|---|---|---|
| `image` | file (PNG/JPG) | required; single product image |
| `test_logs` | file (CSV) | optional; columns below |
| `known_issues` | file(s) (md/txt/pdf) | optional; indexed into vector store |
| `question` | str | optional; defaults to "Analyze defects + root cause" |
| `category` | str | MVTec category hint, e.g. `"hazelnut"` |

`test_logs` CSV columns: `unit_id, timestamp, station, measure_name,
measure_value, spec_low, spec_high, pass_fail`.

## Output schema (`AnalysisResponse`)

`request_id, category, anomaly_score (0–1), defect_label, defect_regions
[bbox + score], heatmap_path, related_known_issues [{title, snippet, source,
similarity}], root_cause_hypothesis {summary, likely_causes[], confidence,
evidence[]}, next_actions [str], report_markdown, warnings []`.

All tool outputs are Pydantic models. If no API key → deterministic fallback
(template hypothesis + retrieval-only report), flagged in `warnings`.

## 5 Agent tool signatures (bounded, Pydantic I/O)

```python
def analyze_image_defect(image_path: str, category: str | None) -> ImageDefectResult
# -> anomaly_score, defect_label, regions[bbox,score], heatmap_path

def query_test_logs(question: str, log_table: str) -> TestLogResult
# -> generated_sql, rows[], summary, failed_measures[]  (read-only SQL, allow-listed)

def retrieve_known_issues(query: str, k: int = 5) -> KnownIssuesResult
# -> matches[{title, snippet, source, similarity}]  (vector search, FAISS/Chroma)

def generate_root_cause_hypothesis(ctx: RootCauseInput) -> RootCauseResult
# in: image result + log result + retrieved issues -> summary, likely_causes[], confidence, evidence[]

def generate_engineering_report(ctx: ReportInput) -> ReportResult
# -> report_markdown (sections: summary, evidence, hypothesis, next actions, appendix)
```

## Expected demo flow

1. User uploads a `hazelnut` defect image + sample test-logs CSV + a question.
2. `analyze_image_defect` → anomaly score + heatmap overlay.
3. `query_test_logs` → SQL finds out-of-spec measures for that unit.
4. `retrieve_known_issues` → top matching SOP/known-issue snippets.
5. `generate_root_cause_hypothesis` → ranked likely causes with cited evidence.
6. `generate_engineering_report` → Markdown report (later PDF) + API JSON.
7. Streamlit dashboard renders image, heatmap, table, hypothesis, report.

## Dataset subset (CONFIRMED 2026-06-13)

MVTec AD, **1 category to start: `hazelnut`** (clear defect types: crack, cut,
hole, print) + 10–20 demo images. Add a 2nd category (`metal_nut` or `bottle`)
only if Phase 2 baseline works. **Confirmed by Bao.**

## Acceptance criteria (MVP demo = PASS)

- Backend runs locally (Docker) and accepts image+CSV upload.
- All 5 tools return valid Pydantic objects on the demo input.
- Agent produces one end-to-end `AnalysisResponse` with a non-empty Markdown
  report citing at least one retrieved known issue and one log finding.
- Fallback path works with no API key (flagged in `warnings`).
- 1 screen-recorded demo video of the full flow.

## Risks

- P1 — CV baseline weak/noisy on subset → mitigate: pretrained-embedding +
  PatchCore/PaDiM-style scoring, accept "approximate" framing, no SOTA claim.
- P1 — agent loops / unbounded tool calls → mitigate: fixed tool set, max
  iterations, structured output enforced.
- P2 — scope creep into Azure/React/tests → keep as Phase 4, do not gate demo.
- P2 — no Azure student credit confirmed → demo local Docker first. **Confirm.**
- P3 — CrossRoute close-out slips if FactoryLens starts too deep → keep MVP
  narrow until CrossRoute is chốt.

## First step (smallest high-leverage)

Phase 1 scaffold: FastAPI app + `/analyze` stub + `AnalysisRequest/Response`
Pydantic models + SQLite schema for test logs + MVTec loader + Dockerfile.
No AI yet — lock the contract this spec defines, then Phase 2 (vision) → Phase 3
(agent).

## Open questions to confirm

1. MVTec category: start with `hazelnut`? (or your preferred category)
2. Azure student credit available? (decides if Phase 4 deploy is realistic)
3. LLM provider for Phase 3: OpenAI / Gemini / Azure OpenAI / fallback-only?

## Verdict

PASS WITH GAP — spec is executable for Phase 1; gaps are the 3 open questions
above (dataset category, Azure credit, LLM provider). No code written yet.
