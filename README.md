# FactoryLens AI

AI copilot for **industrial visual defect analysis, root-cause investigation,
and engineering report generation**. Upload a product image (+ optional test
logs and known-issue docs) and a question; get an anomaly assessment, related
known issues, a root-cause hypothesis, next actions, and a structured
engineering report.

> Status: **Phase 1 skeleton** (runnable FastAPI + locked Pydantic contract,
> no AI yet). See `docs/MVP_SPEC.md` for the full input/output contract and the
> 5 agent tools.

## Stack (target)

LangChain · FastAPI · PostgreSQL + pgvector · OpenAI · Docker. Dataset: MVTec AD
subset (`hazelnut` first).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn factorylens.main:app --reload   # http://127.0.0.1:8000/health
pytest
```

`POST /analyze` (multipart: `image`, `test_logs`, `question`, `category`)
currently returns a contract-valid stub.

## Layout

```
working/
  src/factorylens/   # app code (schemas.py = locked contract, main.py = API)
  tests/             # smoke + contract tests
  docs/MVP_SPEC.md   # product + tool contract (source of truth)
```

Internal planning/tracking lives **outside this repo** in `../_tracking/`
(never committed).

## Contributing rule

No direct commits to `main`. Flow: **issue → branch → small commits → PR
(goal / changes / tests / risks + label) → review → squash merge.**
