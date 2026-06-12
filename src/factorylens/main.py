"""FastAPI app — Phase 1 skeleton.

Runnable now: `/health` works; `/analyze` returns a contract-valid stub
(no AI yet). Vision (Phase 2) and the LangChain agent (Phase 3) plug in later.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Form, UploadFile

from factorylens import __version__
from factorylens.schemas import AnalysisResponse

app = FastAPI(title="FactoryLens AI", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    image: UploadFile | None = None,
    test_logs: UploadFile | None = None,
    question: str = Form(default="Analyze defects + root cause"),
    category: str | None = Form(default=None),
) -> AnalysisResponse:
    """Phase 1 stub: validates inputs and returns the contract shape.

    Tools (analyze_image_defect, query_test_logs, retrieve_known_issues,
    generate_root_cause_hypothesis, generate_engineering_report) are wired in
    Phase 2-3. For now we echo a contract-valid skeleton + a warning.
    """
    warnings = ["stub response: AI pipeline not implemented yet (Phase 2-3)"]
    if image is None:
        warnings.append("no image uploaded")
    return AnalysisResponse(
        request_id=str(uuid.uuid4()),
        category=category,
        warnings=warnings,
    )
