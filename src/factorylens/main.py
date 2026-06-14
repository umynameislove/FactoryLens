"""FastAPI app — Phase 1 skeleton.

Runnable now: `/health` works; `/analyze` returns a contract-valid stub
(no AI yet). Vision (Phase 2) and the LangChain agent (Phase 3) plug in later.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from factorylens import __version__
from factorylens.api.uploads import router as uploads_router
from factorylens.db import get_db, init_db
from factorylens.schemas import AnalysisResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize DB objects when configured while preserving app liveness."""

    try:
        await run_in_threadpool(init_db)
    except (OSError, SQLAlchemyError, ValidationError):
        logger.warning(
            "Database initialization did not complete; readiness is unavailable."
        )
    yield


app = FastAPI(
    title="FactoryLens AI",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(uploads_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/readyz", response_model=None)
def readiness(db: Session = Depends(get_db)) -> JSONResponse:
    """Report database readiness without exposing connection details."""

    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "db": "down"},
        )
    return JSONResponse(content={"status": "ready", "db": "up"})


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
