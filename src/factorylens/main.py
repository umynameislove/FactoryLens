"""FastAPI app — Phase 1 skeleton.

Runnable now: `/health` works; `/analyze` returns a contract-valid stub
(no AI yet). Vision (Phase 2) and the LangChain agent (Phase 3) plug in later.
"""

from __future__ import annotations
 
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Form, UploadFile, File, HTTPException
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
from factorylens.config import Settings, get_settings
from factorylens.storage import (
    ImageStorageError,
    UploadValidationError,
    save_upload_image,
)
from factorylens.ingest.logs import LogValidationError, parse_and_ingest_logs
from factorylens.agents.investigation import run_analysis


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
    image: Annotated[UploadFile, File()],
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    test_logs: Annotated[UploadFile | None, File()] = None,
    question: Annotated[str, Form()] = "Analyze defects + root cause",
    category: Annotated[str | None, Form()] = None,
) -> AnalysisResponse:
    """Run the full pipeline: store image, ingest logs, analyze, return report."""
    try:
        _, stored_path, _, _ = save_upload_image(image, settings)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except ImageStorageError:
        raise HTTPException(status_code=500, detail="Image could not be stored.") from None

    if test_logs is not None:
        try:
            parse_and_ingest_logs(test_logs, db, settings)
        except LogValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    return await run_in_threadpool(
        run_analysis, stored_path, db, category=category
    )
