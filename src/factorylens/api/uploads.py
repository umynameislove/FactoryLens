"""Upload endpoints for product images and manufacturing test logs."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from factorylens.config import Settings, get_settings
from factorylens.db import get_db
from factorylens.ingest.logs import LogValidationError, parse_and_ingest_logs
from factorylens.schemas import UploadImageResponse, UploadLogsResponse
from factorylens.storage import (
    ImageStorageError,
    UploadValidationError,
    save_upload_image,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("/image", response_model=UploadImageResponse)
def upload_image(
    file: Annotated[UploadFile, File(...)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UploadImageResponse:
    try:
        image_id, stored_path, content_type, size_bytes = save_upload_image(
            file,
            settings,
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except ImageStorageError as exc:
        logger.error(
            "Image upload storage failed (%s).",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Image could not be stored.",
        ) from None

    return UploadImageResponse(
        image_id=image_id,
        stored_path=stored_path,
        content_type=content_type,
        size_bytes=size_bytes,
    )


@router.post("/logs", response_model=UploadLogsResponse)
def upload_logs(
    file: Annotated[UploadFile, File(...)],
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UploadLogsResponse:
    try:
        return parse_and_ingest_logs(file, db, settings)
    except LogValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except SQLAlchemyError as exc:
        logger.error(
            "Log ingestion database write failed (%s).",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Logs could not be ingested.",
        ) from None
