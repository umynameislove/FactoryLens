"""Lazy SQLAlchemy engine, session factory, and FastAPI dependency."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from typing import Any, cast

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from factorylens.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """Create the engine on first database use, not at module import time."""

    return create_engine(
        get_settings().database_url,
        pool_pre_ping=True,
    )


class _LazyEngine:
    """Proxy that preserves a conventional ``engine`` export without eager config."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_engine(), name)


engine = cast(Engine, _LazyEngine())
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Iterator[Session]:
    """Yield a request-scoped database session and always close it."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
