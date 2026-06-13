"""Public database-layer imports."""

from factorylens.db.base import Base
from factorylens.db.models import TestLog
from factorylens.db.session import SessionLocal, engine, get_db, get_engine


def init_db() -> None:
    """Initialize database objects without importing the CLI module eagerly."""

    from factorylens.db.init_db import init_db as run_init_db

    run_init_db()


__all__ = [
    "Base",
    "SessionLocal",
    "TestLog",
    "engine",
    "get_db",
    "get_engine",
    "init_db",
]
