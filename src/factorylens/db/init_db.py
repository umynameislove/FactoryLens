"""Idempotent database initialization for PostgreSQL and pgvector."""

from sqlalchemy import text

from factorylens.db.base import Base
from factorylens.db.models import TestLog
from factorylens.db.session import get_engine


def init_db() -> None:
    """Enable pgvector on PostgreSQL and create all registered tables."""

    _ = TestLog.__table__
    db_engine = get_engine()
    with db_engine.begin() as connection:
        if connection.dialect.name == "postgresql":
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        Base.metadata.create_all(bind=connection)


if __name__ == "__main__":
    init_db()
