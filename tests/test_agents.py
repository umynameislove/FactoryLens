from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from factorylens.agents.investigation import run_analysis
from factorylens.config import Settings
from factorylens.db.base import Base
from factorylens.db.models import TestLog
from factorylens.schemas import AnalysisResponse
from factorylens.tools import llm as llm_module


class _FakeEmbedder:
    dim = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]


@pytest.fixture
def db() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(
        TestLog(
            unit_id="u1",
            timestamp=datetime.now(timezone.utc),
            station="s1",
            measure_name="weight_g",
            measure_value=9.9,
            spec_low=1.0,
            spec_high=5.0,
            pass_fail="FAIL",
        )
    )
    session.commit()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_fallback_returns_valid_response(db, monkeypatch) -> None:
    # ép KHÔNG key -> A11/A12 dùng fallback, không gọi mạng
    no_key = Settings(database_url="sqlite://", openai_api_key=None, _env_file=None)
    monkeypatch.setattr(llm_module, "get_settings", lambda: no_key)

    result = run_analysis(
        "nonexistent.png",
        db,
        category="hazelnut",
        settings=no_key,
        embedder=_FakeEmbedder(),
        use_agent=False,
    )

    assert isinstance(result, AnalysisResponse)
    assert result.request_id
    assert result.category == "hazelnut"
    assert result.report_markdown        # report luôn có (fallback template)
    assert isinstance(result.warnings, list)