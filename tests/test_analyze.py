from __future__ import annotations

import io
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from factorylens.config import Settings
from factorylens.db.base import Base
from factorylens.db.session import get_db
from factorylens.main import app
from factorylens.schemas import AnalysisResponse
from factorylens.tools import llm as llm_module


@pytest.fixture
def client(monkeypatch) -> Iterator[TestClient]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    test_session = sessionmaker(bind=engine)

    def override_get_db() -> Iterator[Session]:
        with test_session() as session:
            yield session

    # ép no-key -> A11/A12 fallback, không gọi mạng
    no_key = Settings(database_url="sqlite://", openai_api_key=None, _env_file=None)
    monkeypatch.setattr(llm_module, "get_settings", lambda: no_key)

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (120, 120, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_analyze_returns_full_response(client: TestClient) -> None:
    csv = (
        "unit_id,timestamp,station,measure_name,measure_value,spec_low,spec_high,pass_fail\n"
        "u1,2026-06-20T09:00:00+00:00,s1,weight_g,9.9,1.0,5.0,FAIL\n"
    )
    response = client.post(
        "/analyze",
        files={
            "image": ("unit.png", _png_bytes(), "image/png"),
            "test_logs": ("logs.csv", csv, "text/csv"),
        },
        data={"category": "hazelnut", "question": "why defect?"},
    )

    assert response.status_code == 200
    parsed = AnalysisResponse.model_validate(response.json())
    assert parsed.request_id
    assert parsed.category == "hazelnut"
    assert parsed.report_markdown          # fallback report luôn có
    assert isinstance(parsed.warnings, list)


def test_analyze_requires_image(client: TestClient) -> None:
    response = client.post("/analyze", data={"category": "hazelnut"})
    assert response.status_code == 422   # thiếu ảnh -> Unprocessable