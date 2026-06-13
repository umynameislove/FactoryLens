"""Minimal smoke tests proving the skeleton runs and honors the contract."""

from collections.abc import Iterator
from typing import cast

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from factorylens.db.session import get_db
from factorylens.main import app
from factorylens.schemas import AnalysisResponse

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_reports_database_up() -> None:
    test_engine = create_engine("sqlite://")
    test_session = sessionmaker(bind=test_engine)

    def override_get_db() -> Iterator[Session]:
        with test_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        response = client.get("/readyz")
    finally:
        app.dependency_overrides.clear()
        test_engine.dispose()

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "db": "up"}


def test_readyz_hides_database_error_details() -> None:
    class FailingSession:
        def execute(self, _statement: object) -> None:
            raise SQLAlchemyError("contains-sensitive-database-details")

    def override_get_db() -> Iterator[Session]:
        yield cast(Session, FailingSession())

    app.dependency_overrides[get_db] = override_get_db
    try:
        response = client.get("/readyz")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "db": "down"}
    assert "sensitive" not in response.text


def test_analyze_returns_contract_shape() -> None:
    r = client.post(
        "/analyze",
        data={"question": "why defect?", "category": "hazelnut"},
    )
    assert r.status_code == 200
    # Response must validate against the locked contract.
    parsed = AnalysisResponse.model_validate(r.json())
    assert parsed.category == "hazelnut"
    assert parsed.request_id
    assert any("stub" in w for w in parsed.warnings)
