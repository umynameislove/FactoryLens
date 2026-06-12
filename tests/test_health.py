"""Minimal smoke tests proving the skeleton runs and honors the contract."""

from fastapi.testclient import TestClient

from factorylens.main import app
from factorylens.schemas import AnalysisResponse

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_returns_contract_shape() -> None:
    r = client.post("/analyze", data={"question": "why defect?", "category": "hazelnut"})
    assert r.status_code == 200
    # Response must validate against the locked contract.
    parsed = AnalysisResponse.model_validate(r.json())
    assert parsed.category == "hazelnut"
    assert parsed.request_id
    assert any("stub" in w for w in parsed.warnings)
