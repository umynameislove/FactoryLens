from __future__ import annotations

from factorylens.config import Settings
from factorylens.schemas import (
    ImageDefectResult,
    KnownIssueMatch,
    KnownIssuesResult,
    RootCauseHypothesis,
    TestLogResult as LogResult,
)
from factorylens.tools import llm as llm_module
from factorylens.tools.report import generate_engineering_report


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
        json_mode: bool = True,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "model": model,
                "temperature": temperature,
                "json_mode": json_mode,
            }
        )
        return self.response


def _image_result() -> ImageDefectResult:
    return ImageDefectResult(
        anomaly_score=0.82,
        defect_label="hazelnut:defect",
        defect_regions=[],
        heatmap_path="heatmaps/sample.png",
    )


def _log_result() -> LogResult:
    return LogResult(
        generated_sql="SELECT ...",
        rows=[],
        row_count=3,
        failed_measures=["weight_g"],
        summary="Found 3 rows; 1 FAIL across measures: weight_g.",
    )


def _known_issues() -> KnownIssuesResult:
    return KnownIssuesResult(
        query="hazelnut defect",
        matches=[
            KnownIssueMatch(
                title="Dirty transfer contact",
                snippet="Transfer contact contamination can leave visible marks.",
                source="KI-008-print-dirty-contact.md",
                similarity=0.91,
            ),
            KnownIssueMatch(
                title="Sorter guide contact",
                snippet="A worn guide may cut or scrape the shell.",
                source="KI-004-cut-sorter-guide.md",
                similarity=0.72,
            ),
            KnownIssueMatch(
                title="Impact chip",
                snippet="Impact can create a shell void or chip.",
                source="KI-006-hole-impact-chip.md",
                similarity=0.55,
            ),
        ],
    )


def _hypothesis() -> RootCauseHypothesis:
    return RootCauseHypothesis(
        summary="Likely transfer contamination with a supporting log failure.",
        likely_causes=["Dirty transfer contact"],
        confidence=0.67,
        evidence=["weight_g failed", "KI-008 matched"],
    )


def test_fake_llm_markdown_is_returned_unchanged() -> None:
    markdown = (
        "## Summary\nok\n\n"
        "## Evidence\nok\n\n"
        "## Root-cause hypothesis\nok\n\n"
        "## Recommended next actions\nok\n\n"
        "## Appendix\nok"
    )
    fake = _FakeLLM(markdown)

    result = generate_engineering_report(
        _image_result(),
        _log_result(),
        _known_issues(),
        _hypothesis(),
        llm=fake,
        model="test-model",
        temperature=0.1,
    )

    assert result == markdown
    assert fake.calls[0]["json_mode"] is False
    assert "## Summary" in str(fake.calls[0]["system"])
    assert "## Appendix" in str(fake.calls[0]["system"])
    assert "weight_g" in str(fake.calls[0]["user"])
    assert _hypothesis().summary in str(fake.calls[0]["user"])


def test_no_key_uses_deterministic_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: Settings(
            database_url="sqlite://",
            openai_api_key=None,
            _env_file=None,
        ),
    )

    result = generate_engineering_report(
        _image_result(),
        _log_result(),
        _known_issues(),
        _hypothesis(),
        llm=None,
    )

    for section in [
        "## Summary",
        "## Evidence",
        "## Root-cause hypothesis",
        "## Recommended next actions",
        "## Appendix",
    ]:
        assert section in result
    assert "weight_g" in result
    assert _hypothesis().summary in result


def test_blank_llm_response_falls_back() -> None:
    result = generate_engineering_report(
        _image_result(),
        _log_result(),
        _known_issues(),
        _hypothesis(),
        llm=_FakeLLM("   "),
        model="test-model",
    )

    assert "Heuristic report (LLM unavailable)" in result
    assert "weight_g" in result
    assert _hypothesis().summary in result
