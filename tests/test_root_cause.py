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
from factorylens.tools.root_cause import generate_root_cause_hypothesis


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
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "model": model,
                "temperature": temperature,
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
            KnownIssueMatch(
                title="Extra issue",
                snippet="This fourth match should not be included in the prompt.",
                source="KI-extra.md",
                similarity=0.4,
            ),
        ],
    )


def test_valid_json_from_fake_llm_is_parsed() -> None:
    fake = _FakeLLM(
        """
        {
          "summary": "Likely transfer contamination with one failed measure.",
          "likely_causes": ["Dirty transfer contact"],
          "confidence": 0.67,
          "evidence": ["weight_g failed", "KI-008 matched"]
        }
        """
    )

    result = generate_root_cause_hypothesis(
        _image_result(),
        _log_result(),
        _known_issues(),
        llm=fake,
        model="test-model",
        temperature=0.1,
    )

    assert isinstance(result, RootCauseHypothesis)
    assert result.summary.startswith("Likely transfer contamination")
    assert result.likely_causes == ["Dirty transfer contact"]
    assert result.confidence == 0.67
    assert "weight_g failed" in result.evidence
    assert fake.calls[0]["model"] == "test-model"
    assert fake.calls[0]["temperature"] == 0.1
    assert "Use ONLY the provided evidence" in str(fake.calls[0]["system"])
    assert "weight_g" in str(fake.calls[0]["user"])
    assert "Dirty transfer contact" in str(fake.calls[0]["user"])
    assert "Extra issue" not in str(fake.calls[0]["user"])


def test_no_key_uses_deterministic_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: Settings(
            database_url="sqlite://",
            openai_api_key=None,
            _env_file=None,
        ),
    )

    result = generate_root_cause_hypothesis(
        _image_result(),
        _log_result(),
        _known_issues(),
        llm=None,
    )

    assert result.confidence == 0.3
    assert result.likely_causes == [
        "Dirty transfer contact",
        "Sorter guide contact",
    ]
    assert "weight_g" in " ".join(result.evidence)
    assert "Heuristic fallback" in result.summary


def test_invalid_llm_json_falls_back() -> None:
    result = generate_root_cause_hypothesis(
        _image_result(),
        _log_result(),
        _known_issues(),
        llm=_FakeLLM("not json"),
        model="test-model",
    )

    assert result.confidence == 0.3
    assert "weight_g" in " ".join(result.evidence)
