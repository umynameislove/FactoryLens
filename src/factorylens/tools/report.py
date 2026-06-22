"""Engineering-report generation tool with LLM and deterministic fallback."""

from __future__ import annotations

import logging

from factorylens.config import get_settings
from factorylens.schemas import (
    ImageDefectResult,
    KnownIssuesResult,
    RootCauseHypothesis,
    TestLogResult,
)
from factorylens.tools.llm import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


_SECTIONS = (
    "## Summary",
    "## Evidence",
    "## Root-cause hypothesis",
    "## Recommended next actions",
    "## Appendix",
)


def generate_engineering_report(
    image_result: ImageDefectResult,
    log_result: TestLogResult,
    known_issues: KnownIssuesResult,
    hypothesis: RootCauseHypothesis,
    *,
    llm: LLMClient | None = None,
    model: str | None = None,
    temperature: float = 0.3,
) -> str:
    try:
        client = llm if llm is not None else get_llm_client()
    except Exception:  # noqa: BLE001
        logger.warning("Engineering-report LLM client unavailable; using fallback.")
        return _fallback_report(image_result, log_result, known_issues, hypothesis)

    if client is None:
        return _fallback_report(image_result, log_result, known_issues, hypothesis)

    system, user = _build_prompts(
        image_result,
        log_result,
        known_issues,
        hypothesis,
    )
    try:
        report = client.complete(
            system,
            user,
            model=model or get_settings().openai_model,
            temperature=temperature,
            json_mode=False,
        )
        if not report.strip():
            return _fallback_report(
                image_result,
                log_result,
                known_issues,
                hypothesis,
            )
        return report
    except Exception:  # noqa: BLE001
        logger.warning("Engineering-report LLM generation failed; using fallback.")
        return _fallback_report(image_result, log_result, known_issues, hypothesis)


def _build_prompts(
    image_result: ImageDefectResult,
    log_result: TestLogResult,
    known_issues: KnownIssuesResult,
    hypothesis: RootCauseHypothesis,
) -> tuple[str, str]:
    section_list = ", ".join(_SECTIONS)
    system = (
        "Write a concise industrial defect engineering report in Markdown using "
        "ONLY the provided data. Do NOT invent measurements, defects, known "
        "issues, or sources. If evidence is weak, say so. The report MUST "
        f"contain exactly these sections in order: {section_list}."
    )

    issue_lines = []
    for match in known_issues.matches[:3]:
        issue_lines.append(
            "- "
            f"title={match.title}; "
            f"source={match.source}; "
            f"similarity={match.similarity:.3f}; "
            f"snippet={match.snippet}"
        )

    user = (
        "Image evidence:\n"
        f"- anomaly_score={image_result.anomaly_score:.4f}\n"
        f"- defect_label={image_result.defect_label or 'none'}\n\n"
        "Log evidence:\n"
        f"- failed_measures={_csv_or_none(log_result.failed_measures)}\n"
        f"- summary={log_result.summary or 'none'}\n\n"
        "Top known issues:\n"
        f"{chr(10).join(issue_lines) if issue_lines else '- none'}\n\n"
        "Root-cause hypothesis:\n"
        f"- summary={hypothesis.summary}\n"
        f"- likely_causes={_csv_or_none(hypothesis.likely_causes)}\n"
        f"- confidence={hypothesis.confidence:.3f}\n"
        f"- evidence={_csv_or_none(hypothesis.evidence)}"
    )
    return system, user


def _fallback_report(
    image_result: ImageDefectResult,
    log_result: TestLogResult,
    known_issues: KnownIssuesResult,
    hypothesis: RootCauseHypothesis,
) -> str:
    failed_count = len(log_result.failed_measures)
    closest_issue = (
        f"{known_issues.matches[0].title} ({known_issues.matches[0].source})"
        if known_issues.matches
        else "none"
    )
    return "\n\n".join(
        [
            "## Summary\n"
            "Heuristic report (LLM unavailable). "
            f"Anomaly score {image_result.anomaly_score:.4f}; "
            f"defect label {image_result.defect_label or 'none'}; "
            f"{failed_count} failed measure(s); "
            f"closest known issue: {closest_issue}.",
            "## Evidence\n"
            f"- Image: anomaly_score={image_result.anomaly_score:.4f}; "
            f"defect_label={image_result.defect_label or 'none'}\n"
            f"- Logs: summary={log_result.summary or 'none'}; "
            f"failed_measures={_csv_or_none(log_result.failed_measures)}\n"
            f"{_known_issue_bullets(known_issues)}",
            "## Root-cause hypothesis\n"
            f"- Summary: {hypothesis.summary}\n"
            f"- Likely causes: {_csv_or_none(hypothesis.likely_causes)}\n"
            f"- Confidence: {hypothesis.confidence:.3f}\n"
            f"- Evidence: {_csv_or_none(hypothesis.evidence)}",
            f"## Recommended next actions\n{_next_action_bullets(hypothesis)}",
            "## Appendix\n"
            f"- Generated SQL: `{log_result.generated_sql or 'none'}`\n"
            f"- Heatmap path: `{image_result.heatmap_path or 'none'}`",
        ]
    )


def _known_issue_bullets(known_issues: KnownIssuesResult) -> str:
    if not known_issues.matches:
        return "- Known issues: none"
    return "\n".join(
        "- Known issue: "
        f"{match.title} ({match.source}, similarity={match.similarity:.3f}) - "
        f"{match.snippet}"
        for match in known_issues.matches[:3]
    )


def _next_action_bullets(hypothesis: RootCauseHypothesis) -> str:
    if not hypothesis.likely_causes:
        return "- Manual review required."
    return "\n".join(
        f"- Investigate and verify: {cause}" for cause in hypothesis.likely_causes
    )


def _csv_or_none(values: list[str]) -> str:
    return ", ".join(values) if values else "none"
