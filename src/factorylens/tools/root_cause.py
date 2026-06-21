"""Root-cause hypothesis tool with OpenAI and deterministic fallback paths."""

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


def generate_root_cause_hypothesis(
    image_result: ImageDefectResult,
    log_result: TestLogResult,
    known_issues: KnownIssuesResult,
    *,
    llm: LLMClient | None = None,
    model: str | None = None,
    temperature: float = 0.3,
) -> RootCauseHypothesis:
    try:
        client = llm if llm is not None else get_llm_client()
    except Exception:
        logger.warning("Root-cause LLM client unavailable; using fallback.")
        return _fallback_hypothesis(image_result, log_result, known_issues)

    if client is None:
        return _fallback_hypothesis(image_result, log_result, known_issues)

    system, user = _build_prompts(image_result, log_result, known_issues)
    try:
        selected_model = model or get_settings().openai_model
        raw = client.complete(
            system,
            user,
            model=selected_model,
            temperature=temperature,
        )
        return RootCauseHypothesis.model_validate_json(raw)
    except Exception:
        logger.warning("Root-cause LLM generation failed; using fallback.")
        return _fallback_hypothesis(image_result, log_result, known_issues)


def _build_prompts(
    image_result: ImageDefectResult,
    log_result: TestLogResult,
    known_issues: KnownIssuesResult,
) -> tuple[str, str]:
    system = (
        "You are FactoryLens, an industrial root-cause assistant. Use ONLY the "
        "provided evidence. Do NOT invent measurements, defects, known issues, "
        "or sources. Lower confidence when evidence is weak, missing, or "
        "contradictory. "
        "Each evidence item MUST reference a specific provided datum (a measure "
        "name, a known-issue source ID, or the anomaly score). If there is no "
        "failed measure and no relevant known issue, return low confidence and "
        "state the evidence is insufficient. "
        "Return a JSON object with EXACTLY these keys: "
        "summary (str), likely_causes (array of str), confidence (number 0..1), "
        "evidence (array of short strings citing the data)."
    )

    issue_lines = []
    for match in known_issues.matches[:3]:
        issue_lines.append(
            "- "
            f"title={match.title}; "
            f"similarity={match.similarity:.3f}; "
            f"source={match.source}; "
            f"snippet={match.snippet}"
        )
    issues_text = "\n".join(issue_lines) if issue_lines else "- none"
    failed_measures = ", ".join(log_result.failed_measures) or "none"

    user = (
        "Image evidence:\n"
        f"- anomaly_score={image_result.anomaly_score:.4f}\n"
        f"- defect_label={image_result.defect_label or 'none'}\n\n"
        "Log evidence:\n"
        f"- failed_measures={failed_measures}\n"
        f"- summary={log_result.summary or 'none'}\n\n"
        "Top known issues:\n"
        f"{issues_text}"
    )
    return system, user


def _fallback_hypothesis(
    image_result: ImageDefectResult,
    log_result: TestLogResult,
    known_issues: KnownIssuesResult,
) -> RootCauseHypothesis:
    failed_count = len(log_result.failed_measures)
    likely_causes = [match.title for match in known_issues.matches[:2]]
    evidence = [f"Failed measure: {name}" for name in log_result.failed_measures]
    if known_issues.matches:
        closest = known_issues.matches[0]
        evidence.append(f"Closest known issue: {closest.title} ({closest.source})")

    return RootCauseHypothesis(
        summary=(
            "Heuristic fallback hypothesis based on "
            f"anomaly score {image_result.anomaly_score:.4f} and "
            f"{failed_count} failed measure(s)."
        ),
        likely_causes=likely_causes,
        confidence=0.3,
        evidence=evidence,
    )
