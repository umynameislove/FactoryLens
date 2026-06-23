"""LangChain agent ráp 5 tool FactoryLens (create_agent v1)."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from factorylens.tools.embedder import get_default_embedder

from sqlalchemy.orm import Session

from factorylens.config import Settings, get_settings
from factorylens.schemas import (
    AnalysisResponse,
    ImageDefectResult,
    KnownIssuesResult,
    TestLogResult,
)
from factorylens.schemas import KnownIssuesInput, QueryTestLogsInput
from factorylens.tools.analyze_image import analyze_image_defect
from factorylens.tools.query_logs import query_test_logs
from factorylens.tools.retrieve_known_issues import retrieve_known_issues
from factorylens.tools.root_cause import generate_root_cause_hypothesis
from factorylens.tools.report import generate_engineering_report

logger = logging.getLogger(__name__)


@dataclass
class Investigation:
    """'Sổ điều tra' — tool ghi kết quả structured vào đây."""

    image_path: str
    category: str | None
    db: Session
    settings: Settings
    embedder: object | None = None  # tiêm để test; None -> default
    image_result: ImageDefectResult | None = None
    log_result: TestLogResult | None = None
    known_issues: KnownIssuesResult | None = None
    warnings: list[str] = field(default_factory=list)


def _build_tools(inv: Investigation):
    from langchain.tools import tool

    @tool
    def inspect_image() -> str:
        """Run anomaly detection on the uploaded product image."""
        res = analyze_image_defect(inv.image_path, inv.category, inv.settings)
        inv.image_result = res
        inv.warnings.extend(res.warnings)
        return f"anomaly_score={res.anomaly_score:.3f}; defect_label={res.defect_label}"

    @tool
    def query_logs(unit_id: str | None = None) -> str:
        """Query manufacturing test logs; optionally filter by unit_id."""
        res = query_test_logs(
            QueryTestLogsInput(unit_id=unit_id, failed_only=True), inv.db
        )
        inv.log_result = res
        inv.warnings.extend(res.warnings)
        return res.summary or "no failing measures"

    @tool
    def search_known_issues(query: str) -> str:
        """Retrieve known issues similar to a defect description."""
        res = retrieve_known_issues(
            KnownIssuesInput(query=query),
            inv.db,
            embedder=inv.embedder or get_default_embedder(),
        )
        inv.known_issues = res
        inv.warnings.extend(res.warnings)
        return "; ".join(m.title for m in res.matches) or "no match"

    return [inspect_image, query_logs, search_known_issues]


def _run_agent(inv: Investigation, model: str) -> None:
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0.3)
    agent = create_agent(
        llm,
        tools=_build_tools(inv),
        system_prompt=(
            "You are FactoryLens. Investigate the product defect: first inspect the "
            "image, then query the failing test logs, then search known issues using "
            "the observed defect. Use ONLY the tools. Be concise."
        ),
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "Analyze this unit's defect."}]},
        {"recursion_limit": 12},  # chặn vòng lặp (bounded, ~5 bước tool)
    )


def _finalize(inv: Investigation) -> AnalysisResponse:
    """Chạy A11 + A12 trên dữ liệu đã thu, ráp response. Dùng cho cả agent path lẫn fallback."""
    image = inv.image_result or ImageDefectResult(anomaly_score=0.0)
    logs = inv.log_result or TestLogResult(generated_sql="")
    issues = inv.known_issues or KnownIssuesResult(query="")

    hypothesis = generate_root_cause_hypothesis(image, logs, issues)
    report_md = generate_engineering_report(image, logs, issues, hypothesis)

    return AnalysisResponse(
        request_id=str(uuid.uuid4()),
        category=inv.category,
        anomaly_score=image.anomaly_score,
        defect_label=image.defect_label,
        defect_regions=image.defect_regions,
        heatmap_path=image.heatmap_path,
        related_known_issues=issues.matches,
        root_cause_hypothesis=hypothesis,
        next_actions=hypothesis.likely_causes,
        report_markdown=report_md,
        warnings=list(dict.fromkeys(inv.warnings)),  # bỏ trùng, giữ thứ tự
    )


def run_analysis(
    image_path: str,
    db: Session,
    *,
    category: str | None = None,
    settings: Settings | None = None,
    embedder: object | None = None,
    use_agent: bool | None = None,
) -> AnalysisResponse:
    s = settings or get_settings()
    inv = Investigation(
        image_path=image_path, category=category, db=db, settings=s, embedder=embedder
    )

    has_key = bool(s.openai_api_key and s.openai_api_key.strip())
    want_agent = has_key if use_agent is None else use_agent

    if want_agent:
        try:
            _run_agent(inv, s.openai_model)
        except Exception:  # noqa: BLE001
            logger.warning("Agent run failed; falling back to deterministic pipeline.")
            inv.warnings.append("Agent unavailable; used deterministic pipeline.")
            _deterministic_collect(inv)
    else:
        inv.warnings.append("No API key; deterministic pipeline used.")
        _deterministic_collect(inv)

    return _finalize(inv)


def _deterministic_collect(inv: Investigation) -> None:
    inv.image_result = analyze_image_defect(inv.image_path, inv.category, inv.settings)
    inv.log_result = query_test_logs(QueryTestLogsInput(failed_only=True), inv.db)
    inv.known_issues = retrieve_known_issues(
        KnownIssuesInput(
            query=inv.image_result.defect_label or inv.category or "defect"
        ),
        inv.db,
        embedder=inv.embedder or get_default_embedder(),
    )
    inv.warnings.extend(inv.image_result.warnings)
    inv.warnings.extend(inv.log_result.warnings)
    inv.warnings.extend(inv.known_issues.warnings)
