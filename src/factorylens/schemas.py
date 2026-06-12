"""Pydantic contracts for FactoryLens. Source of truth: docs/MVP_SPEC.md.

These models lock the API/tool contract for Phase 1 before any AI is added.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# --- API request / response ------------------------------------------------


class AnalysisRequest(BaseModel):
    """Parsed analysis request (files handled separately as uploads)."""

    question: str = Field(
        default="Analyze defects + root cause",
        description="User question for the copilot.",
    )
    category: str | None = Field(
        default=None, description="MVTec category hint, e.g. 'hazelnut'."
    )


class DefectRegion(BaseModel):
    bbox: list[int] = Field(description="[x1, y1, x2, y2] in pixels.")
    score: float = Field(ge=0.0, le=1.0)


class KnownIssueMatch(BaseModel):
    title: str
    snippet: str
    source: str
    similarity: float = Field(ge=0.0, le=1.0)


class RootCauseHypothesis(BaseModel):
    summary: str
    likely_causes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    request_id: str
    category: str | None = None
    anomaly_score: float = Field(ge=0.0, le=1.0, default=0.0)
    defect_label: str | None = None
    defect_regions: list[DefectRegion] = Field(default_factory=list)
    heatmap_path: str | None = None
    related_known_issues: list[KnownIssueMatch] = Field(default_factory=list)
    root_cause_hypothesis: RootCauseHypothesis | None = None
    next_actions: list[str] = Field(default_factory=list)
    report_markdown: str | None = None
    warnings: list[str] = Field(default_factory=list)
