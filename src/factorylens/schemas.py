"""Pydantic contracts for FactoryLens. Source of truth: docs/MVP_SPEC.md.

These models lock the API/tool contract for Phase 1 before any AI is added.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal

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


class ImageDefectResult(BaseModel):
    """Structured output from the standalone image-defect tool."""

    anomaly_score: float = Field(ge=0.0, le=1.0)
    defect_label: str | None = None
    defect_regions: list[DefectRegion] = Field(default_factory=list)
    heatmap_path: str | None = None
    warnings: list[str] = Field(default_factory=list)


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


class UploadImageResponse(BaseModel):
    image_id: str
    stored_path: str
    content_type: str
    size_bytes: int = Field(ge=0)


class LogIngestError(BaseModel):
    row: int
    error: str


class UploadLogsResponse(BaseModel):
    rows_received: int = Field(ge=0)
    rows_ingested: int = Field(ge=0)
    rows_rejected: int = Field(ge=0)
    errors: list[LogIngestError] = Field(default_factory=list)

class QueryTestLogsInput(BaseModel):
    unit_id: str | None = None
    station: str | None = None
    measure_name: str | None = None
    pass_fail: Literal["PASS","FAIL"] | None = None
    failed_only: bool = False
    limit: int = Field(default = 100, ge = 1, le = 1000)

class TestLogRow(BaseModel):
    unit_id: str
    station: str
    measure_name: str
    measure_value: float
    spec_low: float | None = None
    spec_high: float | None = None
    pass_fail: str
    timestamp: str


class TestLogResult(BaseModel):
    generated_sql: str
    rows: list[TestLogRow] = Field(default_factory=list)
    row_count: int = 0
    failed_measures: list[str] = Field(default_factory=list)
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)


class KnownIssuesInput(BaseModel):
    """Tham số gọi tool retrieve known-issues."""
    query: str = Field(min_length=1, description="Triệu chứng/câu hỏi cần tra.")
    top_k: int = Field(default=4, ge=1, le=20)
    defect_type: str | None = Field(
        default=None,
        description="Lọc theo loại lỗi: crack/cut/hole/print. None = không lọc.",
    )


class KnownIssuesResult(BaseModel):
    query: str
    matches: list[KnownIssueMatch] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)