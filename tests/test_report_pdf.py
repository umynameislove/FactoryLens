import pytest

pytest.importorskip("fpdf")
pytest.importorskip("markdown")

from factorylens.reporting.pdf import render_report_pdf


def test_render_report_pdf_returns_pdf_bytes() -> None:
    markdown_report = """
# FactoryLens Report

## Summary

This is a **small** engineering report.

- anomaly score reviewed
- failed measure checked
"""

    rendered = render_report_pdf(markdown_report)

    assert isinstance(rendered, bytes)
    assert rendered.startswith(b"%PDF-")
    assert len(rendered) > 200


def test_render_report_pdf_rejects_empty() -> None:
    with pytest.raises(ValueError, match="markdown_text must not be empty"):
        render_report_pdf("   ")
