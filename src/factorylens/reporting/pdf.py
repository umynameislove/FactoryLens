"""PDF rendering for FactoryLens engineering reports."""

from __future__ import annotations


def render_report_pdf(
    markdown_text: str,
    *,
    title: str = "FactoryLens Engineering Report",
) -> bytes:
    """Render ASCII/Latin-1 Markdown report text to PDF bytes using core fonts."""

    if not markdown_text.strip():
        raise ValueError("markdown_text must not be empty")

    import markdown
    from fpdf import FPDF

    html = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    pdf = FPDF()
    pdf.set_title(title)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.write_html(html)
    return bytes(pdf.output())
