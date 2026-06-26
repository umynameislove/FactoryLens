from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factorylens.agents.investigation import run_analysis  # noqa: E402
from factorylens.db.session import SessionLocal  # noqa: E402
from factorylens.schemas import AnalysisResponse  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    if isinstance(value, str):
        return [value] if value else []
    return [value]


def clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(number, 0.0), 1.0)


def resolve_existing_path(path_value: Any) -> Path | None:
    if not path_value:
        return None

    raw_path = Path(str(path_value)).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend([Path.cwd() / raw_path, PROJECT_ROOT / raw_path])

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def save_uploaded_file_to_temp(uploaded_file: Any) -> Path:
    suffix = Path(uploaded_file.name or "").suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        suffix = ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return Path(handle.name)


def find_demo_scenarios() -> dict[str, dict[str, Path | None]]:
    scenarios_root = PROJECT_ROOT / "assets" / "demo" / "scenarios"
    if not scenarios_root.is_dir():
        return {}

    scenarios: dict[str, dict[str, Path | None]] = {}
    for directory in sorted(path for path in scenarios_root.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        csv_files = sorted(path for path in directory.glob("*.csv") if path.is_file())
        scenario_doc = directory / "scenario.md"
        scenarios[directory.name] = {
            "image": images[0] if images else None,
            "csv": csv_files[0] if csv_files else None,
            "scenario": scenario_doc if scenario_doc.is_file() else None,
        }
    return scenarios


def render_input_preview(image_source: Any, csv_source: Any) -> None:
    st.subheader("Input Preview")
    image_column, logs_column = st.columns(2)

    with image_column:
        st.caption("Product image")
        if image_source is None:
            st.info("No image selected.")
        else:
            st.image(image_source, use_column_width=True)

    with logs_column:
        st.caption("CSV logs")
        if csv_source is None:
            st.info("No CSV logs selected.")
            return
        try:
            dataframe = _read_csv_dataframe(csv_source)
        except Exception as exc:  # noqa: BLE001
            st.warning("CSV preview unavailable.")
            st.caption(str(exc))
            return

        preview = dataframe.head(100)
        if "pass_fail" in preview.columns:
            styled = preview.style.apply(_highlight_fail_rows, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.dataframe(preview, use_container_width=True, hide_index=True)


def render_summary(result: AnalysisResponse) -> None:
    st.subheader("Analysis Summary")
    score = get_field(result, "anomaly_score")
    label = get_field(result, "defect_label") or "N/A"
    score_text = f"{float(score):.4f}" if isinstance(score, int | float) else "N/A"

    score_column, label_column = st.columns(2)
    score_column.metric("Anomaly score", score_text)
    label_column.metric("Defect label", label)


def render_visuals(result: AnalysisResponse, original_image: Any) -> None:
    st.subheader("Visual Evidence")
    original_column, heatmap_column = st.columns(2)

    with original_column:
        st.caption("Original image")
        if original_image is None:
            st.info("No original image available.")
        else:
            st.image(original_image, use_column_width=True)

    with heatmap_column:
        st.caption("Heatmap")
        heatmap_path = resolve_existing_path(get_field(result, "heatmap_path"))
        if heatmap_path is None:
            st.info("No heatmap available.")
        else:
            st.image(str(heatmap_path), use_column_width=True)


def render_known_issues(result: AnalysisResponse) -> None:
    st.subheader("Related Known Issues")
    issues = as_list(get_field(result, "related_known_issues"))
    if not issues:
        st.info("No related known issues found.")
        return

    for index, issue in enumerate(issues):
        title = get_field(issue, "title", "N/A")
        source = get_field(issue, "source", "N/A")
        snippet = get_field(issue, "snippet", "")
        similarity = clamp01(get_field(issue, "similarity", 0.0))

        if index:
            st.divider()
        st.markdown(f"**{title}**")
        st.caption(f"Source: {source}")
        if snippet:
            st.write(snippet)
        st.caption(f"Similarity: {similarity:.2f}")
        st.progress(similarity)


def render_hypothesis(result: AnalysisResponse) -> None:
    st.subheader("Root-Cause Hypothesis")
    hypothesis = get_field(result, "root_cause_hypothesis")
    if hypothesis is None:
        st.info("No root-cause hypothesis available.")
        return

    summary = get_field(hypothesis, "summary", "N/A")
    confidence = clamp01(get_field(hypothesis, "confidence", 0.0))
    likely_causes = as_list(get_field(hypothesis, "likely_causes"))
    evidence = as_list(get_field(hypothesis, "evidence"))

    st.write(summary)
    st.metric("Confidence", f"{confidence:.2f}")

    st.markdown("**Likely causes**")
    if likely_causes:
        for cause in likely_causes:
            st.markdown(f"- {cause}")
    else:
        st.write("N/A")

    if evidence:
        st.markdown("**Evidence**")
        for item in evidence:
            st.markdown(f"- {item}")


def render_report(result: AnalysisResponse) -> None:
    st.subheader("Engineering Report")
    report_markdown = get_field(result, "report_markdown")
    if not isinstance(report_markdown, str) or not report_markdown.strip():
        st.info("No report generated.")
        return

    st.markdown(report_markdown)
    st.download_button(
        "Download report (.md)",
        data=report_markdown,
        file_name="factorylens_report.md",
        mime="text/markdown",
    )


def render_warnings(result: AnalysisResponse) -> None:
    warnings = as_list(get_field(result, "warnings"))
    if not warnings:
        return

    st.subheader("Warnings")
    for warning in warnings:
        st.warning(str(warning))


def _read_csv_dataframe(csv_source: Any) -> pd.DataFrame:
    if isinstance(csv_source, bytes):
        return pd.read_csv(io.BytesIO(csv_source))
    return pd.read_csv(csv_source)


def _highlight_fail_rows(row: pd.Series) -> list[str]:
    value = str(row.get("pass_fail", "")).upper()
    color = "background-color: #fff4d6" if value == "FAIL" else ""
    return [color for _ in row]


def _selected_scenario_assets(
    scenarios: dict[str, dict[str, Path | None]],
    selected_name: str,
) -> dict[str, Path | None]:
    if selected_name == "None":
        return {"image": None, "csv": None, "scenario": None}
    return scenarios.get(selected_name, {"image": None, "csv": None, "scenario": None})


def _cleanup_temp_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def main() -> None:
    st.set_page_config(page_title="FactoryLens AI Dashboard", layout="wide")
    st.title("FactoryLens AI Dashboard")
    st.caption(
        "Upload product evidence, run analysis, and review technical findings."
    )

    scenarios = find_demo_scenarios()
    scenario_names = ["None", *scenarios.keys()]
    selected_scenario = st.sidebar.selectbox("Demo scenario", scenario_names)
    scenario_assets = _selected_scenario_assets(scenarios, selected_scenario)

    uploaded_image = st.sidebar.file_uploader(
        "Upload product image",
        type=["png", "jpg", "jpeg"],
    )
    uploaded_csv = st.sidebar.file_uploader("Upload CSV logs", type=["csv"])
    question = st.sidebar.text_area(
        "Investigation question",
        value="Analyze defects + root cause",
        height=90,
    )
    category = st.sidebar.selectbox("Category", ["hazelnut"], index=0)
    analyze_clicked = st.sidebar.button("Analyze", type="primary")

    image_preview: bytes | Path | None = (
        uploaded_image.getvalue()
        if uploaded_image is not None
        else scenario_assets["image"]
    )
    csv_preview: bytes | Path | None = (
        uploaded_csv.getvalue() if uploaded_csv is not None else scenario_assets["csv"]
    )

    render_input_preview(image_preview, csv_preview)

    if not analyze_clicked:
        return

    if image_preview is None:
        st.error("Product image is required.")
        return

    temp_image_path: Path | None = None
    analysis_image_path: Path
    db = None
    try:
        if uploaded_image is not None:
            temp_image_path = save_uploaded_file_to_temp(uploaded_image)
            analysis_image_path = temp_image_path
        else:
            analysis_image_path = Path(image_preview)

        # thêm cấu hình DB, memory bank, known-issue vectors vô đây nha cu.
        db = SessionLocal()
        # TODO(Bao): pass question and uploaded CSV path into run_analysis when
        # Lane A supports those inputs.
        _ = question
        result = run_analysis(
            str(analysis_image_path),
            db,
            category=category,
        )
    except Exception:  # noqa: BLE001
        st.error("Analysis failed.")
        return
    finally:
        if db is not None:
            db.close()
        _cleanup_temp_file(temp_image_path)

    render_summary(result)
    render_visuals(result, image_preview)
    render_known_issues(result)
    render_hypothesis(result)
    render_report(result)
    render_warnings(result)


if __name__ == "__main__":
    main()
