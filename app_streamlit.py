from __future__ import annotations

import hashlib
import html
import io
import os
import re
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
EMPTY_SCENARIO = "None"
RESULT_STATE_KEY = "factorylens_analysis_result"
RESULT_SIGNATURE_KEY = "factorylens_analysis_signature"
RESULT_IMAGE_KEY = "factorylens_analysis_image"
SCENARIO_SOURCE_PATTERN = re.compile(r"Primary original source:\s*`([^`]+)`")


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


def _scenario_source_image(scenario_doc: Path) -> Path | None:
    try:
        scenario_text = scenario_doc.read_text(encoding="utf-8")
    except OSError:
        return None

    match = SCENARIO_SOURCE_PATTERN.search(scenario_text)
    if match is None:
        return None
    return resolve_existing_path(match.group(1))


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
        documented_image = (
            _scenario_source_image(scenario_doc) if scenario_doc.is_file() else None
        )
        scenarios[directory.name] = {
            "image": images[0] if images else documented_image,
            "csv": csv_files[0] if csv_files else None,
            "scenario": scenario_doc if scenario_doc.is_file() else None,
        }
    return scenarios


def _selected_scenario_assets(
    scenarios: dict[str, dict[str, Path | None]],
    selected_name: str,
) -> dict[str, Path | None]:
    if selected_name == EMPTY_SCENARIO:
        return {"image": None, "csv": None, "scenario": None}
    return scenarios.get(
        selected_name,
        {"image": None, "csv": None, "scenario": None},
    )


def _cleanup_temp_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _read_csv_dataframe(csv_source: Any) -> pd.DataFrame:
    if isinstance(csv_source, bytes):
        return pd.read_csv(io.BytesIO(csv_source))
    return pd.read_csv(csv_source)


def _highlight_fail_rows(row: pd.Series) -> list[str]:
    value = str(row.get("pass_fail", "")).upper()
    style = "background-color: #3b1010; color: #ffffff; font-weight: 700"
    return [style if value == "FAIL" else "" for _ in row]


def _input_signature(
    selected_scenario: str,
    image_source: bytes | Path | None,
    csv_source: bytes | Path | None,
    question: str,
    category: str,
) -> str:
    digest = hashlib.sha256()
    for value in (selected_scenario, question, category):
        digest.update(value.encode("utf-8", errors="replace"))
        digest.update(b"\0")

    for source in (image_source, csv_source):
        if isinstance(source, bytes):
            digest.update(source)
        elif source is not None:
            digest.update(str(source).encode("utf-8", errors="replace"))
            try:
                digest.update(str(source.stat().st_mtime_ns).encode("ascii"))
            except OSError:
                pass
        digest.update(b"\0")
    return digest.hexdigest()


def _format_scenario_name(name: str) -> str:
    return name.replace("_", " ").upper()


def _safe_text(value: Any) -> str:
    return html.escape(str(value))


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --fl-black: #000000;
            --fl-soft-black: #0a0a0a;
            --fl-panel: #0d0d0d;
            --fl-white: #ffffff;
            --fl-muted: #b6b6ba;
            --fl-hairline: #3a3a3f;
            --fl-font: "D-DIN", "Arial Narrow", Arial, Verdana, sans-serif;
        }

        html, body, [class*="css"], .stApp {
            font-family: var(--fl-font);
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: var(--fl-black);
            color: var(--fl-white);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"],
        #MainMenu,
        footer {
            visibility: hidden;
        }

        .stApp [data-testid="stMainBlockContainer"],
        .stApp .block-container {
            box-sizing: border-box;
            margin-left: auto;
            margin-right: auto;
            max-width: 1360px;
            padding-bottom: 5rem;
            padding-left: clamp(24px, 4vw, 72px);
            padding-right: clamp(24px, 4vw, 72px);
            padding-top: 2.2rem;
            width: 100%;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--fl-white) !important;
            font-family: "D-DIN-Bold", "Arial Narrow", Arial, Verdana, sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: 0.075rem !important;
            line-height: 1.08 !important;
            text-transform: uppercase;
        }

        p, label, li, [data-testid="stCaptionContainer"] {
            color: var(--fl-muted);
            letter-spacing: 0.02rem;
        }

        a {
            color: var(--fl-white) !important;
            text-decoration: underline;
        }

        hr {
            border-color: var(--fl-hairline) !important;
            margin: 2rem 0 !important;
        }

        [data-testid="stSidebar"] {
            background: #050505;
            border-right: 1px solid var(--fl-hairline);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.6rem;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stWidgetLabel"] p {
            color: var(--fl-white) !important;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.09rem !important;
            text-transform: uppercase;
        }

        .fl-nav {
            align-items: center;
            border-bottom: 1px solid var(--fl-hairline);
            display: flex;
            justify-content: space-between;
            margin-bottom: 4.2rem;
            padding-bottom: 1.1rem;
        }

        .fl-wordmark {
            color: var(--fl-white);
            font-family: "D-DIN-Bold", "Arial Narrow", Arial, sans-serif;
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.2rem;
        }

        .fl-nav-meta,
        .fl-eyebrow,
        .fl-panel-label,
        .fl-sidebar-kicker {
            color: var(--fl-muted);
            font-size: 0.72rem;
            letter-spacing: 0.11rem;
            line-height: 1.8;
            text-transform: uppercase;
        }

        .fl-hero {
            margin-bottom: 4.8rem;
            max-width: 1040px;
        }

        .fl-hero h1 {
            font-size: clamp(3rem, 6.3vw, 5rem) !important;
            letter-spacing: 0.1rem !important;
            line-height: 0.95 !important;
            margin: 0.8rem 0 1.2rem;
        }

        .fl-hero p {
            font-size: 1rem;
            line-height: 1.7;
            margin: 0;
            max-width: 700px;
        }

        .fl-section-head {
            border-top: 1px solid var(--fl-hairline);
            margin: 4rem 0 1.8rem;
            padding-top: 1rem;
        }

        .fl-section-head h2 {
            font-size: clamp(1.8rem, 3.2vw, 3rem) !important;
            margin: 0.55rem 0 0.65rem;
        }

        .fl-section-head p {
            line-height: 1.6;
            margin: 0;
            max-width: 760px;
        }

        .fl-sidebar-brand {
            border-bottom: 1px solid var(--fl-hairline);
            margin-bottom: 1.5rem;
            padding: 0.25rem 0 1.4rem;
        }

        .fl-sidebar-brand strong {
            color: var(--fl-white);
            display: block;
            font-family: "D-DIN-Bold", "Arial Narrow", Arial, sans-serif;
            font-size: 1.2rem;
            letter-spacing: 0.12rem;
            margin-top: 0.35rem;
            text-transform: uppercase;
        }

        .fl-sidebar-note,
        .fl-empty,
        .fl-notice {
            background: var(--fl-soft-black);
            border: 1px solid var(--fl-hairline);
            color: var(--fl-muted);
            font-size: 0.78rem;
            letter-spacing: 0.035rem;
            line-height: 1.55;
            padding: 0.9rem 1rem;
        }

        .fl-sidebar-note {
            margin-top: 1.2rem;
        }

        .fl-panel-label {
            border-bottom: 1px solid var(--fl-hairline);
            color: var(--fl-white);
            margin-bottom: 1rem;
            padding-bottom: 0.55rem;
        }

        .fl-callout {
            border-left: 2px solid var(--fl-white);
            margin: 1rem 0;
            padding: 0.1rem 0 0.1rem 1.2rem;
        }

        .fl-callout p {
            color: var(--fl-white);
            line-height: 1.65;
            margin: 0;
        }

        .fl-meta-line {
            color: var(--fl-muted);
            font-size: 0.72rem;
            letter-spacing: 0.09rem;
            margin: 1rem 0;
            text-transform: uppercase;
        }

        .fl-issue {
            border-bottom: 1px solid var(--fl-hairline);
            margin-bottom: 1.4rem;
            padding-bottom: 1.4rem;
        }

        .fl-issue h4 {
            font-size: 1rem !important;
            margin: 0.2rem 0 0.35rem;
        }

        .fl-issue p {
            line-height: 1.6;
            margin: 0.25rem 0;
        }

        [data-testid="stMetric"] {
            background: var(--fl-panel);
            border: 1px solid var(--fl-hairline);
            min-height: 128px;
            padding: 1.15rem 1.25rem;
        }

        [data-testid="stMetricLabel"] p {
            color: var(--fl-muted) !important;
            font-size: 0.68rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.1rem !important;
            text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
            color: var(--fl-white) !important;
            font-family: "D-DIN-Bold", "Arial Narrow", Arial, sans-serif;
            font-size: clamp(1.45rem, 2.2vw, 2.25rem) !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: transparent !important;
            border: 1px solid var(--fl-white) !important;
            border-radius: 32px !important;
            color: var(--fl-white) !important;
            font-family: var(--fl-font) !important;
            font-size: 0.74rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.09rem !important;
            min-height: 48px;
            padding: 0.75rem 1.5rem !important;
            text-transform: uppercase;
            transition: background 150ms ease, color 150ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: var(--fl-white) !important;
            color: var(--fl-black) !important;
        }

        .stButton > button:focus,
        .stDownloadButton > button:focus {
            box-shadow: 0 0 0 2px var(--fl-black), 0 0 0 3px var(--fl-white) !important;
        }

        [data-baseweb="select"] > div,
        [data-baseweb="base-input"],
        [data-baseweb="input"] > div,
        textarea {
            background: var(--fl-panel) !important;
            border-color: var(--fl-hairline) !important;
            border-radius: 4px !important;
            color: var(--fl-white) !important;
        }

        input, textarea,
        [data-baseweb="select"] span {
            color: var(--fl-white) !important;
            font-family: var(--fl-font) !important;
        }

        [data-baseweb="popover"],
        [role="listbox"],
        [role="option"] {
            background: var(--fl-panel) !important;
            color: var(--fl-white) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--fl-panel) !important;
            border: 1px dashed var(--fl-hairline) !important;
            border-radius: 4px !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background: transparent !important;
            border: 1px solid var(--fl-white) !important;
            color: var(--fl-white) !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border: 1px solid var(--fl-hairline);
        }

        [data-testid="stImage"] img {
            border: 1px solid var(--fl-hairline);
            border-radius: 0;
        }

        [data-testid="stProgress"] > div > div {
            background: var(--fl-white) !important;
        }

        [data-testid="stProgress"] > div {
            background: #242424 !important;
        }

        [data-testid="stExpander"] {
            background: var(--fl-soft-black);
            border: 1px solid var(--fl-hairline) !important;
            border-radius: 0 !important;
        }

        [data-testid="stExpander"] summary p {
            color: var(--fl-white) !important;
            font-size: 0.74rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.08rem !important;
            text-transform: uppercase;
        }

        [data-testid="stAlert"] {
            background: var(--fl-soft-black) !important;
            border: 1px solid var(--fl-hairline) !important;
            border-radius: 0 !important;
            color: var(--fl-white) !important;
            filter: grayscale(1);
        }

        .stSpinner > div {
            border-top-color: var(--fl-white) !important;
        }

        @media (max-width: 768px) {
            .stApp [data-testid="stMainBlockContainer"],
            .stApp .block-container {
                padding-bottom: 3rem;
                padding-top: 1.4rem;
            }

            .fl-nav {
                margin-bottom: 3rem;
            }

            .fl-nav-meta {
                display: none;
            }

            .fl-hero {
                margin-bottom: 3rem;
            }

            .fl-hero h1 {
                font-size: 2.7rem !important;
            }

            .fl-section-head {
                margin-top: 3rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_cockpit_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --fl-panel: #0b0b0c;
            --fl-panel-raised: #101011;
            --fl-muted: #a7a7ac;
            --fl-dim: #6e6e74;
            --fl-hairline: #29292d;
            --fl-hairline-strong: #46464c;
            --fl-display: "D-DIN-Bold", "Arial Narrow", Arial, Verdana, sans-serif;
        }

        .stApp [data-testid="stMainBlockContainer"],
        .stApp .block-container {
            padding-bottom: 6rem;
            padding-top: 1.7rem;
        }

        [data-testid="stSidebar"] {
            background: #050505 !important;
            border-right-color: var(--fl-hairline-strong);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 1.35rem 1.35rem 2rem;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.7rem;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stWidgetLabel"] p {
            font-size: 0.68rem !important;
            letter-spacing: 0.105rem !important;
        }

        .fl-topbar {
            align-items: center;
            border-bottom: 1px solid var(--fl-hairline);
            display: flex;
            justify-content: space-between;
            min-height: 42px;
            padding-bottom: 0.9rem;
        }

        .fl-wordmark {
            font-family: var(--fl-display);
            font-size: 0.92rem;
            letter-spacing: 0.23rem;
            text-transform: uppercase;
        }

        .fl-wordmark span {
            color: var(--fl-dim);
            font-weight: 400;
            margin-left: 0.65rem;
        }

        .fl-topbar-status {
            align-items: center;
            color: var(--fl-muted);
            display: flex;
            font-size: 0.65rem;
            gap: 0.65rem;
            letter-spacing: 0.11rem;
            text-transform: uppercase;
        }

        .fl-status-dot {
            background: var(--fl-white);
            border-radius: 50%;
            display: inline-block;
            height: 6px;
            width: 6px;
        }

        .fl-hero {
            border-bottom: 1px solid var(--fl-hairline);
            display: grid;
            gap: clamp(2rem, 6vw, 7rem);
            grid-template-columns: minmax(0, 1fr) minmax(250px, 0.34fr);
            margin-bottom: 1.2rem;
            margin-left: auto;
            margin-right: auto;
            max-width: 1360px;
            padding: clamp(4.3rem, 8vw, 7.2rem) 0 3.2rem;
            width: 100%;
        }

        .fl-hero h1 {
            font-size: clamp(3.4rem, 6.4vw, 5.7rem) !important;
            margin: 0.75rem 0 1.3rem;
            max-width: 980px;
        }

        .fl-hero p {
            color: #d0d0d3;
            font-size: 0.98rem;
            max-width: 660px;
        }

        .fl-mission-card {
            align-self: end;
            border-left: 1px solid var(--fl-hairline-strong);
            padding-left: 1.4rem;
        }

        .fl-mission-card-title {
            color: var(--fl-white);
            font-family: var(--fl-display);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12rem;
            margin-bottom: 0.85rem;
            text-transform: uppercase;
        }

        .fl-mission-row {
            align-items: center;
            border-top: 1px solid var(--fl-hairline);
            display: flex;
            justify-content: space-between;
            padding: 0.72rem 0;
        }

        .fl-mission-row span,
        .fl-mission-row strong {
            font-size: 0.64rem;
            letter-spacing: 0.09rem;
            text-transform: uppercase;
        }

        .fl-mission-row span {
            color: var(--fl-dim);
        }

        .fl-mission-row strong {
            color: var(--fl-white);
        }

        .fl-mission-strip {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            margin-bottom: 4.8rem;
        }

        .fl-strip-cell {
            border-right: 1px solid var(--fl-hairline);
            color: var(--fl-muted);
            font-size: 0.66rem;
            letter-spacing: 0.095rem;
            padding: 0.45rem 1rem 0.45rem 0;
            text-transform: uppercase;
        }

        .fl-strip-cell:not(:first-child) {
            padding-left: 1rem;
        }

        .fl-strip-cell:last-child {
            border-right: 0;
        }

        .fl-strip-cell strong {
            color: var(--fl-white);
            margin-left: 0.45rem;
        }

        .fl-section-head {
            align-items: end;
            border-top-color: var(--fl-hairline-strong);
            display: grid;
            gap: 2rem;
            grid-template-columns: minmax(200px, 0.34fr) minmax(0, 1fr);
            margin: 4.8rem 0 1.7rem;
            padding-top: 1.15rem;
        }

        .fl-section-head h2 {
            font-size: clamp(2rem, 3.25vw, 3.35rem) !important;
            letter-spacing: 0.095rem !important;
            margin: 0;
        }

        .fl-section-head p {
            margin-top: 0.55rem;
            max-width: 680px;
        }

        .fl-sidebar-brand {
            border-bottom-color: var(--fl-hairline-strong);
            margin-bottom: 0.7rem;
            padding: 0.15rem 0 1.25rem;
        }

        .fl-sidebar-brand strong {
            font-family: var(--fl-display);
            font-size: 1.3rem;
            letter-spacing: 0.105rem;
            line-height: 1.05;
            margin-top: 0.45rem;
        }

        .fl-sidebar-status {
            align-items: center;
            color: var(--fl-muted);
            display: flex;
            font-size: 0.62rem;
            justify-content: space-between;
            letter-spacing: 0.095rem;
            margin-top: 0.9rem;
            text-transform: uppercase;
        }

        .fl-control-group {
            border-top: 1px solid var(--fl-hairline);
            margin-top: 0.55rem;
            padding: 0.95rem 0 0.75rem;;
        }

        .fl-control-index {
            color: var(--fl-dim);
            font-size: 0.62rem;
            letter-spacing: 0.1rem;
            margin-bottom: 0.15rem;
            text-transform: uppercase;
        }

        .fl-control-title {
            color: var(--fl-white);
            font-family: var(--fl-display);
            font-size: 0.77rem;
            font-weight: 700;
            letter-spacing: 0.09rem;
            text-transform: uppercase;
        }

        .fl-control-copy {
            color: var(--fl-dim);
            font-size: 0.7rem;
            line-height: 1.45;
            margin-top: 0.25rem;
        }

        .fl-sidebar-note {
            border-left: 2px solid var(--fl-white);
            font-size: 0.7rem;
            margin-top: 0.7rem;
        }

        .fl-frame-head {
            align-items: center;
            background: var(--fl-panel);
            border: 1px solid var(--fl-hairline-strong);
            border-bottom: 0;
            display: flex;
            justify-content: space-between;
            min-height: 48px;
            padding: 0.7rem 0.9rem;
        }

        .fl-frame-title {
            color: var(--fl-white);
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.1rem;
            text-transform: uppercase;
        }

        .fl-frame-state {
            color: var(--fl-dim);
            font-size: 0.61rem;
            letter-spacing: 0.09rem;
            text-transform: uppercase;
        }

        .fl-empty {
            align-items: center;
            display: flex;
            justify-content: center;
            min-height: 128px;
            text-align: center;
            text-transform: uppercase;
        }

        .fl-standby {
            margin-top: 1.5rem;
        }

        .fl-metric-card {
            background: var(--fl-panel);
            border: 1px solid var(--fl-hairline);
            min-height: 154px;
            padding: 1.15rem 1.25rem 1.2rem;
            position: relative;
        }

        .fl-metric-card::before {
            background: var(--fl-white);
            content: "";
            height: 1px;
            left: -1px;
            position: absolute;
            top: -1px;
            width: 44px;
        }

        .fl-metric-label {
            color: var(--fl-muted);
            font-size: 0.64rem;
            font-weight: 700;
            letter-spacing: 0.105rem;
            text-transform: uppercase;
        }

        .fl-metric-value {
            color: var(--fl-white);
            font-family: var(--fl-display);
            font-size: clamp(1.75rem, 2.8vw, 2.7rem);
            font-weight: 700;
            letter-spacing: 0.05rem;
            line-height: 1;
            margin: 1.1rem 0 0.95rem;
            overflow-wrap: anywhere;
            text-transform: uppercase;
        }

        .fl-metric-note {
            color: var(--fl-dim);
            font-size: 0.62rem;
            letter-spacing: 0.075rem;
            text-transform: uppercase;
        }

        .fl-mini-metrics {
            display: grid;
            gap: 1px;
            grid-template-columns: repeat(2, 1fr);
            margin-bottom: 0.9rem;
        }

        .fl-mini-metric {
            background: var(--fl-panel);
            border: 1px solid var(--fl-hairline);
            padding: 0.8rem 0.9rem;
        }

        .fl-mini-metric span {
            color: var(--fl-dim);
            display: block;
            font-size: 0.58rem;
            letter-spacing: 0.09rem;
            text-transform: uppercase;
        }

        .fl-mini-metric strong {
            color: var(--fl-white);
            display: block;
            font-family: var(--fl-display);
            font-size: 1.35rem;
            letter-spacing: 0.04rem;
            margin-top: 0.25rem;
        }

        .fl-summary-card {
            background: var(--fl-panel-raised);
            border: 1px solid var(--fl-hairline-strong);
            border-left: 3px solid var(--fl-white);
            min-height: 154px;
            padding: 1.35rem 1.5rem;
        }

        .fl-summary-card p {
            color: var(--fl-white);
            font-size: 1.02rem;
            line-height: 1.7;
            margin: 0.7rem 0 0;
        }

        .fl-item-card {
            align-items: start;
            background: var(--fl-panel);
            border: 1px solid var(--fl-hairline);
            display: grid;
            gap: 0.9rem;
            grid-template-columns: 30px minmax(0, 1fr);
            margin-bottom: 0.55rem;
            padding: 0.9rem 1rem;
        }

        .fl-item-number {
            color: var(--fl-dim);
            font-size: 0.62rem;
            letter-spacing: 0.08rem;
            padding-top: 0.15rem;
        }

        .fl-item-copy {
            color: #d1d1d4;
            font-size: 0.84rem;
            line-height: 1.55;
        }

        .fl-issue-card {
            background: var(--fl-panel);
            border: 1px solid var(--fl-hairline);
            margin-bottom: 0.85rem;
            min-height: 210px;
            padding: 1.2rem 1.25rem;
        }

        .fl-issue-topline,
        .fl-similarity-row {
            align-items: center;
            display: flex;
            justify-content: space-between;
        }

        .fl-issue-index,
        .fl-issue-source,
        .fl-similarity-row span {
            color: var(--fl-dim);
            font-size: 0.59rem;
            letter-spacing: 0.09rem;
            text-transform: uppercase;
        }

        .fl-issue-card h3 {
            font-size: 1.05rem !important;
            line-height: 1.25 !important;
            margin: 1rem 0 0.55rem;
        }

        .fl-issue-card p {
            color: var(--fl-muted);
            font-size: 0.79rem;
            line-height: 1.55;
            margin: 0;
            min-height: 3.6rem;
        }

        .fl-similarity-row {
            margin-top: 1.1rem;
        }

        .fl-similarity-row strong {
            color: var(--fl-white);
            font-family: var(--fl-display);
            font-size: 0.76rem;
        }

        .fl-progress-track {
            background: #252528;
            height: 3px;
            margin-top: 0.55rem;
            overflow: hidden;
            width: 100%;
        }

        .fl-progress-fill {
            background: var(--fl-white);
            height: 100%;
        }

        .fl-report-bar {
            align-items: center;
            background: var(--fl-panel-raised);
            border: 1px solid var(--fl-hairline-strong);
            border-bottom: 0;
            display: flex;
            justify-content: space-between;
            padding: 0.85rem 1rem;
        }

        .fl-report-bar strong,
        .fl-report-bar span {
            font-size: 0.64rem;
            letter-spacing: 0.1rem;
            text-transform: uppercase;
        }

        .fl-report-bar strong {
            color: var(--fl-white);
        }

        .fl-report-bar span {
            color: var(--fl-dim);
        }

        [data-testid="stExpander"] {
            background: var(--fl-panel) !important;
            border-color: var(--fl-hairline-strong) !important;
        }

        [data-testid="stAlert"] {
            background: var(--fl-panel) !important;
            border-color: var(--fl-hairline-strong) !important;
            border-left: 2px solid var(--fl-white) !important;
        }

        [data-testid="stAlert"] p {
            color: #d6d6d8 !important;
            font-size: 0.78rem !important;
            line-height: 1.55 !important;
        }

        [data-testid="stDataFrame"] {
            border-color: var(--fl-hairline-strong);
        }

        @media (max-width: 900px) {
            .stApp [data-testid="stMainBlockContainer"],
            .stApp .block-container {
                padding-bottom: 3.5rem;
                padding-top: 1.35rem;
            }

            .fl-hero {
                gap: 2.5rem;
                grid-template-columns: 1fr;
                padding: 3.5rem 0 2.5rem;
            }

            .fl-mission-card {
                border-left: 0;
                border-top: 1px solid var(--fl-hairline-strong);
                padding-left: 0;
                padding-top: 1.2rem;
            }

            .fl-section-head {
                gap: 0.4rem;
                grid-template-columns: 1fr;
                margin-top: 3.5rem;
            }

            .fl-mission-strip {
                margin-bottom: 3.5rem;
            }
        }

        @media (max-width: 600px) {
            .fl-hero h1 {
                font-size: 2.55rem !important;
            }

            .fl-topbar-status,
            .fl-wordmark span {
                display: none;
            }

            .fl-mission-strip {
                grid-template-columns: 1fr;
            }

            .fl-strip-cell,
            .fl-strip-cell:not(:first-child) {
                border-bottom: 1px solid var(--fl-hairline);
                border-right: 0;
                padding: 0.65rem 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    st.markdown(
        """
        <div class="fl-topbar">
            <div class="fl-wordmark">FACTORYLENS <span>FL / 01</span></div>
            <div class="fl-topbar-status">
                <span class="fl-status-dot"></span>
                VISUAL AUDIT NODE ONLINE
            </div>
        </div>
        <section class="fl-hero">
            <div>
                <div class="fl-eyebrow">INDUSTRIAL INVESTIGATION PLATFORM</div>
                <h1>VISUAL AUDIT<br>MISSION CONTROL</h1>
                <p>
                    Inspect product imagery, correlate production evidence, and
                    convert anomaly signals into an engineering-ready assessment.
                </p>
            </div>
            <aside class="fl-mission-card">
                <div class="fl-mission-card-title">MISSION STATUS</div>
                <div class="fl-mission-row"><span>NODE</span><strong>FL-VISION-01</strong></div>
                <div class="fl-mission-row"><span>PIPELINE</span><strong>ARMED</strong></div>
                <div class="fl-mission-row"><span>MODE</span><strong>ROOT CAUSE</strong></div>
                <div class="fl-mission-row"><span>OUTPUT</span><strong>AUDIT RECORD</strong></div>
            </aside>
        </section>
        <div class="fl-mission-strip">
            <div class="fl-strip-cell">INPUT <strong>IMAGE + LOGS</strong></div>
            <div class="fl-strip-cell">ANALYSIS <strong>VISION + RETRIEVAL</strong></div>
            <div class="fl-strip-cell">DELIVERABLE <strong>ENGINEERING REPORT</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(
    sequence: str,
    title: str,
    description: str | None = None,
) -> None:
    description_markup = f"<p>{_safe_text(description)}</p>" if description else ""
    st.markdown(
        f"""
        <section class="fl-section-head">
            <div class="fl-eyebrow">{_safe_text(sequence)}</div>
            <div class="fl-section-copy">
                <h2>{_safe_text(title)}</h2>
                {description_markup}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_empty(message: str) -> None:
    st.markdown(
        f'<div class="fl-empty">{_safe_text(message)}</div>',
        unsafe_allow_html=True,
    )


def render_frame_header(title: str, state: str) -> None:
    st.markdown(
        f"""
        <div class="fl-frame-head">
            <span class="fl-frame-title">{_safe_text(title)}</span>
            <span class="fl-frame-state">{_safe_text(state)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: Any, note: str) -> None:
    st.markdown(
        f"""
        <article class="fl-metric-card">
            <div class="fl-metric-label">{_safe_text(label)}</div>
            <div class="fl-metric-value">{_safe_text(value)}</div>
            <div class="fl-metric-note">{_safe_text(note)}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def render_item_cards(items: list[Any], empty_message: str) -> None:
    if not items:
        render_empty(empty_message)
        return

    cards = "".join(
        f"""
        <div class="fl-item-card">
            <div class="fl-item-number">{index:02d}</div>
            <div class="fl-item-copy">{_safe_text(item)}</div>
        </div>
        """
        for index, item in enumerate(items, start=1)
    )
    st.markdown(cards, unsafe_allow_html=True)


def render_input_preview(
    image_source: bytes | Path | None,
    csv_source: bytes | Path | None,
    scenario_doc: Path | None,
    selected_scenario: str,
) -> None:
    render_section_header(
        "01 / EVIDENCE INTAKE",
        "INPUT PREVIEW",
        "Verify the selected product evidence before initiating the analysis sequence.",
    )

    image_column, logs_column = st.columns(2, gap="large")
    with image_column:
        image_state = "FRAME READY" if image_source is not None else "AWAITING INPUT"
        render_frame_header("01-A / INPUT FRAME", image_state)
        if image_source is None:
            if selected_scenario == EMPTY_SCENARIO:
                render_empty("NO PRODUCT IMAGE SELECTED.")
            else:
                render_empty(
                    "SCENARIO IMAGE IS NOT AVAILABLE LOCALLY. UPLOAD A PRODUCT "
                    "IMAGE TO CONTINUE."
                )
        else:
            st.image(image_source, use_column_width=True)

    with logs_column:
        log_state = "LOG STREAM READY" if csv_source is not None else "AWAITING INPUT"
        render_frame_header("01-B / PRODUCTION LOGS", log_state)
        if csv_source is None:
            render_empty("NO CSV LOGS SELECTED.")
        else:
            try:
                dataframe = _read_csv_dataframe(csv_source)
            except Exception as exc:  # noqa: BLE001
                render_empty("CSV PREVIEW UNAVAILABLE.")
                st.caption(str(exc))
            else:
                fail_count = 0
                if "pass_fail" in dataframe.columns:
                    fail_count = int(
                        dataframe["pass_fail"].astype(str).str.upper().eq("FAIL").sum()
                    )
                st.markdown(
                    f"""
                    <div class="fl-mini-metrics">
                        <div class="fl-mini-metric">
                            <span>RECORDS INGESTED</span>
                            <strong>{len(dataframe):,}</strong>
                        </div>
                        <div class="fl-mini-metric">
                            <span>FAIL SIGNALS</span>
                            <strong>{fail_count:,}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                preview = dataframe.head(100)
                if "pass_fail" in preview.columns:
                    st.dataframe(
                        preview.style.apply(_highlight_fail_rows, axis=1),
                        use_container_width=True,
                        hide_index=True,
                        height=320,
                    )
                else:
                    st.dataframe(
                        preview,
                        use_container_width=True,
                        hide_index=True,
                        height=320,
                    )
                if len(dataframe) > len(preview):
                    st.caption(f"Showing the first {len(preview):,} records.")

    if scenario_doc is not None:
        with st.expander(
            f"SCENARIO BRIEF / {_format_scenario_name(selected_scenario)}"
        ):
            try:
                st.markdown(scenario_doc.read_text(encoding="utf-8"))
            except OSError:
                render_empty("SCENARIO BRIEF UNAVAILABLE.")


def render_summary(result: AnalysisResponse) -> None:
    render_section_header(
        "02 / INSPECTION OUTCOME",
        "ANALYSIS SUMMARY",
        "Primary classification and anomaly signal returned by the inspection pipeline.",
    )
    score = get_field(result, "anomaly_score")
    label = get_field(result, "defect_label") or "N/A"
    category = get_field(result, "category") or "N/A"
    score_text = f"{float(score):.4f}" if isinstance(score, int | float) else "N/A"

    score_column, label_column, category_column = st.columns(3)
    with score_column:
        render_metric_card("ANOMALY SCORE", score_text, "NORMALIZED MODEL SIGNAL")
    with label_column:
        render_metric_card("DEFECT LABEL", str(label).upper(), "PRIMARY CLASSIFICATION")
    with category_column:
        render_metric_card("PRODUCT CLASS", str(category).upper(), "INSPECTION CATEGORY")

    request_id = get_field(result, "request_id")
    if request_id:
        st.markdown(
            f'<div class="fl-meta-line">REQUEST ID / {_safe_text(request_id)}</div>',
            unsafe_allow_html=True,
        )


def render_visuals(result: AnalysisResponse, original_image: Any) -> None:
    render_section_header(
        "03 / VISUAL EVIDENCE",
        "INSPECTION FRAME",
        "Compare the source image with the model-generated localization evidence.",
    )
    original_column, heatmap_column = st.columns(2, gap="large")

    with original_column:
        source_state = "CAPTURE LOCKED" if original_image is not None else "NO SIGNAL"
        render_frame_header("ORIGINAL FRAME", source_state)
        if original_image is None:
            render_empty("NO ORIGINAL IMAGE AVAILABLE.")
        else:
            st.image(original_image, use_column_width=True)

    with heatmap_column:
        heatmap_path = resolve_existing_path(get_field(result, "heatmap_path"))
        heatmap_state = "LOCALIZATION READY" if heatmap_path else "FALLBACK MODE"
        render_frame_header("HEATMAP ANALYSIS", heatmap_state)
        if heatmap_path is None:
            render_empty("NO HEATMAP GENERATED / FALLBACK ANALYSIS REMAINS AVAILABLE.")
        else:
            st.image(str(heatmap_path), use_column_width=True)


def render_hypothesis(result: AnalysisResponse) -> None:
    render_section_header(
        "04 / ENGINEERING ASSESSMENT",
        "ROOT-CAUSE HYPOTHESIS",
        "Evidence-backed interpretation of the most likely production failure mode.",
    )
    hypothesis = get_field(result, "root_cause_hypothesis")
    if hypothesis is None:
        render_empty("NO ROOT-CAUSE HYPOTHESIS AVAILABLE.")
        return

    summary = get_field(hypothesis, "summary", "N/A")
    confidence = clamp01(get_field(hypothesis, "confidence", 0.0))
    likely_causes = as_list(get_field(hypothesis, "likely_causes"))
    evidence = as_list(get_field(hypothesis, "evidence"))

    confidence_column, summary_column = st.columns([1, 2], gap="large")
    with confidence_column:
        render_metric_card(
            "HYPOTHESIS CONFIDENCE",
            f"{confidence:.2f}",
            "MODEL-SUPPORTED ASSESSMENT",
        )
        st.progress(confidence)
    with summary_column:
        st.markdown(
            f"""
            <article class="fl-summary-card">
                <div class="fl-metric-label">PRIMARY ASSESSMENT</div>
                <p>{_safe_text(summary)}</p>
            </article>
            """,
            unsafe_allow_html=True,
        )

    cause_column, evidence_column = st.columns(2, gap="large")
    with cause_column:
        st.markdown("### LIKELY CAUSES")
        render_item_cards(likely_causes, "NO LIKELY CAUSES RETURNED.")

    with evidence_column:
        st.markdown("### SUPPORTING EVIDENCE")
        render_item_cards(evidence, "NO SUPPORTING EVIDENCE RETURNED.")


def render_known_issues(result: AnalysisResponse) -> None:
    render_section_header(
        "05 / KNOWLEDGE CORRELATION",
        "RELATED KNOWN ISSUES",
        "Historical issue matches ranked by semantic similarity to this inspection.",
    )
    issues = as_list(get_field(result, "related_known_issues"))
    if not issues:
        render_empty("NO RELATED KNOWN ISSUES FOUND.")
        return

    issue_columns = st.columns(2, gap="large")
    for index, issue in enumerate(issues, start=1):
        title = get_field(issue, "title", "N/A")
        source = get_field(issue, "source", "N/A")
        snippet = get_field(issue, "snippet", "")
        similarity = clamp01(get_field(issue, "similarity", 0.0))
        snippet_text = snippet or "No issue synopsis was returned by the knowledge base."
        similarity_percent = similarity * 100

        with issue_columns[(index - 1) % len(issue_columns)]:
            st.markdown(
                f"""
                <article class="fl-issue-card">
                    <div class="fl-issue-topline">
                        <span class="fl-issue-index">CORRELATION / {index:02d}</span>
                        <span class="fl-issue-source">{_safe_text(source)}</span>
                    </div>
                    <h3>{_safe_text(title)}</h3>
                    <p>{_safe_text(snippet_text)}</p>
                    <div class="fl-similarity-row">
                        <span>SIMILARITY</span>
                        <strong>{similarity:.2f}</strong>
                    </div>
                    <div class="fl-progress-track">
                        <div class="fl-progress-fill" style="width: {similarity_percent:.1f}%"></div>
                    </div>
                </article>
                """,
                unsafe_allow_html=True,
            )


def render_report(result: AnalysisResponse) -> None:
    render_section_header(
        "06 / TECHNICAL RECORD",
        "ENGINEERING REPORT",
        "Generated assessment prepared for review, handoff, and audit traceability.",
    )
    report_markdown = get_field(result, "report_markdown")
    if not isinstance(report_markdown, str) or not report_markdown.strip():
        render_empty("NO ENGINEERING REPORT GENERATED.")
        return

    st.markdown(
        """
        <div class="fl-report-bar">
            <strong>REPORT CHANNEL / READY</strong>
            <span>MARKDOWN TECHNICAL RECORD</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("READ FULL ENGINEERING REPORT", expanded=True):
        st.markdown(report_markdown)
    st.download_button(
        "DOWNLOAD TECHNICAL RECORD / MD",
        data=report_markdown,
        file_name="factorylens_report.md",
        mime="text/markdown",
    )


def render_warnings(result: AnalysisResponse) -> None:
    warnings = as_list(get_field(result, "warnings"))
    if not warnings:
        return

    render_section_header(
        "07 / SYSTEM NOTICES",
        "ANALYSIS WARNINGS",
        "Pipeline conditions that may affect interpretation of the final assessment.",
    )
    for warning in warnings:
        st.warning(str(warning))


def run_analysis_request(
    image_preview: bytes | Path,
    uploaded_image: Any,
    category: str,
) -> AnalysisResponse | None:
    temp_image_path: Path | None = None
    db = None
    try:
        if uploaded_image is not None:
            temp_image_path = save_uploaded_file_to_temp(uploaded_image)
            analysis_image_path = temp_image_path
        else:
            analysis_image_path = Path(image_preview)

        db = SessionLocal()
        return run_analysis(
            str(analysis_image_path),
            db,
            category=category,
        )
    except Exception as exc:  # noqa: BLE001
        st.error("Analysis failed. Verify the database and model configuration.")
        with st.expander("TECHNICAL DETAILS"):
            st.code(str(exc))
        return None
    finally:
        if db is not None:
            db.close()
        _cleanup_temp_file(temp_image_path)


def render_sidebar_group(index: str, title: str, copy: str) -> None:
    st.sidebar.markdown(
        f"""
        <section class="fl-control-group">
            <div class="fl-control-index">{_safe_text(index)}</div>
            <div class="fl-control-title">{_safe_text(title)}</div>
            <div class="fl-control-copy">{_safe_text(copy)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(
    scenarios: dict[str, dict[str, Path | None]],
) -> tuple[str, Any, Any, str, str, bool]:
    st.sidebar.markdown(
        """
        <div class="fl-sidebar-brand">
            <div class="fl-sidebar-kicker">MISSION CONTROL</div>
            <strong>INVESTIGATION<br>CONTROL DECK</strong>
            <div class="fl-sidebar-status">
                <span>FL / VISUAL NODE</span>
                <span>ONLINE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar_group(
        "01 / MISSION PROFILE",
        "DEMO SCENARIO",
        "Load a prepared evidence package or begin with a clean investigation.",
    )
    scenario_names = [EMPTY_SCENARIO, *scenarios.keys()]
    selected_scenario = st.sidebar.selectbox(
        "Demo scenario",
        scenario_names,
        format_func=_format_scenario_name,
    )

    render_sidebar_group(
        "02 / EVIDENCE CHANNELS",
        "INPUT PAYLOAD",
        "Attach the visual frame and optional production log stream.",
    )
    uploaded_image = st.sidebar.file_uploader(
        "Product image",
        type=["png", "jpg", "jpeg"],
    )
    uploaded_csv = st.sidebar.file_uploader("CSV logs", type=["csv"])

    render_sidebar_group(
        "03 / ANALYSIS DIRECTIVE",
        "QUERY PARAMETERS",
        "Define the investigation objective and inspection category.",
    )
    question = st.sidebar.text_area(
        "Investigation question",
        value="Analyze defects + root cause",
        height=92,
    )
    category = st.sidebar.selectbox("Category", ["hazelnut"], index=0)

    render_sidebar_group(
        "04 / EXECUTION",
        "LAUNCH SEQUENCE",
        "Run the visual audit and assemble the engineering record.",
    )
    analyze_clicked = st.sidebar.button("RUN ANALYSIS", use_container_width=True)

    st.sidebar.markdown(
        """
        <div class="fl-sidebar-note">
            CURRENT ANALYSIS LANE USES THE PRODUCT IMAGE AND CATEGORY. THE CSV AND
            INVESTIGATION QUESTION REMAIN VISIBLE AS REVIEW CONTEXT.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return (
        selected_scenario,
        uploaded_image,
        uploaded_csv,
        question,
        category,
        analyze_clicked,
    )


def main() -> None:
    st.set_page_config(
        page_title="FactoryLens / Industrial Diagnostics",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    inject_cockpit_styles()
    render_app_header()

    scenarios = find_demo_scenarios()
    (
        selected_scenario,
        uploaded_image,
        uploaded_csv,
        question,
        category,
        analyze_clicked,
    ) = render_sidebar(scenarios)
    scenario_assets = _selected_scenario_assets(scenarios, selected_scenario)

    image_preview: bytes | Path | None = (
        uploaded_image.getvalue()
        if uploaded_image is not None
        else scenario_assets["image"]
    )
    csv_preview: bytes | Path | None = (
        uploaded_csv.getvalue()
        if uploaded_csv is not None
        else scenario_assets["csv"]
    )
    signature = _input_signature(
        selected_scenario,
        image_preview,
        csv_preview,
        question,
        category,
    )

    render_input_preview(
        image_preview,
        csv_preview,
        scenario_assets["scenario"],
        selected_scenario,
    )

    if analyze_clicked:
        if image_preview is None:
            st.error("A product image is required before analysis can begin.")
            st.session_state.pop(RESULT_STATE_KEY, None)
            st.session_state.pop(RESULT_SIGNATURE_KEY, None)
            st.session_state.pop(RESULT_IMAGE_KEY, None)
        else:
            with st.spinner("RUNNING INSPECTION SEQUENCE"):
                result = run_analysis_request(
                    image_preview,
                    uploaded_image,
                    category,
                )
            if result is not None:
                st.session_state[RESULT_STATE_KEY] = result
                st.session_state[RESULT_SIGNATURE_KEY] = signature
                st.session_state[RESULT_IMAGE_KEY] = image_preview
            else:
                st.session_state.pop(RESULT_STATE_KEY, None)
                st.session_state.pop(RESULT_SIGNATURE_KEY, None)
                st.session_state.pop(RESULT_IMAGE_KEY, None)

    result = st.session_state.get(RESULT_STATE_KEY)
    result_signature = st.session_state.get(RESULT_SIGNATURE_KEY)
    if result is not None and result_signature == signature:
        result_image = st.session_state.get(RESULT_IMAGE_KEY, image_preview)
        render_summary(result)
        render_visuals(result, result_image)
        render_hypothesis(result)
        render_known_issues(result)
        render_report(result)
        render_warnings(result)
    elif result is not None:
        st.markdown(
            """
            <div class="fl-notice">
                INPUTS HAVE CHANGED. RUN ANALYSIS TO REFRESH THE ENGINEERING RESULT.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="fl-empty fl-standby">
                ANALYSIS SEQUENCE STANDBY / PROVIDE A PRODUCT IMAGE AND RUN ANALYSIS.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
