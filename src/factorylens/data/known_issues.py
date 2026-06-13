"""Load simple known-issue Markdown docs for later retrieval/indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REQUIRED_FIELDS = (
    "Defect type",
    "Symptom",
    "Likely root cause",
    "Recommended action / SOP",
    "Severity",
)


@dataclass(frozen=True)
class KnownIssueDoc:
    title: str
    defect_type: str
    symptom: str
    likely_root_cause: str
    recommended_action: str
    severity: str
    source: str
    text: str


def load_known_issues(directory: str) -> list[KnownIssueDoc]:
    """Parse known-issue Markdown files from a directory.

    Files are expected to use the fixed template in ``assets/known_issues``.
    ``INDEX.md`` is skipped.
    """

    docs: list[KnownIssueDoc] = []
    for path in sorted(Path(directory).glob("*.md")):
        if path.name == "INDEX.md":
            continue
        docs.append(_parse_known_issue(path))
    return docs


def _parse_known_issue(path: Path) -> KnownIssueDoc:
    raw_text = path.read_text(encoding="utf-8").strip()
    lines = raw_text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{path} must start with a '# ' title")

    title = lines[0].removeprefix("# ").strip()
    fields = _parse_fields(lines[1:], path)

    return KnownIssueDoc(
        title=title,
        defect_type=fields["Defect type"],
        symptom=fields["Symptom"],
        likely_root_cause=fields["Likely root cause"],
        recommended_action=fields["Recommended action / SOP"],
        severity=fields["Severity"],
        source=path.name,
        text=raw_text,
    )


def _parse_fields(lines: list[str], path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"{path} has non-template line: {line}")
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()

    missing = [field for field in REQUIRED_FIELDS if not fields.get(field)]
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")

    return fields
