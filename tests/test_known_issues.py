from __future__ import annotations

from factorylens.data.known_issues import load_known_issues


def test_load_known_issues_parses_demo_corpus():
    docs = load_known_issues("assets/known_issues")

    assert len(docs) >= 6
    assert {doc.defect_type for doc in docs} == {"crack", "cut", "hole", "print"}
    assert {doc.severity for doc in docs} <= {"low", "medium", "high"}
    assert all(doc.title.startswith("KI-") for doc in docs)
    assert all(doc.source.endswith(".md") for doc in docs)
    assert all("Recommended action / SOP:" in doc.text for doc in docs)


def test_known_issues_include_index_file():
    index = open("assets/known_issues/INDEX.md", encoding="utf-8").read()

    assert "KI-001" in index
    assert "KI-008" in index
    assert "Defect type" in index
