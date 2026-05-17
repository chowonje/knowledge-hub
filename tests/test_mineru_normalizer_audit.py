from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.mineru_normalizer_audit import (
    MINERU_NORMALIZER_AUDIT_SCHEMA_ID,
    build_mineru_normalizer_audit,
    write_mineru_normalizer_audit_reports,
)


def _write_fixture(parsed_dir: Path, *, with_page: bool = False) -> None:
    parsed_dir.mkdir(parents=True)
    markdown = """# Attention Is All You Need

## 1 Introduction

![](images/figure1.jpg)
Figure 1: The Transformer - model architecture.

$$
Attention(Q,K,V)=softmax(QK^T)V
$$

Table 1: Maximum path lengths.
<table><tr><td>Layer Type</td><td>Maximum Path Length</td></tr><tr><td>Self-Attention</td><td>O(1)</td></tr></table>
"""
    page = 1 if with_page else None
    elements = [
        {"type": "title", "text": "", "page": page, "bbox": [10, 10, 200, 30], "reading_order": 1},
        {"type": "text", "text": "Attention Is All You Need", "page": page, "bbox": [10, 10, 200, 30]},
        {"type": "title", "text": "", "page": page, "bbox": [10, 50, 140, 65], "reading_order": 2},
        {"type": "text", "text": "1 Introduction", "page": page, "bbox": [10, 50, 140, 65]},
        {"type": "image", "text": "", "page": page, "bbox": [20, 90, 200, 220], "reading_order": 3},
        {"type": "image_caption", "text": "", "page": page, "bbox": [20, 225, 200, 240], "reading_order": 4},
        {"type": "interline_equation", "text": "", "page": page, "bbox": [30, 260, 180, 280], "reading_order": 5},
        {
            "type": "interline_equation",
            "text": "Attention(Q,K,V)=softmax(QK^T)V",
            "page": page,
            "bbox": [30, 260, 180, 280],
        },
        {"type": "table_caption", "text": "", "page": page, "bbox": [20, 300, 200, 320], "reading_order": 6},
        {"type": "table", "text": "", "page": page, "bbox": [20, 325, 260, 380], "reading_order": 7},
        {"type": "table_body", "text": "", "page": page, "bbox": [20, 325, 260, 380], "reading_order": 7},
    ]
    (parsed_dir / "document.md").write_text(markdown, encoding="utf-8")
    (parsed_dir / "document.json").write_text(
        json.dumps({"markdown_text": markdown, "elements": elements, "parser_meta": {"parser": "mineru"}}),
        encoding="utf-8",
    )
    (parsed_dir / "manifest.json").write_text(
        json.dumps(
            {
                "paper_id": "1706.03762",
                "parser_meta": {"parser": "mineru"},
                "markdown_path": str(parsed_dir / "document.md"),
                "json_path": str(parsed_dir / "document.json"),
            }
        ),
        encoding="utf-8",
    )


def test_mineru_normalizer_audit_emits_non_strict_candidates(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "parsed" / "1706.03762"
    _write_fixture(parsed_dir)

    payload = build_mineru_normalizer_audit(parsed_dir)

    assert payload["schema"] == MINERU_NORMALIZER_AUDIT_SCHEMA_ID
    assert payload["sourceParser"] == "mineru"
    assert validate_payload(payload, MINERU_NORMALIZER_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["sectionCandidates"] == 2
    assert payload["counts"]["tableCandidates"] == 1
    assert payload["counts"]["equationCandidates"] == 1
    assert payload["counts"]["figureCaptionCandidates"] == 1
    assert payload["counts"]["readingOrderCandidates"] == 8
    assert payload["provenance"]["allCandidatesNonStrict"] is True
    assert payload["provenance"]["anyCitationGrade"] is False
    assert payload["provenance"]["originalCharsStartEndAvailable"] is False
    assert payload["provenance"]["pageRecovered"] is False

    for candidate in payload["candidates"]:
        assert candidate["source_parser"] == "mineru"
        assert candidate["evidence_tier"] == "candidate_only"
        assert candidate["strict"] is False
        assert candidate["citation_grade"] is False
        assert "sourceContentHash" not in candidate
        assert "chars:start-end" not in json.dumps(candidate)
        assert "no_original_chars_start_end" in candidate["non_strict_reason"]
        assert "markdown_offsets_not_original_source_chars" in candidate["non_strict_reason"]


def test_mineru_normalizer_table_candidate_is_not_table_cell_evidence(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "parsed" / "1706.03762"
    _write_fixture(parsed_dir)

    payload = build_mineru_normalizer_audit(parsed_dir)
    table = next(item for item in payload["candidates"] if item["candidate_type"] == "table_candidate")

    assert table["layout_element_ids"]
    assert table["tableHtmlPresent"] is True
    assert table["tableCellCitationGrade"] is False
    assert table["tableRows"][0][0]["text"] == "Layer Type"
    assert "not_table_cell_citation_grade" in table["non_strict_reason"]
    assert "no_cell_bbox_or_row_column_provenance" in table["non_strict_reason"]


def test_mineru_normalizer_missing_page_keeps_bbox_non_strict(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "parsed" / "1706.03762"
    _write_fixture(parsed_dir)

    payload = build_mineru_normalizer_audit(parsed_dir)
    linked = [item for item in payload["candidates"] if item["bbox"] is not None]

    assert linked
    assert all(item["page"] is None for item in linked)
    assert all("page_not_recovered" in item["non_strict_reason"] for item in linked)
    assert all("bbox_only_non_strict" in item["non_strict_reason"] for item in linked)
    assert payload["provenance"]["pageRecoveryStatus"] == "unavailable_no_page_fields"


def test_mineru_normalizer_reports_page_only_when_layout_page_exists(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "parsed" / "1706.03762"
    _write_fixture(parsed_dir, with_page=True)

    payload = build_mineru_normalizer_audit(parsed_dir)

    assert payload["provenance"]["pageRecovered"] is True
    assert payload["provenance"]["allCandidatesNonStrict"] is True
    assert payload["counts"]["citationGradeCandidates"] == 0
    assert all(item["strict"] is False for item in payload["candidates"])


def test_mineru_normalizer_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "parsed" / "1706.03762"
    _write_fixture(parsed_dir)
    payload = build_mineru_normalizer_audit(parsed_dir)

    paths = write_mineru_normalizer_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"candidates", "summary", "markdown"}
    candidates = json.loads(Path(paths["candidates"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert candidates["schema"] == MINERU_NORMALIZER_AUDIT_SCHEMA_ID
    assert len(candidates["candidates"]) == payload["counts"]["totalCandidates"]
    assert validate_payload(summary, MINERU_NORMALIZER_AUDIT_SCHEMA_ID, strict=True).ok
    assert "All emitted records are `candidate_only`" in markdown
