from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.mineru_source_alignment_audit import (
    MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID,
    build_mineru_source_alignment_audit,
    write_mineru_source_alignment_reports,
)


def _write_pilot_fixture(
    root: Path,
    *,
    markdown: str | None = None,
    source_hash: str | None = "hash-source",
) -> None:
    paper_id = "paper-1"
    parsed_dir = root / "parser-runs" / "pymupdf" / "parsed" / paper_id
    normalizer_dir = root / "normalizer" / paper_id
    parsed_dir.mkdir(parents=True)
    normalizer_dir.mkdir(parents=True)
    body = markdown or """# paper-1

## Page 1

Repeated caption.

## Page 2

Intro Title

Table 1: Result table.

Figure 1: Architecture diagram.

The model uses scaled dot product attention.
"""
    parser_meta = {"parser": "pymupdf"}
    if source_hash is not None:
        parser_meta["sourceContentHash"] = source_hash
    (parsed_dir / "document.md").write_text(body, encoding="utf-8")
    (parsed_dir / "document.json").write_text(
        json.dumps(
            {
                "markdown_text": body,
                "elements": [
                    {"type": "paragraph", "text": "Repeated caption.", "page": 1},
                    {
                        "type": "paragraph",
                        "text": "Intro Title Table 1: Result table. Figure 1: Architecture diagram.",
                        "page": 2,
                    },
                ],
                "parser_meta": parser_meta,
            }
        ),
        encoding="utf-8",
    )
    (parsed_dir / "manifest.json").write_text(
        json.dumps({"paper_id": paper_id, "parser_meta": parser_meta}),
        encoding="utf-8",
    )
    candidates = [
        {
            "candidate_id": "paper-1:section:0001",
            "candidate_type": "section_candidate",
            "source_parser": "mineru",
            "paper_id": paper_id,
            "text": "Intro Title",
            "markdown_locator": {"locatorKind": "generated_markdown"},
            "layout_element_ids": ["mineru:1"],
            "bbox": [1, 2, 3, 4],
            "page": None,
            "confidence": 0.78,
            "link_reason": "exact_markdown_text_to_layout_text",
            "non_strict_reason": ["no_original_chars_start_end"],
            "evidence_tier": "candidate_only",
            "strict": False,
            "citation_grade": False,
        },
        {
            "candidate_id": "paper-1:section:0002",
            "candidate_type": "section_candidate",
            "source_parser": "mineru",
            "paper_id": paper_id,
            "text": "Repeated caption.",
            "markdown_locator": {},
            "layout_element_ids": ["mineru:2"],
            "confidence": 0.78,
            "non_strict_reason": [],
        },
        {
            "candidate_id": "paper-1:table:0001",
            "candidate_type": "table_candidate",
            "source_parser": "mineru",
            "paper_id": paper_id,
            "text": "Table 1: Result table.",
            "tableRows": [[{"text": "Model"}, {"text": "Score"}]],
            "tableCellCitationGrade": False,
            "layout_element_ids": ["mineru:table:1"],
            "confidence": 0.45,
            "non_strict_reason": ["not_table_cell_citation_grade"],
        },
        {
            "candidate_id": "paper-1:figure-caption:0001",
            "candidate_type": "figure_caption_candidate",
            "source_parser": "mineru",
            "paper_id": paper_id,
            "text": "Figure 1: Architecture diagram.",
            "layout_element_ids": ["mineru:figure:1"],
            "confidence": 0.45,
            "non_strict_reason": ["caption_to_figure_link_incomplete"],
        },
        {
            "candidate_id": "paper-1:equation:0001",
            "candidate_type": "equation_candidate",
            "source_parser": "mineru",
            "paper_id": paper_id,
            "text": "softmax(QK^T)",
            "layout_element_ids": ["mineru:eq:1"],
            "confidence": 0.5,
            "non_strict_reason": ["equation_quote_candidate_only"],
        },
        {
            "candidate_id": "paper-1:reading-order:0001",
            "candidate_type": "reading_order_candidate",
            "source_parser": "mineru",
            "paper_id": paper_id,
            "text": "layout-only row",
            "layout_element_ids": ["mineru:reading:1"],
            "confidence": 0.3,
            "non_strict_reason": ["reading_order_layout_candidate_only"],
        },
    ]
    (normalizer_dir / "mineru-normalizer-candidates.json").write_text(
        json.dumps(
            {
                "schema": "knowledge-hub.paper.mineru-normalizer-audit.v1",
                "status": "ok",
                "paperId": paper_id,
                "candidates": candidates,
            }
        ),
        encoding="utf-8",
    )


def _build(root: Path) -> dict:
    return build_mineru_source_alignment_audit(input_root=root, paper_ids=["paper-1"])


def _candidate(payload: dict, candidate_id: str) -> dict:
    return next(item for item in payload["candidates"] if item["candidate_id"] == candidate_id)


def test_source_alignment_exact_span_recovers_page_and_hash_but_stays_non_strict(tmp_path: Path) -> None:
    _write_pilot_fixture(tmp_path)

    payload = _build(tmp_path)
    section = _candidate(payload, "paper-1:section:0001")

    assert payload["schema"] == MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID
    assert validate_payload(payload, MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID, strict=True).ok
    assert section["alignment_status"] == "aligned"
    assert section["alignment_method"] == "exact"
    assert section["chars_start"] is not None
    assert section["chars_end"] is not None
    assert section["page"] == 2
    assert section["sourceContentHash"] == "hash-source"
    assert section["classification"] == "potential_strict_candidate"
    assert section["strict_requirements_met"] is True
    assert section["strict_eligible"] is False
    assert section["citation_grade"] is False
    assert section["evidence_tier"] == "source_alignment_candidate_only"
    assert "runtime_promotion_disabled_for_tranche" in section["strict_blockers"]
    assert payload["counts"]["strictEligibleCandidates"] == 0


def test_ambiguous_candidate_is_not_given_chars_or_strict_status(tmp_path: Path) -> None:
    markdown = """# paper-1

## Page 1

Repeated caption.

## Page 2

Repeated caption.
"""
    _write_pilot_fixture(tmp_path, markdown=markdown)

    payload = _build(tmp_path)
    ambiguous = _candidate(payload, "paper-1:section:0002")

    assert ambiguous["alignment_status"] == "ambiguous"
    assert ambiguous["chars_start"] is None
    assert ambiguous["chars_end"] is None
    assert ambiguous["page"] is None
    assert ambiguous["classification"] == "blocked"
    assert "ambiguous_match" in ambiguous["strict_blockers"]


def test_normalized_or_ordered_token_alignment_remains_non_strict(tmp_path: Path) -> None:
    markdown = """# paper-1

## Page 1

Repeated caption.

## Page 2

Intro    Title

Table 1: Result table.

Figure 1: Architecture diagram.

The model uses scaled dot-product attention in this paragraph.
"""
    _write_pilot_fixture(tmp_path, markdown=markdown)

    payload = _build(tmp_path)
    section = _candidate(payload, "paper-1:section:0001")
    equation = _candidate(payload, "paper-1:equation:0001")

    assert section["alignment_status"] == "aligned"
    assert section["alignment_method"] == "normalized"
    assert section["page"] == 2
    assert section["strict_requirements_met"] is False
    assert "fuzzy_or_ambiguous_alignment" in section["strict_blockers"]
    assert equation["alignment_status"] == "failed"
    assert "equation_alignment_missing" in equation["strict_blockers"]


def test_missing_page_or_source_hash_blocks_strict_requirements(tmp_path: Path) -> None:
    markdown = """# paper-1

Intro Title
"""
    _write_pilot_fixture(tmp_path, markdown=markdown, source_hash=None)

    payload = _build(tmp_path)
    section = _candidate(payload, "paper-1:section:0001")

    assert section["alignment_status"] == "aligned"
    assert section["page"] is None
    assert section["sourceContentHash"] is None
    assert section["classification"] == "text_aligned_non_strict"
    assert "missing_page" in section["strict_blockers"]
    assert "missing_source_content_hash" in section["strict_blockers"]
    assert payload["counts"]["strictEligibleCandidates"] == 0


def test_table_and_figure_candidates_keep_artifact_specific_blockers(tmp_path: Path) -> None:
    _write_pilot_fixture(tmp_path)

    payload = _build(tmp_path)
    table = _candidate(payload, "paper-1:table:0001")
    figure = _candidate(payload, "paper-1:figure-caption:0001")

    assert table["alignment_status"] == "aligned"
    assert table["tableCellCitationGrade"] is False
    assert table["classification"] == "page_recovered_non_strict"
    assert "table_cell_provenance_missing" in table["strict_blockers"]
    assert figure["alignment_status"] == "aligned"
    assert figure["classification"] == "page_recovered_non_strict"
    assert "figure_region_link_incomplete" in figure["strict_blockers"]
    assert all(item["strict_eligible"] is False for item in payload["candidates"])


def test_report_writer_emits_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    _write_pilot_fixture(tmp_path / "input")
    payload = build_mineru_source_alignment_audit(input_root=tmp_path / "input", paper_ids=["paper-1"])

    paths = write_mineru_source_alignment_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID, strict=True).ok
    assert "All outputs remain `source_alignment_candidate_only`" in markdown
