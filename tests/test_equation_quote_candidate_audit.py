from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.equation_quote_candidate_audit import (
    EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID,
    build_equation_quote_candidate_report,
    write_equation_quote_candidate_reports,
)


def _source_alignment_fixture(root: Path, *, candidates: list[dict] | None = None) -> Path:
    payload = {
        "schema": "knowledge-hub.paper.mineru-source-alignment-audit.v1",
        "status": "ok",
        "candidates": candidates if candidates is not None else _source_candidates(),
    }
    path = root / "mineru-source-alignment-report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _source_candidates() -> list[dict]:
    return [
        {
            "candidate_id": "paper-1:equation:0001",
            "candidate_type": "equation_candidate",
            "paper_id": "paper-1",
            "source_parser": "mineru",
            "candidate_text": "\\mathrm { A } = B\\tag{1}",
            "alignment_status": "failed",
            "alignment_method": "none",
            "alignment_reason": "no_canonical_text_match",
            "chars_start": None,
            "chars_end": None,
            "page": None,
            "sourceContentHash": "hash-source",
            "sourceContentHashSource": "manifest",
            "confidence": 0.0,
            "source_span_locator": {},
            "mineruCandidate": {
                "layout_element_ids": ["mineru:element:1"],
                "bbox": [10, 20, 200, 240],
                "link_reason": "ordinal_equation_like_layout_match_without_page",
            },
            "classification": "blocked",
            "strict_blockers": [
                "runtime_promotion_disabled_for_tranche",
                "text_alignment_not_available",
                "missing_chars_start_end",
                "missing_page",
                "equation_alignment_missing",
                "markdown_offsets_are_generated_not_original_pdf_offsets",
            ],
            "strict_eligible": False,
            "citation_grade": False,
        },
        {
            "candidate_id": "paper-1:equation:0002",
            "candidate_type": "equation_candidate",
            "paper_id": "paper-1",
            "source_parser": "mineru",
            "candidate_text": "x = y",
            "alignment_status": "aligned",
            "alignment_method": "exact",
            "alignment_reason": "single_exact_text_match",
            "chars_start": 50,
            "chars_end": 55,
            "page": 2,
            "sourceContentHash": "hash-source",
            "sourceContentHashSource": "manifest",
            "confidence": 0.99,
            "source_span_locator": {
                "path": "document.md",
                "locatorKind": "canonical_generated_markdown",
                "chars": {"start": 50, "end": 55},
            },
            "mineruCandidate": {
                "layout_element_ids": ["mineru:element:2"],
                "bbox": [30, 40, 250, 280],
                "link_reason": "ordinal_equation_like_layout_match_without_page",
            },
            "classification": "page_recovered_non_strict",
            "strict_blockers": ["runtime_promotion_disabled_for_tranche"],
            "strict_eligible": False,
            "citation_grade": False,
        },
        {
            "candidate_id": "paper-1:section:0001",
            "candidate_type": "section_candidate",
            "paper_id": "paper-1",
            "candidate_text": "1. Introduction",
        },
    ]


def test_equation_quote_report_emits_only_equation_candidates_and_validates_schema(tmp_path: Path) -> None:
    source_path = _source_alignment_fixture(tmp_path)

    payload = build_equation_quote_candidate_report(source_path)

    assert payload["schema"] == EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID
    assert validate_payload(payload, EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["inputCandidateCount"] == 3
    assert payload["counts"]["equationQuoteCandidates"] == 2
    assert payload["counts"]["alignedEquationQuoteCandidates"] == 1
    assert payload["counts"]["layoutRegionCandidateCount"] == 2
    assert payload["counts"]["byAlignmentStatus"] == {"failed": 1, "aligned": 1}


def test_failed_equation_alignment_is_blocked_not_green(tmp_path: Path) -> None:
    source_path = _source_alignment_fixture(tmp_path)

    payload = build_equation_quote_candidate_report(source_path)
    blocked = next(item for item in payload["candidates"] if item["equation_label"] == "tag:1")

    assert blocked["canonical_alignment_status"] == "failed"
    assert blocked["chars_start"] is None
    assert blocked["chars_end"] is None
    assert blocked["page"] is None
    assert blocked["readiness"] == "blocked_alignment_incomplete"
    assert blocked["layout_region_candidate_present"] is True
    assert blocked["equation_region_link_verified"] is False
    assert blocked["equation_semantics_interpreted"] is False
    assert "equation_text_alignment_not_available" in blocked["strict_blockers"]
    assert "missing_chars_start_end" in blocked["strict_blockers"]
    assert blocked["strict_eligible"] is False
    assert blocked["citation_grade"] is False


def test_aligned_equation_quote_still_does_not_interpret_or_promote(tmp_path: Path) -> None:
    source_path = _source_alignment_fixture(tmp_path)

    payload = build_equation_quote_candidate_report(source_path)
    aligned = next(item for item in payload["candidates"] if item["equation_text"] == "x = y")

    assert aligned["sourceContentHash"] == "hash-source"
    assert aligned["chars_start"] == 50
    assert aligned["chars_end"] == 55
    assert aligned["page"] == 2
    assert aligned["evidence_tier"] == "equation_quote_candidate_only"
    assert aligned["equation_semantics_interpreted"] is False
    assert aligned["strict_eligible"] is False
    assert aligned["citation_grade"] is False
    assert "equation_semantics_not_interpreted" in aligned["strict_blockers"]
    assert payload["policy"]["equationInterpretationAllowed"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False


def test_equation_quote_report_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    source_path = _source_alignment_fixture(tmp_path / "input")
    payload = build_equation_quote_candidate_report(source_path)

    paths = write_equation_quote_candidate_reports(payload, tmp_path / "reports")

    assert set(paths) == {"candidates", "summary", "markdown"}
    candidates = json.loads(Path(paths["candidates"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(candidates, EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert "They are not strict evidence" in markdown
    assert "No equation is interpreted" in markdown
