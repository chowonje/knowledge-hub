from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.figure_caption_candidate_audit import (
    FIGURE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID,
    build_figure_caption_candidate_report,
    write_figure_caption_candidate_reports,
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
            "candidate_id": "paper-1:figure-caption:0001",
            "candidate_type": "figure_caption_candidate",
            "paper_id": "paper-1",
            "source_parser": "mineru",
            "candidate_text": "Figure 1: The model architecture.",
            "alignment_status": "aligned",
            "alignment_method": "exact",
            "alignment_reason": "single_exact_text_match",
            "chars_start": 100,
            "chars_end": 133,
            "page": 3,
            "sourceContentHash": "hash-source",
            "sourceContentHashSource": "manifest",
            "confidence": 0.99,
            "source_span_locator": {
                "path": "document.md",
                "locatorKind": "canonical_generated_markdown",
                "chars": {"start": 100, "end": 133},
            },
            "mineruCandidate": {
                "layout_element_ids": ["mineru:element:1", "mineru:element:2"],
                "bbox": [10, 20, 200, 240],
                "link_reason": "ordinal_figure_like_layout_match_without_page",
            },
            "classification": "page_recovered_non_strict",
            "strict_blockers": [
                "runtime_promotion_disabled_for_tranche",
                "figure_region_link_incomplete",
                "markdown_offsets_are_generated_not_original_pdf_offsets",
            ],
            "strict_eligible": False,
            "citation_grade": False,
        },
        {
            "candidate_id": "paper-1:figure-caption:0002",
            "candidate_type": "figure_caption_candidate",
            "paper_id": "paper-1",
            "source_parser": "mineru",
            "candidate_text": "Figure 2: A failed caption alignment.",
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
                "layout_element_ids": ["mineru:element:3"],
                "bbox": [30, 40, 250, 280],
                "link_reason": "ordinal_figure_like_layout_match_without_page",
            },
            "classification": "blocked",
            "strict_blockers": [
                "runtime_promotion_disabled_for_tranche",
                "text_alignment_not_available",
                "missing_chars_start_end",
                "missing_page",
                "figure_region_link_incomplete",
                "markdown_offsets_are_generated_not_original_pdf_offsets",
            ],
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


def test_figure_caption_report_emits_only_figure_candidates_and_validates_schema(tmp_path: Path) -> None:
    source_path = _source_alignment_fixture(tmp_path)

    payload = build_figure_caption_candidate_report(source_path)

    assert payload["schema"] == FIGURE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID
    assert validate_payload(payload, FIGURE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["inputCandidateCount"] == 3
    assert payload["counts"]["figureCaptionCandidates"] == 2
    assert payload["counts"]["alignedCaptionSpanCandidates"] == 1
    assert payload["counts"]["layoutRegionCandidateCount"] == 2
    assert payload["counts"]["byAlignmentStatus"] == {"aligned": 1, "failed": 1}


def test_aligned_caption_with_hash_page_span_and_bbox_remains_non_strict(tmp_path: Path) -> None:
    source_path = _source_alignment_fixture(tmp_path)

    payload = build_figure_caption_candidate_report(source_path)
    candidate = next(item for item in payload["candidates"] if item["figure_label"] == "Figure 1")

    assert candidate["figure_label"] == "Figure 1"
    assert candidate["caption_text"] == "The model architecture."
    assert candidate["sourceContentHash"] == "hash-source"
    assert candidate["chars_start"] == 100
    assert candidate["chars_end"] == 133
    assert candidate["page"] == 3
    assert candidate["bbox"] == [10.0, 20.0, 200.0, 240.0]
    assert candidate["layout_region_candidate_present"] is True
    assert candidate["figure_region_link_verified"] is False
    assert candidate["evidence_tier"] == "figure_caption_candidate_only"
    assert candidate["strict_eligible"] is False
    assert candidate["citation_grade"] is False
    assert "figure_region_link_incomplete" in candidate["strict_blockers"]
    assert payload["policy"]["strictEvidenceCreated"] is False
    assert payload["policy"]["parserRoutingChanged"] is False


def test_failed_caption_alignment_is_reported_as_blocked_not_green(tmp_path: Path) -> None:
    source_path = _source_alignment_fixture(tmp_path)

    payload = build_figure_caption_candidate_report(source_path)
    blocked = next(item for item in payload["candidates"] if item["figure_label"] == "Figure 2")

    assert blocked["canonical_alignment_status"] == "failed"
    assert blocked["chars_start"] is None
    assert blocked["chars_end"] is None
    assert blocked["page"] is None
    assert blocked["readiness"] == "blocked_alignment_incomplete"
    assert "caption_text_alignment_not_available" in blocked["strict_blockers"]
    assert "missing_chars_start_end" in blocked["strict_blockers"]
    assert "missing_page" in blocked["strict_blockers"]
    assert blocked["strict_eligible"] is False
    assert blocked["citation_grade"] is False


def test_figure_caption_report_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    source_path = _source_alignment_fixture(tmp_path / "input")
    payload = build_figure_caption_candidate_report(source_path)

    paths = write_figure_caption_candidate_reports(payload, tmp_path / "reports")

    assert set(paths) == {"candidates", "summary", "markdown"}
    candidates = json.loads(Path(paths["candidates"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(candidates, FIGURE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, FIGURE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID, strict=True).ok
    assert "They are not strict evidence" in markdown
    assert "caption text span is not enough" in markdown
