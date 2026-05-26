from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.figure_region_link_feasibility_audit import (
    FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID,
    build_figure_region_link_feasibility_audit,
    write_figure_region_link_feasibility_audit_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _reports(root: Path, *, wrong_schema: bool = False, normalizer_page: int | None = None) -> tuple[Path, Path]:
    figure_schema = "knowledge-hub.paper.figure-caption-candidate-report.v1"
    source_schema = "knowledge-hub.paper.mineru-source-alignment-audit.v1"
    if wrong_schema:
        figure_schema = "example.wrong.figure"
        source_schema = "example.wrong.source"
    normalizer_path = _write(
        root,
        "normalizer/paper-1/mineru-normalizer-candidates.json",
        {
            "schema": "knowledge-hub.paper.mineru-normalizer-audit.v1",
            "candidates": [
                {
                    "candidate_id": "paper-1:figure-caption:0001",
                    "paper_id": "paper-1",
                    "candidate_type": "figure_caption_candidate",
                    "text": "Figure 1: Model diagram.",
                    "layout_element_ids": ["mineru:figure:1", "mineru:caption:1"],
                    "bbox": [10, 20, 200, 220],
                    "page": normalizer_page,
                    "link_reason": "ordinal_figure_like_layout_match_without_page",
                }
            ],
        },
    )
    figure_path = _write(
        root,
        "figure-caption-candidates.json",
        {
            "schema": figure_schema,
            "candidates": [
                {
                    "candidate_id": "figurecaption:paper-1:0001",
                    "candidate_type": "figure_caption_candidate",
                    "source_candidate_id": "paper-1:figure-caption:0001",
                    "paper_id": "paper-1",
                    "candidate_text": "Figure 1: Model diagram.",
                    "figure_label": "Figure 1",
                    "caption_text": "Model diagram.",
                    "canonical_alignment_status": "aligned",
                    "alignment_method": "exact",
                    "chars_start": 100,
                    "chars_end": 123,
                    "page": 4,
                    "sourceContentHash": "source-hash",
                    "sourceContentHashSource": "manifest",
                    "layout_element_ids": ["mineru:figure:1", "mineru:caption:1"],
                    "bbox": [10, 20, 200, 220],
                    "confidence": 0.99,
                    "strict_blockers": ["figure_region_link_incomplete"],
                    "strict_eligible": False,
                    "citation_grade": False,
                },
                {
                    "candidate_id": "figurecaption:paper-1:0002",
                    "candidate_type": "figure_caption_candidate",
                    "source_candidate_id": "paper-1:figure-caption:9999",
                    "paper_id": "paper-1",
                    "candidate_text": "Figure 2: Missing layout.",
                    "figure_label": "Figure 2",
                    "caption_text": "Missing layout.",
                    "canonical_alignment_status": "failed",
                    "alignment_method": "none",
                    "chars_start": None,
                    "chars_end": None,
                    "page": None,
                    "sourceContentHash": "source-hash",
                    "layout_element_ids": ["mineru:figure:2"],
                    "bbox": [30, 40, 250, 280],
                    "confidence": 0.0,
                    "strict_blockers": ["missing_chars_start_end", "missing_page"],
                    "strict_eligible": False,
                    "citation_grade": False,
                },
            ],
        },
    )
    source_path = _write(
        root,
        "mineru-source-alignment-report.json",
        {
            "schema": source_schema,
            "papers": [
                {
                    "paperId": "paper-1",
                    "input": {"mineruNormalizerCandidatesPath": str(normalizer_path)},
                }
            ],
        },
    )
    return figure_path, source_path


def test_figure_region_link_feasibility_reports_caption_and_layout_but_no_verified_region(tmp_path: Path) -> None:
    figure_path, source_path = _reports(tmp_path)

    payload = build_figure_region_link_feasibility_audit(
        figure_caption_report=figure_path,
        mineru_source_alignment_report=source_path,
    )

    assert payload["schema"] == FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID
    assert validate_payload(payload, FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["auditedFigureCaptionCandidates"] == 2
    assert payload["counts"]["normalizerFigureCaptionMatches"] == 1
    assert payload["counts"]["captionSourceSpanCandidates"] == 1
    assert payload["counts"]["layoutRegionCandidates"] == 1
    assert payload["counts"]["regionPageRecoveredCandidates"] == 0
    assert payload["counts"]["figureRegionLinkVerifiedCandidates"] == 0
    assert payload["counts"]["strictEligibleCandidates"] == 0


def test_layout_bbox_without_region_page_remains_non_strict(tmp_path: Path) -> None:
    figure_path, source_path = _reports(tmp_path)

    payload = build_figure_region_link_feasibility_audit(
        figure_caption_report=figure_path,
        mineru_source_alignment_report=source_path,
    )
    row = next(item for item in payload["rows"] if item["normalizer_match"])

    assert row["caption_source_span_available"] is True
    assert row["layout_region_candidate_present"] is True
    assert row["region_page_recovered"] is False
    assert row["figure_region_type_verified"] is False
    assert row["figure_region_link_verified"] is False
    assert row["feasibility_status"] == "figure_region_candidate_no_region_page"
    assert row["evidence_tier"] == "figure_region_link_feasibility_candidate_only"
    assert row["strict_eligible"] is False
    assert row["citation_grade"] is False
    assert "figure_region_page_missing" in row["strict_blockers"]
    assert "figure_region_type_unverified" in row["strict_blockers"]
    assert payload["policy"]["figureRegionEvidenceCreated"] is False
    assert payload["policy"]["strictEvidenceCreated"] is False


def test_region_page_match_still_requires_type_verification(tmp_path: Path) -> None:
    figure_path, source_path = _reports(tmp_path, normalizer_page=4)

    payload = build_figure_region_link_feasibility_audit(
        figure_caption_report=figure_path,
        mineru_source_alignment_report=source_path,
    )
    row = next(item for item in payload["rows"] if item["normalizer_match"])

    assert row["region_page_recovered"] is True
    assert row["caption_page_matches_region_page"] is True
    assert row["figure_region_type_verified"] is False
    assert row["feasibility_status"] == "figure_region_candidate_type_unverified"
    assert row["figure_region_link_verified"] is False
    assert row["strict_eligible"] is False


def test_missing_normalizer_caption_is_blocked_not_green(tmp_path: Path) -> None:
    figure_path, source_path = _reports(tmp_path)

    payload = build_figure_region_link_feasibility_audit(
        figure_caption_report=figure_path,
        mineru_source_alignment_report=source_path,
    )
    blocked = next(item for item in payload["rows"] if item["source_candidate_id"] == "paper-1:figure-caption:9999")

    assert blocked["normalizer_match"] is False
    assert blocked["layout_region_candidate_present"] is False
    assert blocked["feasibility_status"] == "blocked_missing_normalizer_caption"
    assert "missing_normalizer_figure_caption_candidate" in blocked["strict_blockers"]
    assert blocked["strict_eligible"] is False


def test_figure_region_link_feasibility_blocks_wrong_input_schema_ids(tmp_path: Path) -> None:
    figure_path, source_path = _reports(tmp_path, wrong_schema=True)

    payload = build_figure_region_link_feasibility_audit(
        figure_caption_report=figure_path,
        mineru_source_alignment_report=source_path,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert set(payload["gate"]["schemaViolations"]) == {
        "figure_caption_candidate_report_schema_mismatch",
        "mineru_source_alignment_report_schema_mismatch",
    }


def test_figure_region_link_feasibility_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    figure_path, source_path = _reports(tmp_path / "input")
    payload = build_figure_region_link_feasibility_audit(
        figure_caption_report=figure_path,
        mineru_source_alignment_report=source_path,
    )

    paths = write_figure_region_link_feasibility_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"audit", "summary", "markdown"}
    audit = json.loads(Path(paths["audit"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(audit, FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID, strict=True).ok
    assert "does not verify the visual figure/image region" in markdown
