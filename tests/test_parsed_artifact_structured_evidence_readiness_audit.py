from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_structured_evidence_readiness_audit import (
    PARSED_ARTIFACT_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID,
    STATUS_BLOCKED_AMBIGUOUS_PDF_REGION,
    STATUS_BLOCKED_MANUAL_OR_LATER_EXTRACTOR,
    STATUS_BLOCKED_MISSING_LABEL_NUMBER_DISAMBIGUATION,
    STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_BLOCKED_PDF_REGION_ONLY,
    STATUS_READY,
    build_parsed_artifact_structured_evidence_readiness_audit,
    write_parsed_artifact_structured_evidence_readiness_audit_reports,
)
from knowledge_hub.papers.sectionspan_candidate_audit import SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sectionspan_row(
    *,
    candidate_id: str,
    source_hash: str,
    chars_start: int | None,
    chars_end: int | None,
    page: int | None,
    blockers: list[str] | None,
    status_hint: str = "",
    source_file: str | None = None,
    row_id: str = "",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "paper_id": "paper-1",
        "source_candidate_id": row_id or candidate_id,
        "sourceContentHash": source_hash,
        "candidate_type": "section_span_candidate",
        "chars_start": chars_start,
        "chars_end": chars_end,
        "page": page,
        "source_file": source_file or f"{candidate_id}.pdf",
        "source_tex_row_id": f"src-{candidate_id}",
        "strict_blockers": blockers or [],
        "non_strict_reason": [
            status_hint,
        ] if status_hint else [],
        "bbox": [0, 0, 10, 10] if page is not None and chars_start is None else [],
        "blockIndexes": [1, 2] if page is not None and chars_start is None else [],
        "source_span_created": False,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "parser_routing_changed": False,
        "answer_integration_changed": False,
    }


def _sectionspan_report(
    root: Path,
    *,
    wrong_schema: bool = False,
    legacy_schema: bool = False,
) -> Path:
    if wrong_schema:
        schema = "example.wrong.schema"
    elif legacy_schema:
        schema = "knowledge-hub.paper.tex-sectionspan-candidate-report.v1"
    else:
        schema = SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID
    rows = [
        _sectionspan_row(
            candidate_id="ready-section",
            row_id="ready-section",
            source_hash="hash-ready",
            chars_start=1,
            chars_end=10,
            page=1,
            blockers=[],
        ),
        _sectionspan_row(
            candidate_id="missing-hash",
            row_id="missing-hash",
            source_hash="",
            chars_start=1,
            chars_end=10,
            page=1,
            blockers=[],
            status_hint="missing source hash",
        ),
        _sectionspan_row(
            candidate_id="ambiguous-pdf",
            row_id="ambiguous-pdf",
            source_hash="hash-amb",
            chars_start=1,
            chars_end=10,
            page=2,
            blockers=["ambiguous_pdf_region"],
        ),
        _sectionspan_row(
            candidate_id="missing-label-number",
            row_id="missing-label-number",
            source_hash="hash-label",
            chars_start=1,
            chars_end=10,
            page=2,
            blockers=["missing_label_number_disambiguation"],
        ),
        _sectionspan_row(
            candidate_id="pdf-region-only",
            row_id="pdf-region-only",
            source_hash="hash-pdf",
            chars_start=None,
            chars_end=None,
            page=3,
            blockers=[],
        ),
        _sectionspan_row(
            candidate_id="manual-or-later",
            row_id="manual-or-later",
            source_hash="hash-manual",
            chars_start=1,
            chars_end=10,
            page=4,
            blockers=["manual_or_later_extractor_review_required"],
        ),
    ]
    return _write(
        root / "sectionspan.json",
        {
            "schema": schema,
            "candidates": rows,
        },
    )


def test_parsed_artifact_structured_evidence_readiness_audit_classifies_all_readiness_rows(tmp_path: Path) -> None:
    report_path = _sectionspan_report(tmp_path)

    payload = build_parsed_artifact_structured_evidence_readiness_audit(
        sectionspan_candidate_report=report_path,
    )

    assert payload["schema"] == PARSED_ARTIFACT_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID
    assert payload["status"] == "ok"
    assert validate_payload(payload, PARSED_ARTIFACT_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["inputRows"] == 6
    assert payload["counts"]["targetRows"] == 6
    by_status = {
        row["readiness_status"]: row["recommended_action"]
        for row in payload["rows"]
    }
    assert by_status[STATUS_READY] == "queue_for_explicit_source_span_promotion_executor_review"
    assert by_status[STATUS_BLOCKED_MISSING_SOURCE_HASH] == "recover_source_content_hash_before_promotion"
    assert by_status[STATUS_BLOCKED_AMBIGUOUS_PDF_REGION] == "resolve_ambiguous_pdf_region_before_promotion"
    assert by_status[STATUS_BLOCKED_MISSING_LABEL_NUMBER_DISAMBIGUATION] == "run_label_number_disambiguation_before_promotion"
    assert by_status[STATUS_BLOCKED_PDF_REGION_ONLY] == "recover_original_source_span_or_offset_authority_before_promotion"
    assert by_status[STATUS_BLOCKED_MANUAL_OR_LATER_EXTRACTOR] == "manual_or_later_extractor_review_required"

    assert payload["counts"]["promotionReviewReadyCandidateOnlyRows"] == 1
    assert payload["counts"]["blockedMissingSourceHashRows"] == 1
    assert payload["counts"]["blockedAmbiguousPdfRegionRows"] == 1
    assert payload["counts"]["blockedMissingLabelNumberRows"] == 1
    assert payload["counts"]["blockedPdfRegionOnlyRows"] == 1
    assert payload["counts"]["blockedManualOrLaterExtractorRows"] == 1



def test_parsed_artifact_structured_evidence_readiness_audit_blocks_wrong_parent_schema_ids(tmp_path: Path) -> None:
    report_path = _sectionspan_report(tmp_path, wrong_schema=True)

    payload = build_parsed_artifact_structured_evidence_readiness_audit(
        sectionspan_candidate_report=report_path,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["decision"] == "blocked"
    assert payload["gate"]["readyForPromotionReadinessReview"] is False
    assert "sectionspan_schema_mismatch" in payload["gate"]["schemaViolations"]



def test_parsed_artifact_structured_evidence_readiness_audit_accepts_legacy_tex_sectionspan_schema(tmp_path: Path) -> None:
    report_path = _sectionspan_report(tmp_path, legacy_schema=True)
    payload = build_parsed_artifact_structured_evidence_readiness_audit(
        sectionspan_candidate_report=report_path,
    )

    assert payload["status"] == "ok"
    assert "sectionspan_schema_mismatch" not in payload["warnings"]



def test_parsed_artifact_structured_evidence_readiness_audit_writer_outputs_schema_valid_report_summary_and_markdown(tmp_path: Path) -> None:
    report_path = _sectionspan_report(tmp_path)
    payload = build_parsed_artifact_structured_evidence_readiness_audit(
        sectionspan_candidate_report=report_path,
    )
    paths = write_parsed_artifact_structured_evidence_readiness_audit_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert validate_payload(report, PARSED_ARTIFACT_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID, strict=True).ok
    assert validate_payload(summary, PARSED_ARTIFACT_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID, strict=True).ok
    assert "Parsed Artifact Structured Evidence Readiness Audit" in markdown
    assert "blocked missing source hash rows" in markdown
