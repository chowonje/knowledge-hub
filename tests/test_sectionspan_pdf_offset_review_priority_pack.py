from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_review_priority_pack import (
    SECTIONSPAN_PDF_OFFSET_REVIEW_PRIORITY_PACK_SCHEMA_ID,
    build_sectionspan_pdf_offset_review_priority_pack,
    write_sectionspan_pdf_offset_review_priority_pack_reports,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _record_row(
    row_id: str,
    *,
    paper_id: str = "1706.03762",
    text: str = "1. Introduction",
    section_type: str = "numbered_section",
    section_level: int = 1,
    match_method: str = "exact",
    recorded_decision: str = "needs_review",
) -> dict:
    return {
        "record_row_id": row_id,
        "source_decision_row_id": f"decision:{row_id}",
        "source_gate_row_id": f"gate:{row_id}",
        "source_review_card_id": f"review:{row_id}",
        "source_sectionspan_candidate_id": f"sectionspan:{paper_id}:{row_id}",
        "paper_id": paper_id,
        "candidate_text": text,
        "section_type": section_type,
        "section_level": section_level,
        "canonical_span": {
            "chars_start": 10,
            "chars_end": 25,
            "page": 1,
            "sourceContentHash": "hash1",
        },
        "original_pdf_span": {
            "originalPdfCharsStart": 9,
            "originalPdfCharsEnd": 24,
            "page": 1,
            "sourceContentHash": "hash1",
            "matchMethod": match_method,
            "matchConfidence": 1.0 if match_method == "exact" else 0.9,
        },
        "page_agreement": True,
        "source_hash_agreement": True,
        "recorded_decision": recorded_decision,
        "evidence_tier": "sectionspan_pdf_offset_review_decision_record_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }


def _decision_record(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-review-decision-record.v1",
        "status": "decision_record_required",
        "counts": {
            "recordRows": len(rows),
            "needsReviewRows": sum(1 for item in rows if item.get("recorded_decision") == "needs_review"),
            "approvedForLaterPromotionDesignRows": 0,
            "rejectedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "decisionRecordReady": True,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "decisionRecordOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "decisionRecords": rows,
    }
    payload.update(overrides)
    return payload


def test_priority_pack_selects_bounded_initial_review_rows_and_validates_schema(tmp_path: Path) -> None:
    rows = [
        _record_row("a1", paper_id="1706.03762", text="Abstract", section_type="abstract", section_level=0),
        _record_row("a2", paper_id="1506.02640", text="Abstract", section_type="abstract", section_level=0),
        _record_row("n1", paper_id="1706.03762", text="1. Introduction", section_level=1),
        _record_row("n2", paper_id="1506.02640", text="1. Introduction", section_level=1),
        _record_row("n3", paper_id="1706.03762", text="2. Background", section_level=1, match_method="normalized_whitespace_case"),
        _record_row("b1", paper_id="1706.03762", text="References", section_type="backmatter", section_level=0),
    ]
    record = _write(tmp_path, "record.json", _decision_record(rows))

    payload = build_sectionspan_pdf_offset_review_priority_pack(
        sectionspan_pdf_offset_review_decision_record_report=record,
        max_initial_cards=4,
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_REVIEW_PRIORITY_PACK_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_REVIEW_PRIORITY_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "priority_pack_ready"
    assert payload["counts"]["priorityCardRows"] == 6
    assert payload["counts"]["selectedInitialReviewRows"] == 4
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["runtimePromotionAllowed"] is False
    selected = [item for item in payload["priorityCards"] if item["selected_for_initial_review"]]
    assert len(selected) == 4
    assert any(item["section_type"] == "abstract" for item in selected)
    assert all(item["strict_eligible"] is False for item in selected)


def test_priority_pack_uses_only_needs_review_rows(tmp_path: Path) -> None:
    rows = [
        _record_row("pending"),
        _record_row("approved", recorded_decision="approved_for_later_promotion_design"),
        _record_row("rejected", recorded_decision="rejected_keep_candidate_only"),
    ]
    record = _write(tmp_path, "record.json", _decision_record(rows))

    payload = build_sectionspan_pdf_offset_review_priority_pack(
        sectionspan_pdf_offset_review_decision_record_report=record
    )

    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_REVIEW_PRIORITY_PACK_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["priorityCardRows"] == 1
    assert payload["priorityCards"][0]["source_decision_record_row_id"] == "pending"
    assert payload["priorityCards"][0]["recorded_decision"] == "needs_review"


def test_priority_pack_reports_no_pending_rows_without_creating_evidence(tmp_path: Path) -> None:
    record = _write(
        tmp_path,
        "record.json",
        _decision_record([_record_row("approved", recorded_decision="approved_for_later_promotion_design")]),
    )

    payload = build_sectionspan_pdf_offset_review_priority_pack(
        sectionspan_pdf_offset_review_decision_record_report=record
    )

    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_REVIEW_PRIORITY_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "no_pending_review_rows"
    assert payload["counts"]["priorityCardRows"] == 0
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["policy"]["strictEvidenceCreated"] is False


def test_priority_pack_blocks_unsafe_decision_record(tmp_path: Path) -> None:
    record = _write(
        tmp_path,
        "record.json",
        _decision_record(
            [_record_row("pending")],
            schema="example.wrong.decision-record.v1",
            policy={
                "reportOnly": True,
                "decisionRecordOnly": True,
                "strictEvidenceCreated": True,
                "runtimePromotionAllowed": False,
                "parserRoutingChanged": False,
                "canonicalParsedArtifactsWritten": False,
                "databaseMutation": False,
                "reindexOrReembed": False,
                "answerIntegrationChanged": False,
            },
        ),
    )

    payload = build_sectionspan_pdf_offset_review_priority_pack(
        sectionspan_pdf_offset_review_decision_record_report=record
    )

    assert payload["status"] == "blocked"
    assert "sectionspan_pdf_offset_review_decision_record_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "decisionRecord_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_priority_pack_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    record = _write(tmp_path / "input", "record.json", _decision_record([_record_row("pending")]))
    payload = build_sectionspan_pdf_offset_review_priority_pack(
        sectionspan_pdf_offset_review_decision_record_report=record
    )

    paths = write_sectionspan_pdf_offset_review_priority_pack_reports(payload, tmp_path / "reports")

    assert set(paths) == {"priorityPack", "summary", "markdown"}
    priority_pack = json.loads(Path(paths["priorityPack"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(priority_pack, SECTIONSPAN_PDF_OFFSET_REVIEW_PRIORITY_PACK_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["needsReviewRows"] == 1
    assert "This pack is a review-prioritization report only" in markdown
