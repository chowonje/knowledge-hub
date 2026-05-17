from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.sectionspan_pdf_offset_selected_review_evidence_pack import (
    SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_EVIDENCE_PACK_SCHEMA_ID,
    build_sectionspan_pdf_offset_selected_review_evidence_pack,
    write_sectionspan_pdf_offset_selected_review_evidence_pack_reports,
)
from knowledge_hub.papers.source_text import source_hash_for_path


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _source(root: Path, name: str = "paper.pdf", content: str = "pdf bytes") -> Path:
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path


def _template_row(source_hash: str, **overrides: object) -> dict:
    payload = {
        "decision_row_id": "sectionspan-pdf-offset-selected-review-decision:0001",
        "source_selected_review_card_id": "sectionspan-pdf-offset-selected-review-card:0001",
        "source_sectionspan_candidate_id": "sectionspan:1706.03762:0001",
        "paper_id": "1706.03762",
        "candidate_text": "1. Introduction",
        "section_type": "numbered_section",
        "section_level": 1,
        "review_priority": "P0",
        "canonical_span": {
            "chars_start": 10,
            "chars_end": 25,
            "page": 1,
            "sourceContentHash": source_hash,
        },
        "original_pdf_span": {
            "originalPdfCharsStart": 6,
            "originalPdfCharsEnd": 21,
            "page": 1,
            "sourceContentHash": source_hash,
            "matchMethod": "exact",
            "matchConfidence": 1.0,
        },
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }
    payload.update(overrides)
    return payload


def _template(rows: list[dict], **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-template.v1",
        "status": "selected_decision_template_ready",
        "counts": {
            "templateRows": len(rows),
            "pendingDecisionRows": len(rows),
            "approvedRows": 0,
            "rejectedRows": 0,
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "selectedDecisionTemplateReady": True,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "selectedDecisionTemplateOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "decisionRows": rows,
    }
    payload.update(overrides)
    return payload


def _recovery(source: Path, source_hash: str, **overrides: object) -> dict:
    payload = {
        "schema": "knowledge-hub.paper.sectionspan-pdf-offset-recovery-dry-run.v1",
        "status": "dry_run_complete",
        "counts": {
            "strictEligibleRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
        },
        "gate": {
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
        },
        "policy": {
            "reportOnly": True,
            "dryRunOnly": True,
            "applyExecuted": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "paperContexts": {
            "1706.03762": {
                "status": "ok",
                "sourcePdfPath": str(source),
                "sourceContentHash": source_hash,
                "pageCount": 1,
                "pagesWithText": 1,
            }
        },
    }
    payload.update(overrides)
    return payload


def test_selected_review_evidence_pack_extracts_context_and_suggests_without_decision(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source_hash = source_hash_for_path(str(source))
    template = _write(tmp_path, "template.json", _template([_template_row(source_hash)]))
    recovery = _write(tmp_path, "recovery.json", _recovery(source, source_hash))

    payload = build_sectionspan_pdf_offset_selected_review_evidence_pack(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        sectionspan_pdf_offset_recovery_dry_run_report=recovery,
        pdf_page_text_loader=lambda _path: [{"page": 1, "text": "Intro\n1. Introduction\nBody"}],
    )

    assert payload["schema"] == SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_EVIDENCE_PACK_SCHEMA_ID
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_EVIDENCE_PACK_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "selected_review_evidence_pack_ready"
    assert payload["counts"]["reviewContextReadyRows"] == 1
    assert payload["counts"]["suggestedApproveForLaterPromotionDesignRows"] == 1
    assert payload["counts"]["strictEligibleRows"] == 0
    assert payload["gate"]["humanReviewComplete"] is False
    row = payload["evidenceRows"][0]
    assert row["matched_text"] == "1. Introduction"
    assert row["review_suggestion"] == "approve_for_later_promotion_design"
    assert row["strict_eligible"] is False
    assert "review_suggestions_are_not_human_review_decisions" in row["non_strict_reason"]


def test_selected_review_evidence_pack_accepts_normalized_page_text_but_remains_non_strict(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source_hash = source_hash_for_path(str(source))
    template = _write(
        tmp_path,
        "template.json",
        _template([
            _template_row(
                source_hash,
                candidate_text="1 Introduction",
                original_pdf_span={
                    "originalPdfCharsStart": 6,
                    "originalPdfCharsEnd": 20,
                    "page": 1,
                    "sourceContentHash": source_hash,
                    "matchMethod": "normalized_whitespace_case",
                    "matchConfidence": 0.95,
                },
            )
        ]),
    )
    recovery = _write(tmp_path, "recovery.json", _recovery(source, source_hash))

    payload = build_sectionspan_pdf_offset_selected_review_evidence_pack(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        sectionspan_pdf_offset_recovery_dry_run_report=recovery,
        pdf_page_text_loader=lambda _path: [{"page": 1, "text": "Intro\n1\nIntroduction\nBody"}],
    )

    row = payload["evidenceRows"][0]
    assert validate_payload(payload, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_EVIDENCE_PACK_SCHEMA_ID, strict=True).ok
    assert row["context_match_method"] == "normalized_whitespace_case"
    assert row["page_text_match"] is True
    assert row["citation_grade"] is False
    assert row["runtime_evidence"] is False


def test_selected_review_evidence_pack_blocks_mismatched_context(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source_hash = source_hash_for_path(str(source))
    template = _write(tmp_path, "template.json", _template([_template_row(source_hash)]))
    recovery = _write(tmp_path, "recovery.json", _recovery(source, source_hash))

    payload = build_sectionspan_pdf_offset_selected_review_evidence_pack(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        sectionspan_pdf_offset_recovery_dry_run_report=recovery,
        pdf_page_text_loader=lambda _path: [{"page": 1, "text": "Intro\n2. Background\nBody"}],
    )

    row = payload["evidenceRows"][0]
    assert row["review_context_status"] == "blocked_page_text_mismatch"
    assert row["review_suggestion"] == "needs_review"
    assert row["strict_eligible"] is False


def test_selected_review_evidence_pack_blocks_unsafe_upstream(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source_hash = source_hash_for_path(str(source))
    template = _write(
        tmp_path,
        "template.json",
        _template(
            [_template_row(source_hash)],
            schema="example.wrong.template.v1",
            policy={
                "reportOnly": True,
                "selectedDecisionTemplateOnly": True,
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
    recovery = _write(tmp_path, "recovery.json", _recovery(source, source_hash))

    payload = build_sectionspan_pdf_offset_selected_review_evidence_pack(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        sectionspan_pdf_offset_recovery_dry_run_report=recovery,
        pdf_page_text_loader=lambda _path: [{"page": 1, "text": "Intro\n1. Introduction\nBody"}],
    )

    assert payload["status"] == "blocked"
    assert "sectionspan_pdf_offset_selected_review_decision_template_schema_mismatch" in payload["gate"]["unsafeUpstreamFlags"]
    assert "selectedDecisionTemplate_strictEvidenceCreated_true" in payload["gate"]["unsafeUpstreamFlags"]


def test_selected_review_evidence_pack_writer_outputs_schema_valid_json_and_markdown(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source_hash = source_hash_for_path(str(source))
    template = _write(tmp_path / "input", "template.json", _template([_template_row(source_hash)]))
    recovery = _write(tmp_path / "input", "recovery.json", _recovery(source, source_hash))
    payload = build_sectionspan_pdf_offset_selected_review_evidence_pack(
        sectionspan_pdf_offset_selected_review_decision_template_report=template,
        sectionspan_pdf_offset_recovery_dry_run_report=recovery,
        pdf_page_text_loader=lambda _path: [{"page": 1, "text": "Intro\n1. Introduction\nBody"}],
    )

    paths = write_sectionspan_pdf_offset_selected_review_evidence_pack_reports(payload, tmp_path / "reports")

    assert set(paths) == {"report", "summary", "markdown"}
    report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert validate_payload(report, SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_EVIDENCE_PACK_SCHEMA_ID, strict=True).ok
    assert summary["counts"]["reviewContextReadyRows"] == 1
    assert "Suggestions are not decisions" in markdown
