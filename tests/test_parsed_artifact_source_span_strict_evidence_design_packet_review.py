from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_design_packet_review import (
    PACKET_REVIEW_STATUS_BLOCKED_MISSING_CHARS,
    PACKET_REVIEW_STATUS_BLOCKED_RUNTIME_FLAGS,
    PACKET_REVIEW_STATUS_READY,
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_PACKET_REVIEW_SCHEMA_ID,
    PACKET_REVIEW_RECOMMENDED_ACTION_READY,
    build_parsed_artifact_source_span_strict_evidence_design_packet_review,
)
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_design_review_reconciliation import (
    FINAL_STATUS_READY,
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_RECONCILIATION_SCHEMA_ID,
    RECOMMENDED_ACTION_READY as RECONCILIATION_ACTION_READY,
    SOURCE_ORIGINAL_DESIGN_REVIEW,
)


def _final_ready_row(*, index: int = 1, strict_eligible: bool = False) -> dict:
    return {
        "reconciliation_row_id": f"reconciliation:ready:{index:04d}",
        "source": SOURCE_ORIGINAL_DESIGN_REVIEW,
        "review_row_id": f"review:{index:04d}",
        "design_row_id": f"offset-design:{index:04d}",
        "disambiguation_row_id": "",
        "sourceSpanId": f"source-span:paper-1:section:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:section:{index}",
        "paper_id": "paper-1",
        "artifact_type": "section",
        "sourceContentHash": "hash-paper-1",
        "source_file": "",
        "text_surface": "Introduction",
        "proposed_chars": {
            "start": 10,
            "end": 22,
            "basis": "sourceContentHash",
            "normalization": "nfkc_whitespace_casefold_v1",
            "expectedSubstringSha256": "abc123",
            "sourceContentHash": "hash-paper-1",
        },
        "final_status": FINAL_STATUS_READY,
        "recommended_action": RECONCILIATION_ACTION_READY,
        "readyForStrictEvidenceDesignReview": True,
        "strictEligible": strict_eligible,
        "strictEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "sourceSpanUpdatedRows": 0,
    }


def _manual_row(*, index: int = 9) -> dict:
    return {
        "reconciliation_row_id": f"reconciliation:blocked:{index:04d}",
        "source": "disambiguation_design",
        "review_row_id": f"review:{index:04d}",
        "design_row_id": f"offset-design:{index:04d}",
        "sourceSpanId": f"source-span:paper-1:section:blocked-{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:section:blocked-{index}",
        "paper_id": "paper-1",
        "artifact_type": "section",
        "sourceContentHash": "hash-paper-1",
        "source_file": "",
        "final_status": "blocked_still_ambiguous_after_disambiguation",
        "recommended_action": "queue_for_manual_or_later_extractor_disambiguation",
        "readyForStrictEvidenceDesignReview": False,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "sourceSpanUpdatedRows": 0,
    }


def _reconciliation_report(
    *,
    ready_rows: list[dict],
    manual_rows: list[dict],
) -> dict:
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_RECONCILIATION_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "designReviewReportPath": "/tmp/review.json",
            "designReviewSchema": "knowledge-hub.paper.parsed-artifact-source-span-strict-evidence-design-review.v1",
            "disambiguationReportPath": "/tmp/disambiguation.json",
            "disambiguationSchema": "knowledge-hub.paper.parsed-artifact-source-span-text-match-disambiguation-design.v1",
        },
        "counts": {
            "inputDesignReviewRows": len(ready_rows) + len(manual_rows),
            "inputDisambiguationRows": len(manual_rows),
            "readyFromOriginalDesignReviewRows": len(ready_rows),
            "readyFromDisambiguationRows": 0,
            "finalReadyForStrictEvidenceDesignReviewRows": len(ready_rows),
            "stillBlockedAmbiguousRows": len(manual_rows),
            "blockedInputSchemaViolationRows": 0,
            "sourceSpanUpdatedRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "schemaViolationCount": 0,
            "byPaperId": {},
            "byArtifactType": {},
            "bySource": {},
            "byFinalStatus": {},
            "byRecommendedAction": {},
        },
        "gate": {
            "designReviewPacketReconciled": True,
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "parsed_artifact_source_span_strict_evidence_design_review_reconciliation_ready",
            "recommendedNextTranche": "parsed_artifact_source_span_strict_evidence_design_packet_review",
        },
        "policy": {
            "reportOnly": True,
            "sourceSpanStoreWrite": False,
            "strictEvidenceCreated": False,
            "strictEligibleMutation": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "warnings": [],
        "finalReadyRows": ready_rows,
        "manualOrExtractorFollowUpRows": manual_rows,
        "rows": [*ready_rows, *manual_rows],
    }


def test_packet_review_marks_valid_rows_ready(tmp_path: Path) -> None:
    reconciliation_path = tmp_path / "reconciliation.json"
    reconciliation_path.write_text(
        json.dumps(
            _reconciliation_report(
                ready_rows=[_final_ready_row(index=1), _final_ready_row(index=2)],
                manual_rows=[_manual_row()],
            )
        ),
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_design_packet_review(
        reconciliation_report_path=reconciliation_path,
    )

    assert report["counts"]["packetCandidateRows"] == 2
    assert report["counts"]["designPacketReviewReadyRows"] == 2
    assert report["counts"]["excludedManualOrExtractorRows"] == 1
    assert report["gate"]["designPacketReviewReady"] is True
    assert report["gate"]["recommendedNextTranche"] == "parsed_artifact_strict_evidence_record_contract"
    assert all(row["packet_review_status"] == PACKET_REVIEW_STATUS_READY for row in report["packetRows"])
    assert all(
        row["recommended_action"] == PACKET_REVIEW_RECOMMENDED_ACTION_READY for row in report["packetRows"]
    )


def test_packet_review_blocks_runtime_flag_violation(tmp_path: Path) -> None:
    violating_row = _final_ready_row(index=1)
    violating_row["runtimeEvidence"] = True
    reconciliation_path = tmp_path / "reconciliation.json"
    reconciliation_path.write_text(
        json.dumps(
            _reconciliation_report(
                ready_rows=[violating_row],
                manual_rows=[],
            )
        ),
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_design_packet_review(
        reconciliation_report_path=reconciliation_path,
    )

    assert report["counts"]["designPacketReviewReadyRows"] == 0
    assert report["counts"]["blockedRuntimeOrStrictFlagViolationRows"] == 1
    assert report["packetRows"][0]["packet_review_status"] == PACKET_REVIEW_STATUS_BLOCKED_RUNTIME_FLAGS
    assert report["gate"]["designPacketReviewReady"] is False


def test_packet_review_blocks_missing_chars(tmp_path: Path) -> None:
    bad_row = _final_ready_row(index=3)
    bad_row["proposed_chars"] = {"basis": "sourceContentHash"}
    reconciliation_path = tmp_path / "reconciliation.json"
    reconciliation_path.write_text(
        json.dumps(_reconciliation_report(ready_rows=[bad_row], manual_rows=[])),
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_design_packet_review(
        reconciliation_report_path=reconciliation_path,
    )

    assert report["packetRows"][0]["packet_review_status"] == PACKET_REVIEW_STATUS_BLOCKED_MISSING_CHARS
    assert report["gate"]["designPacketReviewReady"] is False


def test_packet_review_report_validates_against_schema(tmp_path: Path) -> None:
    reconciliation_path = tmp_path / "reconciliation.json"
    reconciliation_path.write_text(
        json.dumps(_reconciliation_report(ready_rows=[_final_ready_row()], manual_rows=[])),
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_design_packet_review(
        reconciliation_report_path=reconciliation_path,
    )
    validation = validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_PACKET_REVIEW_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors
