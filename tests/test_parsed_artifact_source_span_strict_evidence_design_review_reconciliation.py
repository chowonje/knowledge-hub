from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_design_review import (
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID,
    REVIEW_STATUS_BLOCKED_NON_UNIQUE,
    REVIEW_STATUS_READY,
)
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_design_review_reconciliation import (
    FINAL_STATUS_BLOCKED_AMBIGUOUS,
    FINAL_STATUS_READY,
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_RECONCILIATION_SCHEMA_ID,
    RECOMMENDED_ACTION_BLOCKED,
    RECOMMENDED_ACTION_READY,
    SOURCE_DISAMBIGUATION_DESIGN,
    SOURCE_ORIGINAL_DESIGN_REVIEW,
    build_parsed_artifact_source_span_strict_evidence_design_review_reconciliation,
)
from knowledge_hub.papers.parsed_artifact_source_span_text_match_disambiguation_design import (
    DISAMBIGUATION_STATUS_BLOCKED_STILL_NON_UNIQUE,
    DISAMBIGUATION_STATUS_CANDIDATE,
    PARSED_ARTIFACT_SOURCE_SPAN_TEXT_MATCH_DISAMBIGUATION_DESIGN_SCHEMA_ID,
)


def _review_row(*, index: int, review_status: str, text_surface: str = "Introduction") -> dict:
    ready = review_status == REVIEW_STATUS_READY
    return {
        "review_row_id": f"review:{index:04d}",
        "design_row_id": f"offset-design:{index:04d}",
        "sourceSpanId": f"source-span:paper-1:section:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:section:{index}",
        "paper_id": "paper-1",
        "artifact_type": "section",
        "source_candidate_id": f"candidate-{index}",
        "sourceContentHash": "hash-paper-1",
        "source_file": "",
        "text_surface": text_surface,
        "design_status": "offset_authority_design_candidate_only",
        "review_status": review_status,
        "review_blockers": ["strict_evidence_design_review_only"],
        "proposed_chars": {
            "start": 10,
            "end": 22,
            "basis": "sourceContentHash",
            "normalization": "nfkc_whitespace_casefold_v1",
            "expectedSubstringSha256": "abc123",
            "sourceContentHash": "hash-paper-1",
        },
        "locator": {"page": 1, "bbox": [], "blockIndexes": [], "chars": {"start": None, "end": None}},
        "recommended_action": "queue_for_strict_evidence_design_review_packet_only",
        "readyForStrictEvidenceDesignReview": ready,
        "disambiguationRequired": review_status == REVIEW_STATUS_BLOCKED_NON_UNIQUE,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "sourceSpanUpdatedRows": 0,
    }


def _design_review_report(*rows: dict) -> dict:
    row_list = list(rows)
    ambiguous = [row for row in row_list if row.get("review_status") == REVIEW_STATUS_BLOCKED_NON_UNIQUE]
    ready = [row for row in row_list if row.get("review_status") == REVIEW_STATUS_READY]
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "offsetAuthorityDesignReportPath": "/tmp/offset.json",
            "offsetAuthorityDesignSchema": "knowledge-hub.paper.parsed-artifact-source-span-original-source-offset-authority-design.v1",
            "requestedPaperIds": [],
        },
        "counts": {
            "inputRows": len(row_list),
            "readyForStrictEvidenceDesignReviewRows": len(ready),
            "blockedNonUniqueTextMatchRows": len(ambiguous),
            "blockedMissingExpectedSubstringHashRows": 0,
            "blockedMissingCharsBasisRows": 0,
            "blockedMissingCharsOffsetsRows": 0,
            "blockedRequiresManualOrLaterExtractorReviewRows": 0,
            "blockedUnsupportedArtifactTypeRows": 0,
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
            "byReviewStatus": {},
            "byRecommendedAction": {},
        },
        "gate": {
            "designReviewPacketReady": bool(ready),
            "disambiguationQueuePresent": bool(ambiguous),
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "parsed_artifact_source_span_strict_evidence_design_review_ready",
            "recommendedNextTranche": "parsed_artifact_source_span_text_match_disambiguation_design",
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
        "sampleBlockers": [],
        "readyDesignRows": ready,
        "ambiguousDisambiguationRows": ambiguous,
        "rows": row_list,
    }


def _disambiguation_row(
    *,
    index: int,
    disambiguation_status: str,
    review_row_id: str,
    source_span_suffix: str,
) -> dict:
    return {
        "disambiguation_row_id": f"disambig:{index:04d}",
        "review_row_id": review_row_id,
        "design_row_id": f"offset-design:{index:04d}",
        "sourceSpanId": f"source-span:paper-1:section:{source_span_suffix}",
        "candidateRecordId": f"source-span-candidate:paper-1:section:{source_span_suffix}",
        "paper_id": "paper-1",
        "artifact_type": "section",
        "source_candidate_id": f"candidate-{index}",
        "sourceContentHash": "hash-paper-1",
        "source_file": "",
        "text_surface": "Ambiguous Term",
        "locator": {"page": 2, "bbox": [], "blockIndexes": [], "chars": {"start": None, "end": None}},
        "disambiguation_status": disambiguation_status,
        "disambiguation_blockers": ["text_match_disambiguation_design_only"],
        "disambiguation_method": "page_context_unique_match",
        "proposed_chars": {
            "start": 100,
            "end": 115,
            "basis": "sourceContentHash",
            "normalization": "nfkc_whitespace_casefold_v1",
            "expectedSubstringSha256": "def456",
            "sourceContentHash": "hash-paper-1",
        },
        "recommended_action": "queue_for_strict_evidence_design_review_after_disambiguation_design",
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "sourceSpanUpdatedRows": 0,
    }


def _disambiguation_report(
    *,
    candidates: list[dict],
    still_ambiguous: list[dict],
) -> dict:
    rows = [*candidates, *still_ambiguous]
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_TEXT_MATCH_DISAMBIGUATION_DESIGN_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "designReviewReportPath": "/tmp/review.json",
            "designReviewSchema": PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID,
            "papersDir": "/tmp/papers",
            "parsedRoot": "/tmp/papers/parsed",
            "requestedPaperIds": [],
        },
        "counts": {
            "inputRows": 102,
            "targetRows": len(rows),
            "disambiguationDesignCandidateOnlyRows": len(candidates),
            "blockedStillNonUniqueAfterLocatorContextRows": len(still_ambiguous),
            "blockedMissingLocatorContextRows": 0,
            "blockedMissingCandidateMatchOffsetsRows": 0,
            "blockedSourceTextUnavailableRows": 0,
            "blockedRequiresManualOrLaterExtractorReviewRows": 0,
            "blockedInputSchemaViolationRows": 0,
            "sourceSpanUpdatedRows": 0,
            "strictEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "schemaViolationCount": 0,
            "byArtifactType": {},
            "byDisambiguationStatus": {},
            "byRecommendedAction": {},
        },
        "gate": {
            "disambiguationDesignComplete": True,
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "parsed_artifact_source_span_text_match_disambiguation_design_ready",
            "recommendedNextTranche": "parsed_artifact_source_span_strict_evidence_design_review_reconciliation",
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
        "disambiguationDesignRows": candidates,
        "stillAmbiguousRows": still_ambiguous,
        "rows": rows,
    }


def test_reconciliation_combines_ready_and_disambiguated_rows(tmp_path: Path) -> None:
    review_path = tmp_path / "review.json"
    disambiguation_path = tmp_path / "disambiguation.json"

    review_path.write_text(
        json.dumps(
            _design_review_report(
                _review_row(index=1, review_status=REVIEW_STATUS_READY, text_surface="Ready One"),
                _review_row(index=2, review_status=REVIEW_STATUS_BLOCKED_NON_UNIQUE, text_surface="Ambiguous"),
            )
        ),
        encoding="utf-8",
    )
    disambiguation_path.write_text(
        json.dumps(
            _disambiguation_report(
                candidates=[
                    _disambiguation_row(
                        index=2,
                        disambiguation_status=DISAMBIGUATION_STATUS_CANDIDATE,
                        review_row_id="review:0002",
                        source_span_suffix="ambiguous",
                    )
                ],
                still_ambiguous=[],
            )
        ),
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_design_review_reconciliation(
        design_review_report_path=review_path,
        disambiguation_report_path=disambiguation_path,
    )

    assert report["counts"]["readyFromOriginalDesignReviewRows"] == 1
    assert report["counts"]["readyFromDisambiguationRows"] == 1
    assert report["counts"]["finalReadyForStrictEvidenceDesignReviewRows"] == 2
    assert report["counts"]["stillBlockedAmbiguousRows"] == 0
    assert report["counts"]["strictEvidenceCreatedRows"] == 0
    assert report["gate"]["designReviewPacketReconciled"] is True
    assert report["gate"]["recommendedNextTranche"] == (
        "parsed_artifact_source_span_strict_evidence_design_packet_review"
    )

    sources = {row["source"] for row in report["finalReadyRows"]}
    assert sources == {SOURCE_ORIGINAL_DESIGN_REVIEW, SOURCE_DISAMBIGUATION_DESIGN}
    assert all(row["final_status"] == FINAL_STATUS_READY for row in report["finalReadyRows"])
    assert all(row["recommended_action"] == RECOMMENDED_ACTION_READY for row in report["finalReadyRows"])


def test_reconciliation_preserves_still_blocked_rows(tmp_path: Path) -> None:
    review_path = tmp_path / "review.json"
    disambiguation_path = tmp_path / "disambiguation.json"

    review_path.write_text(
        json.dumps(_design_review_report(_review_row(index=1, review_status=REVIEW_STATUS_READY))),
        encoding="utf-8",
    )
    disambiguation_path.write_text(
        json.dumps(
            _disambiguation_report(
                candidates=[],
                still_ambiguous=[
                    _disambiguation_row(
                        index=9,
                        disambiguation_status=DISAMBIGUATION_STATUS_BLOCKED_STILL_NON_UNIQUE,
                        review_row_id="review:0009",
                        source_span_suffix="still-blocked",
                    )
                ],
            )
        ),
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_design_review_reconciliation(
        design_review_report_path=review_path,
        disambiguation_report_path=disambiguation_path,
    )

    assert report["counts"]["finalReadyForStrictEvidenceDesignReviewRows"] == 1
    assert report["counts"]["stillBlockedAmbiguousRows"] == 1
    assert report["manualOrExtractorFollowUpRows"][0]["final_status"] == FINAL_STATUS_BLOCKED_AMBIGUOUS
    assert report["manualOrExtractorFollowUpRows"][0]["recommended_action"] == RECOMMENDED_ACTION_BLOCKED


def test_reconciliation_report_validates_against_schema(tmp_path: Path) -> None:
    review_path = tmp_path / "review.json"
    disambiguation_path = tmp_path / "disambiguation.json"
    review_path.write_text(
        json.dumps(_design_review_report(_review_row(index=1, review_status=REVIEW_STATUS_READY))),
        encoding="utf-8",
    )
    disambiguation_path.write_text(
        json.dumps(_disambiguation_report(candidates=[], still_ambiguous=[])),
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_design_review_reconciliation(
        design_review_report_path=review_path,
        disambiguation_report_path=disambiguation_path,
    )
    validation = validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_RECONCILIATION_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors
