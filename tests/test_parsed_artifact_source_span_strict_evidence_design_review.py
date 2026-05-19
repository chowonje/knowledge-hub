from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_original_source_offset_authority_design import (
    DESIGN_STATUS_BLOCKED_NON_UNIQUE_TEXT_MATCH,
    DESIGN_STATUS_OFFSET_AUTHORITY_CANDIDATE,
    PARSED_ARTIFACT_SOURCE_SPAN_ORIGINAL_SOURCE_OFFSET_AUTHORITY_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_design_review import (
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID,
    REVIEW_STATUS_BLOCKED_NON_UNIQUE,
    REVIEW_STATUS_READY,
    build_parsed_artifact_source_span_strict_evidence_design_review,
    write_parsed_artifact_source_span_strict_evidence_design_review_reports,
)


def _design_row(
    *,
    index: int = 1,
    design_status: str = DESIGN_STATUS_OFFSET_AUTHORITY_CANDIDATE,
    artifact_type: str = "section",
    proposed_chars: dict | None = None,
) -> dict:
    return {
        "design_row_id": f"offset-design:{index:04d}",
        "policy_gate_row_id": f"gate:{index:04d}",
        "sourceSpanId": f"source-span:paper-1:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:{artifact_type}:{index}",
        "paper_id": "paper-1",
        "artifact_type": artifact_type,
        "source_candidate_id": f"candidate-{index}",
        "sourceContentHash": f"hash-{index}",
        "source_file": "",
        "text_surface": "Introduction",
        "design_status": design_status,
        "design_blockers": ["original_source_offset_authority_design_only"],
        "proposed_chars": proposed_chars
        or {
            "start": 10,
            "end": 22,
            "basis": "sourceContentHash",
            "normalization": "nfkc_whitespace_casefold_v1",
            "expectedSubstringSha256": "abc123",
            "sourceContentHash": f"hash-{index}",
        },
        "locator": {"page": 1, "bbox": [], "blockIndexes": []},
        "source_resolution": {"canonicalTextLength": 100},
        "recommended_action": "queue_for_strict_evidence_design_review_packet_only",
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "sourceSpanUpdatedRows": 0,
    }


def _design_report(*rows: dict) -> dict:
    row_list = list(rows)
    ready = sum(1 for row in row_list if row.get("design_status") == DESIGN_STATUS_OFFSET_AUTHORITY_CANDIDATE)
    ambiguous = sum(
        1 for row in row_list if row.get("design_status") == DESIGN_STATUS_BLOCKED_NON_UNIQUE_TEXT_MATCH
    )
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_ORIGINAL_SOURCE_OFFSET_AUTHORITY_DESIGN_SCHEMA_ID,
        "status": "blocked",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "policyGateReportPath": "/tmp/gate.json",
            "policyGateSchema": "knowledge-hub.paper.parsed-artifact-source-span-strict-evidence-policy-gate-v2-typed.v1",
            "papersDir": "/tmp/papers",
            "parsedRoot": "/tmp/papers/parsed",
            "sectionspanCandidateReportPath": "/tmp/sectionspan.json",
            "figureCaptionCandidateReportPath": "/tmp/figure.json",
            "requestedPaperIds": [],
        },
        "counts": {
            "inputRows": len(row_list),
            "targetRows": len(row_list),
            "offsetAuthorityDesignCandidateOnlyRows": ready,
            "blockedMissingSourceHashRows": 0,
            "blockedMissingSourceFileRows": 0,
            "blockedMissingLocatorContextRows": 0,
            "blockedMissingTextSurfaceRows": 0,
            "blockedSourceTextUnavailableRows": 0,
            "blockedNonUniqueTextMatchRows": ambiguous,
            "blockedRequiresManualOrLaterExtractorReviewRows": 0,
            "blockedHashBasisUnavailableRows": 0,
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
            "byDesignStatus": {},
            "byRecommendedAction": {},
        },
        "gate": {
            "offsetAuthorityDesignComplete": False,
            "readyForStrictEvidenceDesignReview": False,
            "strictEvidenceCreated": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "blocked",
            "recommendedNextTranche": "parsed_artifact_source_span_strict_evidence_design_review",
        },
        "policy": {
            "reportOnly": True,
            "sourceSpanStoreWrite": False,
            "strictEvidenceCreated": False,
            "strictEligibleMutation": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "warnings": [],
        "rows": row_list,
    }


def test_design_review_ready_and_ambiguous_rows(tmp_path: Path) -> None:
    design_path = tmp_path / "design.json"
    design_path.write_text(
        json.dumps(
            _design_report(
                _design_row(index=1),
                _design_row(
                    index=2,
                    design_status=DESIGN_STATUS_BLOCKED_NON_UNIQUE_TEXT_MATCH,
                    proposed_chars={},
                ),
            )
        ),
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_design_review(
        offset_authority_design_report_path=design_path,
    )

    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 2
    assert report["counts"]["readyForStrictEvidenceDesignReviewRows"] == 1
    assert report["counts"]["blockedNonUniqueTextMatchRows"] == 1
    assert len(report["readyDesignRows"]) == 1
    assert len(report["ambiguousDisambiguationRows"]) == 1
    assert report["rows"][0]["review_status"] == REVIEW_STATUS_READY
    assert report["rows"][1]["review_status"] == REVIEW_STATUS_BLOCKED_NON_UNIQUE
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_design_review_blocks_invalid_proposed_chars(tmp_path: Path) -> None:
    design_path = tmp_path / "design.json"
    design_path.write_text(
        json.dumps(
            _design_report(
                _design_row(
                    index=1,
                    proposed_chars={
                        "start": 10,
                        "end": 20,
                        "basis": "wrong",
                        "normalization": "nfkc_whitespace_casefold_v1",
                        "expectedSubstringSha256": "abc",
                    },
                )
            )
        ),
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_design_review(
        offset_authority_design_report_path=design_path,
    )

    assert report["counts"]["readyForStrictEvidenceDesignReviewRows"] == 0
    assert report["rows"][0]["review_status"] == "blocked_missing_chars_basis"


def test_write_design_review_reports(tmp_path: Path) -> None:
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(_design_report(_design_row(index=1))), encoding="utf-8")
    report = build_parsed_artifact_source_span_strict_evidence_design_review(
        offset_authority_design_report_path=design_path,
    )
    out_dir = tmp_path / "out"
    paths = write_parsed_artifact_source_span_strict_evidence_design_review_reports(report, out_dir)
    assert Path(paths["report"]).name == "parsed-artifact-source-span-strict-evidence-design-review.json"
    loaded = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert loaded["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID
