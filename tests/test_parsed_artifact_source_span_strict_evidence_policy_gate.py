from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_promotion_readback_review import (
    PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
    READBACK_STATUS_VALIDATED,
)
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_policy_gate import (
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
    POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY,
    POLICY_STATUS_BLOCKED_RUNTIME_OR_STRICT_FLAG,
    POLICY_STATUS_STRICT_POLICY_CANDIDATE_ONLY,
    _classify_policy_row,
    build_parsed_artifact_source_span_strict_evidence_policy_gate,
    write_parsed_artifact_source_span_strict_evidence_policy_gate_reports,
)


def _readback_row(
    *,
    index: int = 1,
    readback_status: str = READBACK_STATUS_VALIDATED,
    locator: dict | None = None,
    strict_eligible: bool = False,
) -> dict:
    return {
        "review_row_id": f"parsed-artifact-source-span-promotion-readback-review:{index:04d}",
        "sourceSpanId": f"source-span:paper-1:section:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:section:{index}",
        "runId": "run-1",
        "paper_id": "paper-1",
        "artifact_type": "section",
        "source_candidate_id": f"candidate-{index}",
        "sourceContentHash": f"source-hash-{index}",
        "source_file": "paper-1.pdf",
        "locator": locator
        or {
            "page": index,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "blockIndexes": [index],
            "chars": {"start": 10, "end": 20},
        },
        "idempotencyKey": f"idem-{index}",
        "source_span_store_path": "/tmp/paper-1.jsonl",
        "source_span_store_line": index,
        "readback_status": readback_status,
        "review_blockers": [],
        "readback_validated": readback_status == READBACK_STATUS_VALIDATED,
        "sourceSpanCreated": False,
        "strictEligible": strict_eligible,
        "citationGrade": False,
        "runtimeEvidence": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "canonicalParsedArtifactsWritten": False,
        "recommended_action": "queue_for_explicit_strict_evidence_policy_gate",
    }


def _readback_report(*rows: dict) -> dict:
    row_list = list(rows)
    validated_rows = sum(
        1 for row in row_list if row.get("readback_status") == READBACK_STATUS_VALIDATED
    )
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
        "status": "ok" if validated_rows == len(row_list) else "blocked",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "papersDir": "/tmp/papers",
            "sourceSpanStoreRoot": "/tmp/papers/structured_evidence/source_span",
            "runManifestPath": "",
            "requestedRunId": "",
            "requestedRunManifestPath": "",
            "requestedPaperIds": [],
            "runIdentity": {
                "requestedRunId": "",
                "requestedRunManifestPath": "",
                "runIdFilterMode": "none",
                "manifestCandidateRunIds": [],
                "observedRecordRunIds": [],
                "resolvedRecordRunIds": [],
                "resolution": "",
            },
        },
        "counts": {
            "inputRows": len(row_list),
            "sourceSpanRecordRows": len(row_list),
            "readbackValidatedRows": validated_rows,
            "blockedSchemaViolationRows": 0,
            "blockedMissingSourceHashRows": 0,
            "blockedMissingLocatorRows": 0,
            "blockedDuplicateIdempotencyKeyRows": 0,
            "blockedRuntimeOrStrictFlagRows": 0,
            "sourceSpanCreatedRows": 0,
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
            "byReadbackStatus": {},
            "byRecommendedAction": {},
        },
        "gate": {
            "readyForStrictEvidenceGate": True,
            "sourceSpanCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "parsed_artifact_source_span_promotion_readback_review_ready",
            "recommendedNextTranche": "parsed_artifact_source_span_strict_evidence_policy_gate",
        },
        "policy": {
            "reportOnly": True,
            "readbackOnly": True,
            "sourceSpanStoreWrite": False,
            "candidateStoreWrite": False,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
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
        "rows": row_list,
    }


def test_strict_evidence_policy_gate_marks_char_offset_rows_ready(tmp_path: Path) -> None:
    readback_path = tmp_path / "readback.json"
    readback_path.write_text(
        json.dumps(_readback_report(_readback_row(index=1)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_policy_gate(
        readback_report_path=readback_path,
    )

    assert report["status"] == "ok"
    assert report["counts"]["strictPolicyCandidateOnlyRows"] == 1
    assert report["counts"]["blockedMissingOffsetAuthorityRows"] == 0
    assert report["counts"]["strictEvidenceCreatedRows"] == 0
    assert report["rows"][0]["policy_gate_status"] == POLICY_STATUS_STRICT_POLICY_CANDIDATE_ONLY
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_strict_evidence_policy_gate_blocks_page_only_offset_authority(tmp_path: Path) -> None:
    readback_path = tmp_path / "readback.json"
    readback_path.write_text(
        json.dumps(
            _readback_report(
                _readback_row(
                    index=1,
                    locator={
                        "page": 1,
                        "bbox": [],
                        "blockIndexes": [],
                        "chars": {"start": None, "end": None},
                    },
                )
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_policy_gate(
        readback_report_path=readback_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["strictPolicyCandidateOnlyRows"] == 0
    assert report["counts"]["blockedMissingOffsetAuthorityRows"] == 1
    assert report["rows"][0]["policy_gate_status"] == POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY
    assert report["rows"][0]["strictEligible"] is False
    assert report["rows"][0]["strictEvidenceCreated"] is False


def test_strict_evidence_policy_gate_blocks_page_bbox_without_chars(tmp_path: Path) -> None:
    readback_path = tmp_path / "readback.json"
    readback_path.write_text(
        json.dumps(
            _readback_report(
                _readback_row(
                    index=1,
                    locator={
                        "page": 1,
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                        "blockIndexes": [1],
                        "chars": {"start": None, "end": None},
                    },
                )
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_policy_gate(
        readback_report_path=readback_path,
    )

    assert report["counts"]["strictPolicyCandidateOnlyRows"] == 0
    assert report["counts"]["blockedMissingOffsetAuthorityRows"] == 1
    assert report["rows"][0]["offset_authority_mode"] == "page_bbox_non_strict_candidate_only"


def test_strict_evidence_policy_gate_blocks_readback_not_ready(tmp_path: Path) -> None:
    readback_path = tmp_path / "readback.json"
    readback_path.write_text(
        json.dumps(
            _readback_report(
                _readback_row(index=1, readback_status="blocked_schema_violation"),
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_policy_gate(
        readback_report_path=readback_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedReadbackNotReadyRows"] == 1
    assert report["counts"]["strictEvidenceCreatedRows"] == 0


def test_classify_policy_row_blocks_runtime_or_strict_flags() -> None:
    row = _readback_row(index=1)
    row["strictEligible"] = True
    status, blockers, offset_mode = _classify_policy_row(row)
    assert status == POLICY_STATUS_BLOCKED_RUNTIME_OR_STRICT_FLAG
    assert "strictEligible_true" in blockers
    assert offset_mode == "chars_offset_authority"


def test_strict_evidence_policy_gate_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    readback_path = tmp_path / "readback.json"
    readback_path.write_text(
        json.dumps(_readback_report(_readback_row(index=1)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = build_parsed_artifact_source_span_strict_evidence_policy_gate(
        readback_report_path=readback_path,
    )

    paths = write_parsed_artifact_source_span_strict_evidence_policy_gate_reports(
        report,
        tmp_path / "reports",
    )

    written_report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert "blocked missing offset authority rows" in markdown
    assert validate_payload(
        written_report,
        PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok
