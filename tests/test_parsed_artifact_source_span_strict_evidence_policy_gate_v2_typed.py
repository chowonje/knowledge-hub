from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_promotion_readback_review import (
    PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
    READBACK_STATUS_VALIDATED,
)
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed import (
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_V2_TYPED_SCHEMA_ID,
    POLICY_STATUS_BLOCKED_LOCATOR_BASIS_UNKNOWN,
    POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY,
    POLICY_STATUS_BLOCKED_MISSING_STRUCTURED_AUTHORITY,
    POLICY_STATUS_STRICT_STRUCTURED,
    POLICY_STATUS_STRICT_TEXT,
    build_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed,
    write_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed_reports,
)


def _readback_row(
    *,
    index: int = 1,
    artifact_type: str = "section",
    readback_status: str = READBACK_STATUS_VALIDATED,
    locator: dict | None = None,
) -> dict:
    return {
        "review_row_id": f"parsed-artifact-source-span-promotion-readback-review:{index:04d}",
        "sourceSpanId": f"source-span:paper-1:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:{artifact_type}:{index}",
        "runId": "run-1",
        "paper_id": "paper-1",
        "artifact_type": artifact_type,
        "source_candidate_id": f"candidate-{index}",
        "sourceContentHash": f"source-hash-{index}",
        "source_file": "paper-1.pdf",
        "locator": locator
        or {
            "page": index,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "blockIndexes": [index],
            "chars": {
                "start": 10,
                "end": 20,
                "basis": "sourceContentHash",
                "normalization": "nfkc",
                "expectedSubstringSha256": "abc123",
            },
        },
        "idempotencyKey": f"idem-{index}",
        "source_span_store_path": "/tmp/paper-1.jsonl",
        "source_span_store_line": index,
        "readback_status": readback_status,
        "review_blockers": [],
        "readback_validated": readback_status == READBACK_STATUS_VALIDATED,
        "sourceSpanCreated": False,
        "strictEligible": False,
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
            "readyForStrictEvidenceGate": validated_rows == len(row_list),
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


def test_typed_strict_policy_gate_marks_text_rows_ready(tmp_path: Path) -> None:
    readback_path = tmp_path / "readback.json"
    readback_path.write_text(
        json.dumps(_readback_report(_readback_row(index=1)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed(
        readback_report_path=readback_path,
    )

    assert report["status"] == "ok"
    assert report["counts"]["strictTextPolicyCandidateOnlyRows"] == 1
    assert report["counts"]["strictEvidenceCreatedRows"] == 0
    assert report["rows"][0]["policy_gate_status"] == POLICY_STATUS_STRICT_TEXT
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_V2_TYPED_SCHEMA_ID,
        strict=True,
    ).ok


def test_typed_strict_policy_gate_blocks_page_only_text_rows(tmp_path: Path) -> None:
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

    report = build_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed(
        readback_report_path=readback_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["strictTextPolicyCandidateOnlyRows"] == 0
    assert report["counts"]["blockedMissingOffsetAuthorityRows"] == 1
    assert report["rows"][0]["policy_gate_status"] == POLICY_STATUS_BLOCKED_MISSING_OFFSET_AUTHORITY


def test_typed_strict_policy_gate_blocks_missing_chars_basis(tmp_path: Path) -> None:
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
                        "chars": {
                            "start": 1,
                            "end": 5,
                            "basis": "pdf_bbox",
                            "normalization": "nfkc",
                            "expectedSubstringSha256": "",
                        },
                    },
                )
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed(
        readback_report_path=readback_path,
    )

    assert report["counts"]["strictTextPolicyCandidateOnlyRows"] == 0
    assert report["rows"][0]["policy_gate_status"] == POLICY_STATUS_BLOCKED_LOCATOR_BASIS_UNKNOWN


def test_typed_strict_policy_gate_blocks_table_without_structured_authority(
    tmp_path: Path,
) -> None:
    readback_path = tmp_path / "readback.json"
    readback_path.write_text(
        json.dumps(
            _readback_report(_readback_row(index=1, artifact_type="table")),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed(
        readback_report_path=readback_path,
    )

    assert report["counts"]["strictStructuredPolicyCandidateOnlyRows"] == 0
    assert report["counts"]["blockedMissingStructuredAuthorityRows"] == 1
    assert report["rows"][0]["policy_gate_status"] == POLICY_STATUS_BLOCKED_MISSING_STRUCTURED_AUTHORITY


def test_typed_strict_policy_gate_accepts_structured_table_authority(tmp_path: Path) -> None:
    row = _readback_row(index=1, artifact_type="table")
    row["tableId"] = "table-1"
    row["tableRow"] = "2"
    row["tableCol"] = "3"
    row["cellRawText"] = "42"
    row["cellNormalizedValue"] = "42"
    row["cellContentHash"] = "cell-hash"
    readback_path = tmp_path / "readback.json"
    readback_path.write_text(
        json.dumps(_readback_report(row), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed(
        readback_report_path=readback_path,
    )

    assert report["status"] == "ok"
    assert report["counts"]["strictStructuredPolicyCandidateOnlyRows"] == 1
    assert report["rows"][0]["policy_gate_status"] == POLICY_STATUS_STRICT_STRUCTURED


def test_typed_strict_policy_gate_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    readback_path = tmp_path / "readback.json"
    readback_path.write_text(
        json.dumps(_readback_report(_readback_row(index=1)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = build_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed(
        readback_report_path=readback_path,
    )

    paths = write_parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed_reports(
        report,
        tmp_path / "reports",
    )

    written_report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert validate_payload(
        written_report,
        PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_POLICY_GATE_V2_TYPED_SCHEMA_ID,
        strict=True,
    ).ok
