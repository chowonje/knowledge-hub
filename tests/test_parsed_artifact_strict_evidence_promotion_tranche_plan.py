from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_strict_evidence_policy_gate import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
    POLICY_STATUS_CANDIDATE_ONLY,
    build_parsed_artifact_strict_evidence_policy_gate,
    write_parsed_artifact_strict_evidence_policy_gate_reports,
)
from knowledge_hub.papers.parsed_artifact_strict_evidence_promotion_tranche_plan import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID,
    PLAN_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT,
    PLAN_STATUS_CANDIDATE_ONLY,
    TRANCHE_FIGURE_CAPTION_PILOT,
    TRANCHE_TEXT_SECTION_PILOT,
    build_parsed_artifact_strict_evidence_promotion_tranche_plan,
    write_parsed_artifact_strict_evidence_promotion_tranche_plan_reports,
)
from tests.test_parsed_artifact_strict_evidence_policy_gate import (
    _ready_readback_report,
    _write_readback_report,
)


def _synthetic_policy_gate_row(
    *,
    index: int = 1,
    artifact_type: str = "section",
    paper_id: str = "paper-1",
) -> dict:
    return {
        "policy_gate_row_id": f"parsed-artifact-strict-evidence-policy-gate:{index:04d}",
        "readback_review_row_id": f"parsed-artifact-strict-evidence-promotion-readback-review:{index:04d}",
        "strictEvidenceId": f"strict-evidence:{paper_id}:{artifact_type}:{index}",
        "sourceSpanId": f"source-span:{paper_id}:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:{paper_id}:{artifact_type}:{index}",
        "runId": "run-1",
        "paper_id": paper_id,
        "artifact_type": artifact_type,
        "sourceContentHash": "hash-1",
        "idempotencyKey": f"strict-idem-{index}",
        "strict_evidence_store_path": f"/tmp/strict/{paper_id}.jsonl",
        "strict_evidence_store_line": index,
        "readback_status": "readback_validated",
        "readback_validated": True,
        "source_span_reference_found": True,
        "source_span_reference_hash_match": True,
        "verbatimSubstringSha256": "abc123",
        "authority_chars": {
            "start": 0,
            "end": 12,
            "basis": "sourceContentHash",
            "normalization": "nfkc_whitespace_casefold_v1",
            "expectedSubstringSha256": "abc123",
        },
        "policy_gate_status": POLICY_STATUS_CANDIDATE_ONLY,
        "policy_blockers": [],
        "strictEvidencePolicyCandidateOnly": True,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "canonicalParsedArtifactsWritten": False,
        "sourceSpanUpdatedRows": 0,
        "recommended_action": "queue_for_explicit_strict_evidence_promotion_tranche",
    }


def _synthetic_policy_gate_report(rows: list[dict]) -> dict:
    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "readbackReportPath": "/tmp/readback.json",
            "readbackSchema": "knowledge-hub.paper.parsed-artifact-strict-evidence-promotion-readback-review.v1",
            "readbackReportStatus": "ok",
            "requestedPaperIds": [],
        },
        "counts": {
            "inputRows": len(rows),
            "readbackValidatedRows": len(rows),
            "strictEvidencePolicyCandidateOnlyRows": len(rows),
            "blockedInputReportSchemaViolationRows": 0,
            "blockedReadbackNotValidatedRows": 0,
            "blockedMissingVerbatimHashRows": 0,
            "blockedMissingAuthorityCharsRows": 0,
            "blockedInvalidAuthorityBasisRows": 0,
            "blockedUnsupportedNormalizationRows": 0,
            "blockedRuntimeOrCitationFlagViolationRows": 0,
            "blockedMissingSourceSpanReferenceRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "sourceSpanUpdatedRows": 0,
            "schemaViolationCount": 0,
            "byPaperId": {},
            "byArtifactType": {},
            "byPolicyGateStatus": {POLICY_STATUS_CANDIDATE_ONLY: len(rows)},
            "byRecommendedAction": {},
        },
        "gate": {
            "strictEvidencePolicyGateReady": True,
            "strictEligibleMutationAllowed": False,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "parsed_artifact_strict_evidence_policy_gate_ready",
            "recommendedNextTranche": "parsed_artifact_strict_evidence_promotion_tranche_plan",
        },
        "policy": {
            "reportOnly": True,
            "policyGateOnly": True,
            "strictEvidenceStoreWrite": False,
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
        "rows": rows,
    }


def test_promotion_tranche_plan_assigns_section_and_figure_pilots(tmp_path: Path) -> None:
    policy_gate_path = tmp_path / "policy-gate.json"
    _write_readback_report(
        policy_gate_path,
        _synthetic_policy_gate_report(
            [
                _synthetic_policy_gate_row(index=1, artifact_type="section"),
                _synthetic_policy_gate_row(index=2, artifact_type="figure"),
            ]
        ),
    )

    report = build_parsed_artifact_strict_evidence_promotion_tranche_plan(
        policy_gate_report_path=policy_gate_path,
    )

    assert report["schema"] == PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID
    assert report["status"] == "ok"
    assert report["counts"]["plannedPromotionRows"] == 2
    assert report["counts"]["sectionPilotRows"] == 1
    assert report["counts"]["figureCaptionPilotRows"] == 1
    assert report["gate"]["promotionTranchePlanReady"] is True
    assert report["gate"]["recommendedNextTranche"] == TRANCHE_TEXT_SECTION_PILOT
    assert len(report["blockedLater"]) >= 2
    tranches = {item["trancheId"]: item for item in report["tranches"]}
    assert tranches[TRANCHE_TEXT_SECTION_PILOT]["rowCount"] == 1
    assert tranches[TRANCHE_FIGURE_CAPTION_PILOT]["rowCount"] == 1
    assert {row["plan_status"] for row in report["rows"]} == {PLAN_STATUS_CANDIDATE_ONLY}
    assert validate_payload(
        report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID,
        strict=True,
    ).ok


def test_promotion_tranche_plan_blocks_policy_gate_and_artifact_violations(tmp_path: Path) -> None:
    rows = [
        _synthetic_policy_gate_row(index=1, artifact_type="section"),
        _synthetic_policy_gate_row(index=2, artifact_type="equation"),
    ]
    rows[0]["policy_gate_status"] = "blocked_missing_verbatim_hash"
    rows[0]["strictEvidencePolicyCandidateOnly"] = False
    policy_gate_path = tmp_path / "policy-gate.json"
    _write_readback_report(policy_gate_path, _synthetic_policy_gate_report(rows))

    report = build_parsed_artifact_strict_evidence_promotion_tranche_plan(
        policy_gate_report_path=policy_gate_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["plannedPromotionRows"] == 0
    assert report["counts"]["blockedUnsupportedArtifactTypeRows"] == 1
    assert validate_payload(
        report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID,
        strict=True,
    ).ok


def test_promotion_tranche_plan_blocks_unsupported_artifact_type_when_candidate(tmp_path: Path) -> None:
    policy_gate_path = tmp_path / "policy-gate.json"
    _write_readback_report(
        policy_gate_path,
        _synthetic_policy_gate_report([_synthetic_policy_gate_row(index=1, artifact_type="table")]),
    )

    report = build_parsed_artifact_strict_evidence_promotion_tranche_plan(
        policy_gate_report_path=policy_gate_path,
    )

    assert report["status"] == "blocked"
    assert report["rows"][0]["plan_status"] == PLAN_STATUS_BLOCKED_UNSUPPORTED_ARTIFACT
    assert validate_payload(
        report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID,
        strict=True,
    ).ok


def test_promotion_tranche_plan_blocks_invalid_input_schema(tmp_path: Path) -> None:
    policy_gate_report = _synthetic_policy_gate_report([_synthetic_policy_gate_row()])
    policy_gate_report["schema"] = "wrong.schema"
    policy_gate_path = tmp_path / "policy-gate.json"
    _write_readback_report(policy_gate_path, policy_gate_report)

    report = build_parsed_artifact_strict_evidence_promotion_tranche_plan(
        policy_gate_report_path=policy_gate_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["gate"]["promotionTranchePlanReady"] is False
    assert validate_payload(
        report,
        PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID,
        strict=True,
    ).ok


def test_promotion_tranche_plan_from_integrated_policy_gate(tmp_path: Path) -> None:
    readback_report_path = tmp_path / "readback.json"
    _write_readback_report(readback_report_path, _ready_readback_report(tmp_path))
    policy_gate_report = build_parsed_artifact_strict_evidence_policy_gate(
        readback_report_path=readback_report_path,
    )
    policy_gate_path = tmp_path / "policy-gate.json"
    write_parsed_artifact_strict_evidence_policy_gate_reports(policy_gate_report, policy_gate_path)

    report = build_parsed_artifact_strict_evidence_promotion_tranche_plan(
        policy_gate_report_path=policy_gate_path / "parsed-artifact-strict-evidence-policy-gate.json",
    )

    assert report["status"] == "ok"
    assert report["counts"]["plannedPromotionRows"] == 2
    assert report["counts"]["sectionPilotRows"] == 2
    assert report["counts"]["strictEvidenceWriteRows"] == 0


def test_promotion_tranche_plan_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    policy_gate_path = tmp_path / "policy-gate.json"
    _write_readback_report(
        policy_gate_path,
        _synthetic_policy_gate_report([_synthetic_policy_gate_row()]),
    )
    report = build_parsed_artifact_strict_evidence_promotion_tranche_plan(
        policy_gate_report_path=policy_gate_path,
    )
    paths = write_parsed_artifact_strict_evidence_promotion_tranche_plan_reports(
        report,
        tmp_path / "reports",
    )
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert validate_payload(
        written,
        PARSED_ARTIFACT_STRICT_EVIDENCE_PROMOTION_TRANCHE_PLAN_SCHEMA_ID,
        strict=True,
    ).ok
