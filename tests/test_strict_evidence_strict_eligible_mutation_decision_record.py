from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_post_pilot_promotion_hold_review import (
    HOLD_STATUS_ACTIVE,
    STRICT_EVIDENCE_POST_PILOT_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
)
from knowledge_hub.papers.strict_evidence_strict_eligible_mutation_decision_record import (
    DECISION_SEPARATE_ELIGIBILITY_RECORD,
    DECISION_STATUS_CANDIDATE_ONLY,
    DEFAULT_HOLD_REVIEW_REPORT_PATH,
    STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID,
    build_strict_evidence_strict_eligible_mutation_decision_record,
    write_strict_evidence_strict_eligible_mutation_decision_record_reports,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _future_promotion_readiness_checklist() -> list[dict]:
    return [
        {
            "id": f"item-{index}",
            "title": f"Checklist item {index}",
            "status": "pending",
            "requiredBeforePromotion": True,
        }
        for index in range(8)
    ]


def _blocked_downstream_gate_matrix() -> dict:
    return {
        name: {
            "allowed": False,
            "ready": False,
            "reason": f"blocked_until_explicit_{name}_tranche",
            "holdActive": True,
        }
        for name in (
            "strictEligibleMutation",
            "citationGradeEvidence",
            "runtimeEvidence",
            "parserRouting",
            "answerIntegration",
            "databaseMutation",
            "reindexOrReembed",
        )
    }


def _synthetic_hold_row(*, index: int, artifact_type: str) -> dict:
    return {
        "hold_row_id": f"strict-evidence-post-pilot-promotion-hold-review:{index:04d}",
        "completion_row_id": f"completion:{index:04d}",
        "readback_row_id": f"readback:{index:04d}",
        "strictEvidenceId": f"strict-evidence:paper-1:{artifact_type}:{index}",
        "sourceSpanId": f"source-span:paper-1:{artifact_type}:{index}",
        "candidateRecordId": f"candidate:paper-1:{artifact_type}:{index}",
        "paper_id": "paper-1",
        "artifact_type": artifact_type,
        "manifestType": (
            "strict_evidence_text_section_pilot_executor_apply"
            if artifact_type == "section"
            else "strict_evidence_figure_caption_pilot_executor_apply"
        ),
        "completion_status": "strict_evidence_pilot_tranche_complete_candidate_only",
        "hold_status": HOLD_STATUS_ACTIVE,
        "hold_blockers": [],
        "postPilotPromotionHoldActive": True,
        "strictEligibleMutationAllowed": False,
        "citationGradeAllowed": False,
        "runtimeEvidenceAllowed": False,
        "parserRoutingAllowed": False,
        "answerIntegrationAllowed": False,
        "databaseMutationAllowed": False,
        "reindexOrReembedAllowed": False,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "recommended_action": "post_pilot_promotion_hold_active",
    }


def _synthetic_hold_review_report(rows: list[dict]) -> dict:
    section_rows = sum(1 for row in rows if row.get("artifact_type") == "section")
    figure_rows = sum(1 for row in rows if row.get("artifact_type") == "figure")
    return {
        "schema": STRICT_EVIDENCE_POST_PILOT_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-20T00:00:00Z",
        "input": {
            "completionGateReportPath": "/tmp/completion.json",
            "completionGateReportSchema": "knowledge-hub.paper.strict-evidence-pilot-tranche-completion-gate.v1",
            "completionGateReportStatus": "ok",
            "completionDecision": "strict_evidence_pilot_tranche_complete_candidate_only",
            "requestedPaperIds": [],
            "sectionRunManifestPath": "/tmp/section.json",
            "figureCaptionRunManifestPath": "/tmp/figure.json",
            "expectedPolicyCandidateRows": len(rows),
            "expectedHoldActiveRows": len(rows),
        },
        "counts": {
            "inputPolicyCandidateRows": len(rows),
            "validatedPilotRows": len(rows),
            "holdActiveRows": len(rows),
            "sectionHoldRows": section_rows,
            "figureCaptionHoldRows": figure_rows,
            "blockedCompletionGateNotCandidateRows": 0,
            "blockedPilotNotCompleteRows": 0,
            "blockedDownstreamGateAlreadyEnabledRows": 0,
            "blockedStoreRowCountChangedRows": 0,
            "blockedInputSchemaViolationRows": 0,
            "strictEligibleMutationAllowedRows": 0,
            "citationGradeAllowedRows": 0,
            "runtimeEvidenceAllowedRows": 0,
            "parserRoutingAllowedRows": 0,
            "answerIntegrationAllowedRows": 0,
            "strictEvidenceStoreRows": 99 if len(rows) == 99 else len(rows),
            "sourceSpanStoreRows": 102 if len(rows) == 99 else len(rows) + 3,
            "strictEvidenceWriteRows": 0,
            "strictEvidenceCreatedRows": 0,
            "citationGradeEvidenceCreatedRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "parserRoutingChangedRows": 0,
            "answerIntegrationChangedRows": 0,
            "databaseMutationRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "sourceSpanUpdatedRows": 0,
            "manifestWriteRows": 0,
            "reindexOrReembedRows": 0,
            "schemaViolationCount": 0,
            "downstreamGateAllowedCount": 0,
            "byArtifactType": {"section": section_rows, "figure": figure_rows},
            "byHoldStatus": {HOLD_STATUS_ACTIVE: len(rows)},
            "byRecommendedAction": {"post_pilot_promotion_hold_active": len(rows)},
        },
        "blockedDownstreamGateMatrix": _blocked_downstream_gate_matrix(),
        "futurePromotionReadinessChecklist": _future_promotion_readiness_checklist(),
        "noMutationPolicyMatrix": {
            "reportOnly": True,
            "holdReviewOnly": True,
            "manifestWrite": False,
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
        "gate": {
            "postPilotPromotionHoldReviewReady": True,
            "holdDecision": "strict_evidence_post_pilot_promotion_hold_active",
            "strictEligibleMutationAllowed": False,
            "citationGradeAllowed": False,
            "runtimeEvidenceAllowed": False,
            "parserRoutingAllowed": False,
            "answerIntegrationAllowed": False,
            "databaseMutationAllowed": False,
            "reindexOrReembedAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "sourceSpanStoreWriteAllowed": False,
            "runManifestWriteAllowed": False,
            "schemaViolations": [],
            "recommendedNextTranche": "strict_evidence_strict_eligible_mutation_decision_record",
        },
        "policy": {"reportOnly": True, "holdReviewOnly": True},
        "warnings": [],
        "rows": rows,
    }


def test_decision_record_prefers_append_only_eligibility_records(tmp_path: Path) -> None:
    rows = [
        _synthetic_hold_row(index=index, artifact_type="section")
        for index in range(1, 46)
    ] + [
        _synthetic_hold_row(index=index, artifact_type="figure")
        for index in range(1, 55)
    ]
    hold_path = tmp_path / "hold-review.json"
    _write_json(hold_path, _synthetic_hold_review_report(rows))

    report = build_strict_evidence_strict_eligible_mutation_decision_record(
        hold_review_report_path=hold_path,
    )

    assert report["status"] == "ok"
    counts = report["counts"]
    assert counts["decisionRecordCandidateOnlyRows"] == 99
    assert counts["sectionDecisionRows"] == 45
    assert counts["figureCaptionDecisionRows"] == 54
    assert counts["strictEligibleMutationAllowedRows"] == 0
    assert counts["eligibilityRecordWriteRows"] == 0
    assert report["gate"]["decision"] == DECISION_SEPARATE_ELIGIBILITY_RECORD
    assert report["gate"]["strictEvidenceInPlaceMutationAllowed"] is False
    assert report["gate"]["strictEligibleBooleanMutationAllowed"] is False
    assert report["gate"]["recommendedNextTranche"] == "strict_evidence_eligibility_record_contract"
    assert {row["decision_status"] for row in report["rows"]} == {DECISION_STATUS_CANDIDATE_ONLY}
    assert validate_payload(
        report,
        STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID,
        strict=True,
    ).ok


def test_decision_record_blocks_invalid_hold_review_report(tmp_path: Path) -> None:
    hold_path = tmp_path / "hold-review.json"
    _write_json(hold_path, {"schema": "wrong.schema", "status": "ok", "rows": []})

    report = build_strict_evidence_strict_eligible_mutation_decision_record(
        hold_review_report_path=hold_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["gate"]["recommendedNextTranche"] == (
        "strict_evidence_post_pilot_promotion_hold_review_repair"
    )
    assert validate_payload(
        report,
        STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID,
        strict=True,
    ).ok


def test_decision_record_writer_outputs_schema_valid_reports(tmp_path: Path, monkeypatch) -> None:
    rows = [
        _synthetic_hold_row(index=1, artifact_type="section"),
        _synthetic_hold_row(index=1, artifact_type="figure"),
    ]
    payload = _synthetic_hold_review_report(rows)
    payload["counts"]["strictEvidenceStoreRows"] = 2
    payload["counts"]["sourceSpanStoreRows"] = 5
    hold_path = tmp_path / "hold-review.json"
    _write_json(hold_path, payload)

    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_strict_eligible_mutation_decision_record.EXPECTED_POLICY_CANDIDATE_ROWS",
        2,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_strict_eligible_mutation_decision_record.EXPECTED_SECTION_MANIFEST_ROWS",
        1,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_strict_eligible_mutation_decision_record.EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS",
        1,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_strict_eligible_mutation_decision_record.EXPECTED_STRICT_EVIDENCE_STORE_ROWS",
        2,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_strict_eligible_mutation_decision_record.EXPECTED_SOURCE_SPAN_STORE_ROWS",
        5,
    )

    report = build_strict_evidence_strict_eligible_mutation_decision_record(
        hold_review_report_path=hold_path,
    )
    paths = write_strict_evidence_strict_eligible_mutation_decision_record_reports(
        report,
        tmp_path / "reports",
    )
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))

    assert written["status"] == "ok"
    assert Path(paths["summary"]).is_file()
    assert Path(paths["markdown"]).is_file()
    assert validate_payload(
        written,
        STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID,
        strict=True,
    ).ok


def test_decision_record_integrated_measured_local_report() -> None:
    if not DEFAULT_HOLD_REVIEW_REPORT_PATH.is_file():
        return

    report = build_strict_evidence_strict_eligible_mutation_decision_record()
    assert report["status"] == "ok"
    assert report["counts"]["decisionRecordCandidateOnlyRows"] == 99
    assert report["gate"]["strictEligibleMutationAllowed"] is False
    assert validate_payload(
        report,
        STRICT_EVIDENCE_STRICT_ELIGIBLE_MUTATION_DECISION_RECORD_SCHEMA_ID,
        strict=True,
    ).ok
