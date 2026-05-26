from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_pilot_tranche_completion_gate import (
    COMPLETION_STATUS_COMPLETE,
    STRICT_EVIDENCE_PILOT_TRANCHE_COMPLETION_GATE_SCHEMA_ID,
)
from knowledge_hub.papers.strict_evidence_post_pilot_promotion_hold_review import (
    DEFAULT_COMPLETION_GATE_REPORT_PATH,
    HOLD_STATUS_ACTIVE,
    HOLD_STATUS_BLOCKED_INPUT_SCHEMA,
    STRICT_EVIDENCE_POST_PILOT_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
    build_strict_evidence_post_pilot_promotion_hold_review,
    write_strict_evidence_post_pilot_promotion_hold_review_reports,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _blocked_later_gates() -> dict:
    return {
        name: {
            "ready": False,
            "allowed": False,
            "reason": f"blocked_until_explicit_post_pilot_{name}_tranche",
        }
        for name in (
            "citationGradeEvidence",
            "runtimeEvidence",
            "parserRouting",
            "answerIntegration",
            "strictEligibleMutation",
        )
    }


def _synthetic_completion_row(*, index: int, artifact_type: str) -> dict:
    manifest_type = (
        "strict_evidence_text_section_pilot_executor_apply"
        if artifact_type == "section"
        else "strict_evidence_figure_caption_pilot_executor_apply"
    )
    return {
        "completion_row_id": f"strict-evidence-pilot-tranche-completion-gate:{index:04d}",
        "readback_row_id": f"strict-evidence-pilot-tranche-manifest-readback-review:{index:04d}",
        "strictEvidenceId": f"strict-evidence:paper-1:{artifact_type}:{index}",
        "sourceSpanId": f"source-span:paper-1:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:{artifact_type}:{index}",
        "paper_id": "paper-1",
        "artifact_type": artifact_type,
        "manifestType": manifest_type,
        "completion_status": COMPLETION_STATUS_COMPLETE,
        "completion_blockers": [],
        "pilotTrancheCompleteCandidateOnly": True,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "recommended_action": "strict_evidence_pilot_tranche_complete_candidate_only",
    }


def _synthetic_completion_gate_report(rows: list[dict]) -> dict:
    section_rows = sum(1 for row in rows if row.get("artifact_type") == "section")
    figure_rows = sum(1 for row in rows if row.get("artifact_type") == "figure")
    return {
        "schema": STRICT_EVIDENCE_PILOT_TRANCHE_COMPLETION_GATE_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "manifestReadbackReportPath": "/tmp/readback.json",
            "manifestReadbackReportSchema": "readback.schema",
            "manifestReadbackReportStatus": "ok",
            "requestedPaperIds": [],
            "sectionRunManifestPath": "/tmp/section-manifest.json",
            "figureCaptionRunManifestPath": "/tmp/figure-manifest.json",
            "expectedPolicyCandidateRows": len(rows),
            "expectedSectionValidatedRows": section_rows,
            "expectedFigureCaptionValidatedRows": figure_rows,
            "expectedStrictEvidenceStoreRows": 99 if len(rows) == 99 else len(rows),
            "expectedSourceSpanStoreRows": 102 if len(rows) == 99 else len(rows) + 3,
        },
        "counts": {
            "inputPolicyCandidateRows": len(rows),
            "validatedPilotRows": len(rows),
            "sectionValidatedRows": section_rows,
            "figureCaptionValidatedRows": figure_rows,
            "completionCandidateOnlyRows": len(rows),
            "blockedManifestReadbackNotValidatedRows": 0,
            "blockedMissingPolicyCandidateRows": 0,
            "blockedUnexpectedManifestRows": 0,
            "blockedDuplicateStrictEvidenceIdRows": 0,
            "blockedStoreRowCountChangedRows": 0,
            "blockedRuntimeOrCitationFlagViolationRows": 0,
            "blockedInputSchemaViolationRows": 0,
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
            "schemaViolationCount": 0,
            "byArtifactType": {"section": section_rows, "figure": figure_rows},
            "byCompletionStatus": {COMPLETION_STATUS_COMPLETE: len(rows)},
            "byRecommendedAction": {COMPLETION_STATUS_COMPLETE: len(rows)},
        },
        "blockedLaterGates": _blocked_later_gates(),
        "noMutationPolicyMatrix": {
            "reportOnly": True,
            "completionGateOnly": True,
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
            "pilotTrancheCompletionGateReady": True,
            "completionDecision": COMPLETION_STATUS_COMPLETE,
            "strictEligibleMutationAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "sourceSpanStoreWriteAllowed": False,
            "runManifestWriteAllowed": False,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "recommendedNextTranche": "strict_evidence_post_pilot_promotion_hold_review",
        },
        "policy": {
            "reportOnly": True,
            "completionGateOnly": True,
            "strictEvidenceStoreWrite": False,
            "sourceSpanStoreWrite": False,
            "reindexOrReembed": False,
            "databaseMutation": False,
        },
        "warnings": [],
        "rows": rows,
    }


def test_hold_review_marks_full_pilot_on_hold(tmp_path: Path) -> None:
    rows = [
        _synthetic_completion_row(index=index, artifact_type="section")
        for index in range(1, 46)
    ] + [
        _synthetic_completion_row(index=index, artifact_type="figure")
        for index in range(1, 55)
    ]
    completion_path = tmp_path / "completion-gate.json"
    _write_json(completion_path, _synthetic_completion_gate_report(rows))

    report = build_strict_evidence_post_pilot_promotion_hold_review(
        completion_gate_report_path=completion_path,
    )

    assert report["status"] == "ok"
    counts = report["counts"]
    assert counts["inputPolicyCandidateRows"] == 99
    assert counts["validatedPilotRows"] == 99
    assert counts["holdActiveRows"] == 99
    assert counts["sectionHoldRows"] == 45
    assert counts["figureCaptionHoldRows"] == 54
    assert counts["strictEligibleMutationAllowedRows"] == 0
    assert counts["citationGradeAllowedRows"] == 0
    assert counts["runtimeEvidenceAllowedRows"] == 0
    assert counts["parserRoutingAllowedRows"] == 0
    assert counts["answerIntegrationAllowedRows"] == 0
    assert counts["strictEvidenceStoreRows"] == 99
    assert counts["sourceSpanStoreRows"] == 102
    assert report["gate"]["holdDecision"] == "strict_evidence_post_pilot_promotion_hold_active"
    assert len(report["futurePromotionReadinessChecklist"]) >= 8
    assert validate_payload(
        report,
        STRICT_EVIDENCE_POST_PILOT_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_hold_review_blocks_invalid_completion_gate_report(tmp_path: Path) -> None:
    completion_path = tmp_path / "completion-gate.json"
    _write_json(completion_path, {"schema": "wrong.schema", "status": "ok", "rows": []})

    report = build_strict_evidence_post_pilot_promotion_hold_review(
        completion_gate_report_path=completion_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0


def test_hold_review_writer_outputs_schema_valid_reports(tmp_path: Path, monkeypatch) -> None:
    rows = [
        _synthetic_completion_row(index=1, artifact_type="section"),
        _synthetic_completion_row(index=1, artifact_type="figure"),
    ]
    completion_path = tmp_path / "completion-gate.json"
    payload = _synthetic_completion_gate_report(rows)
    payload["counts"]["strictEvidenceStoreRows"] = 2
    payload["counts"]["sourceSpanStoreRows"] = 5
    payload["input"]["expectedStrictEvidenceStoreRows"] = 2
    payload["input"]["expectedSourceSpanStoreRows"] = 5
    _write_json(completion_path, payload)

    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_post_pilot_promotion_hold_review.EXPECTED_POLICY_CANDIDATE_ROWS",
        2,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_post_pilot_promotion_hold_review.EXPECTED_SECTION_MANIFEST_ROWS",
        1,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_post_pilot_promotion_hold_review.EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS",
        1,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_post_pilot_promotion_hold_review.EXPECTED_STRICT_EVIDENCE_STORE_ROWS",
        2,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_post_pilot_promotion_hold_review.EXPECTED_SOURCE_SPAN_STORE_ROWS",
        5,
    )

    report = build_strict_evidence_post_pilot_promotion_hold_review(
        completion_gate_report_path=completion_path,
    )
    paths = write_strict_evidence_post_pilot_promotion_hold_review_reports(
        report,
        tmp_path / "reports",
    )
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert written["status"] == "ok"
    assert {row["hold_status"] for row in written["rows"]} == {HOLD_STATUS_ACTIVE}
    assert validate_payload(
        written,
        STRICT_EVIDENCE_POST_PILOT_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_hold_review_integrated_measured_local_report() -> None:
    if not DEFAULT_COMPLETION_GATE_REPORT_PATH.is_file():
        return

    report = build_strict_evidence_post_pilot_promotion_hold_review()
    assert report["status"] == "ok"
    assert report["counts"]["holdActiveRows"] == 99
    assert validate_payload(
        report,
        STRICT_EVIDENCE_POST_PILOT_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
