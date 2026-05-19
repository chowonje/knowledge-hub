from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_pilot_tranche_completion_gate import (
    COMPLETION_STATUS_BLOCKED_INPUT_SCHEMA,
    COMPLETION_STATUS_COMPLETE,
    DEFAULT_MANIFEST_READBACK_REPORT_PATH,
    STRICT_EVIDENCE_PILOT_TRANCHE_COMPLETION_GATE_SCHEMA_ID,
    build_strict_evidence_pilot_tranche_completion_gate,
    write_strict_evidence_pilot_tranche_completion_gate_reports,
)
from knowledge_hub.papers.strict_evidence_pilot_tranche_manifest_readback_review import (
    READBACK_STATUS_VALIDATED,
    STRICT_EVIDENCE_PILOT_TRANCHE_MANIFEST_READBACK_REVIEW_SCHEMA_ID,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _synthetic_readback_row(*, index: int, artifact_type: str) -> dict:
    manifest_type = (
        "strict_evidence_text_section_pilot_executor_apply"
        if artifact_type == "section"
        else "strict_evidence_figure_caption_pilot_executor_apply"
    )
    return {
        "readback_row_id": f"strict-evidence-pilot-tranche-manifest-readback-review:{index:04d}",
        "policy_gate_row_id": f"parsed-artifact-strict-evidence-policy-gate:{index:04d}",
        "strictEvidenceId": f"strict-evidence:paper-1:{artifact_type}:{index}",
        "sourceSpanId": f"source-span:paper-1:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:{artifact_type}:{index}",
        "paper_id": "paper-1",
        "artifact_type": artifact_type,
        "manifestType": manifest_type,
        "readback_status": READBACK_STATUS_VALIDATED,
        "readback_blockers": [],
        "policy_gate_status": "strict_evidence_policy_candidate_only",
        "sectionManifestFound": artifact_type == "section",
        "figureCaptionManifestFound": artifact_type == "figure",
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "recommended_action": "pilot_manifest_readback_validated",
    }


def _synthetic_readback_report(rows: list[dict]) -> dict:
    section_rows = sum(1 for row in rows if row.get("artifact_type") == "section")
    figure_rows = sum(1 for row in rows if row.get("artifact_type") == "figure")
    return {
        "schema": STRICT_EVIDENCE_PILOT_TRANCHE_MANIFEST_READBACK_REVIEW_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "sectionApplyReportPath": "/tmp/section-apply.json",
            "figureCaptionApplyReportPath": "/tmp/figure-apply.json",
            "policyGateReportPath": "/tmp/policy-gate.json",
            "papersDir": "/tmp/papers",
            "sectionRunManifestPath": "/tmp/section-manifest.json",
            "figureCaptionRunManifestPath": "/tmp/figure-manifest.json",
            "expectedPolicyCandidateRows": len(rows),
            "expectedSectionManifestRows": section_rows,
            "expectedFigureCaptionManifestRows": figure_rows,
            "expectedStrictEvidenceStoreRows": 99 if len(rows) == 99 else len(rows),
            "expectedSourceSpanStoreRows": 102 if len(rows) == 99 else len(rows) + 3,
        },
        "counts": {
            "inputPolicyCandidateRows": len(rows),
            "sectionManifestRows": section_rows,
            "figureCaptionManifestRows": figure_rows,
            "combinedManifestRows": len(rows),
            "pilotManifestReadbackValidatedRows": len(rows),
            "missingPolicyCandidateRows": 0,
            "unexpectedManifestRows": 0,
            "duplicateStrictEvidenceIdRows": 0,
            "blockedManifestMissingRows": 0,
            "blockedManifestSchemaOrShapeViolationRows": 0,
            "blockedArtifactTypeMismatchRows": 0,
            "blockedRuntimeOrCitationFlagViolationRows": 0,
            "blockedStoreRowCountChangedRows": 0,
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
            "byPaperId": {"paper-1": len(rows)},
            "byArtifactType": {"section": section_rows, "figure": figure_rows},
            "byManifestType": {},
            "byReadbackStatus": {READBACK_STATUS_VALIDATED: len(rows)},
            "byRecommendedAction": {"pilot_manifest_readback_validated": len(rows)},
        },
        "diagnostics": {
            "duplicateStrictEvidenceIds": [],
            "missingPolicyCandidateIds": [],
            "unexpectedManifestRowIds": [],
        },
        "noMutationPolicyMatrix": {
            "readbackOnly": True,
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
            "pilotManifestReadbackReviewReady": True,
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
            "decision": "strict_evidence_pilot_tranche_manifest_readback_validated",
            "recommendedNextTranche": "strict_evidence_pilot_tranche_completion_gate",
        },
        "policy": {"reportOnly": True, "readbackOnly": True},
        "warnings": [],
        "rows": rows,
    }


def test_completion_gate_marks_full_pilot_complete(tmp_path: Path, monkeypatch) -> None:
    rows = [
        _synthetic_readback_row(index=index, artifact_type="section")
        for index in range(1, 46)
    ] + [
        _synthetic_readback_row(index=index, artifact_type="figure")
        for index in range(1, 55)
    ]
    readback_path = tmp_path / "readback.json"
    _write_json(readback_path, _synthetic_readback_report(rows))

    report = build_strict_evidence_pilot_tranche_completion_gate(
        manifest_readback_report_path=readback_path,
    )

    assert report["status"] == "ok"
    counts = report["counts"]
    assert counts["inputPolicyCandidateRows"] == 99
    assert counts["validatedPilotRows"] == 99
    assert counts["sectionValidatedRows"] == 45
    assert counts["figureCaptionValidatedRows"] == 54
    assert counts["completionCandidateOnlyRows"] == 99
    assert counts["strictEvidenceStoreRows"] == 99
    assert counts["sourceSpanStoreRows"] == 102
    assert report["gate"]["completionDecision"] == "strict_evidence_pilot_tranche_complete_candidate_only"
    assert report["gate"]["pilotTrancheCompletionGateReady"] is True
    assert report["blockedLaterGates"]["citationGradeEvidence"]["allowed"] is False
    assert validate_payload(
        report,
        STRICT_EVIDENCE_PILOT_TRANCHE_COMPLETION_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_completion_gate_blocks_invalid_readback_report(tmp_path: Path) -> None:
    readback_path = tmp_path / "readback.json"
    _write_json(readback_path, {"schema": "wrong.schema", "status": "ok", "rows": []})

    report = build_strict_evidence_pilot_tranche_completion_gate(
        manifest_readback_report_path=readback_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0


def test_completion_gate_writer_outputs_schema_valid_reports(tmp_path: Path, monkeypatch) -> None:
    rows = [
        _synthetic_readback_row(index=1, artifact_type="section"),
        _synthetic_readback_row(index=1, artifact_type="figure"),
    ]
    readback_path = tmp_path / "readback.json"
    report_payload = _synthetic_readback_report(rows)
    report_payload["counts"]["strictEvidenceStoreRows"] = 2
    report_payload["counts"]["sourceSpanStoreRows"] = 5
    _write_json(readback_path, report_payload)

    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_pilot_tranche_completion_gate.EXPECTED_POLICY_CANDIDATE_ROWS",
        2,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_pilot_tranche_completion_gate.EXPECTED_SECTION_MANIFEST_ROWS",
        1,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_pilot_tranche_completion_gate.EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS",
        1,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_pilot_tranche_completion_gate.EXPECTED_STRICT_EVIDENCE_STORE_ROWS",
        2,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_pilot_tranche_completion_gate.EXPECTED_SOURCE_SPAN_STORE_ROWS",
        5,
    )

    report = build_strict_evidence_pilot_tranche_completion_gate(
        manifest_readback_report_path=readback_path,
    )
    paths = write_strict_evidence_pilot_tranche_completion_gate_reports(
        report,
        tmp_path / "reports",
    )
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert written["status"] == "ok"
    assert {row["completion_status"] for row in written["rows"]} == {COMPLETION_STATUS_COMPLETE}
    assert validate_payload(
        written,
        STRICT_EVIDENCE_PILOT_TRANCHE_COMPLETION_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_completion_gate_integrated_measured_local_report() -> None:
    if not DEFAULT_MANIFEST_READBACK_REPORT_PATH.is_file():
        return

    report = build_strict_evidence_pilot_tranche_completion_gate()
    assert report["status"] == "ok"
    assert report["counts"]["completionCandidateOnlyRows"] == 99
    assert validate_payload(
        report,
        STRICT_EVIDENCE_PILOT_TRANCHE_COMPLETION_GATE_SCHEMA_ID,
        strict=True,
    ).ok
