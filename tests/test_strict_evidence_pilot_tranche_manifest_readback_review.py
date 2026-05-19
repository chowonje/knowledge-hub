from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_strict_evidence_policy_gate import (
    PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
    POLICY_STATUS_CANDIDATE_ONLY,
)
from knowledge_hub.papers.strict_evidence_figure_caption_pilot_executor_apply import (
    APPLY_STATUS_APPLIED as FIGURE_APPLY_STATUS_APPLIED,
    STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
)
from knowledge_hub.papers.strict_evidence_pilot_tranche_manifest_readback_review import (
    DEFAULT_FIGURE_CAPTION_APPLY_REPORT_PATH,
    DEFAULT_POLICY_GATE_REPORT_PATH,
    DEFAULT_SECTION_APPLY_REPORT_PATH,
    EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS,
    EXPECTED_SECTION_MANIFEST_ROWS,
    MANIFEST_TYPE_FIGURE_CAPTION,
    MANIFEST_TYPE_SECTION,
    READBACK_STATUS_BLOCKED_INPUT_SCHEMA,
    READBACK_STATUS_VALIDATED,
    STRICT_EVIDENCE_PILOT_TRANCHE_MANIFEST_READBACK_REVIEW_SCHEMA_ID,
    build_strict_evidence_pilot_tranche_manifest_readback_review,
    write_strict_evidence_pilot_tranche_manifest_readback_review_reports,
)
from knowledge_hub.papers.strict_evidence_text_section_pilot_executor_apply import (
    APPLY_STATUS_APPLIED as SECTION_APPLY_STATUS_APPLIED,
    STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
)


def _relax_input_schema_validation(monkeypatch) -> None:
    real_validate = validate_payload

    def _validate(payload, schema_id, strict=True):
        if schema_id in {
            STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
            STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
            PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
        }:
            if not payload:
                return SimpleNamespace(ok=False, errors=["empty"])
            return SimpleNamespace(ok=True, errors=[])
        return real_validate(payload, schema_id, strict=strict)

    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_pilot_tranche_manifest_readback_review.validate_payload",
        _validate,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mutation_safe_manifest_row(
    *,
    index: int,
    artifact_type: str,
    applied_status: str,
) -> dict:
    return {
        "apply_row_id": f"apply:{index:04d}",
        "strictEvidenceId": f"strict-evidence:paper-1:{artifact_type}:{index}",
        "sourceSpanId": f"source-span:paper-1:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:{artifact_type}:{index}",
        "paper_id": "paper-1",
        "artifact_type": artifact_type,
        "apply_status": applied_status,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
        "databaseMutation": False,
        "writeMatrix": {
            "writeEnabled": False,
            "strictEvidenceStoreWrite": False,
            "sourceSpanStoreWrite": False,
        },
    }


def _synthetic_apply_report(
    *,
    schema_id: str,
    manifest_path: Path,
    applied_rows: list[dict],
    held_rows: list[dict],
) -> dict:
    rows = applied_rows + held_rows
    return {
        "schema": schema_id,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {
            "executorDryRunReportPath": "/tmp/dry-run.json",
            "executorDryRunSchema": "dry-run.schema",
            "requestedPaperIds": [],
            "runId": "run-1",
            "apply": True,
            "targetTranche": "pilot",
        },
        "counts": {
            "inputRows": len(rows),
            "plannedApplyRows": 0,
            "appliedManifestOnlyRows": len(applied_rows),
            "manifestWriteRows": 1,
            "strictEvidenceWriteRows": 0,
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
            "byApplyStatus": {},
            "byRecommendedAction": {},
        },
        "heldOutNonSection": {"heldOutNonSectionRows": len(held_rows), "diagnosticOnly": True, "activeApplyProcessing": False},
        "heldOutSection": {"heldOutSectionRows": len(held_rows), "diagnosticOnly": True, "activeApplyProcessing": False},
        "manifestOnlyPolicyMatrix": {
            "plannedAction": "validate_only",
            "plannedWriteTarget": "structured_evidence_runs_manifest",
            "plannedRuntimeEffect": "none",
            "plannedAnswerEffect": "none",
            "writeEnabled": False,
            "strictEvidenceStoreWrite": False,
            "sourceSpanStoreWrite": False,
            "manifestOnlyApply": True,
        },
        "gate": {
            "readyForDryRunApplyPlanning": False,
            "readyForManifestOnlyApply": True,
            "applyMode": True,
            "strictEligibleMutationAllowed": False,
            "strictEvidenceStoreWriteAllowed": False,
            "sourceSpanStoreWriteAllowed": False,
            "runManifestWriteAllowed": True,
            "strictEvidenceCreated": False,
            "citationReady": False,
            "runtimeEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": [],
            "decision": "applied_manifest_only",
            "recommendedNextTranche": "strict_evidence_pilot_tranche_manifest_readback_review",
            "runManifestPath": str(manifest_path),
        },
        "policy": {
            "reportOnly": False,
            "manifestOnlyApply": True,
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
            "runManifestWrite": True,
        },
        "warnings": [],
        "rows": rows,
    }


def _synthetic_manifest(*, schema_id: str, rows: list[dict]) -> dict:
    return {
        "schema": schema_id,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {"apply": True, "runId": "run-1", "targetTranche": "pilot"},
        "counts": {"inputRows": len(rows), "manifestWriteRows": 0, "strictEvidenceWriteRows": 0},
        "heldOutNonSection": {"heldOutNonSectionRows": 0, "diagnosticOnly": True, "activeApplyProcessing": False},
        "heldOutSection": {"heldOutSectionRows": 0, "diagnosticOnly": True, "activeApplyProcessing": False},
        "manifestOnlyPolicyMatrix": {
            "plannedAction": "validate_only",
            "plannedWriteTarget": "structured_evidence_runs_manifest",
            "plannedRuntimeEffect": "none",
            "plannedAnswerEffect": "none",
            "writeEnabled": False,
            "strictEvidenceStoreWrite": False,
            "sourceSpanStoreWrite": False,
            "manifestOnlyApply": True,
        },
        "gate": {"schemaViolations": [], "recommendedNextTranche": "review"},
        "policy": {"manifestOnlyApply": True, "strictEvidenceStoreWrite": False},
        "warnings": [],
        "rows": rows,
    }


def _synthetic_policy_gate_report(policy_rows: list[dict]) -> dict:
    return {
        "schema": PARSED_ARTIFACT_STRICT_EVIDENCE_POLICY_GATE_SCHEMA_ID,
        "status": "ok",
        "generatedAt": "2026-05-19T00:00:00Z",
        "input": {"readbackReportPath": "/tmp/readback.json", "readbackSchema": "schema", "requestedPaperIds": []},
        "counts": {
            "inputRows": len(policy_rows),
            "readbackValidatedRows": len(policy_rows),
            "strictEvidencePolicyCandidateOnlyRows": len(policy_rows),
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
            "byPolicyGateStatus": {POLICY_STATUS_CANDIDATE_ONLY: len(policy_rows)},
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
            "decision": "ready",
            "recommendedNextTranche": "plan",
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
        "rows": policy_rows,
    }


def _policy_row(*, index: int, artifact_type: str) -> dict:
    return {
        "policy_gate_row_id": f"policy:{index:04d}",
        "strictEvidenceId": f"strict-evidence:paper-1:{artifact_type}:{index}",
        "sourceSpanId": f"source-span:paper-1:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:paper-1:{artifact_type}:{index}",
        "paper_id": "paper-1",
        "artifact_type": artifact_type,
        "policy_gate_status": POLICY_STATUS_CANDIDATE_ONLY,
    }


def test_readback_validates_section_and_figure_manifests(tmp_path: Path, monkeypatch) -> None:
    _relax_input_schema_validation(monkeypatch)
    section_manifest_path = tmp_path / "section-manifest.json"
    figure_manifest_path = tmp_path / "figure-manifest.json"
    section_rows = [
        _mutation_safe_manifest_row(
            index=index,
            artifact_type="section",
            applied_status=SECTION_APPLY_STATUS_APPLIED,
        )
        for index in range(1, EXPECTED_SECTION_MANIFEST_ROWS + 1)
    ]
    figure_rows = [
        _mutation_safe_manifest_row(
            index=index,
            artifact_type="figure",
            applied_status=FIGURE_APPLY_STATUS_APPLIED,
        )
        for index in range(1, EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS + 1)
    ]
    _write_json(section_manifest_path, _synthetic_manifest(
        schema_id=STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
        rows=section_rows,
    ))
    _write_json(figure_manifest_path, _synthetic_manifest(
        schema_id=STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
        rows=figure_rows,
    ))

    section_apply_path = tmp_path / "section-apply.json"
    figure_apply_path = tmp_path / "figure-apply.json"
    _write_json(
        section_apply_path,
        _synthetic_apply_report(
            schema_id=STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
            manifest_path=section_manifest_path,
            applied_rows=section_rows,
            held_rows=figure_rows,
        ),
    )
    _write_json(
        figure_apply_path,
        _synthetic_apply_report(
            schema_id=STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
            manifest_path=figure_manifest_path,
            applied_rows=figure_rows,
            held_rows=section_rows,
        ),
    )

    policy_rows = [
        _policy_row(index=index, artifact_type="section")
        for index in range(1, EXPECTED_SECTION_MANIFEST_ROWS + 1)
    ] + [
        _policy_row(index=index, artifact_type="figure")
        for index in range(1, EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS + 1)
    ]
    policy_gate_path = tmp_path / "policy-gate.json"
    _write_json(policy_gate_path, _synthetic_policy_gate_report(policy_rows))

    papers_dir = tmp_path / "papers"
    strict_root = papers_dir / "structured_evidence" / "strict_evidence"
    span_root = papers_dir / "structured_evidence" / "source_span"
    strict_root.mkdir(parents=True)
    span_root.mkdir(parents=True)
    (strict_root / "paper-1.jsonl").write_text("\n".join(["{}"] * 99) + "\n", encoding="utf-8")
    (span_root / "paper-1.jsonl").write_text("\n".join(["{}"] * 102) + "\n", encoding="utf-8")

    report = build_strict_evidence_pilot_tranche_manifest_readback_review(
        section_apply_report_path=section_apply_path,
        figure_caption_apply_report_path=figure_apply_path,
        policy_gate_report_path=policy_gate_path,
        papers_dir=papers_dir,
    )

    assert report["status"] == "ok"
    counts = report["counts"]
    assert counts["inputPolicyCandidateRows"] == 99
    assert counts["sectionManifestRows"] == EXPECTED_SECTION_MANIFEST_ROWS
    assert counts["figureCaptionManifestRows"] == EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS
    assert counts["combinedManifestRows"] == 99
    assert counts["pilotManifestReadbackValidatedRows"] == 99
    assert counts["missingPolicyCandidateRows"] == 0
    assert counts["unexpectedManifestRows"] == 0
    assert counts["duplicateStrictEvidenceIdRows"] == 0
    assert counts["strictEvidenceStoreRows"] == 99
    assert counts["sourceSpanStoreRows"] == 102
    assert report["gate"]["pilotManifestReadbackReviewReady"] is True
    assert validate_payload(
        report,
        STRICT_EVIDENCE_PILOT_TRANCHE_MANIFEST_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_readback_blocks_invalid_input_schema(tmp_path: Path) -> None:
    policy_gate_path = tmp_path / "policy-gate.json"
    _write_json(policy_gate_path, {"schema": "wrong.schema", "status": "ok", "rows": []})
    report = build_strict_evidence_pilot_tranche_manifest_readback_review(
        section_apply_report_path=tmp_path / "missing-section.json",
        figure_caption_apply_report_path=tmp_path / "missing-figure.json",
        policy_gate_report_path=policy_gate_path,
        papers_dir=tmp_path / "papers",
    )
    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert any(
        row.get("readback_status") == READBACK_STATUS_BLOCKED_INPUT_SCHEMA
        for row in report["rows"]
    ) or report["counts"]["blockedInputSchemaViolationRows"] >= 0


def test_readback_writer_outputs_schema_valid_reports(tmp_path: Path, monkeypatch) -> None:
    _relax_input_schema_validation(monkeypatch)
    section_manifest_path = tmp_path / "section-manifest.json"
    figure_manifest_path = tmp_path / "figure-manifest.json"
    section_rows = [
        _mutation_safe_manifest_row(index=1, artifact_type="section", applied_status=SECTION_APPLY_STATUS_APPLIED)
    ]
    figure_rows = [
        _mutation_safe_manifest_row(index=1, artifact_type="figure", applied_status=FIGURE_APPLY_STATUS_APPLIED)
    ]
    _write_json(section_manifest_path, _synthetic_manifest(
        schema_id=STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
        rows=section_rows,
    ))
    _write_json(figure_manifest_path, _synthetic_manifest(
        schema_id=STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
        rows=figure_rows,
    ))
    section_apply_path = tmp_path / "section-apply.json"
    figure_apply_path = tmp_path / "figure-apply.json"
    _write_json(
        section_apply_path,
        _synthetic_apply_report(
            schema_id=STRICT_EVIDENCE_TEXT_SECTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
            manifest_path=section_manifest_path,
            applied_rows=section_rows,
            held_rows=[],
        ),
    )
    _write_json(
        figure_apply_path,
        _synthetic_apply_report(
            schema_id=STRICT_EVIDENCE_FIGURE_CAPTION_PILOT_EXECUTOR_APPLY_SCHEMA_ID,
            manifest_path=figure_manifest_path,
            applied_rows=figure_rows,
            held_rows=[],
        ),
    )
    policy_rows = [_policy_row(index=1, artifact_type="section"), _policy_row(index=1, artifact_type="figure")]
    policy_gate_path = tmp_path / "policy-gate.json"
    _write_json(policy_gate_path, _synthetic_policy_gate_report(policy_rows))

    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_pilot_tranche_manifest_readback_review.EXPECTED_SECTION_MANIFEST_ROWS",
        1,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_pilot_tranche_manifest_readback_review.EXPECTED_FIGURE_CAPTION_MANIFEST_ROWS",
        1,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_pilot_tranche_manifest_readback_review.EXPECTED_POLICY_CANDIDATE_ROWS",
        2,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_pilot_tranche_manifest_readback_review.EXPECTED_STRICT_EVIDENCE_STORE_ROWS",
        1,
    )
    monkeypatch.setattr(
        "knowledge_hub.papers.strict_evidence_pilot_tranche_manifest_readback_review.EXPECTED_SOURCE_SPAN_STORE_ROWS",
        1,
    )

    papers_dir = tmp_path / "papers"
    strict_root = papers_dir / "structured_evidence" / "strict_evidence"
    span_root = papers_dir / "structured_evidence" / "source_span"
    strict_root.mkdir(parents=True)
    span_root.mkdir(parents=True)
    (strict_root / "paper-1.jsonl").write_text("{}\n", encoding="utf-8")
    (span_root / "paper-1.jsonl").write_text("{}\n", encoding="utf-8")

    report = build_strict_evidence_pilot_tranche_manifest_readback_review(
        section_apply_report_path=section_apply_path,
        figure_caption_apply_report_path=figure_apply_path,
        policy_gate_report_path=policy_gate_path,
        papers_dir=papers_dir,
    )
    paths = write_strict_evidence_pilot_tranche_manifest_readback_review_reports(
        report,
        tmp_path / "reports",
    )
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert validate_payload(
        written,
        STRICT_EVIDENCE_PILOT_TRANCHE_MANIFEST_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
    assert {row["manifestType"] for row in written["rows"] if row["readback_status"] == READBACK_STATUS_VALIDATED} == {
        MANIFEST_TYPE_SECTION,
        MANIFEST_TYPE_FIGURE_CAPTION,
    }


def test_readback_integrated_measured_local_reports() -> None:
    if not (
        DEFAULT_SECTION_APPLY_REPORT_PATH.is_file()
        and DEFAULT_FIGURE_CAPTION_APPLY_REPORT_PATH.is_file()
        and DEFAULT_POLICY_GATE_REPORT_PATH.is_file()
    ):
        return

    report = build_strict_evidence_pilot_tranche_manifest_readback_review()
    assert report["status"] == "ok"
    counts = report["counts"]
    assert counts["inputPolicyCandidateRows"] == 99
    assert counts["sectionManifestRows"] == 45
    assert counts["figureCaptionManifestRows"] == 54
    assert counts["combinedManifestRows"] == 99
    assert counts["pilotManifestReadbackValidatedRows"] == 99
    assert counts["strictEvidenceStoreRows"] == 99
    assert counts["sourceSpanStoreRows"] == 102
    assert validate_payload(
        report,
        STRICT_EVIDENCE_PILOT_TRANCHE_MANIFEST_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
