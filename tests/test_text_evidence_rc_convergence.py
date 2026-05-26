from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.text_evidence_rc_convergence import (
    TEXT_EVIDENCE_RC_CONVERGENCE_SCHEMA_ID,
    build_text_evidence_rc_convergence_report,
    write_report,
)


def _write_report(root: Path, name: str, payload: dict) -> None:
    path = root / "eval" / "knowledgeos" / "reports" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _minimal_report(schema: str, *, status: str = "ready", rows: int = 1) -> dict:
    return {
        "schema": schema,
        "status": status,
        "rows": [{} for _ in range(rows)],
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
        },
        "privatePathLeakRows": 0,
    }


def test_convergence_report_records_rc_hold_without_mutation(tmp_path: Path) -> None:
    reports = {
        "figure_caption_artifact_vertical_slice.v1.json": "knowledge-hub.paper.figure-caption-artifact-vertical-slice-report.v1",
        "figure_caption_text_qa_readback.v1.json": "knowledge-hub.paper.figure-caption-text-qa-readback-report.v1",
        "text_section_paragraph_span_artifacts.v1.json": "knowledge-hub.paper.text-section-paragraph-span-artifacts-report.v1",
        "text_table_caption_candidate_artifacts.v1.json": "knowledge-hub.paper.text-table-caption-candidate-artifacts-report.v1",
        "text_equation_locator_context_artifacts.v1.json": "knowledge-hub.paper.text-equation-locator-context-artifacts-report.v1",
        "text_complex_qa_eval_alignment.v1.json": "knowledge-hub.paper.text-complex-qa-eval-alignment-report.v1",
        "source_alias_normalization.v1.json": "knowledge-hub.paper.source-alias-normalization-report.v1",
    }
    for name, schema in reports.items():
        _write_report(tmp_path, name, _minimal_report(schema))

    report = build_text_evidence_rc_convergence_report(
        project_root=Path.cwd(),
        canonical_repo=None,
        reports_root=tmp_path / "eval" / "knowledgeos" / "reports",
        include_pr_state=False,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] in {"ready_for_integration_review", "blocked"}
    assert report["phaseRows"] == 9
    assert report["publicRcReady"] is False
    assert report["textOnlyScopeGate"]["available"] is False
    assert report["externalActionPreflight"]["available"] is False
    assert report["pr149CloseApprovalRequest"]["available"] is False
    assert report["pr149CloseReceipt"]["available"] is False
    assert report["scope"]["mergePerformed"] is False
    assert report["scope"]["canonicalCheckoutEdited"] is False
    assert report["mutationCounters"]["vaultScanRows"] == 0
    assert report["privatePathLeakRows"] == 0
    assert validate_payload(report, TEXT_EVIDENCE_RC_CONVERGENCE_SCHEMA_ID, strict=True).ok


def test_writer_keeps_report_sanitized(tmp_path: Path) -> None:
    report = {
        "schema": TEXT_EVIDENCE_RC_CONVERGENCE_SCHEMA_ID,
        "status": "ready_for_integration_review",
        "generatedAt": "2026-05-26T00:00:00Z",
        "scope": {
            "writes": "report_only",
            "mergePerformed": False,
            "cherryPickPerformed": False,
            "worktreeDeletionPerformed": False,
            "canonicalCheckoutEdited": False,
            "visualLayoutBranchDeferred": True,
            "runtimeAnswerVisibleExposureRows": 0,
            "strictEvidencePromotionRows": 0,
        },
        "currentStack": {"branch": "codex/test", "head": "abc1234", "dirtyCount": 0},
        "canonicalCheckout": {"available": False, "branch": "", "head": "", "dirtyCount": 0},
        "pullRequest149": {"available": False, "number": 149},
        "textOnlyScopeGate": {
            "available": False,
            "reportRef": "",
            "status": "",
            "decision": "",
            "textOnlyRcReady": False,
            "publicRcReady": False,
            "textReadyRows": 0,
            "textHoldRows": 0,
            "deferredRows": 0,
            "blockerRows": 0,
            "nextAction": "",
            "privatePathLeakRows": 0,
        },
        "externalActionPreflight": {
            "available": False,
            "reportRef": "",
            "status": "",
            "readyForUserApproval": False,
            "canonicalFingerprintMatches": False,
            "pr149ExpectedHoldState": False,
            "blockerRows": 0,
            "nextAction": "",
            "privatePathLeakRows": 0,
        },
        "pr149CloseApprovalRequest": {
            "available": False,
            "reportRef": "",
            "status": "",
            "safeToExecuteAfterApproval": False,
            "executionStatus": "",
            "recommendedDecision": "",
            "blockerRows": 0,
            "nextAction": "",
            "privatePathLeakRows": 0,
        },
        "pr149CloseReceipt": {
            "available": False,
            "reportRef": "",
            "status": "",
            "executionVerified": False,
            "mergePerformed": False,
            "blockerRows": 0,
            "nextAction": "",
            "privatePathLeakRows": 0,
        },
        "phaseRows": 0,
        "readyPhaseRows": 0,
        "blockedPhaseRows": 0,
        "publicRcReady": False,
        "publicRcBlockerRows": 0,
        "blockers": [],
        "mergeQueue": [],
        "phaseReports": [],
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "mergeRows": 0,
            "cherryPickRows": 0,
            "worktreeDeletionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
            "strictEvidencePromotionRows": 0,
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "reportHash": "sha256:" + "1" * 64,
        "nextAction": "test",
        "warnings": [],
        "schemaErrors": [],
    }
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_report(report, json_path=json_path, markdown_path=md_path)

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
