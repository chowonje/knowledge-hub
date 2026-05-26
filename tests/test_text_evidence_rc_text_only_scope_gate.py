from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.text_evidence_rc_text_only_scope_gate import (
    TEXT_EVIDENCE_RC_TEXT_ONLY_SCOPE_GATE_SCHEMA_ID,
    TEXT_PHASE_REPORTS,
    build_text_evidence_rc_text_only_scope_gate,
    write_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_inputs(root: Path, *, phase_status: str = "ready") -> None:
    for spec in TEXT_PHASE_REPORTS:
        _write_json(
            root / spec["reportRef"],
            {
                "schema": f"schema:{spec['phase']}",
                "status": phase_status,
                "candidateRows": 1,
                "visualUnsupportedRows": 1 if spec["phase"] == "text_complex_qa_eval_alignment" else 0,
                "strictEvidencePromotionRows": 0,
                "runtimeAnswerVisibleExposureRows": 0,
                "privatePathLeakRows": 0,
                "mutationCounters": {"databaseMutationRows": 0},
                "rows": [{"id": spec["phase"]}],
            },
        )
    _write_json(
        root / "text_evidence_rc_convergence.v1.json",
        {
            "schema": "knowledge-hub.paper.text-evidence-rc-convergence-report.v1",
            "status": "ready_for_integration_review",
            "publicRcReady": False,
            "nextAction": "close_pr149_without_merge_before_writing_cleanup_snapshot",
            "blockers": [
                {
                    "blockerId": "canonical_checkout_dirty",
                    "severity": "hold",
                    "reason": "canonical cleanup awaits approval",
                }
            ],
        },
    )


def test_text_only_scope_gate_defers_visual_work_without_public_rc_ready(tmp_path: Path) -> None:
    _write_inputs(tmp_path)

    report = build_text_evidence_rc_text_only_scope_gate(
        reports_root=tmp_path,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["textOnlyRcReady"] is True
    assert report["publicRcReady"] is False
    assert report["decision"] == "text_only_scope_ready_pending_rc_convergence_actions"
    assert report["phaseRows"] == 6
    assert report["textReadyRows"] == 6
    assert report["textHoldRows"] == 0
    assert report["deferredRows"] == 6
    assert report["evidencePolicy"]["allowsVisualInspectionEvidence"] is False
    assert report["evidencePolicy"]["allowsVlmDerivedCitationGradeEvidence"] is False
    assert report["scope"]["visualLayoutBranchDeferred"] is True
    assert report["mutationCounters"]["strictEvidencePromotionRows"] == 0
    assert report["mutationCounters"]["runtimeAnswerVisibleExposureRows"] == 0
    assert report["privatePathLeakRows"] == 0
    assert validate_payload(report, TEXT_EVIDENCE_RC_TEXT_ONLY_SCOPE_GATE_SCHEMA_ID, strict=True).ok


def test_text_only_scope_gate_blocks_when_phase_report_not_ready(tmp_path: Path) -> None:
    _write_inputs(tmp_path, phase_status="blocked")

    report = build_text_evidence_rc_text_only_scope_gate(
        reports_root=tmp_path,
        generated_at="2026-05-26T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["textOnlyRcReady"] is False
    assert report["textHoldRows"] == 6


def test_text_only_scope_gate_writer_is_path_sanitized(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    report = build_text_evidence_rc_text_only_scope_gate(
        reports_root=tmp_path,
        generated_at="2026-05-26T00:00:00Z",
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_report(report, json_path=json_path, markdown_path=md_path)

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert json.loads(json_path.read_text(encoding="utf-8"))["privatePathLeakRows"] == 0
