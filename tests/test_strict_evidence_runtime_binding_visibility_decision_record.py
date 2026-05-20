from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_runtime_binding_post_apply_promotion_hold_review import (
    build_strict_evidence_runtime_binding_post_apply_promotion_hold_review,
    write_strict_evidence_runtime_binding_post_apply_promotion_hold_review_reports,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_visibility_decision_record import (
    DECISION_SEPARATE_RUNTIME_VISIBILITY_RECORD,
    DECISION_STATUS_CANDIDATE_ONLY,
    DECISION_STATUS_BLOCKED_DOWNSTREAM_GATE,
    DECISION_STATUS_BLOCKED_INPUT_SCHEMA,
    DECISION_STATUS_BLOCKED_RUNTIME_OR_ANSWER,
    STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_DECISION_RECORD_SCHEMA_ID,
    build_strict_evidence_runtime_binding_visibility_decision_record,
    write_strict_evidence_runtime_binding_visibility_decision_record_reports,
)
from tests.test_strict_evidence_runtime_binding_post_apply_promotion_hold_review import (
    _readback_report_path,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hold_review_report_path(tmp_path: Path) -> Path:
    readback_path = _readback_report_path(tmp_path)
    hold = build_strict_evidence_runtime_binding_post_apply_promotion_hold_review(
        readback_review_report_path=readback_path,
        expected_input_rows=1,
        expected_runtime_binding_record_rows=1,
        expected_citation_grade_store_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )
    hold_dir = tmp_path / "hold-review"
    write_strict_evidence_runtime_binding_post_apply_promotion_hold_review_reports(
        hold,
        hold_dir,
    )
    return hold_dir / "strict-evidence-runtime-binding-post-apply-promotion-hold-review.json"


def _build_one_row_report(hold_path: Path) -> dict:
    return build_strict_evidence_runtime_binding_visibility_decision_record(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_runtime_binding_record_rows=1,
        expected_section_decision_rows=1,
        expected_figure_caption_decision_rows=0,
        expected_citation_grade_store_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )


def test_visibility_decision_prefers_append_only_visibility_records(tmp_path: Path) -> None:
    hold_path = _hold_review_report_path(tmp_path)

    report = _build_one_row_report(hold_path)

    assert report["status"] == "ok"
    counts = report["counts"]
    assert counts["inputRows"] == 1
    assert counts["visibilityDecisionCandidateOnlyRows"] == 1
    assert counts["sectionDecisionRows"] == 1
    assert counts["figureCaptionDecisionRows"] == 0
    assert counts["runtimeVisibilityRecordWriteAllowedRows"] == 0
    assert counts["runtimeVisibleMutationAllowedRows"] == 0
    assert counts["answerIntegrationVisibleAllowedRows"] == 0
    assert counts["runtimeVisibilityRecordWriteRows"] == 0
    assert counts["runtimeVisibleRows"] == 0
    assert counts["answerIntegrationVisibleRows"] == 0
    assert counts["runtimeEvidenceCreatedRows"] == 0
    assert report["gate"]["decision"] == DECISION_SEPARATE_RUNTIME_VISIBILITY_RECORD
    assert report["gate"]["runtimeBindingInPlaceMutationAllowed"] is False
    assert report["gate"]["runtimeVisibleBooleanMutationAllowed"] is False
    assert report["gate"]["answerIntegrationVisibleBooleanMutationAllowed"] is False
    assert report["gate"]["recommendedNextTranche"] == (
        "strict_evidence_runtime_binding_visibility_record_contract"
    )
    assert {row["decision_status"] for row in report["rows"]} == {DECISION_STATUS_CANDIDATE_ONLY}
    assert validate_payload(
        report,
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_DECISION_RECORD_SCHEMA_ID,
        strict=True,
    ).ok


def test_visibility_decision_blocks_invalid_hold_review_report(tmp_path: Path) -> None:
    hold_path = tmp_path / "bad-hold.json"
    _write_json(hold_path, {"schema": "wrong.schema", "status": "ok", "rows": []})

    report = _build_one_row_report(hold_path)

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["gate"]["recommendedNextTranche"] == (
        "strict_evidence_runtime_binding_visibility_decision_record_input_repair"
    )


def test_visibility_decision_blocks_downstream_gate_enablement(tmp_path: Path) -> None:
    hold_path = _hold_review_report_path(tmp_path)
    payload = json.loads(hold_path.read_text(encoding="utf-8"))
    payload["blockedDownstreamGateMatrix"]["runtimeEvidence"]["allowed"] = True
    _write_json(hold_path, payload)

    report = _build_one_row_report(hold_path)

    assert report["status"] == "blocked"
    assert report["rows"][0]["decision_status"] == DECISION_STATUS_BLOCKED_DOWNSTREAM_GATE
    assert "downstream_gate_already_enabled" in report["rows"][0]["decision_blockers"]
    assert report["counts"]["blockedDownstreamGateAlreadyEnabledRows"] == 1


def test_visibility_decision_blocks_runtime_or_answer_row_flags(tmp_path: Path) -> None:
    hold_path = _hold_review_report_path(tmp_path)
    payload = json.loads(hold_path.read_text(encoding="utf-8"))
    payload["rows"][0]["writeMatrix"] = {
        **dict(payload["rows"][0].get("writeMatrix") or {}),
        "runtimeEvidenceCreated": True,
    }
    _write_json(hold_path, payload)

    report = _build_one_row_report(hold_path)

    assert report["status"] == "blocked"
    assert report["rows"][0]["decision_status"] == DECISION_STATUS_BLOCKED_RUNTIME_OR_ANSWER
    assert "writeMatrix.runtimeEvidenceCreated_true" in report["rows"][0]["decision_blockers"]
    assert report["counts"]["blockedRuntimeOrAnswerFlagViolationRows"] == 1


def test_visibility_decision_blocks_input_schema_violation_rows(tmp_path: Path) -> None:
    hold_path = _hold_review_report_path(tmp_path)
    payload = json.loads(hold_path.read_text(encoding="utf-8"))
    payload["status"] = "blocked"
    _write_json(hold_path, payload)

    report = _build_one_row_report(hold_path)

    assert report["status"] == "blocked"
    assert report["rows"][0]["decision_status"] == DECISION_STATUS_BLOCKED_INPUT_SCHEMA


def test_visibility_decision_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    hold_path = _hold_review_report_path(tmp_path)
    report = _build_one_row_report(hold_path)

    paths = write_strict_evidence_runtime_binding_visibility_decision_record_reports(
        report,
        tmp_path / "reports",
    )

    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert written["schema"] == STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_DECISION_RECORD_SCHEMA_ID
    assert summary["counts"]["visibilityDecisionCandidateOnlyRows"] == 1
    assert "Strict Evidence Runtime Binding Visibility Decision Record" in markdown
    assert validate_payload(
        written,
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_DECISION_RECORD_SCHEMA_ID,
        strict=True,
    ).ok


def test_visibility_decision_integrated_measured_local_report() -> None:
    report = build_strict_evidence_runtime_binding_visibility_decision_record()

    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 99
    assert report["counts"]["runtimeBindingRecordRows"] == 99
    assert report["counts"]["runtimeBindingHoldRows"] == 99
    assert report["counts"]["visibilityDecisionCandidateOnlyRows"] == 99
    assert report["counts"]["sectionDecisionRows"] == 45
    assert report["counts"]["figureCaptionDecisionRows"] == 54
    assert report["counts"]["runtimeVisibleMutationAllowedRows"] == 0
    assert report["counts"]["answerIntegrationVisibleAllowedRows"] == 0
    assert report["counts"]["runtimeVisibilityRecordWriteRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["counts"]["answerIntegrationVisibleRows"] == 0
    assert report["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert report["counts"]["parserRoutingChangedRows"] == 0
    assert report["counts"]["answerIntegrationChangedRows"] == 0
    assert validate_payload(
        report,
        STRICT_EVIDENCE_RUNTIME_BINDING_VISIBILITY_DECISION_RECORD_SCHEMA_ID,
        strict=True,
    ).ok
