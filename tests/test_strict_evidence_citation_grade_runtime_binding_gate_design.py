from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_citation_grade_post_apply_promotion_hold_review import (
    build_strict_evidence_citation_grade_post_apply_promotion_hold_review,
    write_strict_evidence_citation_grade_post_apply_promotion_hold_review_reports,
)
from knowledge_hub.papers.strict_evidence_citation_grade_runtime_binding_gate_design import (
    RUNTIME_BINDING_GATE_DESIGN_STATUS_BLOCKED_DOWNSTREAM_GATE,
    RUNTIME_BINDING_GATE_DESIGN_STATUS_BLOCKED_HOLD_NOT_ACTIVE,
    RUNTIME_BINDING_GATE_DESIGN_STATUS_BLOCKED_INPUT_SCHEMA,
    RUNTIME_BINDING_GATE_DESIGN_STATUS_BLOCKED_RUNTIME_OR_ANSWER,
    RUNTIME_BINDING_GATE_DESIGN_STATUS_CANDIDATE_ONLY,
    RUNTIME_BINDING_STORE,
    STRICT_EVIDENCE_CITATION_GRADE_RUNTIME_BINDING_GATE_DESIGN_SCHEMA_ID,
    build_strict_evidence_citation_grade_runtime_binding_gate_design,
    write_strict_evidence_citation_grade_runtime_binding_gate_design_reports,
)
from tests.test_strict_evidence_citation_grade_post_apply_promotion_hold_review import (
    _readback_report_path,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hold_report_path(tmp_path: Path) -> Path:
    readback_path = _readback_report_path(tmp_path)
    hold = build_strict_evidence_citation_grade_post_apply_promotion_hold_review(
        readback_review_report_path=readback_path,
        expected_input_rows=1,
        expected_citation_grade_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )
    hold_dir = tmp_path / "hold"
    write_strict_evidence_citation_grade_post_apply_promotion_hold_review_reports(
        hold,
        hold_dir,
    )
    return hold_dir / "strict-evidence-citation-grade-post-apply-promotion-hold-review.json"


def test_runtime_binding_gate_design_marks_hold_rows_candidate_only(tmp_path: Path) -> None:
    hold_path = _hold_report_path(tmp_path)

    report = build_strict_evidence_citation_grade_runtime_binding_gate_design(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_runtime_binding_gate_design_rows=1,
        expected_section_rows=1,
        expected_figure_caption_rows=0,
        expected_citation_grade_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "ok"
    counts = report["counts"]
    assert counts["inputRows"] == 1
    assert counts["runtimeBindingGateDesignCandidateOnlyRows"] == 1
    assert counts["sectionRuntimeBindingGateDesignRows"] == 1
    assert counts["figureCaptionRuntimeBindingGateDesignRows"] == 0
    assert counts["runtimeBindingRecordWriteRows"] == 0
    assert counts["runtimeEvidenceCreatedRows"] == 0
    assert counts["answerIntegrationChangedRows"] == 0
    assert report["runtimeBindingGateDesign"]["decision"] == (
        "separate_append_only_runtime_binding_record"
    )
    assert report["gate"]["recommendedNextTranche"] == "strict_evidence_runtime_binding_record_contract"
    row = report["rows"][0]
    assert row["runtime_binding_gate_design_status"] == RUNTIME_BINDING_GATE_DESIGN_STATUS_CANDIDATE_ONLY
    assert row["plannedWriteTarget"] == RUNTIME_BINDING_STORE
    assert row["runtimeBindingRecordWriteAllowed"] is False
    assert row["runtimeEvidenceAllowed"] is False
    assert row["answerIntegrationAllowed"] is False
    assert validate_payload(
        report,
        STRICT_EVIDENCE_CITATION_GRADE_RUNTIME_BINDING_GATE_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_binding_gate_design_blocks_invalid_hold_report(tmp_path: Path) -> None:
    hold_path = tmp_path / "bad-hold.json"
    _write_json(hold_path, {"schema": "wrong.schema", "status": "ok", "rows": [{}]})

    report = build_strict_evidence_citation_grade_runtime_binding_gate_design(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_runtime_binding_gate_design_rows=1,
        expected_section_rows=1,
        expected_figure_caption_rows=0,
        expected_citation_grade_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["rows"][0]["runtime_binding_gate_design_status"] == (
        RUNTIME_BINDING_GATE_DESIGN_STATUS_BLOCKED_INPUT_SCHEMA
    )


def test_runtime_binding_gate_design_blocks_non_active_hold_row(tmp_path: Path) -> None:
    hold_path = _hold_report_path(tmp_path)
    payload = json.loads(hold_path.read_text(encoding="utf-8"))
    payload["rows"][0]["hold_status"] = "blocked_readback_not_validated"
    payload["rows"][0]["postApplyPromotionHoldActive"] = False
    _write_json(hold_path, payload)

    report = build_strict_evidence_citation_grade_runtime_binding_gate_design(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_runtime_binding_gate_design_rows=1,
        expected_section_rows=1,
        expected_figure_caption_rows=0,
        expected_citation_grade_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["rows"][0]["runtime_binding_gate_design_status"] == (
        RUNTIME_BINDING_GATE_DESIGN_STATUS_BLOCKED_HOLD_NOT_ACTIVE
    )


def test_runtime_binding_gate_design_blocks_downstream_gate_enablement(tmp_path: Path) -> None:
    hold_path = _hold_report_path(tmp_path)
    payload = json.loads(hold_path.read_text(encoding="utf-8"))
    payload["gate"]["answerIntegrationAllowed"] = True
    _write_json(hold_path, payload)

    report = build_strict_evidence_citation_grade_runtime_binding_gate_design(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_runtime_binding_gate_design_rows=1,
        expected_section_rows=1,
        expected_figure_caption_rows=0,
        expected_citation_grade_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["rows"][0]["runtime_binding_gate_design_status"] == (
        RUNTIME_BINDING_GATE_DESIGN_STATUS_BLOCKED_DOWNSTREAM_GATE
    )
    assert "downstream_gate_answerIntegration_already_enabled" in report["rows"][0][
        "runtime_binding_gate_design_blockers"
    ]


def test_runtime_binding_gate_design_blocks_runtime_row_flag(tmp_path: Path) -> None:
    hold_path = _hold_report_path(tmp_path)
    payload = json.loads(hold_path.read_text(encoding="utf-8"))
    payload["rows"][0]["runtimeVisible"] = True
    _write_json(hold_path, payload)

    report = build_strict_evidence_citation_grade_runtime_binding_gate_design(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_runtime_binding_gate_design_rows=1,
        expected_section_rows=1,
        expected_figure_caption_rows=0,
        expected_citation_grade_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["rows"][0]["runtime_binding_gate_design_status"] == (
        RUNTIME_BINDING_GATE_DESIGN_STATUS_BLOCKED_RUNTIME_OR_ANSWER
    )
    assert "runtimeVisible_true" in report["rows"][0]["runtime_binding_gate_design_blockers"]


def test_runtime_binding_gate_design_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    hold_path = _hold_report_path(tmp_path)
    report = build_strict_evidence_citation_grade_runtime_binding_gate_design(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_runtime_binding_gate_design_rows=1,
        expected_section_rows=1,
        expected_figure_caption_rows=0,
        expected_citation_grade_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )

    paths = write_strict_evidence_citation_grade_runtime_binding_gate_design_reports(
        report,
        tmp_path / "reports",
    )

    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert written["schema"] == STRICT_EVIDENCE_CITATION_GRADE_RUNTIME_BINDING_GATE_DESIGN_SCHEMA_ID
    assert summary["counts"]["runtimeBindingGateDesignCandidateOnlyRows"] == 1
    assert "Strict Evidence Runtime Binding Gate Design" in markdown
    assert validate_payload(
        written,
        STRICT_EVIDENCE_CITATION_GRADE_RUNTIME_BINDING_GATE_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_binding_gate_design_integrated_measured_local_report() -> None:
    report = build_strict_evidence_citation_grade_runtime_binding_gate_design()

    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 99
    assert report["counts"]["citationGradeRecordRows"] == 99
    assert report["counts"]["runtimeBindingGateDesignCandidateOnlyRows"] == 99
    assert report["counts"]["sectionRuntimeBindingGateDesignRows"] == 45
    assert report["counts"]["figureCaptionRuntimeBindingGateDesignRows"] == 54
    assert report["counts"]["strictEvidenceStoreRows"] == 99
    assert report["counts"]["eligibilityStoreRows"] == 99
    assert report["counts"]["sourceSpanStoreRows"] == 102
    assert report["counts"]["blockedCitationGradeHoldNotActiveRows"] == 0
    assert report["counts"]["blockedDownstreamGateAlreadyEnabledRows"] == 0
    assert report["counts"]["blockedRuntimeOrAnswerFlagViolationRows"] == 0
    assert report["counts"]["runtimeBindingRecordWriteRows"] == 0
    assert report["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert report["counts"]["answerIntegrationChangedRows"] == 0
    assert validate_payload(
        report,
        STRICT_EVIDENCE_CITATION_GRADE_RUNTIME_BINDING_GATE_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok
