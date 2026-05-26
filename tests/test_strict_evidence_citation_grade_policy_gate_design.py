from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_citation_grade_policy_gate_design import (
    CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_DOWNSTREAM_GATE,
    CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_RUNTIME_OR_CITATION,
    CITATION_GRADE_POLICY_DESIGN_STATUS_CANDIDATE_ONLY,
    STRICT_EVIDENCE_CITATION_GRADE_POLICY_GATE_DESIGN_SCHEMA_ID,
    build_strict_evidence_citation_grade_policy_gate_design,
    write_strict_evidence_citation_grade_policy_gate_design_reports,
)
from knowledge_hub.papers.strict_evidence_eligibility_post_apply_promotion_hold_review import (
    build_strict_evidence_eligibility_post_apply_promotion_hold_review,
    write_strict_evidence_eligibility_post_apply_promotion_hold_review_reports,
)
from tests.test_strict_evidence_eligibility_post_apply_promotion_hold_review import (
    _readback_report_path,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hold_review_report_path(tmp_path: Path) -> Path:
    readback_path = _readback_report_path(tmp_path)
    hold = build_strict_evidence_eligibility_post_apply_promotion_hold_review(
        readback_review_report_path=readback_path,
        expected_input_rows=1,
        expected_eligibility_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )
    hold_dir = tmp_path / "hold"
    write_strict_evidence_eligibility_post_apply_promotion_hold_review_reports(
        hold,
        hold_dir,
    )
    return hold_dir / "strict-evidence-eligibility-post-apply-promotion-hold-review.json"


def test_citation_grade_policy_gate_design_marks_candidates_without_writes(tmp_path: Path) -> None:
    hold_path = _hold_review_report_path(tmp_path)

    report = build_strict_evidence_citation_grade_policy_gate_design(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_candidate_rows=1,
        expected_section_rows=1,
        expected_figure_caption_rows=0,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "ok"
    counts = report["counts"]
    assert counts["inputRows"] == 1
    assert counts["eligibilityRecordRows"] == 1
    assert counts["holdActiveRows"] == 1
    assert counts["citationGradePolicyDesignCandidateOnlyRows"] == 1
    assert counts["sectionCitationGradePolicyDesignRows"] == 1
    assert counts["figureCaptionCitationGradePolicyDesignRows"] == 0
    assert counts["citationGradeRecordWriteRows"] == 0
    assert counts["citationGradeEvidenceCreatedRows"] == 0
    assert counts["runtimeEvidenceCreatedRows"] == 0
    assert counts["answerIntegrationChangedRows"] == 0
    assert report["rows"][0]["citation_grade_policy_design_status"] == (
        CITATION_GRADE_POLICY_DESIGN_STATUS_CANDIDATE_ONLY
    )
    assert report["citationGradePolicyDesign"]["decision"] == "separate_append_only_citation_grade_record"
    assert report["citationGradePolicyDesign"]["noAnswerSafetyEvalRequiredBeforeRuntime"] is True
    assert report["gate"]["recommendedNextTranche"] == "strict_evidence_citation_grade_record_contract"
    assert validate_payload(
        report,
        STRICT_EVIDENCE_CITATION_GRADE_POLICY_GATE_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok


def test_citation_grade_policy_gate_design_blocks_invalid_hold_report(tmp_path: Path) -> None:
    hold_path = tmp_path / "bad-hold.json"
    _write_json(hold_path, {"schema": "wrong.schema", "status": "ok", "rows": []})

    report = build_strict_evidence_citation_grade_policy_gate_design(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_candidate_rows=1,
        expected_section_rows=1,
        expected_figure_caption_rows=0,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["gate"]["recommendedNextTranche"] == (
        "strict_evidence_eligibility_post_apply_promotion_hold_repair"
    )


def test_citation_grade_policy_gate_design_blocks_runtime_visible_row_flag(tmp_path: Path) -> None:
    hold_path = _hold_review_report_path(tmp_path)
    payload = json.loads(hold_path.read_text(encoding="utf-8"))
    payload["rows"][0]["runtimeVisible"] = True
    _write_json(hold_path, payload)

    report = build_strict_evidence_citation_grade_policy_gate_design(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_candidate_rows=1,
        expected_section_rows=1,
        expected_figure_caption_rows=0,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["rows"][0]["citation_grade_policy_design_status"] == (
        CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_RUNTIME_OR_CITATION
    )
    assert "runtimeVisible_true" in report["rows"][0]["citation_grade_policy_design_blockers"]
    assert report["counts"]["blockedRuntimeOrCitationFlagViolationRows"] == 1


def test_citation_grade_policy_gate_design_blocks_downstream_gate_enablement(tmp_path: Path) -> None:
    hold_path = _hold_review_report_path(tmp_path)
    payload = json.loads(hold_path.read_text(encoding="utf-8"))
    payload["gate"]["citationReady"] = True
    _write_json(hold_path, payload)

    report = build_strict_evidence_citation_grade_policy_gate_design(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_candidate_rows=1,
        expected_section_rows=1,
        expected_figure_caption_rows=0,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["rows"][0]["citation_grade_policy_design_status"] == (
        CITATION_GRADE_POLICY_DESIGN_STATUS_BLOCKED_DOWNSTREAM_GATE
    )
    assert "downstream_gate_citationGradeEvidence_already_enabled" in report["rows"][0][
        "citation_grade_policy_design_blockers"
    ]


def test_citation_grade_policy_gate_design_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    hold_path = _hold_review_report_path(tmp_path)
    report = build_strict_evidence_citation_grade_policy_gate_design(
        hold_review_report_path=hold_path,
        expected_input_rows=1,
        expected_candidate_rows=1,
        expected_section_rows=1,
        expected_figure_caption_rows=0,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )

    paths = write_strict_evidence_citation_grade_policy_gate_design_reports(
        report,
        tmp_path / "reports",
    )

    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert written["schema"] == STRICT_EVIDENCE_CITATION_GRADE_POLICY_GATE_DESIGN_SCHEMA_ID
    assert summary["counts"]["citationGradePolicyDesignCandidateOnlyRows"] == 1
    assert "Strict Evidence Citation-Grade Policy Gate Design" in markdown
    assert validate_payload(
        written,
        STRICT_EVIDENCE_CITATION_GRADE_POLICY_GATE_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok


def test_citation_grade_policy_gate_design_integrated_measured_local_report() -> None:
    report = build_strict_evidence_citation_grade_policy_gate_design()

    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 99
    assert report["counts"]["eligibilityRecordRows"] == 99
    assert report["counts"]["holdActiveRows"] == 99
    assert report["counts"]["citationGradePolicyDesignCandidateOnlyRows"] == 99
    assert report["counts"]["sectionCitationGradePolicyDesignRows"] == 45
    assert report["counts"]["figureCaptionCitationGradePolicyDesignRows"] == 54
    assert report["counts"]["strictEvidenceStoreRows"] == 99
    assert report["counts"]["sourceSpanStoreRows"] == 102
    assert report["counts"]["blockedPostApplyHoldNotActiveRows"] == 0
    assert report["counts"]["blockedDownstreamGateAlreadyEnabledRows"] == 0
    assert report["counts"]["blockedRuntimeOrCitationFlagViolationRows"] == 0
    assert report["counts"]["citationGradeRecordWriteRows"] == 0
    assert report["counts"]["citationGradeEvidenceCreatedRows"] == 0
    assert report["counts"]["runtimeEvidenceCreatedRows"] == 0
    assert report["counts"]["answerIntegrationChangedRows"] == 0
    assert validate_payload(
        report,
        STRICT_EVIDENCE_CITATION_GRADE_POLICY_GATE_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok
