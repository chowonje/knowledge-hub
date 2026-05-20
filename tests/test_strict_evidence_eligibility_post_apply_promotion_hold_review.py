from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_eligibility_executor_apply_readback_review import (
    build_strict_evidence_eligibility_executor_apply_readback_review,
    write_strict_evidence_eligibility_executor_apply_readback_review_reports,
)
from knowledge_hub.papers.strict_evidence_eligibility_post_apply_promotion_hold_review import (
    HOLD_STATUS_ACTIVE,
    HOLD_STATUS_BLOCKED_GATE_ENABLED,
    HOLD_STATUS_BLOCKED_INPUT_SCHEMA,
    HOLD_STATUS_BLOCKED_RUNTIME_OR_CITATION,
    STRICT_EVIDENCE_ELIGIBILITY_POST_APPLY_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
    build_strict_evidence_eligibility_post_apply_promotion_hold_review,
    write_strict_evidence_eligibility_post_apply_promotion_hold_review_reports,
)
from tests.test_strict_evidence_eligibility_executor_apply_readback_review import (
    _fixture_reports,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _readback_report_path(tmp_path: Path) -> Path:
    papers_dir = tmp_path / "papers"
    apply_path, dry_run_path, contract_path = _fixture_reports(tmp_path, papers_dir)
    readback = build_strict_evidence_eligibility_executor_apply_readback_review(
        apply_report_path=apply_path,
        dry_run_report_path=dry_run_path,
        eligibility_record_contract_report_path=contract_path,
        papers_dir=papers_dir,
        expected_input_rows=1,
        expected_eligibility_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )
    readback_dir = tmp_path / "readback"
    write_strict_evidence_eligibility_executor_apply_readback_review_reports(
        readback,
        readback_dir,
    )
    return readback_dir / "strict-evidence-eligibility-executor-apply-readback-review.json"


def test_post_apply_hold_review_marks_readback_validated_rows_on_hold(tmp_path: Path) -> None:
    readback_path = _readback_report_path(tmp_path)

    report = build_strict_evidence_eligibility_post_apply_promotion_hold_review(
        readback_review_report_path=readback_path,
        expected_input_rows=1,
        expected_eligibility_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "ok"
    counts = report["counts"]
    assert counts["inputRows"] == 1
    assert counts["eligibilityRecordRows"] == 1
    assert counts["readbackValidatedRows"] == 1
    assert counts["holdActiveRows"] == 1
    assert counts["sectionHoldRows"] == 1
    assert counts["figureCaptionHoldRows"] == 0
    assert counts["strictEligibleMutationAllowedRows"] == 0
    assert counts["eligibilityRecordWriteAllowedRows"] == 0
    assert counts["citationGradeAllowedRows"] == 0
    assert counts["runtimeEvidenceAllowedRows"] == 0
    assert counts["parserRoutingAllowedRows"] == 0
    assert counts["answerIntegrationAllowedRows"] == 0
    assert counts["eligibilityRecordWriteRows"] == 0
    assert counts["strictEvidenceWriteRows"] == 0
    assert report["rows"][0]["hold_status"] == HOLD_STATUS_ACTIVE
    assert report["gate"]["holdDecision"] == "strict_evidence_eligibility_post_apply_promotion_hold_active"
    assert validate_payload(
        report,
        STRICT_EVIDENCE_ELIGIBILITY_POST_APPLY_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_post_apply_hold_review_blocks_invalid_readback_report(tmp_path: Path) -> None:
    readback_path = tmp_path / "bad-readback.json"
    _write_json(readback_path, {"schema": "wrong.schema", "status": "ok", "rows": []})

    report = build_strict_evidence_eligibility_post_apply_promotion_hold_review(
        readback_review_report_path=readback_path,
        expected_input_rows=1,
        expected_eligibility_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedInputSchemaViolationRows"] == 0
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["gate"]["holdDecision"] == "strict_evidence_eligibility_post_apply_promotion_hold_blocked"


def test_post_apply_hold_review_blocks_illegal_row_flag_without_masking_it(tmp_path: Path) -> None:
    readback_path = _readback_report_path(tmp_path)
    payload = json.loads(readback_path.read_text(encoding="utf-8"))
    payload["rows"][0]["strictEligible"] = True
    _write_json(readback_path, payload)

    report = build_strict_evidence_eligibility_post_apply_promotion_hold_review(
        readback_review_report_path=readback_path,
        expected_input_rows=1,
        expected_eligibility_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["rows"][0]["hold_status"] == HOLD_STATUS_BLOCKED_RUNTIME_OR_CITATION
    assert "strictEligible_true" in report["rows"][0]["hold_blockers"]
    assert report["counts"]["blockedRuntimeOrCitationFlagViolationRows"] == 1


def test_post_apply_hold_review_blocks_downstream_gate_enablement(tmp_path: Path) -> None:
    readback_path = _readback_report_path(tmp_path)
    payload = json.loads(readback_path.read_text(encoding="utf-8"))
    payload["gate"]["citationReady"] = True
    _write_json(readback_path, payload)

    report = build_strict_evidence_eligibility_post_apply_promotion_hold_review(
        readback_review_report_path=readback_path,
        expected_input_rows=1,
        expected_eligibility_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["rows"][0]["hold_status"] == HOLD_STATUS_BLOCKED_GATE_ENABLED
    assert "downstream_gate_citationGradeEvidence_already_enabled" in report["rows"][0]["hold_blockers"]
    assert report["counts"]["blockedDownstreamGateAlreadyEnabledRows"] == 1


def test_post_apply_hold_review_blocks_store_count_drift(tmp_path: Path) -> None:
    readback_path = _readback_report_path(tmp_path)
    payload = json.loads(readback_path.read_text(encoding="utf-8"))
    payload["counts"]["strictEvidenceStoreRows"] = 2
    _write_json(readback_path, payload)

    report = build_strict_evidence_eligibility_post_apply_promotion_hold_review(
        readback_review_report_path=readback_path,
        expected_input_rows=1,
        expected_eligibility_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedStoreRowCountChangedRows"] == 1
    assert "strictEvidenceStoreRows=2_expected_1" in report["rows"][0]["hold_blockers"]


def test_post_apply_hold_review_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    readback_path = _readback_report_path(tmp_path)
    report = build_strict_evidence_eligibility_post_apply_promotion_hold_review(
        readback_review_report_path=readback_path,
        expected_input_rows=1,
        expected_eligibility_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_source_span_store_rows=1,
    )

    paths = write_strict_evidence_eligibility_post_apply_promotion_hold_review_reports(
        report,
        tmp_path / "reports",
    )

    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert written["schema"] == STRICT_EVIDENCE_ELIGIBILITY_POST_APPLY_PROMOTION_HOLD_REVIEW_SCHEMA_ID
    assert summary["counts"]["holdActiveRows"] == 1
    assert "Strict Evidence Eligibility Post-Apply Promotion Hold Review" in markdown
    assert validate_payload(
        written,
        STRICT_EVIDENCE_ELIGIBILITY_POST_APPLY_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_post_apply_hold_review_integrated_measured_local_report() -> None:
    report = build_strict_evidence_eligibility_post_apply_promotion_hold_review()

    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 99
    assert report["counts"]["eligibilityRecordRows"] == 99
    assert report["counts"]["readbackValidatedRows"] == 99
    assert report["counts"]["holdActiveRows"] == 99
    assert report["counts"]["sectionHoldRows"] == 45
    assert report["counts"]["figureCaptionHoldRows"] == 54
    assert report["counts"]["strictEvidenceStoreRows"] == 99
    assert report["counts"]["sourceSpanStoreRows"] == 102
    assert report["counts"]["blockedReadbackNotValidatedRows"] == 0
    assert report["counts"]["blockedDownstreamGateAlreadyEnabledRows"] == 0
    assert report["counts"]["blockedStoreRowCountChangedRows"] == 0
    assert report["counts"]["blockedRuntimeOrCitationFlagViolationRows"] == 0
    assert report["counts"]["eligibilityRecordWriteRows"] == 0
    assert report["counts"]["strictEvidenceWriteRows"] == 0
    assert report["counts"]["strictEligibleMutationRows"] == 0
    assert validate_payload(
        report,
        STRICT_EVIDENCE_ELIGIBILITY_POST_APPLY_PROMOTION_HOLD_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
