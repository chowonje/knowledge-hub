from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_runtime_binding_executor_apply import (
    build_strict_evidence_runtime_binding_executor_apply,
    write_strict_evidence_runtime_binding_executor_apply_reports,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_executor_apply_readback_review import (
    READBACK_STATUS_VALIDATED,
    RUNTIME_BINDING_STORE,
    STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_READBACK_REVIEW_SCHEMA_ID,
    build_strict_evidence_runtime_binding_executor_apply_readback_review,
    write_strict_evidence_runtime_binding_executor_apply_readback_review_reports,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_executor_dry_run import (
    build_strict_evidence_runtime_binding_executor_dry_run,
    write_strict_evidence_runtime_binding_executor_dry_run_reports,
)
from knowledge_hub.papers.strict_evidence_runtime_binding_record_contract import (
    build_strict_evidence_runtime_binding_record_contract,
    write_strict_evidence_runtime_binding_record_contract_reports,
)
from tests.test_strict_evidence_runtime_binding_record_contract import (
    _runtime_binding_gate_design_report_path,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_parent_store_records(papers_dir: Path, runtime_binding_record: dict) -> None:
    paper_id = runtime_binding_record["paperId"]
    artifact_type = runtime_binding_record["artifactType"]
    strict_evidence_id = runtime_binding_record["strictEvidenceId"]
    source_span_id = runtime_binding_record["sourceSpanId"]
    candidate_record_id = runtime_binding_record["candidateRecordId"]
    eligibility_record_id = runtime_binding_record["eligibilityRecordId"]
    citation_grade_record_id = runtime_binding_record["citationGradeRecordId"]

    _write_jsonl(
        papers_dir / "structured_evidence" / "strict_evidence_citation_grade" / f"{paper_id}.jsonl",
        [
            {
                "citationGradeRecordId": citation_grade_record_id,
                "strictEvidenceId": strict_evidence_id,
                "eligibilityRecordId": eligibility_record_id,
                "sourceSpanId": source_span_id,
                "candidateRecordId": candidate_record_id,
                "paperId": paper_id,
                "artifactType": artifact_type,
                "citationGrade": False,
                "runtimeEvidence": False,
                "runtimeVisible": False,
                "answerIntegrationVisible": False,
            }
        ],
    )
    _write_jsonl(
        papers_dir / "structured_evidence" / "strict_evidence" / f"{paper_id}.jsonl",
        [
            {
                "strictEvidenceId": strict_evidence_id,
                "sourceSpanIds": [source_span_id],
                "candidateRecordIds": [candidate_record_id],
                "paperId": paper_id,
                "artifactType": artifact_type,
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "runtimeVisible": False,
                "answerIntegrationVisible": False,
            }
        ],
    )
    _write_jsonl(
        papers_dir / "structured_evidence" / "strict_evidence_eligibility" / f"{paper_id}.jsonl",
        [
            {
                "eligibilityRecordId": eligibility_record_id,
                "strictEvidenceId": strict_evidence_id,
                "sourceSpanId": source_span_id,
                "candidateRecordId": candidate_record_id,
                "paperId": paper_id,
                "artifactType": artifact_type,
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "runtimeVisible": False,
                "answerIntegrationVisible": False,
            }
        ],
    )
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / f"{paper_id}.jsonl",
        [
            {
                "sourceSpanId": source_span_id,
                "candidateRecordId": candidate_record_id,
                "paperId": paper_id,
                "artifactType": artifact_type,
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
                "runtimeVisible": False,
                "answerIntegrationVisible": False,
            }
        ],
    )


def _fixture_reports(tmp_path: Path, papers_dir: Path) -> tuple[Path, Path, Path]:
    gate_design_path = _runtime_binding_gate_design_report_path(tmp_path)

    contract = build_strict_evidence_runtime_binding_record_contract(
        runtime_binding_gate_design_report_path=gate_design_path,
        expected_input_rows=1,
        expected_planned_runtime_binding_rows=1,
    )
    contract_paths = write_strict_evidence_runtime_binding_record_contract_reports(
        contract,
        tmp_path / "contract",
    )
    contract_path = Path(contract_paths["report"])

    dry_run = build_strict_evidence_runtime_binding_executor_dry_run(
        runtime_binding_record_contract_report_path=contract_path,
        runtime_binding_gate_design_report_path=gate_design_path,
        run_id="dry-run-test",
        expected_runtime_binding_gate_design_rows=1,
        expected_section_gate_design_rows=1,
        expected_figure_caption_gate_design_rows=0,
        expected_planned_runtime_binding_rows=1,
    )
    dry_run_paths = write_strict_evidence_runtime_binding_executor_dry_run_reports(
        dry_run,
        tmp_path / "executor-dry-run",
    )
    executor_dry_run_path = Path(dry_run_paths["report"])

    apply_plan = build_strict_evidence_runtime_binding_executor_apply(
        executor_dry_run_report_path=executor_dry_run_path,
        runtime_binding_record_contract_report_path=contract_path,
        run_id="apply-plan-test",
        apply=False,
    )
    apply_plan_paths = write_strict_evidence_runtime_binding_executor_apply_reports(
        apply_plan,
        tmp_path / "apply-plan",
    )
    apply_plan_path = Path(apply_plan_paths["report"])

    apply_report = build_strict_evidence_runtime_binding_executor_apply(
        executor_dry_run_report_path=executor_dry_run_path,
        runtime_binding_record_contract_report_path=contract_path,
        papers_dir=papers_dir,
        run_id="apply-test",
        apply=True,
    )
    runtime_binding_record = apply_report["runtimeBindingRecords"][0]
    _write_parent_store_records(papers_dir, runtime_binding_record)
    apply_paths = write_strict_evidence_runtime_binding_executor_apply_reports(
        apply_report,
        tmp_path / "apply",
    )
    apply_path = Path(apply_paths["report"])
    return apply_path, apply_plan_path, contract_path


def test_readback_review_validates_runtime_binding_store_without_writes(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    apply_path, dry_run_path, contract_path = _fixture_reports(tmp_path, papers_dir)

    report = build_strict_evidence_runtime_binding_executor_apply_readback_review(
        apply_report_path=apply_path,
        dry_run_report_path=dry_run_path,
        runtime_binding_record_contract_report_path=contract_path,
        papers_dir=papers_dir,
        expected_input_rows=1,
        expected_runtime_binding_record_rows=1,
        expected_citation_grade_store_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 1
    assert report["counts"]["runtimeBindingRecordRows"] == 1
    assert report["counts"]["readbackValidatedRows"] == 1
    assert report["counts"]["citationGradeStoreRows"] == 1
    assert report["counts"]["runtimeBindingRecordWriteRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["counts"]["answerIntegrationVisibleRows"] == 0
    assert report["rows"][0]["readback_status"] == READBACK_STATUS_VALIDATED
    assert report["rows"][0]["plannedWriteTarget"] == RUNTIME_BINDING_STORE
    assert validate_payload(
        report,
        STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_readback_review_blocks_missing_eligibility_reference(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    apply_path, dry_run_path, contract_path = _fixture_reports(tmp_path, papers_dir)
    eligibility_path = papers_dir / "structured_evidence" / "strict_evidence_eligibility" / "paper-1.jsonl"
    eligibility_path.write_text("", encoding="utf-8")

    report = build_strict_evidence_runtime_binding_executor_apply_readback_review(
        apply_report_path=apply_path,
        dry_run_report_path=dry_run_path,
        runtime_binding_record_contract_report_path=contract_path,
        papers_dir=papers_dir,
        expected_input_rows=1,
        expected_runtime_binding_record_rows=1,
        expected_citation_grade_store_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=0,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedMissingEligibilityReferenceRows"] == 1
    assert report["counts"]["runtimeBindingRecordWriteRows"] == 0


def test_readback_review_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    apply_path, dry_run_path, contract_path = _fixture_reports(tmp_path, papers_dir)

    report = build_strict_evidence_runtime_binding_executor_apply_readback_review(
        apply_report_path=apply_path,
        dry_run_report_path=dry_run_path,
        runtime_binding_record_contract_report_path=contract_path,
        papers_dir=papers_dir,
        expected_input_rows=1,
        expected_runtime_binding_record_rows=1,
        expected_citation_grade_store_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )
    paths = write_strict_evidence_runtime_binding_executor_apply_readback_review_reports(
        report,
        tmp_path / "reports",
    )
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert summary["counts"]["readbackValidatedRows"] == 1
    assert "Strict Evidence Runtime Binding Executor Apply Readback Review" in markdown
    assert validate_payload(
        written,
        STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_readback_review_integrated_measured_local_report() -> None:
    report = build_strict_evidence_runtime_binding_executor_apply_readback_review()
    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 99
    assert report["counts"]["runtimeBindingRecordRows"] == 99
    assert report["counts"]["readbackValidatedRows"] == 99
    assert report["counts"]["citationGradeStoreRows"] == 99
    assert report["counts"]["strictEvidenceStoreRows"] == 99
    assert report["counts"]["eligibilityStoreRows"] == 99
    assert report["counts"]["sourceSpanStoreRows"] == 102
    assert report["counts"]["blockedMissingRuntimeBindingStoreRows"] == 0
    assert report["counts"]["blockedRuntimeBindingRecordSchemaViolationRows"] == 0
    assert report["counts"]["blockedRuntimeBindingRecordSemanticViolationRows"] == 0
    assert report["counts"]["blockedMissingStrictEvidenceReferenceRows"] == 0
    assert report["counts"]["blockedMissingEligibilityReferenceRows"] == 0
    assert report["counts"]["blockedMissingSourceSpanReferenceRows"] == 0
    assert report["counts"]["blockedMissingCandidateRecordIdRows"] == 0
    assert report["counts"]["blockedCandidateIdentityMismatchRows"] == 0
    assert report["counts"]["blockedDuplicateIdempotencyKeyRows"] == 0
    assert report["counts"]["blockedDuplicateRuntimeBindingRecordIdRows"] == 0
    assert report["counts"]["blockedRuntimeOrAnswerFlagViolationRows"] == 0
    assert report["counts"]["runtimeBindingRecordWriteRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["counts"]["answerIntegrationVisibleRows"] == 0
    assert validate_payload(
        report,
        STRICT_EVIDENCE_RUNTIME_BINDING_EXECUTOR_APPLY_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
