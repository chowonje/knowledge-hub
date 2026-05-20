from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.strict_evidence_citation_grade_executor_apply import (
    build_strict_evidence_citation_grade_executor_apply,
    write_strict_evidence_citation_grade_executor_apply_reports,
)
from knowledge_hub.papers.strict_evidence_citation_grade_executor_apply_readback_review import (
    READBACK_STATUS_VALIDATED,
    STRICT_EVIDENCE_CITATION_GRADE_EXECUTOR_APPLY_READBACK_REVIEW_SCHEMA_ID,
    STRICT_EVIDENCE_CITATION_GRADE_STORE,
    build_strict_evidence_citation_grade_executor_apply_readback_review,
    write_strict_evidence_citation_grade_executor_apply_readback_review_reports,
)
from knowledge_hub.papers.strict_evidence_citation_grade_executor_dry_run import (
    build_strict_evidence_citation_grade_executor_dry_run,
    write_strict_evidence_citation_grade_executor_dry_run_reports,
)
from knowledge_hub.papers.strict_evidence_citation_grade_record_contract import (
    build_sample_citation_grade_record_from_policy_row,
    build_strict_evidence_citation_grade_record_contract,
    write_strict_evidence_citation_grade_record_contract_reports,
)
from tests.test_strict_evidence_citation_grade_record_contract import (
    _policy_gate_design_report_path,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _fixture_reports(tmp_path: Path, papers_dir: Path) -> tuple[Path, Path, Path]:
    policy_path = _policy_gate_design_report_path(tmp_path)
    contract = build_strict_evidence_citation_grade_record_contract(
        policy_gate_design_report_path=policy_path,
    )
    contract_dir = tmp_path / "contract"
    write_strict_evidence_citation_grade_record_contract_reports(contract, contract_dir)
    contract_path = contract_dir / "strict-evidence-citation-grade-record-contract.json"
    dry_run = build_strict_evidence_citation_grade_executor_dry_run(
        citation_grade_record_contract_report_path=contract_path,
        policy_gate_design_report_path=policy_path,
        run_id="dry-run-test",
        expected_policy_candidate_rows=1,
        expected_section_policy_rows=1,
        expected_figure_caption_policy_rows=0,
    )
    dry_run_dir = tmp_path / "dry-run"
    write_strict_evidence_citation_grade_executor_dry_run_reports(dry_run, dry_run_dir)
    dry_run_path = dry_run_dir / "strict-evidence-citation-grade-executor-dry-run.json"

    policy_row = dry_run["rows"][0]
    citation_grade_record = build_sample_citation_grade_record_from_policy_row(
        {
            "policy_design_row_id": policy_row.get("policy_design_row_id"),
            "hold_row_id": policy_row.get("hold_row_id"),
            "strictEvidenceId": policy_row.get("strictEvidenceId"),
            "sourceSpanId": policy_row.get("sourceSpanId"),
            "candidateRecordId": policy_row.get("candidateRecordId"),
            "eligibilityRecordId": policy_row.get("eligibilityRecordId"),
            "paper_id": policy_row.get("paper_id"),
            "artifact_type": policy_row.get("artifact_type"),
            "citation_grade_policy_design_status": policy_row.get("citation_grade_policy_design_status"),
        },
        run_id="apply-test",
    )
    strict_evidence_id = citation_grade_record["strictEvidenceId"]
    source_span_id = citation_grade_record["sourceSpanId"]
    candidate_record_id = citation_grade_record["candidateRecordId"]
    eligibility_record_id = citation_grade_record["eligibilityRecordId"]

    _write_jsonl(
        papers_dir / "structured_evidence" / "strict_evidence" / "paper-1.jsonl",
        [
            {
                "strictEvidenceId": strict_evidence_id,
                "sourceSpanIds": [source_span_id],
                "candidateRecordIds": [candidate_record_id],
                "paperId": "paper-1",
                "artifactType": "section",
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
            }
        ],
    )
    _write_jsonl(
        papers_dir / "structured_evidence" / "strict_evidence_eligibility" / "paper-1.jsonl",
        [
            {
                "eligibilityRecordId": eligibility_record_id,
                "strictEvidenceId": strict_evidence_id,
                "sourceSpanId": source_span_id,
                "candidateRecordId": candidate_record_id,
                "paperId": "paper-1",
                "artifactType": "section",
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
            }
        ],
    )
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-1.jsonl",
        [
            {
                "sourceSpanId": source_span_id,
                "candidateRecordId": candidate_record_id,
                "paperId": "paper-1",
                "artifactType": "section",
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
            }
        ],
    )

    apply_report = build_strict_evidence_citation_grade_executor_apply(
        executor_dry_run_report_path=dry_run_path,
        citation_grade_record_contract_report_path=contract_path,
        papers_dir=papers_dir,
        run_id="apply-test",
        apply=True,
    )
    apply_dir = tmp_path / "apply"
    write_strict_evidence_citation_grade_executor_apply_reports(apply_report, apply_dir)
    apply_path = apply_dir / "strict-evidence-citation-grade-executor-apply.json"
    return apply_path, dry_run_path, contract_path


def test_readback_review_validates_citation_grade_store_without_writes(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    apply_path, dry_run_path, contract_path = _fixture_reports(tmp_path, papers_dir)

    report = build_strict_evidence_citation_grade_executor_apply_readback_review(
        apply_report_path=apply_path,
        dry_run_report_path=apply_path,
        citation_grade_record_contract_report_path=contract_path,
        papers_dir=papers_dir,
        expected_input_rows=1,
        expected_citation_grade_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 1
    assert report["counts"]["citationGradeRecordRows"] == 1
    assert report["counts"]["readbackValidatedRows"] == 1
    assert report["counts"]["citationGradeRecordWriteRows"] == 0
    assert report["counts"]["citationGradeBooleanMutationRows"] == 0
    assert report["rows"][0]["readback_status"] == READBACK_STATUS_VALIDATED
    assert report["rows"][0]["plannedWriteTarget"] == STRICT_EVIDENCE_CITATION_GRADE_STORE
    assert validate_payload(
        report,
        STRICT_EVIDENCE_CITATION_GRADE_EXECUTOR_APPLY_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_readback_review_blocks_missing_eligibility_reference(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    apply_path, dry_run_path, contract_path = _fixture_reports(tmp_path, papers_dir)
    eligibility_path = papers_dir / "structured_evidence" / "strict_evidence_eligibility" / "paper-1.jsonl"
    eligibility_path.write_text("", encoding="utf-8")

    report = build_strict_evidence_citation_grade_executor_apply_readback_review(
        apply_report_path=apply_path,
        dry_run_report_path=apply_path,
        citation_grade_record_contract_report_path=contract_path,
        papers_dir=papers_dir,
        expected_input_rows=1,
        expected_citation_grade_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=0,
        expected_source_span_store_rows=1,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedMissingEligibilityReferenceRows"] == 1


def test_readback_review_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    apply_path, dry_run_path, contract_path = _fixture_reports(tmp_path, papers_dir)

    report = build_strict_evidence_citation_grade_executor_apply_readback_review(
        apply_report_path=apply_path,
        dry_run_report_path=apply_path,
        citation_grade_record_contract_report_path=contract_path,
        papers_dir=papers_dir,
        expected_input_rows=1,
        expected_citation_grade_record_rows=1,
        expected_strict_evidence_store_rows=1,
        expected_eligibility_store_rows=1,
        expected_source_span_store_rows=1,
    )
    paths = write_strict_evidence_citation_grade_executor_apply_readback_review_reports(
        report,
        tmp_path / "reports",
    )
    written = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert validate_payload(
        written,
        STRICT_EVIDENCE_CITATION_GRADE_EXECUTOR_APPLY_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_readback_review_integrated_measured_local_report() -> None:
    report = build_strict_evidence_citation_grade_executor_apply_readback_review()
    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 99
    assert report["counts"]["citationGradeRecordRows"] == 99
    assert report["counts"]["readbackValidatedRows"] == 99
    assert report["counts"]["strictEvidenceStoreRows"] == 99
    assert report["counts"]["eligibilityStoreRows"] == 99
    assert report["counts"]["sourceSpanStoreRows"] == 102
    assert report["counts"]["blockedMissingCitationGradeStoreRows"] == 0
    assert report["counts"]["blockedCitationGradeRecordSchemaViolationRows"] == 0
    assert report["counts"]["blockedCitationGradeRecordSemanticViolationRows"] == 0
    assert report["counts"]["blockedMissingStrictEvidenceReferenceRows"] == 0
    assert report["counts"]["blockedMissingEligibilityReferenceRows"] == 0
    assert report["counts"]["blockedMissingSourceSpanReferenceRows"] == 0
    assert report["counts"]["blockedMissingCandidateRecordIdRows"] == 0
    assert report["counts"]["blockedCandidateIdentityMismatchRows"] == 0
    assert report["counts"]["blockedDuplicateIdempotencyKeyRows"] == 0
    assert report["counts"]["blockedDuplicateCitationGradeRecordIdRows"] == 0
    assert report["counts"]["blockedRuntimeOrCitationFlagViolationRows"] == 0
    assert report["counts"]["citationGradeRecordWriteRows"] == 0
    assert report["counts"]["citationGradeBooleanMutationRows"] == 0
    assert validate_payload(
        report,
        STRICT_EVIDENCE_CITATION_GRADE_EXECUTOR_APPLY_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
