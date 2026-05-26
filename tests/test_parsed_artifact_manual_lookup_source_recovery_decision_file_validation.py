from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_draft import (
    build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft,
    write_parsed_artifact_manual_lookup_source_recovery_decision_file_draft,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_validation import (
    DECISION_APPROVE_SOURCE_URL,
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID,
    ROW_STATUS_INVALID_DISALLOWED_DECISION,
    ROW_STATUS_VALID_APPLY_READY,
    ROW_STATUS_VALID_NEEDS_REVIEW,
    build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation,
    write_parsed_artifact_manual_lookup_source_recovery_decision_file_validation,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_plan import (
    build_parsed_artifact_manual_lookup_source_recovery_plan,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_review_pack import (
    build_parsed_artifact_manual_lookup_source_recovery_review_pack,
)
from knowledge_hub.papers.parsed_artifact_source_recovery_feasibility import (
    build_parsed_artifact_source_recovery_feasibility,
)


def _operator_report_root() -> Path:
    return Path.home() / ("." + "khub") / "reports" / "parsed-artifact-coverage" / "2026-05-21"


def _seed_paper(
    db: SQLiteDatabase,
    *,
    paper_id: str,
    title: str,
    pdf_path: str = "",
    text_path: str = "",
) -> None:
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": title,
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "",
            "pdf_path": pdf_path,
            "text_path": text_path,
            "translated_path": "",
        }
    )


def _draft_bundle(tmp_path: Path) -> tuple[dict, dict]:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    text_source = papers_dir / "source.txt"
    text_source.write_text("text source", encoding="utf-8")
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="custom-missing", title="Custom Missing")
    _seed_paper(
        db,
        paper_id="registered-missing",
        title="Registered Missing",
        pdf_path=str(papers_dir / "registered.pdf"),
    )
    _seed_paper(db, paper_id="text-only", title="Text Only", text_path=str(text_source))
    feasibility = build_parsed_artifact_source_recovery_feasibility(
        sqlite_db=db,
        papers_dir=papers_dir,
        generated_at="2026-05-21T00:00:00+00:00",
    )
    plan = build_parsed_artifact_manual_lookup_source_recovery_plan(
        feasibility_report=feasibility,
        feasibility_report_path=tmp_path / "feasibility.json",
        target_missing_parsed_artifacts=1,
        generated_at="2026-05-21T00:00:00+00:00",
    )
    review_pack = build_parsed_artifact_manual_lookup_source_recovery_review_pack(
        manual_lookup_plan_report=plan,
        manual_lookup_plan_report_path=tmp_path / "plan.json",
        generated_at="2026-05-21T00:00:00+00:00",
    )
    draft_report = build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft(
        review_pack_report=review_pack,
        review_pack_report_path=tmp_path / "review-pack.json",
        generated_at="2026-05-21T00:00:00+00:00",
    )
    decision_file = deepcopy(dict(draft_report.get("decisionFileDraft") or {}))
    return draft_report, decision_file


def test_decision_file_validation_accepts_needs_review_defaults(tmp_path: Path) -> None:
    draft_report, decision_file = _draft_bundle(tmp_path)
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        generated_at="2026-05-21T00:00:00+00:00",
        expected_input_rows=2,
    )

    assert payload["schema"] == PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID
    assert validate_payload(
        payload,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID,
        strict=True,
    ).ok
    assert payload["status"] == "decision_file_validation_ready"
    assert payload["counts"]["inputRows"] == 2
    assert payload["counts"]["validRows"] == 2
    assert payload["counts"]["needsReviewRows"] == 2
    assert payload["counts"]["approvedDecisionRows"] == 0
    assert payload["counts"]["rejectedDecisionRows"] == 0
    assert payload["counts"]["applyReadyRows"] == 0
    assert payload["counts"]["invalidRows"] == 0
    assert payload["gate"]["decisionFileValidationReady"] is True
    assert payload["gate"]["containsOnlyValidatedNeedsReviewRows"] is True
    assert payload["gate"]["recommendedNextTranche"] == (
        "parsed_artifact_manual_lookup_source_recovery_decision_file_human_review"
    )
    assert all(row["validationStatus"] == ROW_STATUS_VALID_NEEDS_REVIEW for row in payload["validationRows"])


def test_decision_file_validation_accepts_valid_approved_decision(tmp_path: Path) -> None:
    draft_report, decision_file = _draft_bundle(tmp_path)
    decision_file["decisions"][0]["decision"] = DECISION_APPROVE_SOURCE_URL
    decision_file["decisions"][0]["approvedSourceType"] = "url"
    decision_file["decisions"][0]["approvedSourceUrl"] = "https://example.com/paper.pdf"
    decision_file["decisions"][0]["approvedSourceContentHash"] = "sha256:" + "a" * 64
    decision_file["decisions"][0]["approvedBy"] = "reviewer"
    decision_file["decisions"][0]["approvedAt"] = "2026-05-21T00:00:00+00:00"
    decision_file["decisions"][0]["notes"] = "approved for later apply"

    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        generated_at="2026-05-21T00:00:00+00:00",
        expected_input_rows=2,
    )

    assert payload["status"] == "decision_file_validation_ready"
    assert payload["counts"]["validRows"] == 2
    assert payload["counts"]["approvedDecisionRows"] == 1
    assert payload["counts"]["applyReadyRows"] == 1
    assert payload["validationRows"][0]["validationStatus"] == ROW_STATUS_VALID_APPLY_READY
    assert payload["gate"]["containsApplyReadyRows"] is True
    assert payload["gate"]["recommendedNextTranche"] == (
        "parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run"
    )


def test_decision_file_validation_blocks_invalid_decision_choice(tmp_path: Path) -> None:
    draft_report, decision_file = _draft_bundle(tmp_path)
    decision_file["decisions"][0]["decision"] = "approve_everything_now"

    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        generated_at="2026-05-21T00:00:00+00:00",
        expected_input_rows=2,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["invalidRows"] == 1
    assert payload["validationRows"][0]["validationStatus"] == ROW_STATUS_INVALID_DISALLOWED_DECISION
    assert payload["gate"]["decisionFileValidationReady"] is False


def test_decision_file_validation_blocks_unsafe_draft_report(tmp_path: Path) -> None:
    draft_report, decision_file = _draft_bundle(tmp_path)
    draft_report["schema"] = "wrong.schema"

    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        generated_at="2026-05-21T00:00:00+00:00",
        expected_input_rows=2,
    )

    assert payload["status"] == "blocked"
    assert payload["inputDecisionFileDraft"]["schemaValidation"]["ok"] is False
    assert payload["gate"]["decisionFileValidationReady"] is False
    assert validate_payload(
        payload,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID,
        strict=True,
    ).ok


def test_decision_file_validation_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    draft_report, decision_file = _draft_bundle(tmp_path)
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        generated_at="2026-05-21T00:00:00+00:00",
        expected_input_rows=2,
    )
    paths = write_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        payload,
        tmp_path / "reports",
    )

    report = json.loads(Path(paths["reportJsonPath"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summaryJsonPath"]).read_text(encoding="utf-8"))
    markdown = Path(paths["reportMarkdownPath"]).read_text(encoding="utf-8")

    assert validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID,
        strict=True,
    ).ok
    assert summary["counts"]["validRows"] == 2
    assert "Decision File Validation" in markdown


def test_decision_file_validation_preserves_zero_mutation_boundary(tmp_path: Path) -> None:
    draft_report, decision_file = _draft_bundle(tmp_path)
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        generated_at="2026-05-21T00:00:00+00:00",
        expected_input_rows=2,
    )

    assert payload["mutationCounters"]["externalLookupRows"] == 0
    assert payload["mutationCounters"]["sourceDownloadRows"] == 0
    assert payload["mutationCounters"]["sourceRegistrationMutationRows"] == 0
    assert payload["mutationCounters"]["parsedArtifactWriteRows"] == 0
    assert payload["mutationCounters"]["vaultReadRows"] == 0
    assert payload["gate"]["sourceRegistrationMutationReady"] is False
    assert payload["gate"]["parsedArtifactMaterializationReady"] is False


@pytest.mark.skip(reason="operator-local integration report is excluded from PR gate")
def test_decision_file_validation_integrated_measured_local_report() -> None:
    report_root = _operator_report_root()
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=json.loads(
            (
                report_root
                / "parsed-artifact-manual-lookup-source-recovery-decision-file-draft/"
                "01-parsed-artifact-manual-lookup-source-recovery-decision-file-draft/"
                "parsed-artifact-manual-lookup-source-recovery-decision-file-draft.json"
            ).read_text(encoding="utf-8")
        ),
        decision_file=json.loads(
            (
                report_root
                / "parsed-artifact-manual-lookup-source-recovery-decision-file-draft/"
                "01-parsed-artifact-manual-lookup-source-recovery-decision-file-draft/"
                "manual-lookup-source-recovery-decisions.draft.json"
            ).read_text(encoding="utf-8")
        ),
    )

    assert payload["status"] == "decision_file_validation_ready"
    assert payload["counts"]["inputRows"] == 15
    assert payload["counts"]["validRows"] == 15
    assert payload["counts"]["needsReviewRows"] == 0
    assert payload["counts"]["approvedDecisionRows"] == 12
    assert payload["counts"]["rejectedDecisionRows"] == 0
    assert payload["counts"]["holdForManualLookupRows"] == 3
    assert payload["counts"]["applyReadyRows"] == 12
    assert payload["counts"]["invalidRows"] == 0
    assert payload["mutationCounters"]["externalLookupRows"] == 0
    assert payload["mutationCounters"]["sourceRegistrationMutationRows"] == 0
    assert payload["mutationCounters"]["parsedArtifactWriteRows"] == 0
    assert validate_payload(
        payload,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID,
        strict=True,
    ).ok


def test_decision_file_draft_writer_then_validation_round_trip(tmp_path: Path) -> None:
    draft_report, _decision_file = _draft_bundle(tmp_path)
    draft_paths = write_parsed_artifact_manual_lookup_source_recovery_decision_file_draft(
        draft_report,
        tmp_path / "draft",
    )
    loaded_draft_report = json.loads(Path(draft_paths["reportJsonPath"]).read_text(encoding="utf-8"))
    loaded_decision_file = json.loads(Path(draft_paths["decisionFileDraftPath"]).read_text(encoding="utf-8"))
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=loaded_draft_report,
        decision_file=loaded_decision_file,
        generated_at="2026-05-21T00:00:00+00:00",
        expected_input_rows=2,
    )

    assert payload["status"] == "decision_file_validation_ready"
    assert payload["counts"]["validRows"] == 2
