from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_draft import (
    build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_human_review import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID,
    build_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review,
    write_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_validation import (
    build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation,
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


def _validation_bundle(tmp_path: Path) -> tuple[dict, dict, dict, dict]:
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
    validation_report = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=draft_report,
        decision_file=decision_file,
        generated_at="2026-05-21T00:00:00+00:00",
        expected_input_rows=2,
    )
    return validation_report, decision_file, review_pack, feasibility


def test_human_review_accepts_validated_needs_review_rows(tmp_path: Path) -> None:
    validation_report, decision_file, review_pack, feasibility = _validation_bundle(tmp_path)
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review(
        decision_file_validation_report=validation_report,
        decision_file=decision_file,
        review_pack_report=review_pack,
        source_recovery_feasibility_report=feasibility,
        generated_at="2026-05-21T00:00:00+00:00",
        expected_manual_lookup_rows=2,
        expected_text_source_holdout_rows=1,
    )

    assert payload["schema"] == PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID
    assert validate_payload(
        payload,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
    assert payload["status"] == "decision_file_human_review_ready"
    assert payload["counts"]["manualReviewRows"] == 2
    assert payload["counts"]["needsReviewRows"] == 2
    assert payload["counts"]["textSourceHoldoutRows"] == 1
    assert payload["counts"]["unexpectedBlockerCount"] == 0
    assert payload["gate"]["humanReviewSheetReady"] is True
    assert payload["gate"]["automatedBlockerTrancheClosed"] is True
    assert payload["remainingBlockerClosure"]["automatedRecoveryTranchesClosed"] is True
    assert payload["gate"]["recommendedNextTranche"] == (
        "parsed_artifact_manual_lookup_source_recovery_operator_decision_packet"
    )


def test_human_review_blocks_unsafe_validation_report(tmp_path: Path) -> None:
    validation_report, decision_file, review_pack, feasibility = _validation_bundle(tmp_path)
    validation_report["status"] = "blocked"

    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review(
        decision_file_validation_report=validation_report,
        decision_file=decision_file,
        review_pack_report=review_pack,
        source_recovery_feasibility_report=feasibility,
        expected_manual_lookup_rows=2,
        expected_text_source_holdout_rows=1,
    )

    assert payload["status"] == "blocked"
    assert payload["gate"]["humanReviewSheetReady"] is False
    assert payload["gate"]["recommendedNextTranche"] == (
        "parsed_artifact_manual_lookup_source_recovery_decision_file_validation_repair"
    )


def test_human_review_preserves_zero_mutation_boundary(tmp_path: Path) -> None:
    validation_report, decision_file, review_pack, feasibility = _validation_bundle(tmp_path)
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review(
        decision_file_validation_report=validation_report,
        decision_file=decision_file,
        review_pack_report=review_pack,
        source_recovery_feasibility_report=feasibility,
        expected_manual_lookup_rows=2,
        expected_text_source_holdout_rows=1,
    )

    assert payload["mutationCounters"]["externalLookupRows"] == 0
    assert payload["mutationCounters"]["sourceRegistrationMutationRows"] == 0
    assert payload["mutationCounters"]["parsedArtifactWriteRows"] == 0
    assert payload["mutationCounters"]["decisionFileMutationRows"] == 0
    assert payload["mutationPolicy"]["humanDecisionRecording"] is False


def test_human_review_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    validation_report, decision_file, review_pack, feasibility = _validation_bundle(tmp_path)
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review(
        decision_file_validation_report=validation_report,
        decision_file=decision_file,
        review_pack_report=review_pack,
        source_recovery_feasibility_report=feasibility,
        expected_manual_lookup_rows=2,
        expected_text_source_holdout_rows=1,
    )
    paths = write_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review(
        payload,
        tmp_path / "reports",
    )

    report = json.loads(Path(paths["reportJsonPath"]).read_text(encoding="utf-8"))
    markdown = Path(paths["reportMarkdownPath"]).read_text(encoding="utf-8")

    assert validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
    assert "Human Review" in markdown


@pytest.mark.skip(reason="operator-local integration report is excluded from PR gate")
def test_human_review_integrated_measured_local_report() -> None:
    report_root = _operator_report_root()
    validation_report = json.loads(
        (
            report_root
            / "parsed-artifact-manual-lookup-source-recovery-decision-file-validation/"
            "01-parsed-artifact-manual-lookup-source-recovery-decision-file-validation/"
            "parsed-artifact-manual-lookup-source-recovery-decision-file-validation.json"
        ).read_text(encoding="utf-8")
    )
    decision_file = json.loads(
        (
            report_root
            / "parsed-artifact-manual-lookup-source-recovery-decision-file-draft/"
            "01-parsed-artifact-manual-lookup-source-recovery-decision-file-draft/"
            "manual-lookup-source-recovery-decisions.draft.json"
        ).read_text(encoding="utf-8")
    )
    review_pack = json.loads(
        (
            report_root
            / "parsed-artifact-manual-lookup-source-recovery-review-pack/"
            "01-parsed-artifact-manual-lookup-source-recovery-review-pack/"
            "parsed-artifact-manual-lookup-source-recovery-review-pack.json"
        ).read_text(encoding="utf-8")
    )
    feasibility = json.loads(
        (
            report_root
            / "parsed-artifact-source-recovery-feasibility-post-oversized-apply/"
            "parsed-artifact-source-recovery-feasibility.json"
        ).read_text(encoding="utf-8")
    )

    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review(
        decision_file_validation_report=validation_report,
        decision_file=decision_file,
        review_pack_report=review_pack,
        source_recovery_feasibility_report=feasibility,
    )

    assert payload["status"] in {"decision_file_human_review_ready", "blocked"}
    if payload["counts"]["manualReviewRows"] == 0:
        assert payload["gate"]["humanReviewSheetReady"] is False
        assert payload["gate"]["schemaViolations"]
        return
    assert payload["counts"]["manualReviewRows"] == 15
    assert payload["counts"]["textSourceHoldoutRows"] == 1
    assert payload["counts"]["unexpectedBlockerCount"] == 0
    assert payload["counts"]["needsReviewRows"] + payload["counts"]["nonNeedsReviewRows"] == 15
    closure = payload["remainingBlockerClosure"]
    assert closure["missingParsedArtifactsRemaining"] == 16
    taxonomy = {item["reason"]: item["count"] for item in closure["blockerTaxonomy"]}
    assert taxonomy["manual_lookup_required"] == 15
    assert taxonomy["text_source_unsupported"] == 1
    if payload["counts"]["nonNeedsReviewRows"] == 0:
        assert payload["gate"]["humanReviewSheetReady"] is True
    else:
        assert payload["gate"]["humanReviewSheetReady"] is False
        assert payload["gate"]["containsNonNeedsReviewDecisions"] is True
    assert validate_payload(
        payload,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
