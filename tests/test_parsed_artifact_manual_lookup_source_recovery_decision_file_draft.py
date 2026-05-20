from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_draft import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_DRAFT_SCHEMA_ID,
    build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft,
    write_parsed_artifact_manual_lookup_source_recovery_decision_file_draft,
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


def _review_pack(tmp_path: Path) -> dict:
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
    return build_parsed_artifact_manual_lookup_source_recovery_review_pack(
        manual_lookup_plan_report=plan,
        manual_lookup_plan_report_path=tmp_path / "plan.json",
        generated_at="2026-05-21T00:00:00+00:00",
    )


def test_decision_file_draft_defaults_all_rows_to_needs_review(tmp_path: Path) -> None:
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft(
        review_pack_report=_review_pack(tmp_path),
        review_pack_report_path=tmp_path / "review-pack.json",
        generated_at="2026-05-21T00:00:00+00:00",
    )

    assert payload["schema"] == PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_DRAFT_SCHEMA_ID
    assert validate_payload(
        payload,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_DRAFT_SCHEMA_ID,
        strict=True,
    ).ok
    assert payload["status"] == "decision_file_draft_ready"
    assert payload["counts"]["inputRows"] == 3
    assert payload["counts"]["sourceReviewCardRows"] == 2
    assert payload["counts"]["draftDecisionRows"] == 2
    assert payload["counts"]["needsReviewRows"] == 2
    assert payload["counts"]["approvedDecisionRows"] == 0
    assert payload["counts"]["applyReadyRows"] == 0
    assert payload["counts"]["textSourceHoldoutRows"] == 1
    assert payload["gate"]["containsOnlyNeedsReviewDefaults"] is True
    assert payload["gate"]["containsAcceptedSourceApprovals"] is False
    assert payload["gate"]["applyReady"] is False
    assert payload["manualLookupDecisionDraftPaperIds"] == ["registered-missing", "custom-missing"]
    assert all(row["decision"] == "needs_review" for row in payload["draftRows"])
    assert all(row["approvedSourceUrl"] == "" for row in payload["draftRows"])
    assert all(row["approvedLocalPdfPath"] == "" for row in payload["draftRows"])
    assert all(row["applyReady"] is False for row in payload["draftRows"])
    assert all(row["decision"] == "needs_review" for row in payload["decisionFileDraft"]["decisions"])


def test_decision_file_draft_blocks_unsafe_review_pack() -> None:
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft(
        review_pack_report={"schema": "wrong"},
        generated_at="2026-05-21T00:00:00+00:00",
    )

    assert payload["status"] == "blocked"
    assert payload["inputReviewPack"]["schemaValidation"]["ok"] is False
    assert payload["counts"]["draftDecisionRows"] == 0
    assert payload["gate"]["decisionFileDraftReady"] is False
    assert "manual_lookup_source_recovery_review_pack_schema_violation" in payload["gate"]["unsafeUpstreamFlags"]
    assert validate_payload(
        payload,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_DRAFT_SCHEMA_ID,
        strict=True,
    ).ok


def test_decision_file_draft_writer_outputs_editable_draft_file(tmp_path: Path) -> None:
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft(
        review_pack_report=_review_pack(tmp_path),
        generated_at="2026-05-21T00:00:00+00:00",
    )

    paths = write_parsed_artifact_manual_lookup_source_recovery_decision_file_draft(payload, tmp_path / "reports")
    report = json.loads(Path(paths["reportJsonPath"]).read_text(encoding="utf-8"))
    draft_file = json.loads(Path(paths["decisionFileDraftPath"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summaryJsonPath"]).read_text(encoding="utf-8"))
    markdown = Path(paths["reportMarkdownPath"]).read_text(encoding="utf-8")

    assert validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_DRAFT_SCHEMA_ID,
        strict=True,
    ).ok
    assert draft_file["draftOnly"] is True
    assert len(draft_file["decisions"]) == 2
    assert all(row["decision"] == "needs_review" for row in draft_file["decisions"])
    assert all(row["approvedSourceUrl"] == "" for row in draft_file["decisions"])
    assert summary["counts"]["draftDecisionRows"] == 2
    assert "editable starting point only" in markdown


def test_decision_file_draft_preserves_zero_mutation_boundary(tmp_path: Path) -> None:
    payload = build_parsed_artifact_manual_lookup_source_recovery_decision_file_draft(
        review_pack_report=_review_pack(tmp_path),
        generated_at="2026-05-21T00:00:00+00:00",
    )

    assert payload["mutationCounters"]["externalLookupRows"] == 0
    assert payload["mutationCounters"]["sourceDownloadRows"] == 0
    assert payload["mutationCounters"]["sourceRegistrationMutationRows"] == 0
    assert payload["mutationCounters"]["parsedArtifactWriteRows"] == 0
    assert payload["mutationCounters"]["vaultReadRows"] == 0
    assert payload["mutationCounters"]["runManifestWriteRows"] == 0
    assert payload["gate"]["sourceRegistrationMutationReady"] is False
    assert payload["gate"]["parsedArtifactMaterializationReady"] is False
    assert payload["gate"]["answerIntegrationReady"] is False
