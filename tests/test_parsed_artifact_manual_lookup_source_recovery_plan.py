from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_plan import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_PLAN_SCHEMA_ID,
    build_parsed_artifact_manual_lookup_source_recovery_plan,
    write_parsed_artifact_manual_lookup_source_recovery_plan,
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


def _feasibility_report(db: SQLiteDatabase, papers_dir: Path) -> dict:
    return build_parsed_artifact_source_recovery_feasibility(
        sqlite_db=db,
        papers_dir=papers_dir,
        generated_at="2026-05-21T00:00:00+00:00",
    )


def test_manual_lookup_plan_classifies_manual_and_text_holdout_rows(tmp_path: Path) -> None:
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

    report = build_parsed_artifact_manual_lookup_source_recovery_plan(
        feasibility_report=_feasibility_report(db, papers_dir),
        feasibility_report_path=tmp_path / "feasibility.json",
        target_missing_parsed_artifacts=1,
        generated_at="2026-05-21T00:00:00+00:00",
    )

    assert report["schema"] == PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_PLAN_SCHEMA_ID
    assert validate_payload(report, PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_PLAN_SCHEMA_ID, strict=True).ok
    assert report["status"] == "manual_lookup_source_recovery_plan_candidate_only"
    assert report["candidatePool"]["inputRows"] == 3
    assert report["candidatePool"]["manualLookupCandidateOnlyRows"] == 2
    assert report["candidatePool"]["textSourceContractHoldRows"] == 1
    assert report["manualLookupCandidatePaperIds"] == ["custom-missing", "registered-missing"]
    assert report["textSourceHoldoutPaperIds"] == ["text-only"]
    assert report["coverageTarget"]["manualRecoveriesNeededForTarget"] == 2
    assert report["coverageTarget"]["targetReachableIfEnoughManualSourcesResolved"] is True
    modes = {row["paperId"]: row["manualLookupMode"] for row in report["manualLookupCandidates"]}
    assert modes == {
        "custom-missing": "no_registered_source_artifact_manual_lookup",
        "registered-missing": "registered_pdf_path_missing_manual_lookup",
    }
    assert report["mutationCounters"]["sourceDownloadRows"] == 0
    assert report["mutationCounters"]["parsedArtifactWriteRows"] == 0
    assert not (papers_dir / "parsed" / "custom-missing").exists()


def test_manual_lookup_plan_blocks_invalid_input_schema(tmp_path: Path) -> None:
    report = build_parsed_artifact_manual_lookup_source_recovery_plan(
        feasibility_report={"schema": "wrong"},
        generated_at="2026-05-21T00:00:00+00:00",
    )

    assert report["status"] == "blocked_input_schema_violation"
    assert report["inputFeasibility"]["schemaValidation"]["ok"] is False
    assert report["candidatePool"]["manualLookupCandidateOnlyRows"] == 0
    assert validate_payload(report, PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_PLAN_SCHEMA_ID, strict=True).ok


def test_manual_lookup_plan_writer_outputs_report_and_candidate_ids(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="manual-paper", title="Manual Paper")
    report = build_parsed_artifact_manual_lookup_source_recovery_plan(
        feasibility_report=_feasibility_report(db, papers_dir),
        generated_at="2026-05-21T00:00:00+00:00",
    )

    paths = write_parsed_artifact_manual_lookup_source_recovery_plan(report, tmp_path / "reports")
    written = json.loads(Path(paths["reportJsonPath"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summaryJsonPath"]).read_text(encoding="utf-8"))

    assert validate_payload(written, PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_PLAN_SCHEMA_ID, strict=True).ok
    assert summary["reportSchema"] == PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_PLAN_SCHEMA_ID
    assert Path(paths["reportMarkdownPath"]).exists()
    assert Path(paths["manualLookupCandidateIdsPath"]).read_text(encoding="utf-8").strip() == "manual-paper"
    assert Path(paths["textSourceHoldoutIdsPath"]).read_text(encoding="utf-8").strip() == ""
