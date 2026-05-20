from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_source_missing_resolution_plan import (
    PARSED_ARTIFACT_SOURCE_MISSING_RESOLUTION_PLAN_SCHEMA_ID,
    build_parsed_artifact_source_missing_resolution_plan,
    write_parsed_artifact_source_missing_resolution_plan,
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


def test_source_missing_resolution_plan_selects_bounded_source_pdf_missing_only(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    existing_pdf = papers_dir / "existing.pdf"
    oversized_pdf = papers_dir / "oversized.pdf"
    text_source = papers_dir / "source.txt"
    existing_pdf.write_bytes(b"%PDF-1.4 small")
    oversized_pdf.write_bytes(b"x" * 32)
    text_source.write_text("plain text", encoding="utf-8")

    _seed_paper(db, paper_id="2600.10001", title="Missing Registered PDF", pdf_path=str(papers_dir / "missing-1.pdf"))
    _seed_paper(db, paper_id="2600.10002", title="No Registered Source")
    _seed_paper(db, paper_id="2600.10003", title="Missing Registered PDF 2", pdf_path=str(papers_dir / "missing-2.pdf"))
    _seed_paper(db, paper_id="2600.10004", title="Oversized", pdf_path=str(oversized_pdf))
    _seed_paper(db, paper_id="2600.10005", title="Text Source", text_path=str(text_source))
    _seed_paper(db, paper_id="2600.10006", title="Eligible Existing PDF", pdf_path=str(existing_pdf))

    report = build_parsed_artifact_source_missing_resolution_plan(
        sqlite_db=db,
        papers_dir=papers_dir,
        candidate_limit=2,
        max_source_pdf_bytes=16,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    assert report["schema"] == PARSED_ARTIFACT_SOURCE_MISSING_RESOLUTION_PLAN_SCHEMA_ID
    assert validate_payload(report, PARSED_ARTIFACT_SOURCE_MISSING_RESOLUTION_PLAN_SCHEMA_ID, strict=True).ok
    assert report["baseline"]["scannedPapers"] == 6
    assert report["baseline"]["missingParsedArtifacts"] == 6
    assert report["sourceBlockerSnapshot"]["sourcePdfMissingRows"] == 3
    assert report["sourceBlockerSnapshot"]["heldOutSourceBlockerRows"] == 2
    assert report["candidatePool"]["selectedCandidateCount"] == 2
    assert report["candidatePool"]["unselectedSourcePdfMissingRows"] == 1
    selected = set(report["selectedCandidatePaperIds"])
    assert selected <= {"2600.10001", "2600.10002", "2600.10003"}
    assert len(selected) == 2
    held_out = report["heldOutSourceBlockerPaperIdsByStatus"]
    assert held_out["source_pdf_oversized"]["paperIds"] == ["2600.10004"]
    assert held_out["text_source_unsupported"]["paperIds"] == ["2600.10005"]
    by_resolution = report["sourcePdfMissingPaperIdsByResolutionPlanStatus"]
    assert by_resolution["registered_pdf_path_missing"]["count"] == 2
    assert by_resolution["no_registered_source_artifact"]["paperIds"] == ["2600.10002"]
    assert report["dryRunMaterializationReadiness"]["ready"] is False
    assert report["dryRunMaterializationReadiness"]["dryRunAttempted"] is False
    assert report["expectedCoverageChangeIfApplied"]["expectedMissingParsedArtifactsReduction"] == 0
    assert report["expectedCoverageChangeIfApplied"]["potentialMissingParsedArtifactsUnlockedAfterSeparateSourceRecovery"] == 2
    assert report["mutationPolicy"]["sourceDownload"] is False
    assert report["mutationPolicy"]["sourcePathRewrite"] is False
    assert report["mutationPolicy"]["sourceRegistrationMutation"] is False
    assert report["mutationCounters"]["sourceDownloadRows"] == 0
    assert report["mutationCounters"]["sourceRegistrationMutationRows"] == 0
    assert not (papers_dir / "parsed" / "2600.10001" / "document.json").exists()


def test_source_missing_resolution_plan_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.10007", title="Missing", pdf_path=str(papers_dir / "missing.pdf"))
    report = build_parsed_artifact_source_missing_resolution_plan(
        sqlite_db=db,
        papers_dir=papers_dir,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    paths = write_parsed_artifact_source_missing_resolution_plan(report, tmp_path / "reports")
    written = json.loads(Path(paths["reportJsonPath"]).read_text(encoding="utf-8"))
    ids_by_status = json.loads(
        Path(paths["sourcePdfMissingIdsByResolutionPlanStatusPath"]).read_text(encoding="utf-8")
    )

    assert validate_payload(written, PARSED_ARTIFACT_SOURCE_MISSING_RESOLUTION_PLAN_SCHEMA_ID, strict=True).ok
    assert Path(paths["reportMarkdownPath"]).exists()
    assert Path(paths["selectedCandidateIdsPath"]).read_text(encoding="utf-8").strip() == "2600.10007"
    assert ids_by_status["registered_pdf_path_missing"]["paperIds"] == ["2600.10007"]
