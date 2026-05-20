from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_source_recovery_feasibility import (
    PARSED_ARTIFACT_SOURCE_RECOVERY_FEASIBILITY_SCHEMA_ID,
    build_parsed_artifact_source_recovery_feasibility,
    write_parsed_artifact_source_recovery_feasibility,
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


def test_source_recovery_feasibility_classifies_local_repair_reacquisition_and_policy_rows(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    archive_dir = papers_dir / "archive"
    archive_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    relocated_pdf = archive_dir / "missing-registered.pdf"
    oversized_pdf = papers_dir / "oversized.pdf"
    text_source = papers_dir / "source.txt"
    existing_pdf = papers_dir / "2505.10005.pdf"
    relocated_pdf.write_bytes(b"%PDF-1.4 relocated")
    oversized_pdf.write_bytes(b"x" * 32)
    text_source.write_text("text source", encoding="utf-8")
    existing_pdf.write_bytes(b"%PDF-1.4 existing")

    _seed_paper(
        db,
        paper_id="2501.10001",
        title="Path Repair",
        pdf_path=str(papers_dir / "missing-registered.pdf"),
    )
    _seed_paper(
        db,
        paper_id="2502.10002",
        title="Arxiv Reacquisition",
        pdf_path=str(papers_dir / "missing-arxiv.pdf"),
    )
    _seed_paper(db, paper_id="custom-paper", title="Manual Lookup")
    _seed_paper(db, paper_id="2503.10003", title="Oversized", pdf_path=str(oversized_pdf))
    _seed_paper(db, paper_id="2504.10004", title="Text Source", text_path=str(text_source))
    _seed_paper(db, paper_id="2505.10005", title="Eligible Existing PDF", pdf_path=str(existing_pdf))

    report = build_parsed_artifact_source_recovery_feasibility(
        sqlite_db=db,
        papers_dir=papers_dir,
        max_source_pdf_bytes=24,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    assert report["schema"] == PARSED_ARTIFACT_SOURCE_RECOVERY_FEASIBILITY_SCHEMA_ID
    assert validate_payload(report, PARSED_ARTIFACT_SOURCE_RECOVERY_FEASIBILITY_SCHEMA_ID, strict=True).ok
    assert report["baseline"]["scannedPapers"] == 6
    assert report["baseline"]["missingParsedArtifacts"] == 6
    assert report["sourceMissingRecoverySummary"]["sourcePdfMissingRows"] == 3
    assert report["localScanSummary"]["pathRepairCandidateRows"] == 1
    assert report["sourceMissingRecoverySummary"]["identifierReacquisitionCandidateRows"] == 1
    assert report["sourceMissingRecoverySummary"]["manualLookupRequiredRows"] == 1
    assert report["oversizedPolicySummary"]["sourcePdfOversizedRows"] == 1
    assert report["textSourcePolicySummary"]["textSourceUnsupportedRows"] == 1
    assert report["pathRepairCandidatePaperIds"] == ["2501.10001"]
    assert report["reacquisitionCandidatePaperIdsByStatus"]["arxiv_pdf_reacquisition_candidate"]["paperIds"] == [
        "2502.10002"
    ]
    assert report["reacquisitionCandidatePaperIdsByStatus"]["manual_lookup_required"]["paperIds"] == [
        "custom-paper"
    ]
    assert report["localPresencePaperIdsByStatus"]["local_pdf_same_basename_found_elsewhere"]["paperIds"] == [
        "2501.10001"
    ]
    assert report["localPresencePaperIdsByStatus"][
        "no_registered_source_artifact_no_local_pdf_candidate_found"
    ]["paperIds"] == ["custom-paper"]
    assert report["oversizedPolicyRows"][0]["applyReadyNow"] is False
    assert report["oversizedPolicyRows"][0]["parserRoutingChangeAllowed"] is False
    assert report["textSourcePolicyRows"][0]["decisionRecommendation"] == "do_not_count_as_current_pdf_backed_parsed_artifact"
    assert report["expectedCoverageChangeIfApplied"]["expectedImmediateMissingParsedArtifactsReduction"] == 0
    assert report["mutationPolicy"]["sourceDownload"] is False
    assert report["mutationCounters"]["sourceDownloadRows"] == 0
    assert report["mutationCounters"]["sourceRegistrationMutationRows"] == 0
    assert not (papers_dir / "parsed" / "2501.10001" / "document.json").exists()


def test_source_recovery_feasibility_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2506.10006", title="Missing", pdf_path=str(papers_dir / "missing.pdf"))
    report = build_parsed_artifact_source_recovery_feasibility(
        sqlite_db=db,
        papers_dir=papers_dir,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    paths = write_parsed_artifact_source_recovery_feasibility(report, tmp_path / "reports")
    written = json.loads(Path(paths["reportJsonPath"]).read_text(encoding="utf-8"))
    reacquisition_ids = json.loads(Path(paths["reacquisitionCandidateIdsByStatusPath"]).read_text(encoding="utf-8"))

    assert validate_payload(written, PARSED_ARTIFACT_SOURCE_RECOVERY_FEASIBILITY_SCHEMA_ID, strict=True).ok
    assert Path(paths["reportMarkdownPath"]).exists()
    assert reacquisition_ids["arxiv_pdf_reacquisition_candidate"]["paperIds"] == ["2506.10006"]
