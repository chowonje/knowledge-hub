from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_coverage_batch_report import (
    PARSED_ARTIFACT_COVERAGE_BATCH_REPORT_SCHEMA_ID,
    build_parsed_artifact_coverage_batch_report,
    write_parsed_artifact_coverage_batch_report,
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


def _write_existing_parse(papers_dir: Path, paper_id: str) -> None:
    target = papers_dir / "parsed" / paper_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "document.md").write_text("# parsed\n", encoding="utf-8")
    (target / "document.json").write_text(json.dumps({"elements": []}), encoding="utf-8")
    (target / "manifest.json").write_text(json.dumps({"paper_id": paper_id}), encoding="utf-8")


def test_coverage_batch_report_selects_existing_pdf_candidates_only(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    first_pdf = papers_dir / "first.pdf"
    second_pdf = papers_dir / "second.pdf"
    text_source = papers_dir / "source.txt"
    first_pdf.write_bytes(b"%PDF-1.4 first")
    second_pdf.write_bytes(b"%PDF-1.4 second")
    text_source.write_text("text", encoding="utf-8")
    _seed_paper(db, paper_id="2600.00001", title="First Missing", pdf_path=str(first_pdf))
    _seed_paper(db, paper_id="2600.00002", title="Second Missing", pdf_path=str(second_pdf))
    _seed_paper(db, paper_id="2600.00003", title="Text Only", text_path=str(text_source))
    _seed_paper(db, paper_id="2600.00004", title="Missing Source", pdf_path=str(papers_dir / "missing.pdf"))
    _seed_paper(db, paper_id="2600.00005", title="Already Parsed", pdf_path=str(first_pdf))
    _write_existing_parse(papers_dir, "2600.00005")

    report = build_parsed_artifact_coverage_batch_report(
        sqlite_db=db,
        papers_dir=papers_dir,
        batch_limit=1,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    assert report["schema"] == PARSED_ARTIFACT_COVERAGE_BATCH_REPORT_SCHEMA_ID
    assert validate_payload(report, PARSED_ARTIFACT_COVERAGE_BATCH_REPORT_SCHEMA_ID, strict=True).ok
    assert report["baseline"]["scannedPapers"] == 5
    assert report["baseline"]["missingParsedArtifacts"] == 4
    assert report["candidatePool"]["eligibleExistingPdf"] == 2
    assert {item["reason"]: item["count"] for item in report["candidatePool"]["sourceArtifactTaxonomy"]}[
        "eligible_existing_pdf"
    ] == 2
    assert report["selectedCandidatePaperIds"] == ["2600.00001"]
    assert report["unselectedEligibleCandidatePaperIds"] == ["2600.00002"]
    assert report["nextRecommendedCoverageTranche"]["paperIds"] == ["2600.00002"]
    assert report["dryRunMaterializationReadiness"]["ready"] is True
    assert report["expectedCoverageChangeIfApplied"]["expectedMissingParsedArtifactsReduction"] == 1
    assert report["mutationCounters"]["databaseMutationRows"] == 0
    assert report["mutationPolicy"]["vaultScan"] is False
    assert not (papers_dir / "parsed" / "2600.00001" / "document.json").exists()


def test_coverage_batch_report_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    source_pdf = papers_dir / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.4")
    _seed_paper(db, paper_id="2600.00006", title="Candidate", pdf_path=str(source_pdf))
    report = build_parsed_artifact_coverage_batch_report(
        sqlite_db=db,
        papers_dir=papers_dir,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    paths = write_parsed_artifact_coverage_batch_report(report, tmp_path / "reports")
    written = json.loads(Path(paths["reportJsonPath"]).read_text(encoding="utf-8"))

    assert validate_payload(written, PARSED_ARTIFACT_COVERAGE_BATCH_REPORT_SCHEMA_ID, strict=True).ok
    assert Path(paths["reportMarkdownPath"]).exists()
    assert Path(paths["selectedCandidateIdsPath"]).read_text(encoding="utf-8").strip() == "2600.00006"
