from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_source_blocker_report import (
    PARSED_ARTIFACT_SOURCE_BLOCKER_REPORT_SCHEMA_ID,
    build_parsed_artifact_source_blocker_report,
    write_parsed_artifact_source_blocker_report,
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


def test_source_blocker_report_classifies_remaining_source_blockers_only(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    small_pdf = papers_dir / "small.pdf"
    oversized_pdf = papers_dir / "oversized.pdf"
    text_source = papers_dir / "source.txt"
    small_pdf.write_bytes(b"%PDF-1.4 small")
    oversized_pdf.write_bytes(b"x" * 32)
    text_source.write_text("text", encoding="utf-8")
    _seed_paper(db, paper_id="2600.00001", title="Eligible", pdf_path=str(small_pdf))
    _seed_paper(db, paper_id="2600.00002", title="Missing Source", pdf_path=str(papers_dir / "missing.pdf"))
    _seed_paper(db, paper_id="2600.00003", title="Oversized", pdf_path=str(oversized_pdf))
    _seed_paper(db, paper_id="2600.00004", title="Text Only", text_path=str(text_source))
    _seed_paper(db, paper_id="2600.00005", title="Already Parsed", pdf_path=str(small_pdf))
    _write_existing_parse(papers_dir, "2600.00005")

    report = build_parsed_artifact_source_blocker_report(
        sqlite_db=db,
        papers_dir=papers_dir,
        max_source_pdf_bytes=16,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    assert report["schema"] == PARSED_ARTIFACT_SOURCE_BLOCKER_REPORT_SCHEMA_ID
    assert validate_payload(report, PARSED_ARTIFACT_SOURCE_BLOCKER_REPORT_SCHEMA_ID, strict=True).ok
    assert report["baseline"]["scannedPapers"] == 5
    assert report["baseline"]["missingParsedArtifacts"] == 4
    assert report["sourceBlockerPool"]["sourceBlockerRows"] == 3
    assert report["sourceBlockerPool"]["eligibleExistingPdfRows"] == 1
    taxonomy = {item["reason"]: item["count"] for item in report["sourceBlockerPool"]["blockerTaxonomy"]}
    assert taxonomy == {
        "source_pdf_missing": 1,
        "source_pdf_oversized": 1,
        "text_source_unsupported": 1,
    }
    assert report["sourceBlockerPaperIdsByStatus"]["source_pdf_missing"]["paperIds"] == ["2600.00002"]
    assert report["sourceBlockerPaperIdsByStatus"]["source_pdf_oversized"]["paperIds"] == ["2600.00003"]
    assert report["sourceBlockerPaperIdsByStatus"]["text_source_unsupported"]["paperIds"] == ["2600.00004"]
    assert report["mutationPolicy"]["sourceDownload"] is False
    assert report["mutationPolicy"]["manualBlockerResolution"] is False
    assert report["mutationCounters"]["sourceDownloadRows"] == 0
    assert report["mutationCounters"]["parsedArtifactWriteRows"] == 0
    assert not (papers_dir / "parsed" / "2600.00001" / "document.json").exists()


def test_source_blocker_report_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.00006", title="Missing", pdf_path=str(papers_dir / "missing.pdf"))
    report = build_parsed_artifact_source_blocker_report(
        sqlite_db=db,
        papers_dir=papers_dir,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    paths = write_parsed_artifact_source_blocker_report(report, tmp_path / "reports")
    written = json.loads(Path(paths["reportJsonPath"]).read_text(encoding="utf-8"))
    ids_by_status = json.loads(Path(paths["sourceBlockerIdsByStatusPath"]).read_text(encoding="utf-8"))

    assert validate_payload(written, PARSED_ARTIFACT_SOURCE_BLOCKER_REPORT_SCHEMA_ID, strict=True).ok
    assert Path(paths["reportMarkdownPath"]).exists()
    assert ids_by_status["source_pdf_missing"]["paperIds"] == ["2600.00006"]
