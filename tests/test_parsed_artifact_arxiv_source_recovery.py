from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_arxiv_source_recovery import (
    PARSED_ARTIFACT_ARXIV_SOURCE_RECOVERY_SCHEMA_ID,
    build_parsed_artifact_arxiv_source_recovery,
    write_parsed_artifact_arxiv_source_recovery,
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


def test_arxiv_source_recovery_dry_run_selects_sorted_limited_candidates(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    existing_pdf = papers_dir / "existing.pdf"
    existing_pdf.write_bytes(b"%PDF-1.4 existing")

    _seed_paper(db, paper_id="2503.10003", title="Third", pdf_path=str(papers_dir / "missing-third.pdf"))
    _seed_paper(db, paper_id="2501.10001", title="First", pdf_path=str(papers_dir / "missing-first.pdf"))
    _seed_paper(db, paper_id="2502.10002", title="Second", pdf_path=str(papers_dir / "missing-second.pdf"))
    _seed_paper(db, paper_id="custom-paper", title="Manual Lookup")
    _seed_paper(db, paper_id="2504.10004", title="Already Parsed", pdf_path=str(existing_pdf))
    _write_existing_parse(papers_dir, "2504.10004")

    report = build_parsed_artifact_arxiv_source_recovery(
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=2,
        apply=False,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    assert report["schema"] == PARSED_ARTIFACT_ARXIV_SOURCE_RECOVERY_SCHEMA_ID
    assert validate_payload(report, PARSED_ARTIFACT_ARXIV_SOURCE_RECOVERY_SCHEMA_ID, strict=True).ok
    assert report["status"] == "ready"
    assert report["selectedCandidatePaperIds"] == ["2501.10001", "2502.10002"]
    assert report["unselectedCandidatePaperIds"] == ["2503.10003"]
    assert report["recoveredPaperIds"] == []
    assert report["counts"]["planned"] == 2
    assert report["mutationPolicy"]["databaseMutation"] is False
    assert report["mutationCounters"]["sourceDownloadRows"] == 0
    assert report["mutationCounters"]["sourceRegistrationMutationRows"] == 0
    assert not (papers_dir / "recovered_sources" / "arxiv" / "2501.10001.pdf").exists()
    assert db.get_paper("2501.10001")["pdf_path"] == str(papers_dir / "missing-first.pdf")


def test_arxiv_source_recovery_apply_downloads_to_safe_target_and_registers(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2501.10001", title="Recover Me", pdf_path=str(papers_dir / "missing.pdf"))

    def fake_download(url: str, target_path: Path, timeout_seconds: float) -> dict[str, Any]:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"%PDF-1.4 recovered")
        return {"sizeBytes": target_path.stat().st_size, "sha256": "fake"}

    report = build_parsed_artifact_arxiv_source_recovery(
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=1,
        apply=True,
        download_fn=fake_download,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    target = papers_dir / "recovered_sources" / "arxiv" / "2501.10001.pdf"
    assert validate_payload(report, PARSED_ARTIFACT_ARXIV_SOURCE_RECOVERY_SCHEMA_ID, strict=True).ok
    assert report["status"] == "applied"
    assert report["recoveredPaperIds"] == ["2501.10001"]
    assert report["mutationPolicy"]["databaseMutation"] is True
    assert report["mutationCounters"]["sourceDownloadRows"] == 1
    assert report["mutationCounters"]["sourceRegistrationMutationRows"] == 1
    assert target.exists()
    assert db.get_paper("2501.10001")["pdf_path"] == str(target)
    assert report["items"][0]["targetSourceArtifact"]["path"] == "papers_dir/recovered_sources/arxiv/2501.10001.pdf"
    assert report["items"][0]["previousSourceArtifact"]["path"] == "papers_dir/missing.pdf"
    assert not (papers_dir / "parsed" / "2501.10001" / "document.json").exists()


def test_arxiv_source_recovery_invalid_download_does_not_register_source(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    missing_path = papers_dir / "missing.pdf"
    _seed_paper(db, paper_id="2501.10001", title="Invalid Download", pdf_path=str(missing_path))

    def fake_bad_download(url: str, target_path: Path, timeout_seconds: float) -> dict[str, Any]:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"not a pdf")
        return {"sizeBytes": target_path.stat().st_size, "sha256": "bad"}

    report = build_parsed_artifact_arxiv_source_recovery(
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=1,
        apply=True,
        download_fn=fake_bad_download,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    assert validate_payload(report, PARSED_ARTIFACT_ARXIV_SOURCE_RECOVERY_SCHEMA_ID, strict=True).ok
    assert report["status"] == "failed"
    assert report["counts"]["failed"] == 1
    assert report["mutationCounters"]["databaseMutationRows"] == 0
    assert report["mutationCounters"]["sourceRegistrationMutationRows"] == 0
    assert db.get_paper("2501.10001")["pdf_path"] == str(missing_path)


def test_arxiv_source_recovery_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2501.10001", title="Writer", pdf_path=str(papers_dir / "missing.pdf"))
    report = build_parsed_artifact_arxiv_source_recovery(
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=1,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    paths = write_parsed_artifact_arxiv_source_recovery(report, tmp_path / "reports")
    written = json.loads(Path(paths["reportJsonPath"]).read_text(encoding="utf-8"))

    assert validate_payload(written, PARSED_ARTIFACT_ARXIV_SOURCE_RECOVERY_SCHEMA_ID, strict=True).ok
    assert Path(paths["reportMarkdownPath"]).exists()
    assert Path(paths["selectedCandidateIdsPath"]).read_text(encoding="utf-8").strip() == "2501.10001"
    assert Path(paths["recoveredPaperIdsPath"]).read_text(encoding="utf-8").strip() == ""
