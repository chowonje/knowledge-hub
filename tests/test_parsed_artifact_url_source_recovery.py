from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_url_source_recovery import (
    PARSED_ARTIFACT_URL_SOURCE_RECOVERY_SCHEMA_ID,
    build_parsed_artifact_url_source_recovery,
    write_parsed_artifact_url_source_recovery,
)


ACL_PAPER_ID = "https___aclanthology_org_2024__1e73f455"


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


def test_url_source_recovery_dry_run_resolves_and_probes_acl_candidate(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id=ACL_PAPER_ID, title="2024.naacl-long.20", pdf_path=str(papers_dir / "missing.pdf"))
    _seed_paper(db, paper_id="manual-paper", title="Manual")

    def fake_probe(url: str, timeout_seconds: float) -> dict[str, Any]:
        return {
            "ok": True,
            "statusCode": 200,
            "contentType": "application/pdf",
            "contentLength": 717417,
            "looksLikePdf": True,
        }

    report = build_parsed_artifact_url_source_recovery(
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=5,
        apply=False,
        probe_network=True,
        probe_fn=fake_probe,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    assert report["schema"] == PARSED_ARTIFACT_URL_SOURCE_RECOVERY_SCHEMA_ID
    assert validate_payload(report, PARSED_ARTIFACT_URL_SOURCE_RECOVERY_SCHEMA_ID, strict=True).ok
    assert report["status"] == "ready"
    assert report["selectedCandidatePaperIds"] == [ACL_PAPER_ID]
    assert report["items"][0]["sourceUrl"] == "https://aclanthology.org/2024.naacl-long.20.pdf"
    assert report["items"][0]["sourceUrlPresence"] == "confirmed_pdf"
    assert report["counts"]["planned"] == 1
    assert report["mutationPolicy"]["databaseMutation"] is False
    assert report["mutationCounters"]["sourceDownloadRows"] == 0
    assert report["mutationCounters"]["sourceRegistrationMutationRows"] == 0
    assert not (papers_dir / "recovered_sources" / "url" / f"{ACL_PAPER_ID}.pdf").exists()
    assert db.get_paper(ACL_PAPER_ID)["pdf_path"] == str(papers_dir / "missing.pdf")


def test_url_source_recovery_apply_downloads_to_safe_target_and_registers(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id=ACL_PAPER_ID, title="2024.naacl-long.20", pdf_path=str(papers_dir / "missing.pdf"))

    def fake_probe(url: str, timeout_seconds: float) -> dict[str, Any]:
        return {
            "ok": True,
            "statusCode": 200,
            "contentType": "application/pdf",
            "contentLength": 20,
            "looksLikePdf": True,
        }

    def fake_download(url: str, target_path: Path, timeout_seconds: float) -> dict[str, Any]:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"%PDF-1.4 recovered")
        return {"sizeBytes": target_path.stat().st_size, "sha256": "fake"}

    report = build_parsed_artifact_url_source_recovery(
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=1,
        apply=True,
        probe_network=True,
        probe_fn=fake_probe,
        download_fn=fake_download,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    target = papers_dir / "recovered_sources" / "url" / f"{ACL_PAPER_ID}.pdf"
    assert validate_payload(report, PARSED_ARTIFACT_URL_SOURCE_RECOVERY_SCHEMA_ID, strict=True).ok
    assert report["status"] == "applied"
    assert report["recoveredPaperIds"] == [ACL_PAPER_ID]
    assert report["mutationPolicy"]["databaseMutation"] is True
    assert report["mutationCounters"]["sourceDownloadRows"] == 1
    assert report["mutationCounters"]["sourceRegistrationMutationRows"] == 1
    assert report["items"][0]["sourceUrlProbeAttempted"] is True
    assert target.exists()
    assert db.get_paper(ACL_PAPER_ID)["pdf_path"] == str(target)
    assert report["items"][0]["targetSourceArtifact"]["path"] == f"papers_dir/recovered_sources/url/{ACL_PAPER_ID}.pdf"
    assert report["items"][0]["previousSourceArtifact"]["path"] == "papers_dir/missing.pdf"
    assert not (papers_dir / "parsed" / ACL_PAPER_ID / "document.json").exists()


def test_url_source_recovery_blocks_unsupported_url_shape(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    paper_id = "https___example_com_paper_1234"
    _seed_paper(db, paper_id=paper_id, title="No Canonical URL", pdf_path=str(papers_dir / "missing.pdf"))

    report = build_parsed_artifact_url_source_recovery(
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=1,
        apply=False,
        generated_at="2026-05-20T00:00:00+00:00",
    )

    assert validate_payload(report, PARSED_ARTIFACT_URL_SOURCE_RECOVERY_SCHEMA_ID, strict=True).ok
    assert report["status"] == "blocked"
    assert report["selectedCandidatePaperIds"] == [paper_id]
    assert report["items"][0]["reason"] == "source_url_unresolved_or_not_allowlisted"
    assert report["mutationCounters"]["databaseMutationRows"] == 0
    assert db.get_paper(paper_id)["pdf_path"] == str(papers_dir / "missing.pdf")


def test_url_source_recovery_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id=ACL_PAPER_ID, title="2024.naacl-long.20", pdf_path=str(papers_dir / "missing.pdf"))

    report = build_parsed_artifact_url_source_recovery(
        sqlite_db=db,
        papers_dir=papers_dir,
        limit=1,
        generated_at="2026-05-20T00:00:00+00:00",
    )
    paths = write_parsed_artifact_url_source_recovery(report, tmp_path / "reports")
    written = json.loads(Path(paths["reportJsonPath"]).read_text(encoding="utf-8"))

    assert validate_payload(written, PARSED_ARTIFACT_URL_SOURCE_RECOVERY_SCHEMA_ID, strict=True).ok
    assert Path(paths["reportMarkdownPath"]).exists()
    assert Path(paths["selectedCandidateIdsPath"]).read_text(encoding="utf-8").strip() == ACL_PAPER_ID
    assert Path(paths["recoveredPaperIdsPath"]).read_text(encoding="utf-8").strip() == ""
