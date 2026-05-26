from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_oversized_pdf_materialization_dry_run import (
    PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_DRY_RUN_SCHEMA_ID,
    build_parsed_artifact_oversized_pdf_materialization_dry_run,
    write_parsed_artifact_oversized_pdf_materialization_dry_run,
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


def test_oversized_pdf_materialization_dry_run_selects_policy_candidates_only(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    selected_pdf = papers_dir / "selected.pdf"
    too_large_pdf = papers_dir / "too-large.pdf"
    eligible_under_base_pdf = papers_dir / "eligible.pdf"
    text_source = papers_dir / "source.txt"
    selected_pdf.write_bytes(b"x" * 32)
    too_large_pdf.write_bytes(b"x" * 80)
    eligible_under_base_pdf.write_bytes(b"x" * 8)
    text_source.write_text("text source", encoding="utf-8")

    _seed_paper(db, paper_id="2600.10001", title="Selected", pdf_path=str(selected_pdf))
    _seed_paper(db, paper_id="2600.10002", title="Too Large", pdf_path=str(too_large_pdf))
    _seed_paper(db, paper_id="2600.10003", title="Already Eligible", pdf_path=str(eligible_under_base_pdf))
    _seed_paper(db, paper_id="2600.10004", title="Text Source", text_path=str(text_source))
    _seed_paper(db, paper_id="2600.10005", title="Already Parsed", pdf_path=str(selected_pdf))
    _write_existing_parse(papers_dir, "2600.10005")

    report = build_parsed_artifact_oversized_pdf_materialization_dry_run(
        sqlite_db=db,
        papers_dir=papers_dir,
        base_max_source_pdf_bytes=10,
        policy_max_source_pdf_bytes=50,
        generated_at="2026-05-21T00:00:00+00:00",
    )

    assert report["schema"] == PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_DRY_RUN_SCHEMA_ID
    assert validate_payload(
        report,
        PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok
    assert report["status"] == "ready"
    assert report["baseline"]["missingParsedArtifacts"] == 4
    assert report["candidatePool"]["sourcePdfOversizedRows"] == 2
    assert report["candidatePool"]["selectedDryRunRows"] == 1
    assert report["selectedCandidatePaperIds"] == ["2600.10001"]
    assert report["blockedCandidates"][0]["paperId"] == "2600.10002"
    assert report["blockedCandidates"][0]["blockedReason"] == "blocked_policy_max_source_pdf_bytes_exceeded"
    assert report["dryRunMaterializationReadiness"]["ready"] is True
    assert report["dryRunMaterializationReadiness"]["counts"]["planned"] == 1
    assert report["expectedCoverageChangeIfApplied"]["expectedMissingParsedArtifactsReduction"] == 1
    assert report["mutationCounters"]["parsedArtifactWriteRows"] == 0
    assert report["mutationPolicy"]["vaultScan"] is False
    assert not (papers_dir / "parsed" / "2600.10001" / "document.json").exists()


def test_oversized_pdf_materialization_dry_run_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    source_pdf = papers_dir / "oversized.pdf"
    source_pdf.write_bytes(b"x" * 32)
    _seed_paper(db, paper_id="2600.10006", title="Candidate", pdf_path=str(source_pdf))

    report = build_parsed_artifact_oversized_pdf_materialization_dry_run(
        sqlite_db=db,
        papers_dir=papers_dir,
        base_max_source_pdf_bytes=10,
        policy_max_source_pdf_bytes=50,
        generated_at="2026-05-21T00:00:00+00:00",
    )
    paths = write_parsed_artifact_oversized_pdf_materialization_dry_run(report, tmp_path / "reports")
    written = json.loads(Path(paths["reportJsonPath"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summaryJsonPath"]).read_text(encoding="utf-8"))

    assert validate_payload(
        written,
        PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok
    assert summary["candidatePool"]["selectedDryRunRows"] == 1
    assert Path(paths["reportMarkdownPath"]).exists()
    assert Path(paths["selectedCandidateIdsPath"]).read_text(encoding="utf-8").strip() == "2600.10006"
