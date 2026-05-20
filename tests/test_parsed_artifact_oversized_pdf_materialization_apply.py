from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers import parsed_materialization
from knowledge_hub.papers.parsed_artifact_oversized_pdf_materialization_apply import (
    PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_APPLY_SCHEMA_ID,
    build_parsed_artifact_oversized_pdf_materialization_apply,
    write_parsed_artifact_oversized_pdf_materialization_apply,
)
from knowledge_hub.papers.parsed_artifact_oversized_pdf_materialization_dry_run import (
    build_parsed_artifact_oversized_pdf_materialization_dry_run,
)


class _FakePyMuPDFAdapter:
    calls: list[dict[str, object]] = []

    def __init__(self, *, papers_dir: str):
        self.papers_dir = Path(papers_dir)

    def ensure_artifacts(self, *, paper_id: str, pdf_path: str, refresh: bool = False, allow_ocr: bool = True):
        self.calls.append(
            {
                "paper_id": paper_id,
                "pdf_path": pdf_path,
                "refresh": refresh,
                "allow_ocr": allow_ocr,
            }
        )
        target = self.papers_dir / "parsed" / paper_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "document.md").write_text(f"# {paper_id}\n\nParsed text.", encoding="utf-8")
        (target / "document.json").write_text(
            json.dumps(
                {
                    "markdown_text": f"# {paper_id}\n\nParsed text.",
                    "elements": [{"type": "paragraph", "text": "Parsed text.", "page": 1}],
                    "parser_meta": {"parser": "pymupdf"},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (target / "manifest.json").write_text(
            json.dumps(
                {
                    "paper_id": paper_id,
                    "parser_meta": {"parser": "pymupdf"},
                    "markdown_path": str(target / "document.md"),
                    "json_path": str(target / "document.json"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _seed_paper(db: SQLiteDatabase, *, paper_id: str, title: str, pdf_path: str = "") -> None:
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
            "text_path": "",
            "translated_path": "",
        }
    )


def _dry_run_report(db: SQLiteDatabase, papers_dir: Path) -> dict:
    return build_parsed_artifact_oversized_pdf_materialization_dry_run(
        sqlite_db=db,
        papers_dir=papers_dir,
        base_max_source_pdf_bytes=10,
        policy_max_source_pdf_bytes=50,
        generated_at="2026-05-21T00:00:00+00:00",
    )


def test_oversized_apply_default_plan_does_not_write(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(parsed_materialization, "PyMuPDFAdapter", _FakePyMuPDFAdapter)
    _FakePyMuPDFAdapter.calls.clear()
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    source_pdf = papers_dir / "oversized.pdf"
    source_pdf.write_bytes(b"x" * 32)
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.20001", title="Oversized", pdf_path=str(source_pdf))
    dry_run = _dry_run_report(db, papers_dir)
    row_before = db.get_paper("2600.20001")

    report = build_parsed_artifact_oversized_pdf_materialization_apply(
        sqlite_db=db,
        papers_dir=papers_dir,
        dry_run_report=dry_run,
        apply=False,
        generated_at="2026-05-21T00:00:00+00:00",
    )

    assert report["schema"] == PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_APPLY_SCHEMA_ID
    assert validate_payload(report, PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_APPLY_SCHEMA_ID, strict=True).ok
    assert report["status"] == "ready"
    assert report["counts"]["planned"] == 1
    assert report["mutationCounters"]["parsedArtifactWriteRows"] == 0
    assert report["baselineBefore"]["missingParsedArtifacts"] == 1
    assert report["baselineAfter"]["missingParsedArtifacts"] == 1
    assert db.get_paper("2600.20001") == row_before
    assert not (papers_dir / "parsed" / "2600.20001" / "document.json").exists()
    assert _FakePyMuPDFAdapter.calls == []


def test_oversized_apply_materializes_only_parsed_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(parsed_materialization, "PyMuPDFAdapter", _FakePyMuPDFAdapter)
    _FakePyMuPDFAdapter.calls.clear()
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    source_pdf = papers_dir / "oversized.pdf"
    source_pdf.write_bytes(b"x" * 32)
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.20002", title="Oversized", pdf_path=str(source_pdf))
    dry_run = _dry_run_report(db, papers_dir)
    row_before = db.get_paper("2600.20002")
    changes_before = db.conn.total_changes

    report = build_parsed_artifact_oversized_pdf_materialization_apply(
        sqlite_db=db,
        papers_dir=papers_dir,
        dry_run_report=dry_run,
        apply=True,
        generated_at="2026-05-21T00:00:00+00:00",
    )

    target = papers_dir / "parsed" / "2600.20002"
    assert validate_payload(report, PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_APPLY_SCHEMA_ID, strict=True).ok
    assert report["status"] == "applied"
    assert report["counts"]["materialized"] == 1
    assert report["materializedPaperIds"] == ["2600.20002"]
    assert report["mutationCounters"]["parsedArtifactWriteRows"] == 1
    assert report["mutationCounters"]["databaseMutationRows"] == 0
    assert report["observedCoverageChange"]["actualMissingParsedArtifactsReduction"] == 1
    assert report["baselineBefore"]["missingParsedArtifacts"] == 1
    assert report["baselineAfter"]["missingParsedArtifacts"] == 0
    assert (target / "document.md").exists()
    assert (target / "document.json").exists()
    assert (target / "manifest.json").exists()
    assert db.conn.total_changes == changes_before
    assert db.get_paper("2600.20002") == row_before
    assert _FakePyMuPDFAdapter.calls == [
        {
            "paper_id": "2600.20002",
            "pdf_path": str(source_pdf),
            "refresh": False,
            "allow_ocr": False,
        }
    ]


def test_oversized_apply_blocks_unready_dry_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(parsed_materialization, "PyMuPDFAdapter", _FakePyMuPDFAdapter)
    _FakePyMuPDFAdapter.calls.clear()
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    dry_run = {
        "schema": "knowledge-hub.paper.parsed-artifact-oversized-pdf-materialization-dry-run.v1",
        "status": "blocked",
        "generatedAt": "2026-05-21T00:00:00+00:00",
        "report": {},
        "baseline": {"scannedPapers": 0, "reportedPapers": 0, "degradedPapers": 0, "missingParsedArtifacts": 0},
        "inputSourceRecoveryFeasibility": {
            "status": "",
            "sourceBlockerSnapshot": {},
            "oversizedPolicySummary": {},
            "sourceMissingRecoverySummary": {},
            "textSourcePolicySummary": {},
        },
        "candidatePool": {
            "inputRows": 0,
            "sourcePdfOversizedRows": 0,
            "oversizedSmallPolicyCandidateRows": 0,
            "selectedDryRunRows": 0,
            "blockedRows": 0,
            "blockerTaxonomy": [],
        },
        "selectedCandidatePaperIds": [],
        "selectedCandidates": [],
        "blockedCandidates": [],
        "dryRunMaterializationReadiness": {
            "ready": False,
            "status": "blocked",
            "counts": {},
            "allSelectedPlanned": False,
            "schemaValidation": {"ok": True, "errors": []},
            "applySkippedReason": "blocked",
        },
        "expectedCoverageChangeIfApplied": {
            "missingParsedArtifactsBefore": 0,
            "expectedMissingParsedArtifactsReduction": 0,
            "expectedMissingParsedArtifactsAfter": 0,
        },
        "nextRecommendedTranche": {"name": "", "candidateCount": 0, "paperIds": [], "rationale": ""},
        "mutationPolicy": {},
        "mutationCounters": {},
        "dryRunMaterialization": {},
        "warnings": [],
    }

    report = build_parsed_artifact_oversized_pdf_materialization_apply(
        sqlite_db=db,
        papers_dir=papers_dir,
        dry_run_report=dry_run,
        apply=True,
        generated_at="2026-05-21T00:00:00+00:00",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["materialized"] == 0
    assert report["warnings"] == ["input_dry_run_report_not_ready"]
    assert _FakePyMuPDFAdapter.calls == []


def test_oversized_apply_writer_outputs_schema_valid_reports(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(parsed_materialization, "PyMuPDFAdapter", _FakePyMuPDFAdapter)
    _FakePyMuPDFAdapter.calls.clear()
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    source_pdf = papers_dir / "oversized.pdf"
    source_pdf.write_bytes(b"x" * 32)
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.20003", title="Writer", pdf_path=str(source_pdf))
    report = build_parsed_artifact_oversized_pdf_materialization_apply(
        sqlite_db=db,
        papers_dir=papers_dir,
        dry_run_report=_dry_run_report(db, papers_dir),
        apply=True,
        generated_at="2026-05-21T00:00:00+00:00",
    )

    paths = write_parsed_artifact_oversized_pdf_materialization_apply(report, tmp_path / "reports")
    written = json.loads(Path(paths["reportJsonPath"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(paths["summaryJsonPath"]).read_text(encoding="utf-8"))

    assert validate_payload(written, PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_APPLY_SCHEMA_ID, strict=True).ok
    assert "schema" not in summary
    assert summary["reportSchema"] == PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_APPLY_SCHEMA_ID
    assert summary["counts"]["materialized"] == 1
    assert Path(paths["reportMarkdownPath"]).exists()
    assert Path(paths["materializedPaperIdsPath"]).read_text(encoding="utf-8").strip() == "2600.20003"
