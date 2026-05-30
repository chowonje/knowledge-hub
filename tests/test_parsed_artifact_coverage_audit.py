from __future__ import annotations

import hashlib
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_coverage_audit import (
    PARSED_ARTIFACT_COVERAGE_AUDIT_SCHEMA_ID,
    build_parsed_artifact_coverage_audit,
    render_parsed_artifact_coverage_audit_markdown,
    write_parsed_artifact_coverage_audit_reports,
)


def _hash_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _seed_paper(db: SQLiteDatabase, *, paper_id: str, title: str, pdf_path: str = "") -> None:
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": title,
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 5,
            "notes": "",
            "pdf_path": pdf_path,
            "text_path": "",
            "translated_path": "",
        }
    )


def _write_parsed_artifact(papers_dir: Path, *, paper_id: str) -> None:
    target = papers_dir / "parsed" / paper_id
    target.mkdir(parents=True, exist_ok=True)
    document_json = target / "document.json"
    document_md = target / "document.md"
    manifest = target / "manifest.json"
    document_md.write_text("# Parsed\n\nBody text.\n", encoding="utf-8")
    document_json.write_text(
        json.dumps(
            {
                "markdown_text": "# Parsed\n\nBody text.\n",
                "elements": [
                    {
                        "type": "heading",
                        "text": "Introduction",
                        "page": 1,
                        "heading_path": ["Introduction"],
                    },
                    {
                        "type": "paragraph",
                        "text": "Body text.",
                        "page": 1,
                        "heading_path": ["Introduction"],
                    },
                ],
                "parser_meta": {"parser": "pymupdf", "page_count": 1, "pages_with_text": 1},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "parser_meta": {"parser": "pymupdf", "page_count": 1, "pages_with_text": 1},
                "markdown_path": str(document_md),
                "json_path": str(document_json),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _artifact(*, artifact_id: str, source_id: str, filename: str, content: bytes, expected_hash: str = "") -> dict:
    return {
        "artifactId": artifact_id,
        "sourceIds": [source_id],
        "expectedFilename": filename,
        "expectedSourceContentHash": expected_hash or _hash_bytes(content),
        "byteLength": len(content),
        "corpusTier": "local_corpus",
    }


def test_parsed_artifact_coverage_audit_splits_coverage_blockers(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))

    ready_bytes = b"ready pdf"
    ready_pdf = papers_dir / "ready.pdf"
    ready_pdf.write_bytes(ready_bytes)
    _seed_paper(db, paper_id="2600.00001", title="Ready Paper", pdf_path=str(ready_pdf))
    _write_parsed_artifact(papers_dir, paper_id="2600.00001")

    missing_parsed_bytes = b"missing parsed pdf"
    missing_parsed_pdf = papers_dir / "missing-parsed.pdf"
    missing_parsed_pdf.write_bytes(missing_parsed_bytes)
    _seed_paper(db, paper_id="2600.00002", title="Missing Parsed Paper", pdf_path=str(missing_parsed_pdf))

    missing_source_bytes = b"missing source pdf"
    _seed_paper(
        db,
        paper_id="2600.00003",
        title="Missing Source Paper",
        pdf_path=str(papers_dir / "missing-source.pdf"),
    )

    mismatch_bytes = b"actual mismatch pdf"
    mismatch_pdf = papers_dir / "hash-mismatch.pdf"
    mismatch_pdf.write_bytes(mismatch_bytes)
    _seed_paper(db, paper_id="2600.00004", title="Hash Mismatch Paper", pdf_path=str(mismatch_pdf))

    unknown_bytes = b"unknown registration pdf"
    unknown_pdf = papers_dir / "unknown.pdf"
    unknown_pdf.write_bytes(unknown_bytes)

    manifest = {
        "schema": "knowledge-hub.corpus-manifest.v1",
        "artifacts": [
            _artifact(
                artifact_id="ready",
                source_id="2600.00001",
                filename=ready_pdf.name,
                content=ready_bytes,
            ),
            _artifact(
                artifact_id="missing_parsed",
                source_id="2600.00002",
                filename=missing_parsed_pdf.name,
                content=missing_parsed_bytes,
            ),
            _artifact(
                artifact_id="missing_source",
                source_id="2600.00003",
                filename="missing-source.pdf",
                content=missing_source_bytes,
            ),
            _artifact(
                artifact_id="hash_mismatch",
                source_id="2600.00004",
                filename=mismatch_pdf.name,
                content=b"expected mismatch pdf",
            ),
            _artifact(
                artifact_id="unknown_status",
                source_id="2600.00005",
                filename=unknown_pdf.name,
                content=unknown_bytes,
            ),
        ],
    }

    report = build_parsed_artifact_coverage_audit(
        sqlite_db=db,
        papers_dir=papers_dir,
        corpus_manifest=manifest,
    )

    assert report["status"] == "ready"
    assert validate_payload(report, PARSED_ARTIFACT_COVERAGE_AUDIT_SCHEMA_ID, strict=True).ok
    assert report["counts"] == {
        "totalCorpusArtifacts": 5,
        "registeredPapers": 4,
        "ready": 1,
        "missingSource": 1,
        "missingParsed": 1,
        "hashMismatch": 1,
        "unknownStatus": 1,
        "parsedDegraded": 0,
    }
    statuses = {item["artifactId"]: item["coverageStatus"] for item in report["items"]}
    assert statuses == {
        "ready": "ready",
        "missing_parsed": "missing_parsed",
        "missing_source": "missing_source",
        "hash_mismatch": "hash_mismatch",
        "unknown_status": "unknown_status",
    }
    assert report["nextParsedCoverageBottleneck"] == "source_hash_mismatch_blocks_trustworthy_parsed_materialization"
    assert report["safety"]["dbMutation"] is False
    assert str(tmp_path) not in json.dumps(report, ensure_ascii=False)


def test_parsed_artifact_coverage_audit_writes_path_safe_json_and_markdown(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    source_bytes = b"source pdf"
    source_pdf = papers_dir / "source.pdf"
    source_pdf.write_bytes(source_bytes)
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.00006", title="Source Ready Missing Parsed", pdf_path=str(source_pdf))
    manifest = {
        "schema": "knowledge-hub.corpus-manifest.v1",
        "artifacts": [
            _artifact(
                artifact_id="source_ready_missing_parsed",
                source_id="2600.00006",
                filename=source_pdf.name,
                content=source_bytes,
            )
        ],
    }

    report = build_parsed_artifact_coverage_audit(
        sqlite_db=db,
        papers_dir=papers_dir,
        corpus_manifest=manifest,
    )
    paths = write_parsed_artifact_coverage_audit_reports(report, tmp_path / "reports")
    written = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert validate_payload(written, PARSED_ARTIFACT_COVERAGE_AUDIT_SCHEMA_ID, strict=True).ok
    assert written["counts"]["missingParsed"] == 1
    assert "Missing parsed" in markdown
    assert "source_ready_missing_parsed" in markdown
    assert str(tmp_path) not in Path(paths["json"]).read_text(encoding="utf-8")
    assert str(tmp_path) not in markdown
    assert str(tmp_path) not in render_parsed_artifact_coverage_audit_markdown(report)
