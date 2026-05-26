from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group
from knowledge_hub.papers.extraction_diagnostics import (
    EXTRACTION_REPORT_SCHEMA_ID,
    build_extraction_report,
    diagnose_paper_parse,
)


class _StubConfig:
    def __init__(self, *, papers_dir: str):
        self._papers_dir = papers_dir

    @property
    def papers_dir(self) -> str:
        return self._papers_dir

    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        _ = args
        return default


class _StubKhub:
    def __init__(self, db: SQLiteDatabase, *, papers_dir: str):
        self._db = db
        self.config = _StubConfig(papers_dir=papers_dir)

    def sqlite_db(self):
        return self._db


def _seed_paper(db: SQLiteDatabase, *, paper_id: str, title: str, year: int = 2026) -> None:
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": title,
            "authors": "A. Researcher",
            "year": year,
            "field": "AI",
            "importance": 4,
            "notes": "",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )


def _write_parse_artifacts(
    papers_dir: Path,
    *,
    paper_id: str,
    parser_meta: dict[str, object],
    elements: list[dict[str, object]],
) -> None:
    target = papers_dir / "parsed" / paper_id
    target.mkdir(parents=True, exist_ok=True)
    markdown_path = target / "document.md"
    json_path = target / "document.json"
    manifest_path = target / "manifest.json"
    markdown_path.write_text("# parsed paper\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "markdown_text": "# parsed paper\n",
                "elements": elements,
                "parser_meta": parser_meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "parser_meta": parser_meta,
                "markdown_path": str(markdown_path),
                "json_path": str(json_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _pymupdf_elements() -> list[dict[str, object]]:
    return [
        {
            "type": "paragraph",
            "text": "Table 1 reports an accuracy value.",
            "page": 1,
            "heading_path": ["Page 1"],
            "reading_order": 0,
        },
        {
            "type": "paragraph",
            "text": "Figure 2 illustrates the method.",
            "page": 2,
            "heading_path": ["Page 2"],
            "reading_order": 1,
        },
    ]


def _opendataloader_elements() -> list[dict[str, object]]:
    return [
        {
            "id": "odl:1:1",
            "type": "heading",
            "text": "Results",
            "page": 1,
            "bbox": [10, 20, 200, 40],
            "heading_path": ["Results"],
            "reading_order": 1,
        },
        {
            "id": "odl:1:2",
            "type": "table_cell",
            "text": "Ours",
            "page": 1,
            "bbox": [10, 50, 60, 70],
            "heading_path": ["Results"],
            "reading_order": 2,
            "linked_content_id": "table:1",
            "row_number": 1,
            "column_number": 0,
        },
        {
            "id": "odl:1:3",
            "type": "table_cell",
            "text": "44.7",
            "page": 1,
            "bbox": [70, 50, 110, 70],
            "heading_path": ["Results"],
            "reading_order": 3,
            "linked_content_id": "table:1",
            "row_number": 1,
            "column_number": 1,
        },
        {
            "id": "odl:1:4",
            "type": "formula",
            "text": "L = - log p(y|x)",
            "page": 1,
            "bbox": [10, 80, 200, 100],
            "heading_path": ["Results"],
            "reading_order": 4,
        },
    ]


def test_pymupdf_page_blob_diagnostic_reports_degraded() -> None:
    diagnostic = diagnose_paper_parse(
        paper_id="2600.00001",
        papers_dir="/tmp/papers",
        manifest={"parser_meta": {"parser": "pymupdf", "page_count": 2, "pages_with_text": 2}},
        document={"elements": _pymupdf_elements(), "parser_meta": {"column_count_detected": 2}},
    )

    assert diagnostic["parser"] == "pymupdf"
    assert diagnostic["pageCount"] == 2
    assert diagnostic["columnCountDetected"] == 2
    assert diagnostic["extractionDegraded"] is True
    assert "page_blob_sections_only" in diagnostic["degradationReasons"]
    assert "multi_column_probe_only" in diagnostic["degradationReasons"]
    assert "tables_caption_only" in diagnostic["degradationReasons"]


def test_opendataloader_structured_elements_count_tables_and_equations() -> None:
    diagnostic = diagnose_paper_parse(
        paper_id="2600.00002",
        papers_dir="/tmp/papers",
        manifest={"parser_meta": {"parser": "opendataloader", "page_count": 1, "pages_with_text": 1}},
        document={"elements": _opendataloader_elements()},
    )

    assert diagnostic["parser"] == "opendataloader"
    assert diagnostic["tablesDetected"] == 1
    assert diagnostic["equationsDetected"] == 1
    assert diagnostic["extractionDegraded"] is False
    assert diagnostic["degradationReasons"] == []


def test_missing_artifacts_are_degraded_without_crashing(tmp_path: Path) -> None:
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.00003", title="Missing Parsed Paper")

    payload = build_extraction_report(sqlite_db=db, papers_dir=tmp_path / "papers", paper_ids=["2600.00003"])

    assert payload["status"] == "degraded"
    assert payload["counts"]["missingParsedArtifacts"] == 1
    diagnostic = payload["papers"][0]["diagnostic"]
    assert diagnostic["extractionDegraded"] is True
    assert "parsed_artifact_missing" in diagnostic["degradationReasons"]


def test_extraction_report_cli_json_and_filters(tmp_path: Path) -> None:
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    papers_dir = tmp_path / "papers"
    _seed_paper(db, paper_id="2600.00001", title="Page Blob Paper", year=2026)
    _seed_paper(db, paper_id="2600.00002", title="Structured Paper", year=2025)
    _seed_paper(db, paper_id="2600.00003", title="Missing Parsed Paper", year=2024)
    _write_parse_artifacts(
        papers_dir,
        paper_id="2600.00001",
        parser_meta={"parser": "pymupdf", "page_count": 2, "pages_with_text": 2, "column_count_detected": 2},
        elements=_pymupdf_elements(),
    )
    _write_parse_artifacts(
        papers_dir,
        paper_id="2600.00002",
        parser_meta={"parser": "opendataloader", "page_count": 1, "pages_with_text": 1},
        elements=_opendataloader_elements(),
    )
    runner = CliRunner()
    khub = _StubKhub(db, papers_dir=str(papers_dir))

    result = runner.invoke(paper_group, ["extraction-report", "--json"], obj={"khub": khub})

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == EXTRACTION_REPORT_SCHEMA_ID
    assert validate_payload(payload, EXTRACTION_REPORT_SCHEMA_ID, strict=True).ok
    assert payload["counts"]["reportedPapers"] == 3
    assert payload["counts"]["degradedPapers"] == 2
    assert str(tmp_path) not in result.output
    assert payload["papers"][0]["artifactPaths"]["artifactDir"].startswith("parsed/")

    one = runner.invoke(
        paper_group,
        ["extraction-report", "--json", "--paper-id", "2600.00002"],
        obj={"khub": khub},
    )
    assert one.exit_code == 0, one.output
    one_payload = json.loads(one.output)
    assert [item["paperId"] for item in one_payload["papers"]] == ["2600.00002"]
    assert one_payload["counts"]["degradedPapers"] == 0

    degraded = runner.invoke(
        paper_group,
        ["extraction-report", "--json", "--degraded-only"],
        obj={"khub": khub},
    )
    assert degraded.exit_code == 0, degraded.output
    degraded_payload = json.loads(degraded.output)
    assert {item["paperId"] for item in degraded_payload["papers"]} == {"2600.00001", "2600.00003"}

    limited = runner.invoke(paper_group, ["extraction-report", "--json", "--limit", "1"], obj={"khub": khub})
    assert limited.exit_code == 0, limited.output
    limited_payload = json.loads(limited.output)
    assert limited_payload["counts"]["reportedPapers"] == 1
