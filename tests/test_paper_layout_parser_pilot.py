from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group
from knowledge_hub.papers import layout_parser_pilot
from knowledge_hub.papers.layout_parser_pilot import (
    LAYOUT_PARSER_PILOT_SCHEMA_ID,
    ParserTimeoutError,
    run_layout_parser_pilot,
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


class _FakeLayoutAdapter:
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
                "papers_dir": str(self.papers_dir),
            }
        )
        target = self.papers_dir / "parsed" / paper_id
        target.mkdir(parents=True, exist_ok=True)
        markdown_text = "# Methods\n\nTable 1 reports 44.7."
        elements = [
            {
                "type": "heading",
                "text": "Methods",
                "page": 1,
                "heading_path": ["Methods"],
                "bbox": [0, 0, 10, 10],
                "reading_order": 1,
            },
            {
                "type": "table_cell",
                "text": "44.7",
                "page": 1,
                "row_number": 1,
                "column_number": 2,
                "linked_content_id": "table:1",
            },
        ]
        parser_meta = {
            "parser": "pymupdf",
            "pageCount": 1,
            "columnCountDetected": 2,
            "readingOrderMethod": "column_probe_only",
        }
        (target / "document.md").write_text(markdown_text, encoding="utf-8")
        (target / "document.json").write_text(
            json.dumps({"markdown_text": markdown_text, "elements": elements, "parser_meta": parser_meta}),
            encoding="utf-8",
        )
        (target / "manifest.json").write_text(
            json.dumps({"paper_id": paper_id, "parser_meta": parser_meta}),
            encoding="utf-8",
        )
        return SimpleNamespace(
            markdown_text=markdown_text,
            elements=elements,
            parser_meta=parser_meta,
            artifact_dir=target,
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


def _available(parser: str) -> dict[str, object]:
    return {"available": parser == "pymupdf", "reason": "" if parser == "pymupdf" else "missing_package", "version": "test", "command": ""}


def test_layout_parser_pilot_plan_does_not_write(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(layout_parser_pilot, "PyMuPDFAdapter", _FakeLayoutAdapter)
    monkeypatch.setattr(layout_parser_pilot, "_parser_availability", _available)
    _FakeLayoutAdapter.calls.clear()
    papers_dir = tmp_path / "papers"
    source_pdf = papers_dir / "Example.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF-1.4")
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.01001", title="Example Paper", pdf_path=str(source_pdf))

    payload = run_layout_parser_pilot(
        sqlite_db=db,
        papers_dir=papers_dir,
        paper_ids=["2600.01001"],
        parsers=["pymupdf"],
        output_dir=tmp_path / "pilot",
    )

    assert payload["schema"] == LAYOUT_PARSER_PILOT_SCHEMA_ID
    assert validate_payload(payload, LAYOUT_PARSER_PILOT_SCHEMA_ID, strict=True).ok
    assert payload["counts"] == {"planned": 1, "ok": 0, "blocked": 0, "failed": 0, "timeout": 0}
    assert payload["papers"][0]["parsers"][0]["status"] == "planned"
    assert payload["request"]["timeoutSeconds"] == 0
    assert not (tmp_path / "pilot").exists()
    assert not (papers_dir / "parsed" / "2600.01001").exists()
    assert _FakeLayoutAdapter.calls == []


def test_layout_parser_pilot_run_writes_only_isolated_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(layout_parser_pilot, "PyMuPDFAdapter", _FakeLayoutAdapter)
    monkeypatch.setattr(layout_parser_pilot, "_parser_availability", _available)
    _FakeLayoutAdapter.calls.clear()
    papers_dir = tmp_path / "papers"
    source_pdf = papers_dir / "Example.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF-1.4")
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.01002", title="Example Paper", pdf_path=str(source_pdf))
    row_before = db.get_paper("2600.01002")
    changes_before = db.conn.total_changes
    output_dir = tmp_path / "pilot"

    payload = run_layout_parser_pilot(
        sqlite_db=db,
        papers_dir=papers_dir,
        paper_ids=["2600.01002"],
        parsers=["pymupdf"],
        output_dir=output_dir,
        run=True,
    )

    isolated_target = output_dir / "pymupdf" / "parsed" / "2600.01002"
    assert payload["status"] == "ok"
    assert payload["counts"]["ok"] == 1
    assert (isolated_target / "document.json").exists()
    assert (isolated_target / "manifest.json").exists()
    assert not (papers_dir / "parsed" / "2600.01002").exists()
    assert db.conn.total_changes == changes_before
    assert db.get_paper("2600.01002") == row_before
    assert _FakeLayoutAdapter.calls[0]["refresh"] is True
    assert _FakeLayoutAdapter.calls[0]["allow_ocr"] is False
    metrics = payload["papers"][0]["parsers"][0]["metrics"]
    assert metrics["realHeadingCount"] >= 1
    assert metrics["tableCellElementCount"] == 1
    assert metrics["columnCountDetected"] == 2
    assert validate_payload(payload, LAYOUT_PARSER_PILOT_SCHEMA_ID, strict=True).ok


def test_layout_parser_pilot_timeout_is_reported_without_green_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(layout_parser_pilot, "_parser_availability", _available)

    def _timeout(**kwargs):  # noqa: ANN002, ANN003
        _ = kwargs
        raise ParserTimeoutError("parser timed out after 1s")

    monkeypatch.setattr(layout_parser_pilot, "_run_parser_with_timeout", _timeout)
    papers_dir = tmp_path / "papers"
    source_pdf = papers_dir / "Example.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF-1.4")
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.01005", title="Timeout Paper", pdf_path=str(source_pdf))

    payload = run_layout_parser_pilot(
        sqlite_db=db,
        papers_dir=papers_dir,
        paper_ids=["2600.01005"],
        parsers=["pymupdf"],
        output_dir=tmp_path / "pilot",
        run=True,
        timeout_seconds=1,
    )

    item = payload["papers"][0]["parsers"][0]
    assert payload["status"] == "failed"
    assert payload["counts"]["timeout"] == 1
    assert item["status"] == "timeout"
    assert item["timeoutSeconds"] == 1
    assert item["durationSeconds"] >= 0
    assert item["artifactDir"] == ""
    assert item["reason"].startswith("parser_timeout:")
    assert not (papers_dir / "parsed" / "2600.01005").exists()
    assert validate_payload(payload, LAYOUT_PARSER_PILOT_SCHEMA_ID, strict=True).ok


def test_layout_parser_pilot_reports_missing_source_and_unavailable_parser(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(layout_parser_pilot, "_parser_availability", _available)
    papers_dir = tmp_path / "papers"
    source_pdf = papers_dir / "Example.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF-1.4")
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.01003", title="Missing Source", pdf_path=str(papers_dir / "missing.pdf"))
    _seed_paper(db, paper_id="2600.01004", title="Unavailable Parser", pdf_path=str(source_pdf))

    missing = run_layout_parser_pilot(
        sqlite_db=db,
        papers_dir=papers_dir,
        paper_ids=["2600.01003"],
        parsers=["pymupdf"],
        output_dir=tmp_path / "pilot",
        run=True,
    )
    unavailable = run_layout_parser_pilot(
        sqlite_db=db,
        papers_dir=papers_dir,
        paper_ids=["2600.01004"],
        parsers=["opendataloader"],
        output_dir=tmp_path / "pilot",
    )

    assert missing["status"] == "blocked"
    assert missing["papers"][0]["parsers"][0]["reason"] == "source_pdf_missing"
    assert unavailable["papers"][0]["parsers"][0]["reason"] == "missing_package"
    assert validate_payload(missing, LAYOUT_PARSER_PILOT_SCHEMA_ID, strict=True).ok
    assert validate_payload(unavailable, LAYOUT_PARSER_PILOT_SCHEMA_ID, strict=True).ok


def test_layout_parser_pilot_cli_json_and_hidden_help(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(layout_parser_pilot, "PyMuPDFAdapter", _FakeLayoutAdapter)
    monkeypatch.setattr(layout_parser_pilot, "_parser_availability", _available)
    _FakeLayoutAdapter.calls.clear()
    papers_dir = tmp_path / "papers"
    source_pdf = papers_dir / "Example.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF-1.4")
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.01004", title="CLI Paper", pdf_path=str(source_pdf))
    khub = _StubKhub(db, papers_dir=str(papers_dir))
    runner = CliRunner()

    result = runner.invoke(
        paper_group,
        [
            "layout-parser-pilot",
            "--paper-id",
            "2600.01004",
            "--parser",
            "pymupdf",
            "--output-dir",
            str(tmp_path / "pilot"),
            "--timeout-seconds",
            "7",
            "--json",
        ],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == LAYOUT_PARSER_PILOT_SCHEMA_ID
    assert payload["counts"]["planned"] == 1
    assert payload["request"]["timeoutSeconds"] == 7
    assert payload["papers"][0]["parsers"][0]["timeoutSeconds"] == 7
    assert validate_payload(payload, LAYOUT_PARSER_PILOT_SCHEMA_ID, strict=True).ok

    help_result = runner.invoke(paper_group, ["--help"], obj={"khub": khub})
    assert help_result.exit_code == 0, help_result.output
    assert "layout-parser-pilot" not in help_result.output
