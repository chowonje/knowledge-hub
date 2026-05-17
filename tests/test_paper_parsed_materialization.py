from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group
from knowledge_hub.papers import parsed_materialization
from knowledge_hub.papers.parsed_materialization import (
    PARSED_MATERIALIZATION_SCHEMA_ID,
    materialize_parsed_artifacts,
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


def _write_existing_parse(papers_dir: Path, paper_id: str, marker: str = "old") -> None:
    target = papers_dir / "parsed" / paper_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "document.md").write_text(marker, encoding="utf-8")
    (target / "document.json").write_text(json.dumps({"marker": marker}), encoding="utf-8")
    (target / "manifest.json").write_text(json.dumps({"marker": marker}), encoding="utf-8")


def test_materialization_dry_run_does_not_write(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(parsed_materialization, "PyMuPDFAdapter", _FakePyMuPDFAdapter)
    _FakePyMuPDFAdapter.calls.clear()
    papers_dir = tmp_path / "papers"
    source_pdf = papers_dir / "Example.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF-1.4")
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.00001", title="Example Paper", pdf_path=str(source_pdf))

    payload = materialize_parsed_artifacts(
        sqlite_db=db,
        papers_dir=papers_dir,
        paper_ids=["2600.00001"],
    )

    assert payload["schema"] == PARSED_MATERIALIZATION_SCHEMA_ID
    assert validate_payload(payload, PARSED_MATERIALIZATION_SCHEMA_ID, strict=True).ok
    assert payload["status"] == "ok"
    assert payload["counts"]["planned"] == 1
    assert payload["items"][0]["status"] == "planned"
    assert not (papers_dir / "parsed" / "2600.00001" / "document.json").exists()
    assert _FakePyMuPDFAdapter.calls == []
    assert str(tmp_path) not in json.dumps(payload)


def test_apply_writes_only_parsed_artifacts_without_db_mutation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(parsed_materialization, "PyMuPDFAdapter", _FakePyMuPDFAdapter)
    _FakePyMuPDFAdapter.calls.clear()
    papers_dir = tmp_path / "papers"
    source_pdf = papers_dir / "Example.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF-1.4")
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.00002", title="Example Paper", pdf_path=str(source_pdf))
    row_before = db.get_paper("2600.00002")
    changes_before = db.conn.total_changes

    payload = materialize_parsed_artifacts(
        sqlite_db=db,
        papers_dir=papers_dir,
        paper_ids=["2600.00002"],
        apply=True,
    )

    target = papers_dir / "parsed" / "2600.00002"
    assert payload["status"] == "ok"
    assert payload["counts"]["materialized"] == 1
    assert (target / "document.md").exists()
    assert (target / "document.json").exists()
    assert (target / "manifest.json").exists()
    assert sorted(path.name for path in target.iterdir()) == ["document.json", "document.md", "manifest.json"]
    assert db.conn.total_changes == changes_before
    assert db.get_paper("2600.00002") == row_before
    assert _FakePyMuPDFAdapter.calls == [
        {
            "paper_id": "2600.00002",
            "pdf_path": str(source_pdf),
            "refresh": False,
            "allow_ocr": False,
        }
    ]


def test_missing_source_artifact_is_reported_as_blocked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(parsed_materialization, "PyMuPDFAdapter", _FakePyMuPDFAdapter)
    _FakePyMuPDFAdapter.calls.clear()
    papers_dir = tmp_path / "papers"
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.00003", title="Missing Source", pdf_path=str(papers_dir / "missing.pdf"))

    payload = materialize_parsed_artifacts(
        sqlite_db=db,
        papers_dir=papers_dir,
        paper_ids=["2600.00003"],
        apply=True,
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["blocked"] == 1
    assert payload["items"][0]["reason"] == "source_pdf_missing"
    assert _FakePyMuPDFAdapter.calls == []


def test_existing_parse_is_not_overwritten_without_overwrite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(parsed_materialization, "PyMuPDFAdapter", _FakePyMuPDFAdapter)
    _FakePyMuPDFAdapter.calls.clear()
    papers_dir = tmp_path / "papers"
    source_pdf = papers_dir / "Example.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF-1.4")
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.00004", title="Existing Parse", pdf_path=str(source_pdf))
    _write_existing_parse(papers_dir, "2600.00004", marker="old")

    skipped = materialize_parsed_artifacts(
        sqlite_db=db,
        papers_dir=papers_dir,
        paper_ids=["2600.00004"],
        apply=True,
    )

    assert skipped["items"][0]["status"] == "skipped_existing"
    assert json.loads((papers_dir / "parsed" / "2600.00004" / "document.json").read_text()) == {"marker": "old"}
    assert _FakePyMuPDFAdapter.calls == []

    overwritten = materialize_parsed_artifacts(
        sqlite_db=db,
        papers_dir=papers_dir,
        paper_ids=["2600.00004"],
        apply=True,
        overwrite=True,
    )

    assert overwritten["items"][0]["status"] == "materialized"
    assert _FakePyMuPDFAdapter.calls[0]["refresh"] is True


def test_materialize_parsed_cli_json_and_hidden_help(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(parsed_materialization, "PyMuPDFAdapter", _FakePyMuPDFAdapter)
    _FakePyMuPDFAdapter.calls.clear()
    papers_dir = tmp_path / "papers"
    source_pdf = papers_dir / "Example.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF-1.4")
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_paper(db, paper_id="2600.00005", title="CLI Paper", pdf_path=str(source_pdf))
    khub = _StubKhub(db, papers_dir=str(papers_dir))
    runner = CliRunner()

    result = runner.invoke(
        paper_group,
        ["materialize-parsed", "--paper-id", "2600.00005", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == PARSED_MATERIALIZATION_SCHEMA_ID
    assert payload["counts"]["planned"] == 1
    assert validate_payload(payload, PARSED_MATERIALIZATION_SCHEMA_ID, strict=True).ok
    assert str(tmp_path) not in result.output

    help_result = runner.invoke(paper_group, ["--help"], obj={"khub": khub})
    assert help_result.exit_code == 0, help_result.output
    assert "materialize-parsed" not in help_result.output
