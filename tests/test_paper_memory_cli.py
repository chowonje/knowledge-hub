from __future__ import annotations

import json

from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.paper_memory_cmd import paper_memory_group


class _StubConfig:
    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        _ = args
        return default


class _StubKhub:
    def __init__(self, db):
        self._db = db
        self.config = _StubConfig()

    def sqlite_db(self):
        return self._db


def _seed(db: SQLiteDatabase):
    db.upsert_paper(
        {
            "arxiv_id": "2603.13017",
            "title": "Personalized Agent Memory",
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 5,
            "notes": "compact retrieval memory",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_note(
        note_id="paper:2603.13017",
        title="[논문] Personalized Agent Memory",
        content="# Title\n\n## 요약\n\n압축형 paper memory card를 생성한다.\n",
        source_type="paper",
        para_category="resource",
        metadata={"arxiv_id": "2603.13017"},
    )


def test_paper_memory_cli_build_show_search_json(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    khub = _StubKhub(db)
    runner = CliRunner()

    built = runner.invoke(paper_memory_group, ["build", "--paper-id", "2603.13017", "--json"], obj={"khub": khub})
    assert built.exit_code == 0
    build_payload = json.loads(built.output)
    assert build_payload["schema"] == "knowledge-hub.paper-memory.build.result.v1"
    assert build_payload["items"][0]["paperId"] == "2603.13017"
    assert "paper" not in build_payload["items"][0]
    assert "formalCause" not in build_payload["items"][0]
    assert "finalCause" not in build_payload["items"][0]
    assert validate_payload(build_payload, build_payload["schema"], strict=True).ok

    shown = runner.invoke(paper_memory_group, ["show", "--paper-id", "2603.13017", "--json"], obj={"khub": khub})
    assert shown.exit_code == 0
    show_payload = json.loads(shown.output)
    assert show_payload["schema"] == "knowledge-hub.paper-memory.card.result.v1"
    assert show_payload["item"]["paperId"] == "2603.13017"
    assert show_payload["item"]["paper"]["paperId"] == "2603.13017"
    assert "formalCause" not in show_payload["item"]
    assert "finalCause" not in show_payload["item"]
    assert validate_payload(show_payload, show_payload["schema"], strict=True).ok

    searched = runner.invoke(
        paper_memory_group,
        ["search", "--query", "paper memory", "--json"],
        obj={"khub": khub},
    )
    assert searched.exit_code == 0
    search_payload = json.loads(searched.output)
    assert search_payload["schema"] == "knowledge-hub.paper-memory.search.result.v1"
    assert search_payload["count"] >= 1
    assert search_payload["items"][0]["sourceNote"]["id"] == "paper:2603.13017"
    assert "formalCause" not in search_payload["items"][0]
    assert "finalCause" not in search_payload["items"][0]
    assert validate_payload(search_payload, search_payload["schema"], strict=True).ok


def test_paper_memory_cli_missing_and_rebuild_paths(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    khub = _StubKhub(db)
    runner = CliRunner()

    missing_build = runner.invoke(
        paper_memory_group,
        ["build", "--paper-id", "missing-paper", "--json"],
        obj={"khub": khub},
    )
    assert missing_build.exit_code != 0
    assert "paper not found" in str(missing_build.exception)

    missing_show = runner.invoke(
        paper_memory_group,
        ["show", "--paper-id", "missing-paper", "--json"],
        obj={"khub": khub},
    )
    assert missing_show.exit_code == 0
    missing_show_payload = json.loads(missing_show.output)
    assert missing_show_payload["status"] == "failed"
    assert missing_show_payload["item"] == {}
    assert validate_payload(missing_show_payload, missing_show_payload["schema"], strict=True).ok

    rebuilt = runner.invoke(
        paper_memory_group,
        ["rebuild", "--all", "--json"],
        obj={"khub": khub},
    )
    assert rebuilt.exit_code == 0
    rebuild_payload = json.loads(rebuilt.output)
    assert rebuild_payload["mode"] == "rebuild_all"
    assert rebuild_payload["count"] == 1
    assert validate_payload(rebuild_payload, rebuild_payload["schema"], strict=True).ok

    missing_all = runner.invoke(
        paper_memory_group,
        ["rebuild", "--json"],
        obj={"khub": khub},
    )
    assert missing_all.exit_code != 0
    assert "--all" in missing_all.output
