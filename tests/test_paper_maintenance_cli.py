from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.core.config import Config
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group


class _StubKhub:
    def __init__(self, config: Config):
        self.config = config


def _config(tmp_path: Path) -> Config:
    config = Config()
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))
    config.set_nested("storage", "papers_dir", str(tmp_path / "papers"))
    config.set_nested("storage", "vector_db", str(tmp_path / "vector_db"))
    config.set_nested("summarization", "provider", "openai")
    config.set_nested("summarization", "model", "gpt-5-mini")
    config.set_nested("obsidian", "vault_path", str(tmp_path / "vault"))
    return config


def test_paper_build_concepts_creates_concept_notes(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    papers_dir = tmp_path / "vault" / "Papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    (papers_dir / "Agent Memory.md").write_text(
        (
            "# Agent Memory\n\n"
            "# 🧩 내가 배워야 할 개념\n"
            "- [[Memory Cards]]\n"
            "## 관련 개념\n"
            "- [[Retrieval-Augmented Generation]]\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_llm",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._batch_describe_concepts",
        lambda llm, batch, all_names: {
            name: {"description": f"{name} description", "related": [n for n in all_names if n != name][:1]}
            for name in batch
        },
    )

    result = CliRunner().invoke(paper_group, ["build-concepts"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    concepts_dir = papers_dir / "Concepts"
    assert (concepts_dir / "Memory Cards.md").exists()
    assert "Memory Cards description" in (concepts_dir / "Memory Cards.md").read_text(encoding="utf-8")


def test_paper_normalize_concepts_dry_run_reports_groups(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    papers_dir = tmp_path / "vault" / "Papers"
    concepts_dir = papers_dir / "Concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    (papers_dir / "Agent Memory.md").write_text(
        (
            "# Agent Memory\n\n"
            "# 🧩 내가 배워야 할 개념\n"
            "- [[Memory Card]]\n"
            "## 관련 개념\n"
            "- [[Memory Cards]]\n"
        ),
        encoding="utf-8",
    )
    (concepts_dir / "Memory Cards.md").write_text("# Memory Cards\n", encoding="utf-8")

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_llm",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._detect_synonym_groups",
        lambda llm, batch: [{"canonical": "Memory Cards", "aliases": ["Memory Card"]}],
    )

    result = CliRunner().invoke(paper_group, ["normalize-concepts", "--dry-run"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    assert "동의어 그룹" in result.output
    assert "Memory Cards" in result.output
    assert "Memory Card" in result.output


def test_paper_resummary_vault_updates_low_quality_notes(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    papers_dir = tmp_path / "vault" / "Papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    note_path = papers_dir / "Agent Memory.md"
    note_path.write_text("# Agent Memory\n\nOld short note.\n", encoding="utf-8")

    class _FakeLLM:
        def summarize_paper(self, text: str, title: str = "", language: str = "ko"):  # noqa: ARG002
            return "Updated summary body with enough content to pass the minimum length threshold for writeback."

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._assess_vault_note_quality",
        lambda content: {"score": 10, "label": "미흡", "color": "yellow", "reason": "짧음"},
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_llm",
        lambda *args, **kwargs: _FakeLLM(),
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._collect_vault_note_text",
        lambda md_path, papers_dir: "source text " * 30,
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._update_vault_note_summary",
        lambda md_path, summary: md_path.write_text(summary, encoding="utf-8"),
    )

    result = CliRunner().invoke(paper_group, ["resummary-vault"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    assert "재요약 완료" in result.output
    assert "Updated summary body" in note_path.read_text(encoding="utf-8")
