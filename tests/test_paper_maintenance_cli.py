from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.core.config import Config
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
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


def test_paper_normalize_concepts_dry_run_includes_db_heuristics(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    papers_dir = tmp_path / "vault" / "Papers"
    concepts_dir = papers_dir / "Concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_ontology_entity(
        entity_id="language_models",
        entity_type="concept",
        canonical_name="Language Models",
        properties={"heuristic_source": "paper_memory_title_fallback"},
        source="test",
    )
    db.upsert_ontology_entity(
        entity_id="llm",
        entity_type="concept",
        canonical_name="Large Language Models",
        source="test",
    )

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_llm",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._detect_synonym_groups",
        lambda llm, batch: [{"canonical": "Large Language Models", "aliases": ["Language Models"]}],
    )

    result = CliRunner().invoke(paper_group, ["normalize-concepts", "--dry-run"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    assert "Large Language Models" in result.output
    assert "Language Models" in result.output


def test_paper_normalize_concepts_updates_memory_cards_and_relations(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    papers_dir = tmp_path / "vault" / "Papers"
    concepts_dir = papers_dir / "Concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    db = SQLiteDatabase(config.sqlite_path)
    paper_id = "2603.13030"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "Language Model Study",
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_ontology_entity(
        entity_id="language_models",
        entity_type="concept",
        canonical_name="Language Models",
        properties={"heuristic_source": "paper_memory_title_fallback"},
        source="test",
    )
    db.upsert_ontology_entity(
        entity_id="llm",
        entity_type="concept",
        canonical_name="Large Language Models",
        source="test",
    )
    db.upsert_paper_memory_card(
        card={
            "memory_id": f"paper-memory:{paper_id}:test",
            "paper_id": paper_id,
            "source_note_id": "",
            "title": "Language Model Study",
            "paper_core": "Study of language models.",
            "problem_context": "",
            "method_core": "",
            "evidence_core": "",
            "limitations": "",
            "concept_links": ["Language Models"],
            "claim_refs": [],
            "published_at": "",
            "evidence_window": "",
            "search_text": "Language Models",
            "quality_flag": "needs_review",
            "version": "paper-memory-v1",
        }
    )
    db.add_relation(
        source_type="paper",
        source_id=paper_id,
        relation="uses",
        target_type="concept",
        target_id="language_models",
        evidence_text=json.dumps({"source": "test", "relation_norm": "uses"}, ensure_ascii=False),
        confidence=0.7,
    )

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_llm",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._detect_synonym_groups",
        lambda llm, batch: [{"canonical": "Large Language Models", "aliases": ["Language Models"]}],
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._merge_obsidian_concept",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._replace_in_paper_notes",
        lambda *args, **kwargs: None,
    )

    result = CliRunner().invoke(paper_group, ["normalize-concepts"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    card = db.get_paper_memory_card(paper_id)
    assert card["concept_links"] == ["Large Language Model"]
    concept_names = [row.get("canonical_name") for row in db.get_paper_concepts(paper_id)]
    assert "Large Language Model" in concept_names
    assert "Language Models" not in concept_names


def test_paper_normalize_concepts_promotes_trusted_title_seed_sources(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    papers_dir = tmp_path / "vault" / "Papers"
    concepts_dir = papers_dir / "Concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_ontology_entity(
        entity_id="ai_agent",
        entity_type="concept",
        canonical_name="AI Agent",
        properties={"heuristic_source": "paper_memory_title_fallback"},
        source="paper_memory_title_fallback",
    )
    db.upsert_ontology_entity(
        entity_id="large_language_model",
        entity_type="concept",
        canonical_name="Large Language Model",
        properties={"heuristic_source": "paper_memory_title_fallback"},
        source="paper_memory_title_fallback",
    )

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_llm",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._detect_synonym_groups",
        lambda llm, batch: [{"canonical": "AI Agent", "aliases": ["AI Agents"]}],
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._merge_obsidian_concept",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._replace_in_paper_notes",
        lambda *args, **kwargs: None,
    )

    result = CliRunner().invoke(paper_group, ["normalize-concepts"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    promoted = db.resolve_entity("AI Agent", entity_type="concept")
    assert promoted is not None
    assert promoted["source"] == "paper_title_seed"
    assert "heuristic_source" not in (promoted.get("properties") or {})


def test_paper_normalize_concepts_applies_curated_memory_cleanup_without_llm_groups(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    papers_dir = tmp_path / "vault" / "Papers"
    concepts_dir = papers_dir / "Concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    db = SQLiteDatabase(config.sqlite_path)
    paper_id = "2604.00001"
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": "LLM-based agents benchmark",
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_paper_memory_card(
        card={
            "memory_id": f"paper-memory:{paper_id}:test",
            "paper_id": paper_id,
            "source_note_id": "",
            "title": "LLM-based agents benchmark",
            "paper_core": "Study of LLM-based agents.",
            "problem_context": "",
            "method_core": "",
            "evidence_core": "",
            "limitations": "",
            "concept_links": ["LLM-based agents", "Benchmark", "Language Models"],
            "claim_refs": [],
            "published_at": "",
            "evidence_window": "",
            "search_text": "LLM-based agents Benchmark Language Models",
            "quality_flag": "needs_review",
            "version": "paper-memory-v1",
        }
    )
    db.upsert_ontology_entity(
        entity_id="benchmark",
        entity_type="concept",
        canonical_name="Benchmark",
        properties={"heuristic_source": "paper_memory_title_fallback"},
        source="paper_memory_title_fallback",
    )

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_llm",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._detect_synonym_groups",
        lambda llm, batch: [],
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._merge_obsidian_concept",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._replace_in_paper_notes",
        lambda *args, **kwargs: None,
    )

    result = CliRunner().invoke(paper_group, ["normalize-concepts"], obj={"khub": _StubKhub(config)})

    assert result.exit_code == 0
    card = db.get_paper_memory_card(paper_id)
    assert card["concept_links"] == ["LLM Agent", "Language Model"]
    assert db.resolve_entity("Benchmark", entity_type="concept") is None
    assert db.resolve_entity("LLM Agent", entity_type="concept") is not None
    assert db.resolve_entity("Language Model", entity_type="concept") is not None


def test_paper_normalize_concepts_passes_provider_override(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    papers_dir = tmp_path / "vault" / "Papers"
    concepts_dir = papers_dir / "Concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_ontology_entity(
        entity_id="language_models",
        entity_type="concept",
        canonical_name="Language Models",
        source="test",
    )
    db.upsert_ontology_entity(
        entity_id="llm",
        entity_type="concept",
        canonical_name="Large Language Models",
        source="test",
    )
    captured: dict[str, str | None] = {}

    def _fake_build_llm(config_obj, provider, model, **kwargs):  # noqa: ARG001
        captured["provider"] = provider
        captured["model"] = model
        return object()

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._build_llm",
        _fake_build_llm,
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._detect_synonym_groups",
        lambda llm, batch: [{"canonical": "Large Language Models", "aliases": ["Language Models"]}],
    )

    result = CliRunner().invoke(
        paper_group,
        ["normalize-concepts", "--dry-run", "--provider", "openai", "--model", "gpt-5-nano"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0
    assert captured == {"provider": "openai", "model": "gpt-5-nano"}


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
