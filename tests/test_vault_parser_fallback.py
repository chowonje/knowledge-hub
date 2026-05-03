from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.models import Document, SourceType
from knowledge_hub.vault import parser as parser_module


def test_parse_frontmatter_fallback_without_dependency(monkeypatch):
    monkeypatch.setattr(parser_module, "frontmatter", None)
    raw = "---\ntitle: Sample\ntags:\n  - ai\n---\nBody text"

    metadata, content = parser_module.ObsidianParser._parse_frontmatter(raw)

    assert metadata["title"] == "Sample"
    assert metadata["tags"] == ["ai"]
    assert content.strip() == "Body text"


def test_parse_frontmatter_falls_back_when_frontmatter_dependency_raises(monkeypatch):
    class _BrokenFrontmatter:
        @staticmethod
        def loads(_raw: str):
            raise ValueError("broken frontmatter")

    monkeypatch.setattr(parser_module, "frontmatter", _BrokenFrontmatter())
    raw = "---\ntitle: Broken\n*\n---\nBody text"

    metadata, content = parser_module.ObsidianParser._parse_frontmatter(raw)

    assert metadata["title"] == "Broken"
    assert content.strip() == "Body text"


def test_parse_frontmatter_loose_salvages_simple_key_values():
    raw = (
        "---\n"
        "@id: macro:overview\n"
        "@type: ConceptCollection\n"
        "카테고리: 거시경제\n"
        "tags: [거시경제, 경제지표, 통화정책]\n"
        "created: 2025-05-22\n"
        "---\n"
        "Body text\n"
    )

    metadata, content = parser_module.ObsidianParser._parse_frontmatter(raw)

    assert metadata["@id"] == "macro:overview"
    assert metadata["@type"] == "ConceptCollection"
    assert metadata["카테고리"] == "거시경제"
    assert metadata["tags"] == ["거시경제", "경제지표", "통화정책"]
    assert str(metadata["created"]) == "2025-05-22"
    assert content.strip() == "Body text"


def test_parse_file_sanitizes_frontmatter_dates(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "dated.md"
    note.write_text(
        "---\n"
        "title: Dated Note\n"
        "created: 2026-03-12\n"
        "aliases:\n"
        "  - first\n"
        "  - second\n"
        "---\n"
        "Body text\n",
        encoding="utf-8",
    )

    parser = parser_module.ObsidianParser(str(vault))
    document = parser.parse_file(note)

    assert document is not None
    assert document.metadata["created"] == "2026-03-12"
    json.dumps(document.metadata, ensure_ascii=False)


def test_parse_file_uses_first_heading_as_title_when_frontmatter_missing(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "untitled.md"
    note.write_text(
        "\n---\n\n# 강화 학습 핵심 요약\n\n본문\n",
        encoding="utf-8",
    )

    parser = parser_module.ObsidianParser(str(vault))
    document = parser.parse_file(note)

    assert document is not None
    assert document.title == "강화 학습 핵심 요약"


def test_chunk_document_emits_stable_scope_ids():
    document = Document(
        content="Intro text before headings.\n\n# Section One\nSection body section body section body.\n",
        metadata={},
        file_path="notes/rag.md",
        title="RAG",
        source_type=SourceType.VAULT,
    )

    chunks = parser_module.ObsidianParser.chunk_document(document, chunk_size=32, chunk_overlap=0)

    assert chunks
    document_chunk = next(chunk for chunk in chunks if chunk["metadata"]["scope_level"] == "document")
    section_chunk = next(chunk for chunk in chunks if chunk["metadata"]["scope_level"] == "section")

    assert document_chunk["metadata"]["document_scope_id"] == "vault:notes/rag.md"
    assert document_chunk["metadata"]["stable_scope_id"] == "vault:notes/rag.md"
    assert document_chunk["metadata"]["section_scope_id"] == ""
    assert section_chunk["metadata"]["document_scope_id"] == "vault:notes/rag.md"
    assert section_chunk["metadata"]["section_scope_id"] == "vault:notes/rag.md::section:Section One"
    assert section_chunk["metadata"]["stable_scope_id"] == "vault:notes/rag.md::section:Section One"


def test_parse_vault_excludes_generated_vendor_paths(tmp_path: Path):
    vault = tmp_path / "vault"
    (vault / ".local-rag" / "node_modules").mkdir(parents=True)
    (vault / "notes").mkdir(parents=True)
    (vault / ".local-rag" / "node_modules" / "ignored.md").write_text("# Ignored\nbody\n", encoding="utf-8")
    (vault / "notes" / "kept.md").write_text("# Kept\nbody\n", encoding="utf-8")

    parser = parser_module.ObsidianParser(str(vault), exclude_folders=["custom"])
    documents = parser.parse_vault()

    assert [doc.file_path for doc in documents] == ["notes/kept.md"]
