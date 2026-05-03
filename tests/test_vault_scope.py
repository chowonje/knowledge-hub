from __future__ import annotations

from knowledge_hub.ai.ask_v2_support import vault_scope_from_query as ask_v2_vault_scope_from_query
from knowledge_hub.domain.vault_knowledge.scope import (
    explicit_vault_scope,
    vault_scope_from_filter,
    vault_scope_from_query,
)


def test_vault_scope_prefers_metadata_filter():
    assert explicit_vault_scope(
        "Please summarize Other Note.md",
        metadata_filter={"file_path": "Project Brief.md"},
    ) == "Project Brief.md"
    assert vault_scope_from_filter({"note_id": "vault:project-brief", "file_path": "Project Brief.md"}) == "vault:project-brief"


def test_vault_scope_extracts_note_id_from_query():
    assert vault_scope_from_query("latest vault:missing-note changes?") == "vault:missing-note"


def test_vault_scope_strips_sentence_punctuation_from_note_id():
    assert vault_scope_from_query("vault:project-brief?") == "vault:project-brief"


def test_vault_scope_extracts_nested_markdown_path_without_sentence_prefix():
    assert vault_scope_from_query("Please summarize AI/RAG/quality.md") == "AI/RAG/quality.md"


def test_vault_scope_extracts_root_markdown_path_with_spaces():
    assert vault_scope_from_query("Please summarize Project Brief.md") == "Project Brief.md"


def test_vault_scope_extracts_temporal_korean_nested_path():
    assert vault_scope_from_query("최근 Missing/NoSuchNote-999.md 업데이트를 알려줘") == "Missing/NoSuchNote-999.md"


def test_ask_v2_vault_scope_wrapper_uses_domain_parser():
    query = "Please summarize Project Brief.md"

    assert ask_v2_vault_scope_from_query(query) == vault_scope_from_query(query)
