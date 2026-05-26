from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "eval/knowledgeos/scripts/collect_vault_default_eval.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("collect_vault_default_eval_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vault_stale_stats_accepts_nested_metadata_file_path(tmp_path: Path):
    module = _load_script()
    note_path = tmp_path / "Projects" / "AI" / "RAG.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text("# RAG\n", encoding="utf-8")
    payload = {
        "sources": [
            {
                "source_type": "vault",
                "metadata": {"file_path": "Projects/AI/RAG.md"},
            }
        ]
    }

    assert module._vault_stale_citation_stats(payload, vault_root=tmp_path) == (1, 0, "0.000000")


def test_vault_stale_stats_accepts_source_ref_alias_without_extension(tmp_path: Path):
    module = _load_script()
    note_path = tmp_path / "Daily" / "retrieval-review.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text("# Review\n", encoding="utf-8")
    payload = {
        "sources": [
            {
                "normalized_source_type": "vault",
                "source_ref": "Daily/retrieval-review#stale citation notes",
            }
        ]
    }

    assert module._vault_stale_citation_stats(payload, vault_root=tmp_path) == (1, 0, "0.000000")


def test_vault_stale_stats_accepts_nested_source_trace_source_id(tmp_path: Path):
    module = _load_script()
    note_path = tmp_path / "Papers" / "pipeline.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text("# Pipeline\n", encoding="utf-8")
    payload = {
        "sources": [
            {
                "source_type": "vault",
                "source_trace": {"sourceId": "Papers/pipeline.md"},
            }
        ]
    }

    assert module._vault_stale_citation_stats(payload, vault_root=tmp_path) == (1, 0, "0.000000")


def test_vault_stale_stats_rejects_absolute_path_outside_vault_root(tmp_path: Path):
    module = _load_script()
    vault_root = tmp_path / "vault"
    outside_path = tmp_path / "outside" / "note.md"
    outside_path.parent.mkdir(parents=True)
    outside_path.write_text("# Outside\n", encoding="utf-8")
    payload = {
        "sources": [
            {
                "source_type": "vault",
                "file_path": str(outside_path),
            }
        ]
    }

    assert module._vault_stale_citation_stats(payload, vault_root=vault_root) == (1, 1, "1.000000")


def test_resolve_vault_root_uses_explicit_root_first(tmp_path: Path):
    module = _load_script()
    explicit = tmp_path / "explicit-vault"
    config = tmp_path / "config-vault"
    explicit.mkdir()
    config.mkdir()

    resolved = module._resolve_vault_root(
        explicit_vault_root=str(explicit),
        config_vault_path=str(config),
        repo_root=tmp_path / "repo",
    )

    assert resolved == explicit.resolve()


def test_resolve_vault_root_falls_back_to_workspace_vault_symlink_shape(tmp_path: Path):
    module = _load_script()
    workspace = tmp_path / "KnowledgeOS"
    repo_root = workspace / ".worktrees" / "knowledge-hub-fix"
    vault = workspace / "vault"
    repo_root.mkdir(parents=True)
    vault.mkdir()

    resolved = module._resolve_vault_root(
        explicit_vault_root="",
        config_vault_path="",
        repo_root=repo_root,
    )

    assert resolved == vault.resolve()


def test_vault_stale_stats_rejects_title_only_source_as_stale(tmp_path: Path):
    module = _load_script()
    payload = {
        "sources": [
            {
                "source_type": "vault",
                "title": "RAG search quality retrospective",
            }
        ]
    }

    assert module._vault_stale_citation_stats(payload, vault_root=tmp_path) == (1, 1, "1.000000")


def test_vault_stale_stats_counts_no_vault_sources_as_empty(tmp_path: Path):
    module = _load_script()
    payload = {"sources": [{"source_type": "paper", "title": "Attention Is All You Need"}]}

    assert module._vault_stale_citation_stats(payload, vault_root=tmp_path) == (0, 0, "")
