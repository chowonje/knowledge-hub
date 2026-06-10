"""Red tests for the 2026-06-11 vault-write hazard closeout.

Covers four hazards from the adversarial architecture review:
1. KoNoteMaterializer must not fall back to ~/Documents/Obsidian Vault.
2. MCP paper-ingest tools must default to create_obsidian_note=False.
3. obsidian.enabled=false must block every vault writer through the
   authoritative adapter chokepoint.
4. Remediation of already-applied notes must require fresh approval.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from knowledge_hub.core.vault_guard import (
    VaultWriteError,
    VaultWriteNotConfiguredError,
    VaultWritesDisabledError,
    ensure_vault_writes_allowed,
)
from knowledge_hub.infrastructure.config import Config
from knowledge_hub.learning.obsidian_writeback import (
    FileSystemVaultAdapter,
    resolve_config_vault_write_adapter,
)
from knowledge_hub.notes.applier import KoNoteApplier
from knowledge_hub.notes.enricher import KoNoteEnricher
from knowledge_hub.notes.materializer import KoNoteMaterializer
from knowledge_hub.notes.models import KoNoteReview
from knowledge_hub.papers.discoverer import DiscoveredPaper
from knowledge_hub.papers.manager import PaperManager


def _config(tmp_path: Path) -> Config:
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    return Config(str(path))


def _enabled_config(tmp_path: Path) -> Config:
    config = _config(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    config.set_nested("obsidian", "enabled", True)
    config.set_nested("obsidian", "vault_path", str(vault))
    return config


def _disabled_config(tmp_path: Path) -> Config:
    config = _config(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    config.set_nested("obsidian", "enabled", False)
    config.set_nested("obsidian", "vault_path", str(vault))
    return config


class _FakeKoNoteRepo:
    def __init__(self) -> None:
        self.payload_updates: list[tuple[int, dict[str, Any]]] = []
        self.status_updates: list[tuple[int, str]] = []
        self.items: dict[int, dict[str, Any]] = {}

    def get_relations(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def get_entity(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        return None

    def update_ko_note_item_payload(self, item_id: int, *, payload: dict, **kwargs: Any) -> bool:
        self.payload_updates.append((int(item_id), payload))
        if int(item_id) in self.items:
            self.items[int(item_id)]["payload_json"] = payload
        return True

    def update_ko_note_item_status(self, item_id: int, *, status: str, **kwargs: Any) -> bool:
        self.status_updates.append((int(item_id), str(status)))
        if int(item_id) in self.items:
            self.items[int(item_id)]["status"] = str(status)
        return True

    def get_ko_note_item(self, item_id: int) -> dict[str, Any] | None:
        return self.items.get(int(item_id))

    def get_ko_note_run(self, run_id: str) -> dict[str, Any] | None:
        return {"run_id": str(run_id)}


# --- hazard 1: documents-vault fallback removed -------------------------------


def test_vault_root_fails_closed_without_vault_path(tmp_path):
    config = _config(tmp_path)
    config.set_nested("obsidian", "enabled", True)
    materializer = KoNoteMaterializer(config, sqlite_db=_FakeKoNoteRepo())
    with pytest.raises(VaultWriteNotConfiguredError):
        materializer._vault_root()


def test_vault_root_fails_closed_when_vault_disabled(tmp_path):
    materializer = KoNoteMaterializer(_disabled_config(tmp_path), sqlite_db=_FakeKoNoteRepo())
    with pytest.raises(VaultWritesDisabledError):
        materializer._vault_root()


def test_vault_root_never_falls_back_to_documents_vault():
    import knowledge_hub.notes.materializer as materializer_module

    assert not hasattr(materializer_module, "DEFAULT_DOCUMENTS_VAULT")


def test_vault_root_resolves_configured_path(tmp_path):
    config = _enabled_config(tmp_path)
    materializer = KoNoteMaterializer(config, sqlite_db=_FakeKoNoteRepo())
    assert materializer._vault_root() == (tmp_path / "vault").resolve()


# --- hazard 3: one authoritative enabled-check --------------------------------


def test_ensure_vault_writes_allowed_blocks_disabled(tmp_path):
    with pytest.raises(VaultWritesDisabledError):
        ensure_vault_writes_allowed(_disabled_config(tmp_path))


def test_ensure_vault_writes_allowed_blocks_unconfigured_path(tmp_path):
    config = _config(tmp_path)
    config.set_nested("obsidian", "enabled", True)
    with pytest.raises(VaultWriteNotConfiguredError):
        ensure_vault_writes_allowed(config)


def test_resolve_config_vault_write_adapter_blocks_disabled(tmp_path):
    with pytest.raises(VaultWritesDisabledError):
        resolve_config_vault_write_adapter(_disabled_config(tmp_path))


def test_resolve_config_vault_write_adapter_blocks_explicit_path_override_when_disabled(tmp_path):
    with pytest.raises(VaultWritesDisabledError):
        resolve_config_vault_write_adapter(
            _disabled_config(tmp_path),
            vault_path=str(tmp_path / "vault"),
        )


def test_resolve_config_vault_write_adapter_allows_enabled(tmp_path):
    adapter = resolve_config_vault_write_adapter(_enabled_config(tmp_path))
    assert isinstance(adapter, FileSystemVaultAdapter)


def test_paper_manager_note_creation_blocked_when_vault_disabled(tmp_path):
    config = _disabled_config(tmp_path)
    manager = PaperManager(
        config=config,
        vector_db=object(),
        sqlite_db=object(),
        embedder=object(),
    )
    paper = DiscoveredPaper(
        arxiv_id="2406.00001",
        title="Guarded Vault Writes",
        authors="A. Author",
        year=2026,
        abstract="abstract",
    )
    assert manager._create_obsidian_note(paper, "summary", "topic") == ""
    assert list((tmp_path / "vault").rglob("*.md")) == []


def test_paper_summary_sync_blocked_when_vault_disabled(tmp_path):
    from knowledge_hub.interfaces.cli.commands.paper_shared_runtime import _update_obsidian_summary

    config = _disabled_config(tmp_path)
    papers_dir = tmp_path / "vault" / "Papers"
    papers_dir.mkdir(parents=True)
    note = papers_dir / "Guarded Vault Writes.md"
    original = "# Note\n\n요약본/번역본이 아직 등록되지 않았습니다\n"
    note.write_text(original, encoding="utf-8")

    _update_obsidian_summary({"title": "Guarded Vault Writes"}, "NEW SUMMARY", config)

    assert note.read_text(encoding="utf-8") == original


def test_apply_lane_fails_closed_when_vault_disabled(tmp_path):
    materializer = KoNoteMaterializer(_disabled_config(tmp_path), sqlite_db=_FakeKoNoteRepo())
    result = KoNoteApplier(materializer).apply(run_id="run_disabled")
    assert result["status"] == "failed"
    assert result["applied"] == 0
    assert any("disabled" in str(warning) for warning in result["warnings"])


# --- hazard 2: MCP ingest tools default to no vault writes --------------------


def _import_tool_specs():
    pytest.importorskip("mcp")
    from knowledge_hub.mcp import tool_specs

    return tool_specs


def test_mcp_paper_ingest_tool_specs_default_to_no_obsidian_note():
    tool_specs = _import_tool_specs()
    tools = tool_specs.build_tools(profile="all")
    for tool_name in ("discover_and_ingest", "run_paper_ingest_flow"):
        tool = next(t for t in tools if t.name == tool_name)
        assert tool.inputSchema["properties"]["create_obsidian_note"]["default"] is False, tool_name


def _paper_handler_ctx(captured: dict[str, Any]) -> dict[str, Any]:
    async def _run_async_tool(*, name, request_echo, sync_job):  # noqa: ANN001
        _ = (name, request_echo)
        payload = await sync_job()
        return "job_1", {"payload": payload}

    def _emit(status, payload, **kwargs):  # noqa: ANN001
        return {"status": status, "payload": payload, "meta": kwargs}

    return {
        "emit": _emit,
        "config": Config(),
        "sqlite_db": object(),
        "searcher": type("Searcher", (), {"database": object(), "embedder": object(), "llm": object()})(),
        "to_bool": lambda value, default=False: default if value is None else bool(value),
        "to_int": lambda value, default=None, minimum=None, maximum=None: default if value is None else int(value),
        "run_async_tool": _run_async_tool,
        "request_echo": {},
        "MCP_TOOL_STATUS_OK": "ok",
        "MCP_TOOL_STATUS_FAILED": "failed",
        "MCP_TOOL_STATUS_QUEUED": "queued",
    }


@pytest.mark.parametrize("tool_name", ["discover_and_ingest", "run_paper_ingest_flow"])
def test_mcp_paper_ingest_handlers_default_to_no_obsidian_note(monkeypatch, tool_name):
    pytest.importorskip("mcp")
    from knowledge_hub.mcp.handlers import paper as paper_handler

    captured: dict[str, Any] = {}

    class _FakeManager:
        def __init__(self, **kwargs):  # noqa: ANN003
            _ = kwargs

        def discover_and_ingest(self, **kwargs):  # noqa: ANN003
            captured.update(kwargs)
            return {"status": "ok", "ingested": [], "failed": [], "warnings": []}

    monkeypatch.setattr("knowledge_hub.papers.manager.PaperManager", _FakeManager)

    result = asyncio.run(
        paper_handler.handle_tool(tool_name, {"topic": "vault guard"}, _paper_handler_ctx(captured))
    )
    assert result["status"] == "queued"
    assert captured["create_obsidian_note"] is False


# --- hazard 4: remediation of applied notes requires fresh approval -----------


def _applied_source_item(staging: Path, final: Path) -> dict[str, Any]:
    return {
        "id": 7,
        "item_type": "source",
        "status": "applied",
        "run_id": "run_orig",
        "title_en": "Guarded Vault Writes",
        "title_ko": "가드된 볼트 쓰기",
        "note_id": "web:guarded-vault-writes",
        "staging_path": str(staging),
        "final_path": str(final),
        "payload_json": {
            "title_en": "Guarded Vault Writes",
            "title_ko": "가드된 볼트 쓰기",
            "note_id": "web:guarded-vault-writes",
            "review": {
                "queue": False,
                "decision": {
                    "status": "approved",
                    "reviewer": "cli-user",
                    "note": "",
                    "reviewedAt": "2026-06-01T00:00:00Z",
                },
            },
        },
    }


def test_remediate_applied_item_requires_fresh_approval(tmp_path):
    config = _enabled_config(tmp_path)
    repo = _FakeKoNoteRepo()
    enricher = KoNoteEnricher(config, sqlite_db=repo)

    staging = tmp_path / "vault" / "LearningHub" / "ai" / "ko_notes" / "note.md"
    staging.parent.mkdir(parents=True)
    staging.write_text("old staging", encoding="utf-8")
    final = tmp_path / "vault" / "Web_Sources" / "note.md"
    final.parent.mkdir(parents=True)
    final_original = "FINAL APPLIED CONTENT"
    final.write_text(final_original, encoding="utf-8")

    item = _applied_source_item(staging, final)
    repo.items[7] = item

    enricher._rewrite_staging_or_final(dict(item), run_id="run_remediate")

    # the applied vault note must not be rewritten without fresh approval
    assert final.read_text(encoding="utf-8") == final_original
    # the item is demoted back into the review lane
    assert (7, "staged") in repo.status_updates
    assert repo.payload_updates, "review payload must be updated for re-approval"
    updated_review = KoNoteReview.from_payload(repo.payload_updates[-1][1])
    assert updated_review.queue is True
    assert updated_review.decision is None or not updated_review.decision.is_approved()


def test_remediate_staged_item_still_rewrites_staging(tmp_path):
    config = _enabled_config(tmp_path)
    repo = _FakeKoNoteRepo()
    enricher = KoNoteEnricher(config, sqlite_db=repo)

    staging = tmp_path / "vault" / "LearningHub" / "ai" / "ko_notes" / "note.md"
    staging.parent.mkdir(parents=True)
    staging.write_text("old staging", encoding="utf-8")
    final = tmp_path / "vault" / "Web_Sources" / "note.md"

    item = _applied_source_item(staging, final)
    item["status"] = "staged"
    repo.items[7] = item

    enricher._rewrite_staging_or_final(dict(item), run_id="run_remediate")

    assert staging.read_text(encoding="utf-8") != "old staging"
    assert not final.exists()
    assert repo.status_updates == []


def test_remediate_rewrite_blocked_when_vault_disabled(tmp_path):
    config = _disabled_config(tmp_path)
    repo = _FakeKoNoteRepo()
    enricher = KoNoteEnricher(config, sqlite_db=repo)

    staging = tmp_path / "vault" / "LearningHub" / "ai" / "ko_notes" / "note.md"
    staging.parent.mkdir(parents=True)
    staging.write_text("old staging", encoding="utf-8")
    item = _applied_source_item(staging, tmp_path / "vault" / "Web_Sources" / "note.md")
    item["status"] = "staged"
    repo.items[7] = item

    with pytest.raises(VaultWriteError):
        enricher._rewrite_staging_or_final(dict(item), run_id="run_remediate")
    assert staging.read_text(encoding="utf-8") == "old staging"
