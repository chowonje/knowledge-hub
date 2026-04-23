from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge_hub.application.context import AppContextFactory


def _resolved_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sync_legacy_state(
    state: Any,
    *,
    config: Any = None,
    sqlite_db: Any = None,
    searcher: Any = None,
    config_path: str | None = None,
) -> None:
    state.config = config
    state.sqlite_db = sqlite_db
    state.searcher = searcher
    state.config_path = config_path


def _get_factory(state: Any) -> AppContextFactory:
    factory = getattr(state, "_app_context_factory", None)
    requested_path = getattr(state, "config_path", None)
    if isinstance(factory, AppContextFactory) and factory.config_path == requested_path:
        return factory
    factory = AppContextFactory(requested_path, project_root=_resolved_project_root())
    setattr(state, "_app_context_factory", factory)
    return factory


def initialize_search_runtime(state: Any):
    app = _get_factory(state).build(require_search=True)
    sync_legacy_state(
        state,
        config=app.config,
        sqlite_db=app.sqlite_db,
        searcher=app.searcher,
        config_path=getattr(state, "config_path", None),
    )
    return app


def initialize_core_runtime(state: Any):
    app = _get_factory(state).build(core_only=True)
    sync_legacy_state(
        state,
        config=app.config,
        sqlite_db=app.sqlite_db,
        searcher=getattr(state, "searcher", None),
        config_path=getattr(state, "config_path", None),
    )
    if getattr(state, "searcher", None) is None:
        state.searcher = None
    return app


def ensure_tool_runtime(
    state: Any,
    *,
    tool_name: str,
    initialize_fn: Any,
    initialize_core_only_fn: Any,
) -> None:
    from knowledge_hub.application.mcp.responses import CORE_ONLY_TOOL_NAMES, JOB_TOOLS

    if tool_name in CORE_ONLY_TOOL_NAMES or tool_name in JOB_TOOLS:
        if getattr(state, "config", None) is None or getattr(state, "sqlite_db", None) is None:
            initialize_core_only_fn()
        return
    if getattr(state, "searcher", None) is None:
        initialize_fn()
