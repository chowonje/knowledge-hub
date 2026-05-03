"""Read-only task context assembly for knowledge + workspace grounded assistance."""

from __future__ import annotations

from typing import Any

from knowledge_hub.application.agent_gateway import build_gateway_metadata
from knowledge_hub.application.context_pack import build_context_pack, classify_task_mode


def build_runtime_diagnostics(config: Any, *, searcher: Any = None) -> dict[str, Any]:
    database = getattr(searcher, "database", None)
    stats: dict[str, Any] = {}
    if database is not None and hasattr(database, "get_stats"):
        try:
            stats = dict(database.get_stats() or {})
        except Exception as error:
            stats = {"error": str(error)}
    return {
        "schema": "knowledge-hub.runtime.diagnostics.v1",
        "status": "ok",
        "providerStates": [],
        "vectorCorpus": {
            "total_documents": int(stats.get("total_documents", 0) or 0),
            "collection_name": str(stats.get("collection_name", "")),
            "available": bool(stats.get("total_documents", 0) and stats.get("collection_name")),
        },
        "warnings": [],
    }


def build_task_context(
    searcher: Any,
    *,
    goal: str,
    repo_path: str | None = None,
    include_workspace: bool = True,
    include_vault: bool = True,
    include_papers: bool = True,
    include_web: bool = True,
    max_workspace_files: int = 8,
    max_project_docs: int = 6,
    max_knowledge_hits: int = 5,
    max_excerpt_chars: int = 1500,
) -> dict[str, Any]:
    pack = build_context_pack(
        searcher,
        query_or_topic=goal,
        target="task",
        repo_path=repo_path,
        include_workspace=include_workspace,
        include_vault=include_vault,
        include_papers=include_papers,
        include_web=include_web,
        max_workspace_files=max_workspace_files,
        max_project_docs=max_project_docs,
        max_knowledge_hits=max_knowledge_hits,
        max_chars=max_excerpt_chars,
    )

    knowledge_hits = [
        {
            "source": source["normalized_source_type"],
            "title": source["title"],
            "snippet": source.get("snippet") or source.get("content") or "",
            "score": source.get("score", 0.0),
            "normalized_source_type": source["normalized_source_type"],
            "scope_level": source.get("scope_level", ""),
            "stable_scope_id": source.get("stable_scope_id", ""),
            "document_scope_id": source.get("document_scope_id", ""),
            "section_scope_id": source.get("section_scope_id", ""),
        }
        for source in pack.get("persistent_sources") or []
    ]
    workspace_files = list(pack.get("workspace_sources") or [])
    project_conventions = list(pack.get("project_conventions") or [])
    warnings = list(pack.get("warnings") or [])
    mode = str(pack.get("mode") or classify_task_mode(goal))

    payload = {
        "schema": "knowledge-hub.task-context.result.v1",
        "status": "ok",
        "goal": goal,
        "mode": mode,
        "repoPath": str(pack.get("repo_path") or ""),
        "knowledge_hits": knowledge_hits,
        "workspace_files": workspace_files,
        "project_conventions": project_conventions,
        "suggested_prompt_context": str(pack.get("prompt_context") or "").strip(),
        "warnings": warnings,
        "runtimeDiagnostics": build_runtime_diagnostics(getattr(searcher, "config", None), searcher=searcher),
    }
    payload["evidenceSummary"] = {
        "persistentKnowledgeCount": len(knowledge_hits),
        "workspaceFileCount": len(workspace_files),
        "workspaceIncluded": bool(include_workspace and workspace_files),
        "repoContextEphemeral": True,
    }
    payload["gateway"] = build_gateway_metadata(surface="task_context", mode="context")
    return payload
