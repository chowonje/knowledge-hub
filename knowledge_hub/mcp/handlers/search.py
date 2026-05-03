from __future__ import annotations

from typing import Any

from knowledge_hub.application.context_pack import normalize_source_type
from knowledge_hub.application.task_context import build_runtime_diagnostics, build_task_context


def _ranking_fields(extras: dict[str, Any] | None, *, source_type: str = "") -> dict[str, Any]:
    data = dict(extras or {})
    ranking = dict(data.get("ranking_signals") or {})
    return {
        "quality_flag": str(data.get("quality_flag") or "unscored"),
        "source_trust_score": data.get("source_trust_score", 0.0),
        "reference_role": str(data.get("reference_role") or ""),
        "reference_tier": str(data.get("reference_tier") or ""),
        "normalized_source_type": str(
            data.get("normalized_source_type") or ranking.get("normalized_source_type") or normalize_source_type(source_type)
        ),
        "duplicate_collapsed": bool(data.get("duplicate_collapsed") or ranking.get("duplicate_collapsed")),
        "top_ranking_signals": list(data.get("top_ranking_signals") or ranking.get("top_ranking_signals") or []),
        "ranking_signals": ranking,
    }


async def handle_tool(name: str, arguments: dict[str, Any], ctx: dict[str, Any]):
    emit = ctx["emit"]
    searcher = ctx["searcher"]
    sqlite_db = ctx["sqlite_db"]
    to_int = ctx["to_int"]
    to_float = ctx["to_float"]
    to_bool = ctx["to_bool"]
    normalize_source = ctx["normalize_source"]
    status_ok = ctx["MCP_TOOL_STATUS_OK"]
    status_failed = ctx["MCP_TOOL_STATUS_FAILED"]

    if name == "search_knowledge":
        query = str(arguments.get("query", "")).strip()
        if not query:
            return emit(status_failed, {"error": "query가 필요합니다."}, status_message="query required")
        top_k = to_int(arguments.get("top_k"), 5, minimum=1, maximum=100)
        source = normalize_source(arguments.get("source"))
        mode = str(arguments.get("mode", "hybrid")).strip().lower()
        alpha = to_float(arguments.get("alpha"), 0.7, minimum=0.0, maximum=1.0)
        results = list(
            searcher.search(
                query,
                top_k=top_k,
                source_type=source,
                retrieval_mode=mode,
                alpha=alpha,
                expand_parent_context=True,
            )
        )
        payload = {
            "query": query,
            "result_count": len(results),
            "runtimeDiagnostics": build_runtime_diagnostics(getattr(searcher, "config", None), searcher=searcher),
            "results": [
                {
                    "title": r.metadata.get("title", "Untitled"),
                    "source_type": r.metadata.get("source_type", ""),
                    "score": r.score,
                    "semantic_score": getattr(r, "semantic_score", 0),
                    "lexical_score": getattr(r, "lexical_score", 0),
                    "mode": getattr(r, "retrieval_mode", mode),
                    "parent_id": r.metadata.get("resolved_parent_id", ""),
                    "parent_label": r.metadata.get("resolved_parent_label", ""),
                    "parent_chunk_span": r.metadata.get("resolved_parent_chunk_span", ""),
                    **_ranking_fields(getattr(r, "lexical_extras", None), source_type=r.metadata.get("source_type", "")),
                    "document": (getattr(r, "document", "") or "")[:200],
                }
                for r in results
            ],
        }
        return emit(status_ok, payload)

    if name == "ask_knowledge":
        question = str(arguments.get("question", "")).strip()
        if not question:
            return emit(status_failed, {"error": "question이 필요합니다."}, status_message="question required")
        top_k = to_int(arguments.get("top_k"), 5, minimum=1, maximum=100)
        min_score = to_float(arguments.get("min_score"), 0.3, minimum=0.0, maximum=1.0)
        source = normalize_source(arguments.get("source"))
        mode = str(arguments.get("mode", "hybrid")).strip().lower()
        alpha = to_float(arguments.get("alpha"), 0.7, minimum=0.0, maximum=1.0)
        result = searcher.generate_answer(
            question,
            top_k=top_k,
            min_score=min_score,
            source_type=source,
            retrieval_mode=mode,
            alpha=alpha,
            allow_external=False,
        )
        payload = result if isinstance(result, dict) else {"answer": str(result), "sources": []}
        payload = {
            "question": question,
            "allowExternal": False,
            "allow_external": False,
            "runtimeDiagnostics": build_runtime_diagnostics(getattr(searcher, "config", None), searcher=searcher),
            **payload,
        }
        return emit(status_ok, payload, artifact=payload)

    if name == "build_task_context":
        goal = str(arguments.get("goal", "")).strip()
        if not goal:
            return emit(status_failed, {"error": "goal이 필요합니다."}, status_message="goal required")
        payload = build_task_context(
            searcher,
            goal=goal,
            repo_path=str(arguments.get("repo_path", "") or "").strip() or None,
            include_workspace=to_bool(arguments.get("include_workspace"), default=True),
            include_vault=to_bool(arguments.get("include_vault"), default=True),
            include_papers=to_bool(arguments.get("include_papers"), default=True),
            include_web=to_bool(arguments.get("include_web"), default=True),
            max_workspace_files=to_int(arguments.get("max_workspace_files"), 8, minimum=1, maximum=32) or 8,
            max_knowledge_hits=to_int(arguments.get("max_knowledge_hits"), 5, minimum=1, maximum=20) or 5,
        )
        return emit(status_ok, payload, artifact=payload)

    if name == "get_hub_stats":
        sql_stats = sqlite_db.get_stats()
        vec_stats = searcher.database.get_stats()
        payload = {
            "notes": sql_stats["notes"],
            "papers": sql_stats["papers"],
            "tags": sql_stats["tags"],
            "links": sql_stats["links"],
            "vector_documents": vec_stats["total_documents"],
            "collection_name": vec_stats["collection_name"],
        }
        return emit(status_ok, payload)

    return None
