from __future__ import annotations

from typing import Any

from knowledge_hub.application.ask_graph import run_ask_graph
from knowledge_hub.application.notebook_workbench import notebook_workbench_chat, notebook_workbench_search
from knowledge_hub.application.transformations import list_transformations, preview_transformation, run_transformation


async def handle_tool(name: str, arguments: dict[str, Any], ctx: dict[str, Any]):
    emit = ctx["emit"]
    searcher = ctx["searcher"]
    sqlite_db = ctx["sqlite_db"]
    config = ctx["config"]
    to_bool = ctx["to_bool"]
    to_int = ctx["to_int"]
    to_float = ctx["to_float"]
    normalize_source = ctx["normalize_source"]

    status_ok = ctx["MCP_TOOL_STATUS_OK"]
    status_failed = ctx["MCP_TOOL_STATUS_FAILED"]

    if name == "transform_list":
        payload = list_transformations()
        return emit(status_ok, payload, artifact=payload)

    if name in {"transform_preview", "transform_run"}:
        transformation_id = str(arguments.get("transformation_id", "")).strip()
        query = str(arguments.get("query", "")).strip()
        if not transformation_id:
            return emit(status_failed, {"error": "transformation_id가 필요합니다."}, status_message="transformation_id required")
        if not query:
            return emit(status_failed, {"error": "query가 필요합니다."}, status_message="query required")
        kwargs = {
            "searcher": searcher,
            "sqlite_db": sqlite_db,
            "transformation_id": transformation_id,
            "query": query,
            "repo_path": str(arguments.get("repo_path", "") or "").strip() or None,
            "include_workspace": to_bool(arguments.get("include_workspace"), default=False),
            "include_vault": to_bool(arguments.get("include_vault"), default=True),
            "include_papers": to_bool(arguments.get("include_papers"), default=True),
            "include_web": to_bool(arguments.get("include_web"), default=True),
            "max_sources": to_int(arguments.get("max_sources"), 6, minimum=1, maximum=20) or 6,
        }
        if name == "transform_preview":
            payload = preview_transformation(**kwargs)
        else:
            payload = run_transformation(
                **kwargs,
                llm=getattr(searcher, "llm", None),
                config=config,
                dry_run=to_bool(arguments.get("dry_run"), default=False),
            )
        tool_status = status_ok if str(payload.get("status")) == "ok" else status_failed
        return emit(tool_status, payload, artifact=payload)

    if name == "ask_graph":
        question = str(arguments.get("question", "")).strip()
        if not question:
            return emit(status_failed, {"error": "question이 필요합니다."}, status_message="question required")
        payload = run_ask_graph(
            searcher,
            question=question,
            source=normalize_source(arguments.get("source")),
            mode=str(arguments.get("mode", "hybrid")).strip().lower() or "hybrid",
            alpha=to_float(arguments.get("alpha"), 0.7, minimum=0.0, maximum=1.0) or 0.7,
            max_steps=to_int(arguments.get("max_steps"), 4, minimum=2, maximum=4) or 4,
            top_k=to_int(arguments.get("top_k"), 5, minimum=1, maximum=20) or 5,
            return_trace=to_bool(arguments.get("return_trace"), default=True),
        )
        return emit(status_ok, payload, artifact=payload)

    if name == "notebook_workbench_search":
        query = str(arguments.get("query", "")).strip()
        if not query:
            return emit(status_failed, {"error": "query가 필요합니다."}, status_message="query required")
        payload = notebook_workbench_search(
            searcher,
            sqlite_db=sqlite_db,
            topic=str(arguments.get("topic", "")).strip() or None,
            query=query,
            selected_source_ids=[str(item) for item in list(arguments.get("selected_source_ids") or [])],
            selected_source_context_modes=dict(arguments.get("selected_source_context_modes") or {}),
            include_vault=to_bool(arguments.get("include_vault"), default=True),
            include_papers=to_bool(arguments.get("include_papers"), default=True),
            include_web=to_bool(arguments.get("include_web"), default=True),
            top_k=to_int(arguments.get("top_k"), 5, minimum=1, maximum=20) or 5,
            mode=str(arguments.get("mode", "hybrid")).strip().lower() or "hybrid",
            alpha=to_float(arguments.get("alpha"), 0.7, minimum=0.0, maximum=1.0) or 0.7,
        )
        return emit(status_ok, payload, artifact=payload)

    if name == "notebook_workbench_chat":
        message = str(arguments.get("message", "")).strip()
        if not message:
            return emit(status_failed, {"error": "message가 필요합니다."}, status_message="message required")
        payload = notebook_workbench_chat(
            searcher,
            sqlite_db=sqlite_db,
            config=config,
            topic=str(arguments.get("topic", "")).strip() or None,
            message=message,
            intent=str(arguments.get("intent", "qa")).strip().lower() or "qa",
            selected_source_ids=[str(item) for item in list(arguments.get("selected_source_ids") or [])],
            selected_source_context_modes=dict(arguments.get("selected_source_context_modes") or {}),
            include_vault=to_bool(arguments.get("include_vault"), default=True),
            include_papers=to_bool(arguments.get("include_papers"), default=True),
            include_web=to_bool(arguments.get("include_web"), default=True),
            top_k=to_int(arguments.get("top_k"), 5, minimum=1, maximum=20) or 5,
            mode=str(arguments.get("mode", "hybrid")).strip().lower() or "hybrid",
            alpha=to_float(arguments.get("alpha"), 0.7, minimum=0.0, maximum=1.0) or 0.7,
        )
        tool_status = status_ok if str(payload.get("status")) == "ok" else status_failed
        return emit(tool_status, payload, artifact=payload)

    return None
