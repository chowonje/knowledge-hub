from __future__ import annotations

import inspect
from typing import Any

from knowledge_hub.application.ask_contracts import ensure_ask_contract_payload, external_policy_contract
from knowledge_hub.application.related_notes import build_related_note_suggestions
from knowledge_hub.application.runtime_diagnostics import build_runtime_diagnostics
from knowledge_hub.application.rag_reports import build_rag_ops_report
from knowledge_hub.application.task_context import build_task_context
from knowledge_hub.ai.retrieval_fit import normalize_source_type
from knowledge_hub.knowledge.graph_signals import analyze_graph_query


def _generate_answer_compat(searcher: Any, query: str, **kwargs: Any):
    generate_answer = getattr(searcher, "generate_answer")
    try:
        signature = inspect.signature(generate_answer)
    except (TypeError, ValueError):
        signature = None

    supported_kwargs = dict(kwargs)
    if signature is not None:
        parameters = signature.parameters
        if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            supported_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    try:
        return generate_answer(query, **supported_kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        fallback_kwargs = dict(supported_kwargs)
        fallback_kwargs.pop("memory_route_mode", None)
        return generate_answer(query, **fallback_kwargs)


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


def _runtime_diagnostics(searcher: Any, *, searcher_error: str = "") -> dict[str, Any]:
    return build_runtime_diagnostics(getattr(searcher, "config", None), searcher=searcher, searcher_error=searcher_error)


def _graph_query_signal(searcher: Any, query: str) -> dict[str, Any]:
    repository = getattr(searcher, "sqlite_db", None)
    if repository is None:
        return {}
    try:
        return analyze_graph_query(query, repository).to_dict()
    except Exception:
        return {}


async def handle_tool(name: str, arguments: dict[str, Any], ctx: dict[str, Any]):
    emit = ctx["emit"]
    searcher = ctx["searcher"]
    sqlite_db = ctx["sqlite_db"]
    normalize_source = ctx["normalize_source"]
    to_int = ctx["to_int"]
    to_float = ctx["to_float"]

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
        results = searcher.search(
            query,
            top_k=top_k,
            source_type=source,
            retrieval_mode=mode,
            alpha=alpha,
            expand_parent_context=True,
        )
        runtime_diagnostics = _runtime_diagnostics(searcher)
        graph_query_signal = _graph_query_signal(searcher, query)
        if not results:
            return emit(
                status_ok,
                {
                    "query": query,
                    "result_count": 0,
                    "results": [],
                    "runtimeDiagnostics": runtime_diagnostics,
                    "graph_query_signal": graph_query_signal,
                },
                status_message="no result",
            )
        payload = {
            "query": query,
            "result_count": len(results),
            "runtimeDiagnostics": runtime_diagnostics,
            "graph_query_signal": graph_query_signal,
            "related_notes": build_related_note_suggestions(results, query=query, limit=5),
            "results": [
                {
                    "title": r.metadata.get("title", "Untitled"),
                    "source_type": r.metadata.get("source_type", ""),
                    "score": r.score,
                    "semantic_score": r.semantic_score,
                    "lexical_score": r.lexical_score,
                    "mode": r.retrieval_mode,
                    "parent_id": r.metadata.get("resolved_parent_id", ""),
                    "parent_label": r.metadata.get("resolved_parent_label", ""),
                    "parent_chunk_span": r.metadata.get("resolved_parent_chunk_span", ""),
                    **_ranking_fields(getattr(r, "lexical_extras", None), source_type=r.metadata.get("source_type", "")),
                    "document": (r.document or "")[:200],
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
        memory_route_mode = str(arguments.get("memory_route_mode", "off")).strip().lower() or "off"
        paper_memory_mode = str(arguments.get("paper_memory_mode", "off")).strip().lower() or "off"
        result = _generate_answer_compat(
            searcher,
            question,
            top_k=top_k,
            min_score=min_score,
            source_type=source,
            retrieval_mode=mode,
            alpha=alpha,
            allow_external=False,
            memory_route_mode=memory_route_mode,
            paper_memory_mode=paper_memory_mode,
        )
        external_policy = external_policy_contract(
            surface="mcp",
            allow_external=False,
            requested=False,
            decision_source="mcp_default_local_only",
        )
        normalized_result = ensure_ask_contract_payload(
            result if isinstance(result, dict) else {"answer": str(result), "sources": []},
            source_type=source,
            memory_route_mode=memory_route_mode,
            paper_memory_mode=paper_memory_mode,
            external_policy=external_policy,
        )
        sources = [
            {
                "title": s.get("title", ""),
                "source_type": s.get("source_type", ""),
                "score": s.get("score", 0),
                "semantic_score": s.get("semantic_score", 0),
                "lexical_score": s.get("lexical_score", 0),
                "mode": s.get("retrieval_mode", mode),
                "parent_id": s.get("parent_id", ""),
                "parent_label": s.get("parent_label", ""),
                "parent_chunk_span": s.get("parent_chunk_span", ""),
                **_ranking_fields(s, source_type=s.get("source_type", "")),
            }
            for s in normalized_result.get("sources", [])
            if isinstance(s, dict)
        ]
        runtime_diagnostics = _runtime_diagnostics(searcher)
        graph_query_signal = _graph_query_signal(searcher, question)
        payload = {
            "question": question,
            "answer": normalized_result.get("answer"),
            "allowExternal": normalized_result.get("allowExternal", False),
            "allow_external": bool(normalized_result.get("allowExternal", False)),
            "externalPolicy": normalized_result.get("externalPolicy", {}),
            "external_policy": normalized_result.get("externalPolicy", {}),
            "sources": sources,
            "evidence": normalized_result.get("evidence", sources),
            "citations": normalized_result.get("citations", []),
            "answer_generation": normalized_result.get("answerGeneration", {}),
            "answer_signals": normalized_result.get("answerSignals", {}),
            "answer_verification": normalized_result.get("answerVerification", {}),
            "answer_rewrite": normalized_result.get("answerRewrite", {}),
            "claim_verification": normalized_result.get("claimVerification", []),
            "claim_consensus": normalized_result.get("claimConsensus", {}),
            "initial_answer_verification": normalized_result.get("initialAnswerVerification", {}),
            "warnings": normalized_result.get("warnings", []),
            "related_clusters": normalized_result.get("related_clusters", []),
            "ontology_entities": normalized_result.get("ontology_entities", []),
            "supporting_beliefs": normalized_result.get("supporting_beliefs", []),
            "contradicting_beliefs": normalized_result.get("contradicting_beliefs", []),
            "belief_updates_suggested": normalized_result.get("belief_updates_suggested", []),
            "memory_route": normalized_result.get("memoryRoute", {}),
            "memory_prefilter": normalized_result.get("memoryPrefilter", {}),
            "memory_relations_used": normalized_result.get("memoryRelationsUsed", []),
            "temporal_signals": normalized_result.get("temporalSignals", {}),
            "paper_memory_prefilter": normalized_result.get("paperMemoryPrefilter", {}),
            "paper_answer_scope": normalized_result.get("paperAnswerScope", {}),
            "evidence_budget": normalized_result.get("evidenceBudget", {}),
            "graph_query_signal": graph_query_signal,
            "runtimeDiagnostics": runtime_diagnostics,
        }
        artifact = {
            "question": question,
            "answer": normalized_result.get("answer"),
            "allowExternal": normalized_result.get("allowExternal", False),
            "allow_external": bool(normalized_result.get("allowExternal", False)),
            "externalPolicy": normalized_result.get("externalPolicy", {}),
            "external_policy": normalized_result.get("externalPolicy", {}),
            "sources": sources,
            "evidence": normalized_result.get("evidence", sources),
            "citations": normalized_result.get("citations", []),
            "answer_generation": normalized_result.get("answerGeneration", {}),
            "answer_signals": normalized_result.get("answerSignals", {}),
            "answer_verification": normalized_result.get("answerVerification", {}),
            "answer_rewrite": normalized_result.get("answerRewrite", {}),
            "claim_verification": normalized_result.get("claimVerification", []),
            "claim_consensus": normalized_result.get("claimConsensus", {}),
            "initial_answer_verification": normalized_result.get("initialAnswerVerification", {}),
            "warnings": normalized_result.get("warnings", []),
            "related_clusters": normalized_result.get("related_clusters", []),
            "ontology_entities": normalized_result.get("ontology_entities", []),
            "supporting_beliefs": normalized_result.get("supporting_beliefs", []),
            "contradicting_beliefs": normalized_result.get("contradicting_beliefs", []),
            "belief_updates_suggested": normalized_result.get("belief_updates_suggested", []),
            "memory_route": normalized_result.get("memoryRoute", {}),
            "memory_prefilter": normalized_result.get("memoryPrefilter", {}),
            "memory_relations_used": normalized_result.get("memoryRelationsUsed", []),
            "temporal_signals": normalized_result.get("temporalSignals", {}),
            "paper_memory_prefilter": normalized_result.get("paperMemoryPrefilter", {}),
            "paper_answer_scope": normalized_result.get("paperAnswerScope", {}),
            "evidence_budget": normalized_result.get("evidenceBudget", {}),
            "graph_query_signal": graph_query_signal,
            "runtimeDiagnostics": runtime_diagnostics,
        }
        return emit(status_ok, payload, artifact=artifact)

    if name == "build_task_context":
        goal = str(arguments.get("goal", "")).strip()
        if not goal:
            return emit(status_failed, {"error": "goal이 필요합니다."}, status_message="goal required")
        payload = build_task_context(
            searcher,
            goal=goal,
            repo_path=str(arguments.get("repo_path", "") or "").strip() or None,
            include_workspace=ctx["to_bool"](arguments.get("include_workspace"), default=True),
            include_vault=ctx["to_bool"](arguments.get("include_vault"), default=True),
            include_papers=ctx["to_bool"](arguments.get("include_papers"), default=True),
            include_web=ctx["to_bool"](arguments.get("include_web"), default=True),
            max_workspace_files=to_int(arguments.get("max_workspace_files"), 8, minimum=1, maximum=32) or 8,
            max_knowledge_hits=to_int(arguments.get("max_knowledge_hits"), 5, minimum=1, maximum=20) or 5,
        )
        return emit(status_ok, payload, artifact=payload)

    if name == "rag_report":
        limit = to_int(arguments.get("limit"), 100, minimum=1, maximum=500)
        days = to_int(arguments.get("days"), 7, minimum=0, maximum=365)
        payload = build_rag_ops_report(searcher.sqlite_db, limit=limit or 100, days=days or 0)
        tool_status = status_ok if str(payload.get("status")) == "ok" else status_failed
        return emit(tool_status, payload)

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
