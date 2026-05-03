from __future__ import annotations

import inspect
from typing import Any

from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.memory_payloads import card_payload
from knowledge_hub.papers.memory_retriever import PaperMemoryRetriever
from knowledge_hub.papers.topic_synthesis import PaperTopicSynthesisService


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


def _pick_paper_candidate(items: list[dict[str, Any]], query: str) -> dict[str, Any] | None:
    if not items:
        return None
    lowered = query.strip().lower()
    if not lowered:
        return items[0]
    for item in items:
        title = str(item.get("title", "")).strip().lower()
        paper_id = str(item.get("arxiv_id", item.get("paper_id", ""))).strip().lower()
        if title == lowered or paper_id == lowered:
            return item
    for item in items:
        title = str(item.get("title", "")).strip().lower()
        if lowered and lowered in title:
            return item
    return items[0]


async def handle_tool(name: str, arguments: dict[str, Any], ctx: dict[str, Any]):
    emit = ctx["emit"]
    config = ctx["config"]
    sqlite_db = ctx["sqlite_db"]
    searcher = ctx["searcher"]
    to_bool = ctx["to_bool"]
    to_int = ctx["to_int"]
    run_async_tool = ctx["run_async_tool"]
    request_echo = ctx["request_echo"]

    status_ok = ctx["MCP_TOOL_STATUS_OK"]
    status_failed = ctx["MCP_TOOL_STATUS_FAILED"]
    status_queued = ctx["MCP_TOOL_STATUS_QUEUED"]

    if name == "build_paper_memory":
        paper_id = str(arguments.get("paper_id", "")).strip()
        if not paper_id:
            return emit(status_failed, {"error": "paper_id가 필요합니다."}, status_message="paper_id required")
        try:
            item = PaperMemoryBuilder(sqlite_db).build_and_store(paper_id=paper_id)
        except Exception as error:
            return emit(status_failed, {"error": str(error)}, status_message="build failed")
        payload = {
            "schema": "knowledge-hub.paper-memory.build.result.v1",
            "status": "ok",
            "mode": "single",
            "count": 1,
            "items": [card_payload(item)] if item else [],
            "warnings": [],
        }
        return emit(status_ok, payload, artifact=payload)

    if name == "get_paper_memory_card":
        paper_id = str(arguments.get("paper_id", "")).strip()
        if not paper_id:
            return emit(status_failed, {"error": "paper_id가 필요합니다."}, status_message="paper_id required")
        item = PaperMemoryRetriever(sqlite_db).get(paper_id)
        payload = {
            "schema": "knowledge-hub.paper-memory.card.result.v1",
            "status": "ok" if item else "failed",
            "item": item or {},
            "warnings": [] if item else [f"paper memory card not found: {paper_id}"],
        }
        return emit(status_ok if item else status_failed, payload, artifact=payload)

    if name == "search_paper_memory":
        query = str(arguments.get("query", "")).strip()
        if not query:
            return emit(status_failed, {"error": "query가 필요합니다."}, status_message="query required")
        limit = to_int(arguments.get("limit"), 10, minimum=1, maximum=100)
        items = PaperMemoryRetriever(sqlite_db).search(query, limit=limit or 10)
        payload = {
            "schema": "knowledge-hub.paper-memory.search.result.v1",
            "status": "ok",
            "query": query,
            "count": len(items),
            "items": items,
            "warnings": [],
        }
        return emit(status_ok, payload, artifact=payload)

    if name == "search_papers":
        query = str(arguments.get("query", "")).strip()
        if not query:
            return emit(status_failed, {"error": "query가 필요합니다."}, status_message="query required")
        papers = sqlite_db.search_papers(query)
        if not papers:
            return emit(status_ok, {"query": query, "paper_count": 0}, status_message="no result")
        payload = {
            "query": query,
            "paper_count": len(papers),
            "items": papers,
        }
        return emit(status_ok, payload)

    if name == "index_paper_keywords":
        arxiv_id = arguments.get("arxiv_id")
        if arxiv_id:
            arxiv_id = str(arxiv_id).strip() or None
        top_k = to_int(arguments.get("top_k"), 12, minimum=1)
        max_links_per_keyword = to_int(arguments.get("max_links_per_keyword"), 5, minimum=0)
        dry_run = to_bool(arguments.get("dry_run"), default=False)

        async def _runner() -> dict[str, Any]:
            from knowledge_hub.papers.manager import PaperManager

            manager = PaperManager(
                config=config,
                vector_db=searcher.database,
                sqlite_db=sqlite_db,
                embedder=searcher.embedder,
            )
            result = manager.sync_translated_keywords(
                arxiv_id=arxiv_id,
                top_k=top_k,
                max_links_per_keyword=max_links_per_keyword,
                dry_run=dry_run,
            )
            return {
                "status": "ok",
                "mode": result["mode"],
                "processed": result["processed"],
                "updated": result["updated"],
                "skipped": result["skipped"],
                "target": "all" if arxiv_id is None else arxiv_id,
                "items": result["items"],
            }

        job_id, queued = await run_async_tool(name=name, request_echo=request_echo, sync_job=_runner)
        return emit(status_queued, queued["payload"], job_id=job_id, status_message="queued")

    if name == "search_authors":
        from knowledge_hub.papers.discoverer import search_authors as _search_authors

        query = str(arguments.get("query", "")).strip()
        if not query:
            return emit(status_failed, {"error": "query가 필요합니다."}, status_message="query required")
        limit = to_int(arguments.get("limit"), 10, minimum=1)
        authors = _search_authors(query, limit=limit)
        if not authors:
            return emit(status_ok, {"query": query, "count": 0, "items": []}, status_message="no result")
        payload = [
            {
                "name": a.name,
                "author_id": a.author_id,
                "paper_count": a.paper_count,
                "citation_count": a.citation_count,
                "h_index": a.h_index,
                "affiliations": a.affiliations,
            }
            for a in authors
        ]
        return emit(status_ok, {"query": query, "items": payload})

    if name == "get_author_papers":
        from knowledge_hub.papers.discoverer import get_author_papers as _get_author_papers

        author_id = str(arguments.get("author_id", "")).strip()
        if not author_id:
            return emit(status_failed, {"error": "author_id가 필요합니다."}, status_message="author_id required")
        limit = to_int(arguments.get("limit"), 20, minimum=1)
        author, papers = _get_author_papers(author_id, limit=limit)
        payload = {
            "author": {
                "name": author.name if author else "",
                "author_id": author_id,
                "paper_count": author.paper_count if author else 0,
                "citation_count": author.citation_count if author else 0,
                "h_index": author.h_index if author else 0,
            }
        }
        if papers:
            payload["papers"] = [
                {
                    "title": p.title,
                    "year": p.year,
                    "citation_count": p.citation_count,
                    "fields_of_study": p.fields_of_study,
                    "arxiv_id": p.arxiv_id or "-",
                }
                for p in papers
            ]
        return emit(status_ok, payload)

    if name == "get_paper_detail":
        from knowledge_hub.papers.discoverer import get_paper_detail as _get_paper_detail

        paper_id = str(arguments.get("paper_id", "")).strip()
        if not paper_id:
            return emit(status_failed, {"error": "paper_id가 필요합니다."}, status_message="paper_id required")
        data = _get_paper_detail(paper_id)
        if not data:
            return emit(status_failed, {"error": f"논문 '{paper_id}'를 찾을 수 없습니다."}, status_message="not found")
        return emit(status_ok, data)

    if name == "paper_lookup_and_summarize":
        paper_id = str(arguments.get("paper_id", "")).strip()
        query = str(arguments.get("query", "")).strip()
        if not paper_id and not query:
            return emit(
                status_failed,
                {"error": "paper_id 또는 query 중 하나가 필요합니다."},
                status_message="paper_id or query required",
            )

        top_k = to_int(arguments.get("top_k"), 5, minimum=1, maximum=20)
        min_score = float(arguments.get("min_score", 0.3) or 0.3)
        mode = str(arguments.get("mode", "hybrid")).strip().lower() or "hybrid"
        if mode not in {"semantic", "keyword", "hybrid"}:
            mode = "hybrid"
        alpha = float(arguments.get("alpha", 0.7) or 0.7)
        alpha = min(1.0, max(0.0, alpha))
        memory_route_mode = str(arguments.get("memory_route_mode", "off")).strip().lower() or "off"
        paper_memory_mode = str(arguments.get("paper_memory_mode", "off")).strip().lower() or "off"

        selected_item: dict[str, Any] | None = None
        selected_paper_id = paper_id
        if not selected_paper_id:
            papers = sqlite_db.search_papers(query)
            if not papers:
                return emit(
                    status_ok,
                    {"query": query, "matched": False, "paper_count": 0, "items": []},
                    status_message="no result",
                )
            selected_item = _pick_paper_candidate(papers, query)
            if selected_item is None:
                return emit(
                    status_ok,
                    {"query": query, "matched": False, "paper_count": len(papers), "items": papers},
                    status_message="no result",
                )
            selected_paper_id = str(selected_item.get("arxiv_id", selected_item.get("paper_id", ""))).strip()

        detail: dict[str, Any] = {}
        if selected_paper_id:
            try:
                from knowledge_hub.papers.discoverer import get_paper_detail as _get_paper_detail

                maybe_detail = _get_paper_detail(selected_paper_id)
                if isinstance(maybe_detail, dict):
                    detail = maybe_detail
            except Exception:
                detail = {}

        if selected_item is None:
            selected_item = {
                "paper_id": selected_paper_id,
                "arxiv_id": selected_paper_id,
                "title": detail.get("title", ""),
            }

        lookup_query = str(
            detail.get("title")
            or selected_item.get("title")
            or selected_paper_id
            or query
        ).strip()

        rag = _generate_answer_compat(
            searcher,
            lookup_query,
            top_k=top_k,
            min_score=min_score,
            source_type="paper",
            retrieval_mode=mode,
            alpha=alpha,
            allow_external=False,
            memory_route_mode=memory_route_mode,
            paper_memory_mode=paper_memory_mode,
            metadata_filter={"source_type": "paper", "arxiv_id": selected_paper_id} if selected_paper_id else None,
        )
        summary = rag if isinstance(rag, dict) else {"answer": str(rag), "sources": []}
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
            }
            for s in summary.get("sources", [])
            if isinstance(s, dict)
        ]
        payload = {
            "matched": True,
            "query": query or lookup_query,
            "paper": {
                "paper_id": selected_paper_id or str(detail.get("paper_id", "")).strip(),
                "arxiv_id": str(detail.get("arxiv_id", selected_item.get("arxiv_id", ""))).strip(),
                "title": str(detail.get("title", selected_item.get("title", ""))).strip(),
                "authors": detail.get("authors"),
                "year": detail.get("year"),
                "abstract": detail.get("abstract"),
                "url": detail.get("url"),
                "citation_count": detail.get("citation_count"),
                "fields_of_study": detail.get("fields_of_study"),
            },
            "summary": {
                "question": lookup_query,
                "answer": summary.get("answer"),
                "sources": sources,
                "classification": "P2",
                "external_calls": False,
                "memory_route": summary.get("memoryRoute", {}),
                "memory_prefilter": summary.get("memoryPrefilter", {}),
                "memory_relations_used": summary.get("memoryRelationsUsed", []),
                "temporal_signals": summary.get("temporalSignals", {}),
                "paper_memory_prefilter": summary.get("paperMemoryPrefilter", {}),
            },
        }
        artifact = {
            "paper": payload["paper"],
            "summary": payload["summary"],
            "classification": "P2",
        }
        return emit(status_ok, payload, artifact=artifact)

    if name == "paper_topic_synthesize":
        query = str(arguments.get("query", "")).strip()
        if not query:
            return emit(status_failed, {"error": "query가 필요합니다."}, status_message="query required")
        source_mode = str(arguments.get("source_mode", "local")).strip().lower() or "local"
        if source_mode not in {"local", "discover", "hybrid"}:
            source_mode = "local"
        candidate_limit = to_int(arguments.get("candidate_limit"), 12, minimum=1, maximum=50) or 12
        selected_limit = to_int(arguments.get("selected_limit"), 6, minimum=1, maximum=20) or 6
        top_k = to_int(arguments.get("top_k"), 8, minimum=1, maximum=30) or 8
        retrieval_mode = str(arguments.get("mode", "hybrid")).strip().lower() or "hybrid"
        if retrieval_mode not in {"semantic", "keyword", "hybrid"}:
            retrieval_mode = "hybrid"
        try:
            alpha = float(arguments.get("alpha", 0.7) or 0.7)
        except Exception:
            alpha = 0.7
        alpha = min(1.0, max(0.0, alpha))
        allow_external = to_bool(arguments.get("allow_external"), default=False)
        llm_mode = str(arguments.get("llm_mode", "auto")).strip().lower() or "auto"
        if llm_mode not in {"auto", "local", "mini", "strong"}:
            llm_mode = "auto"
        provider_override = str(arguments.get("provider", "")).strip() or None
        model_override = str(arguments.get("model", "")).strip() or None
        payload = PaperTopicSynthesisService(
            sqlite_db=sqlite_db,
            searcher=searcher,
            config=config,
        ).synthesize(
            query=query,
            source_mode=source_mode,
            candidate_limit=candidate_limit,
            selected_limit=selected_limit,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            allow_external=allow_external,
            llm_mode=llm_mode,
            provider_override=provider_override,
            model_override=model_override,
        )
        return emit(status_ok, payload, artifact=payload)

    if name == "get_paper_citations":
        from knowledge_hub.papers.discoverer import get_paper_citations as _get_citations

        paper_id = str(arguments.get("paper_id", "")).strip()
        if not paper_id:
            return emit(status_failed, {"error": "paper_id가 필요합니다."}, status_message="paper_id required")
        limit = to_int(arguments.get("limit"), 20, minimum=1)
        _, papers = _get_citations(paper_id, limit=limit)
        payload = {
            "paper_id": paper_id,
            "citations": papers,
        }
        return emit(status_ok, payload)

    if name == "get_paper_references":
        from knowledge_hub.papers.discoverer import get_paper_references as _get_refs

        paper_id = str(arguments.get("paper_id", "")).strip()
        if not paper_id:
            return emit(status_failed, {"error": "paper_id가 필요합니다."}, status_message="paper_id required")
        limit = to_int(arguments.get("limit"), 20, minimum=1)
        _, papers = _get_refs(paper_id, limit=limit)
        payload = {
            "paper_id": paper_id,
            "references": papers,
        }
        return emit(status_ok, payload)

    if name == "analyze_citation_network":
        from knowledge_hub.papers.discoverer import analyze_citation_network as _analyze

        paper_id = str(arguments.get("paper_id", "")).strip()
        if not paper_id:
            return emit(status_failed, {"error": "paper_id가 필요합니다."}, status_message="paper_id required")
        depth = to_int(arguments.get("depth"), 1, minimum=1)
        cit_limit = to_int(arguments.get("citations_limit"), 10, minimum=1)
        ref_limit = to_int(arguments.get("references_limit"), 10, minimum=1)
        payload = _analyze(paper_id, depth=depth, citations_limit=cit_limit, references_limit=ref_limit)
        return emit(status_ok, payload)

    if name == "batch_paper_lookup":
        from knowledge_hub.papers.discoverer import get_papers_batch as _batch

        paper_ids = arguments.get("paper_ids", [])
        if not paper_ids:
            return emit(status_failed, {"error": "paper_ids가 필요합니다."}, status_message="paper_ids required")
        payload = _batch(paper_ids)
        return emit(status_ok, {"items": payload})

    if name == "discover_and_ingest":
        topic = str(arguments.get("topic", "")).strip()
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        max_papers = to_int(arguments.get("max_papers"), 5, minimum=1)
        year_start = arguments.get("year_start")
        if year_start is not None:
            year_start = to_int(year_start, None)
        min_citations = to_int(arguments.get("min_citations"), 0)
        sort_by = str(arguments.get("sort_by", "relevance")).strip()
        if sort_by not in {"relevance", "citationCount"}:
            sort_by = "relevance"
        create_obsidian = to_bool(arguments.get("create_obsidian_note"), default=True)
        gen_summary = to_bool(arguments.get("generate_summary"), default=True)
        judge_enabled = to_bool(arguments.get("judge_enabled"), default=False)
        judge_threshold = float(arguments.get("judge_threshold", 0.62) or 0.62)
        judge_candidates = arguments.get("judge_candidates")
        if judge_candidates is not None:
            judge_candidates = to_int(judge_candidates, None, minimum=1)
        allow_external = to_bool(arguments.get("allow_external"), default=False)
        dry_run = to_bool(arguments.get("dry_run"), default=False)

        async def _runner() -> dict[str, Any]:
            from knowledge_hub.papers.manager import PaperManager

            manager = PaperManager(
                config=config,
                vector_db=searcher.database,
                sqlite_db=sqlite_db,
                embedder=searcher.embedder,
            )
            llm_instance = searcher.llm if gen_summary and not dry_run else None
            judge_llm_instance = searcher.llm if judge_enabled else None
            return manager.discover_and_ingest(
                topic=topic,
                max_papers=max_papers,
                year_start=year_start,
                min_citations=min_citations,
                sort_by=sort_by,
                create_obsidian_note=create_obsidian and not dry_run,
                generate_summary=gen_summary and not dry_run,
                llm=llm_instance,
                judge_enabled=judge_enabled,
                judge_threshold=judge_threshold,
                judge_candidates=judge_candidates,
                allow_external=allow_external,
                judge_llm=judge_llm_instance,
            )

        job_id, queued = await run_async_tool(name=name, request_echo=request_echo, sync_job=_runner)
        return emit(status_queued, queued["payload"], job_id=job_id, status_message="queued")

    if name == "run_paper_ingest_flow":
        topic = str(arguments.get("topic", "")).strip()
        if not topic:
            return emit(status_failed, {"error": "topic이 필요합니다."}, status_message="topic required")
        max_papers = to_int(arguments.get("max_papers"), 5, minimum=1)
        year_start = arguments.get("year_start")
        if year_start is not None:
            year_start = to_int(year_start, None)
        min_citations = to_int(arguments.get("min_citations"), 0)
        sort_by = str(arguments.get("sort_by", "relevance")).strip()
        if sort_by not in {"relevance", "citationCount"}:
            sort_by = "relevance"
        create_obsidian = to_bool(arguments.get("create_obsidian_note"), default=True)
        gen_summary = to_bool(arguments.get("generate_summary"), default=True)
        dry_run = to_bool(arguments.get("dry_run"), default=False)

        async def _runner() -> dict[str, Any]:
            from knowledge_hub.papers.manager import PaperManager

            manager = PaperManager(
                config=config,
                vector_db=searcher.database,
                sqlite_db=sqlite_db,
                embedder=searcher.embedder,
            )
            llm_instance = searcher.llm if gen_summary and not dry_run else None
            result = manager.discover_and_ingest(
                topic=topic,
                max_papers=max_papers,
                year_start=year_start,
                min_citations=min_citations,
                sort_by=sort_by,
                create_obsidian_note=create_obsidian and not dry_run,
                generate_summary=gen_summary and not dry_run,
                llm=llm_instance,
            )
            result["pipeline"] = "run_paper_ingest_flow"
            return result

        job_id, queued = await run_async_tool(name=name, request_echo=request_echo, sync_job=_runner)
        return emit(status_queued, queued["payload"], job_id=job_id, status_message="queued")

    if name == "check_paper_duplicate":
        arxiv_id = str(arguments.get("arxiv_id", "")).strip()
        if not arxiv_id:
            return emit(status_failed, {"error": "arxiv_id가 필요합니다."}, status_message="arxiv_id required")
        from knowledge_hub.papers.manager import PaperManager

        manager = PaperManager(
            config=config,
            vector_db=searcher.database,
            sqlite_db=sqlite_db,
            embedder=searcher.embedder,
        )
        is_dup, reason = manager.is_duplicate(arxiv_id)
        payload = {
            "is_duplicate": is_dup,
            "arxiv_id": arxiv_id,
            "reason": reason,
        }
        if is_dup:
            paper_info = sqlite_db.get_paper(arxiv_id)
            payload["title"] = paper_info["title"] if paper_info else "N/A"
        return emit(status_ok, payload)

    return None
