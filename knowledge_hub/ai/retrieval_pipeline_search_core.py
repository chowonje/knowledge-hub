from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from knowledge_hub.core.models import SearchResult
from knowledge_hub.knowledge.graph_signals import analyze_graph_query


@dataclass(frozen=True)
class RetrievalSearchCoreDeps:
    searcher: Any
    apply_feature_boosts_fn: Callable[[Any, list[SearchResult]], list[SearchResult]]
    expand_query_with_ontology_fn: Callable[[Any, str], list[str]]
    lexical_search_fn: Callable[..., list[SearchResult]]
    semantic_search_fn: Callable[..., list[SearchResult]]
    apply_query_fit_reranking_fn: Callable[..., list[SearchResult]]
    classify_query_intent_fn: Callable[[str], str]
    normalize_source_type_fn: Callable[[str | None], str | None]
    candidate_budgets_for_intent_fn: Callable[..., dict[str, int]]
    paper_definition_rescue_queries_fn: Callable[..., list[str]]
    build_expanded_queries_fn: Callable[..., list[str]]
    build_lexical_query_forms_fn: Callable[..., list[str]]
    family_query_limits_fn: Callable[..., dict[str, int]]
    merge_filter_dicts_fn: Callable[[dict[str, Any] | None, dict[str, Any] | None], dict[str, Any] | None]
    merge_source_filter_fn: Callable[[dict[str, Any] | None, str | None], dict[str, Any] | None]
    safe_float_fn: Callable[[Any, float], float]
    safe_int_fn: Callable[[Any, int], int]
    clean_text_fn: Callable[[str], str]
    result_id_fn: Callable[[SearchResult], str]
    retrieval_sort_key_fn: Callable[..., Any]
    preserve_parent_diversity_fn: Callable[[list[SearchResult]], list[SearchResult]]
    source_label_for_result_fn: Callable[[SearchResult], str]
    top_signal_items_fn: Callable[[list[dict[str, Any]], dict[str, float]], list[dict[str, Any]]]
    ensure_compare_result_coverage_fn: Callable[..., list[SearchResult]]


class RetrievalSearchCore:
    def __init__(self, deps: RetrievalSearchCoreDeps) -> None:
        self._deps = deps

    def graph_candidate_boost(
        self,
        results: list[SearchResult],
        *,
        graph_query_signal: dict[str, Any] | None,
    ) -> list[SearchResult]:
        deps = self._deps
        signal = dict(graph_query_signal or {})
        if not results or not bool(signal.get("is_graph_heavy")):
            return results
        diagnostics = dict(signal.get("diagnostics") or {})
        query_kind = str(diagnostics.get("query_kind") or "")
        hint_dicts = [dict(item) for item in list(signal.get("candidate_hints") or []) if isinstance(item, dict)]
        if not hint_dicts:
            return results

        for item in results:
            metadata = dict(item.metadata or {})
            document_preview = deps.clean_text_fn(getattr(item, "document", ""))[:1200].lower()
            alias_tokens = [
                deps.clean_text_fn(str(alias)).lower()
                for alias in list(metadata.get("aliases") or [])
                if deps.clean_text_fn(str(alias))
            ]
            related_concept_tokens = [
                deps.clean_text_fn(str(concept)).lower()
                for concept in list(metadata.get("related_concepts") or [])
                if deps.clean_text_fn(str(concept))
            ]
            keyword_tokens = [
                deps.clean_text_fn(str(keyword)).lower()
                for keyword in list(metadata.get("keywords") or [])
                if deps.clean_text_fn(str(keyword))
            ]
            identities = {
                deps.clean_text_fn(str(metadata.get("entity_id") or "")).lower(),
                deps.clean_text_fn(str(metadata.get("concept_id") or "")).lower(),
                deps.clean_text_fn(str(metadata.get("canonical_name") or "")).lower(),
                deps.clean_text_fn(str(metadata.get("title") or "")).lower(),
            }
            identities.update(alias_tokens)
            identities.update(related_concept_tokens)
            identities.update(keyword_tokens)
            identities = {token for token in identities if token}
            metadata_text = " ".join(
                [
                    str(metadata.get("title") or ""),
                    str(metadata.get("canonical_name") or ""),
                    str(metadata.get("file_path") or ""),
                    str(metadata.get("section_title") or ""),
                    str(metadata.get("section_path") or ""),
                    document_preview,
                    " ".join(alias_tokens),
                    " ".join(related_concept_tokens),
                    " ".join(keyword_tokens),
                ]
            ).lower()
            boost = 0.0
            matches: list[str] = []
            for hint in hint_dicts:
                canonical = deps.clean_text_fn(str(hint.get("canonical_name") or ""))
                canonical_norm = canonical.lower()
                aliases = [
                    deps.clean_text_fn(str(alias)).lower()
                    for alias in list(hint.get("aliases") or [])
                    if deps.clean_text_fn(str(alias))
                ]
                entity_id = deps.clean_text_fn(str(hint.get("entity_id") or "")).lower()
                hint_score = deps.safe_float_fn(hint.get("score"), 0.0)
                direct_match = bool(
                    (entity_id and entity_id in identities)
                    or (canonical_norm and canonical_norm in identities)
                    or (canonical_norm and canonical_norm in metadata_text)
                    or any(alias and alias in metadata_text for alias in aliases)
                )
                if not direct_match:
                    continue
                candidate_boost = min(0.055, 0.028 + (0.02 * min(1.0, hint_score)))
                if candidate_boost > boost:
                    boost = candidate_boost
                label = canonical or str(hint.get("entity_id") or "").strip()
                if label and label not in matches:
                    matches.append(label)
            if boost <= 0.0:
                continue
            item.score = max(0.0, min(1.0, deps.safe_float_fn(item.score, 0.0) + boost))
            extras = dict(item.lexical_extras or {})
            ranking_signals = dict(extras.get("ranking_signals") or {})
            raw_sort_score = deps.safe_float_fn(extras.get("retrieval_sort_score"), deps.safe_float_fn(item.score, 0.0)) + boost
            ranking_signals["graph_candidate_boost"] = round(boost, 6)
            ranking_signals["graph_query_kind"] = query_kind
            ranking_signals["graph_candidate_matches"] = list(matches[:3])
            ranking_signals["graph_candidate_reduction_applied"] = True
            ranking_signals["ranking_signal_total"] = round(raw_sort_score, 6)
            extras["graph_candidate_boost"] = round(boost, 6)
            extras["graph_candidate_matches"] = list(matches[:3])
            extras["graph_query_kind"] = query_kind
            extras["graph_candidate_reduction_applied"] = True
            extras["retrieval_sort_score"] = round(raw_sort_score, 6)
            extras["retrieval_adjusted_score"] = round(item.score, 6)
            extras["top_ranking_signals"] = deps.top_signal_items_fn(
                list(extras.get("top_ranking_signals") or []),
                {"graph_candidate_boost": boost},
            )
            ranking_signals["top_ranking_signals"] = list(extras["top_ranking_signals"])
            extras["ranking_signals"] = ranking_signals
            item.lexical_extras = extras
        return results

    def run_base_search(
        self,
        *,
        query: str,
        top_k: int,
        source_type: str | None,
        retrieval_mode: str,
        alpha: float,
        semantic_top_k: Optional[int] = None,
        lexical_top_k: Optional[int] = None,
        use_ontology_expansion: bool = True,
        metadata_filter: Optional[dict[str, Any]] = None,
        plan: Any | None = None,
    ) -> tuple[list[SearchResult], dict[str, Any], list[dict[str, Any]]]:
        deps = self._deps
        query_text = str(query or "").strip()
        if not query_text:
            return [], {}, []

        mode = (retrieval_mode or "hybrid").strip().lower()
        if mode not in {"semantic", "keyword", "hybrid"}:
            raise ValueError("retrieval_mode는 semantic|keyword|hybrid 중 하나여야 합니다.")

        filter_dict = dict((plan.metadata_filter_applied if plan else metadata_filter) or {})
        normalized_source = deps.normalize_source_type_fn(source_type)
        if normalized_source and "source_type" not in filter_dict:
            filter_dict["source_type"] = normalized_source
        if not filter_dict:
            filter_dict = None
        source_filtered = bool(deps.normalize_source_type_fn((filter_dict or {}).get("source_type")))

        query_intent = (plan.query_intent if plan else deps.classify_query_intent_fn(query_text)) or "general"
        paper_family = str((plan.paper_family if plan else "") or "").strip().lower()
        query_frame_payload = dict((plan.query_frame if plan else {}) or {})
        resolved_source_ids = [
            deps.clean_text_fn(item)
            for item in list(query_frame_payload.get("resolved_source_ids") or (plan.query_plan if plan else {}).get("resolved_paper_ids") or [])
            if deps.clean_text_fn(item)
        ]
        canonical_entity_ids = [
            deps.clean_text_fn(item)
            for item in list(query_frame_payload.get("canonical_entity_ids") or [])
            if deps.clean_text_fn(item)
        ]
        representative_narrowing = bool(
            normalized_source == "paper"
            and paper_family in {"concept_explainer", "paper_compare"}
            and resolved_source_ids
            and not bool(plan.resolved_source_scope_applied if plan else False)
        )
        source_budgets = dict(
            (plan.candidate_budgets if plan else deps.candidate_budgets_for_intent_fn(query_intent, top_k=top_k, source_scope=normalized_source))
            or {}
        )
        alpha = deps.safe_float_fn(alpha, 0.7)
        semantic_n = max(1, int(semantic_top_k or (top_k * 4 if mode == "hybrid" else top_k)))
        lexical_n = max(1, int(lexical_top_k or (top_k * 4 if mode == "hybrid" else top_k)))

        planned_terms = list((plan.query_plan if plan else {}).get("expanded_terms") or [])[:6]
        ontology_queries = [query_text]
        if use_ontology_expansion and getattr(deps.searcher, "sqlite_db", None):
            try:
                ontology_queries = deps.expand_query_with_ontology_fn(deps.searcher, query_text)
            except Exception:
                ontology_queries = [query_text]
        rescue_queries: list[str] = []
        if normalized_source == "paper" and paper_family == "concept_explainer":
            rescue_queries = deps.paper_definition_rescue_queries_fn(query_text, sqlite_db=getattr(deps.searcher, "sqlite_db", None))
        elif not normalized_source and query_intent == "definition":
            rescue_queries = deps.paper_definition_rescue_queries_fn(query_text, sqlite_db=getattr(deps.searcher, "sqlite_db", None))
        expanded_queries = deps.build_expanded_queries_fn(
            query_text=query_text,
            ontology_queries=ontology_queries,
            planned_terms=planned_terms,
            rescue_queries=rescue_queries,
            normalized_source=normalized_source,
            paper_family=paper_family,
        )
        lexical_query_forms = deps.build_lexical_query_forms_fn(
            query_text=query_text,
            query_frame_payload=query_frame_payload,
            planned_terms=planned_terms,
            rescue_queries=rescue_queries,
            normalized_source=normalized_source,
            paper_family=paper_family,
        )
        limits = deps.family_query_limits_fn(normalized_source=normalized_source, paper_family=paper_family)

        graph_query_signal: dict[str, Any] = {}
        if getattr(deps.searcher, "sqlite_db", None):
            try:
                try:
                    from knowledge_hub.ai import rag as rag_module

                    analyzer = getattr(rag_module, "analyze_graph_query", analyze_graph_query)
                except Exception:
                    analyzer = analyze_graph_query
                graph_query_signal = analyzer(query_text, getattr(deps.searcher, "sqlite_db", None)).to_dict()
            except Exception:
                graph_query_signal = {}

        semantic_hits: list[SearchResult] = []
        lexical_hits: list[SearchResult] = []
        candidate_counts: dict[str, dict[str, int]] = {
            source: {"budget": int(budget), "semantic": 0, "lexical": 0}
            for source, budget in source_budgets.items()
        }

        if mode in {"semantic", "hybrid"}:
            seen_semantic: set[str] = set()
            query_embeddings: dict[str, list[float]] = {}
            for eq in expanded_queries:
                query_embedding = query_embeddings.get(eq, [])
                if not query_embedding:
                    try:
                        query_embedding = getattr(deps.searcher, "embedder").embed_text(eq)
                    except Exception:
                        query_embedding = []
                    query_embeddings[eq] = list(query_embedding or [])
                if not query_embedding:
                    continue
                for hit in deps.semantic_search_fn(deps.searcher, query_embedding=query_embedding, top_k=semantic_n, filter_dict=filter_dict):
                    rid = deps.result_id_fn(hit)
                    if rid in seen_semantic:
                        continue
                    seen_semantic.add(rid)
                    semantic_hits.append(hit)
                if representative_narrowing:
                    for paper_id in resolved_source_ids[: int(limits["representative_scope_limit"])]:
                        scoped_filter = deps.merge_filter_dicts_fn(
                            filter_dict,
                            {"source_type": "paper", "arxiv_id": paper_id},
                        )
                        if scoped_filter is None:
                            continue
                        extra_hits = deps.semantic_search_fn(
                            deps.searcher,
                            query_embedding=query_embedding,
                            top_k=max(2, min(semantic_n, top_k * 2)),
                            filter_dict=scoped_filter,
                        )
                        for hit in extra_hits:
                            rid = deps.result_id_fn(hit)
                            if rid in seen_semantic:
                                continue
                            seen_semantic.add(rid)
                            semantic_hits.append(hit)
                if source_filtered:
                    continue
                for extra_source, budget in source_budgets.items():
                    source_filter = deps.merge_source_filter_fn(filter_dict, extra_source)
                    if source_filter is None or budget <= 0:
                        continue
                    extra_hits = deps.semantic_search_fn(
                        deps.searcher,
                        query_embedding=query_embedding,
                        top_k=max(2, int(budget)),
                        filter_dict=source_filter,
                    )
                    local_count = 0
                    for hit in extra_hits:
                        rid = deps.result_id_fn(hit)
                        if rid in seen_semantic:
                            continue
                        seen_semantic.add(rid)
                        semantic_hits.append(hit)
                        local_count += 1
                    candidate_counts.setdefault(extra_source, {"budget": int(budget), "semantic": 0, "lexical": 0})
                    candidate_counts[extra_source]["semantic"] += local_count

        if mode in {"keyword", "hybrid"}:
            seen_lexical: set[str] = set()
            for lexical_query in lexical_query_forms:
                for hit in deps.lexical_search_fn(deps.searcher, lexical_query, top_k=lexical_n, filter_dict=filter_dict):
                    rid = deps.result_id_fn(hit)
                    if rid in seen_lexical:
                        continue
                    seen_lexical.add(rid)
                    lexical_hits.append(hit)
            if representative_narrowing:
                for paper_id in resolved_source_ids[: int(limits["representative_scope_limit"])]:
                    scoped_filter = deps.merge_filter_dicts_fn(
                        filter_dict,
                        {"source_type": "paper", "arxiv_id": paper_id},
                    )
                    if scoped_filter is None:
                        continue
                    for lexical_query in lexical_query_forms:
                        extra_hits = deps.lexical_search_fn(
                            deps.searcher,
                            lexical_query,
                            top_k=max(2, min(lexical_n, top_k * 2)),
                            filter_dict=scoped_filter,
                        )
                        for hit in extra_hits:
                            rid = deps.result_id_fn(hit)
                            if rid in seen_lexical:
                                continue
                            seen_lexical.add(rid)
                            lexical_hits.append(hit)
            if not source_filtered:
                for extra_source, budget in source_budgets.items():
                    source_filter = deps.merge_source_filter_fn(filter_dict, extra_source)
                    if source_filter is None or budget <= 0:
                        continue
                    local_count = 0
                    for lexical_query in lexical_query_forms:
                        extra_hits = deps.lexical_search_fn(
                            deps.searcher,
                            lexical_query,
                            top_k=max(2, int(budget)),
                            filter_dict=source_filter,
                        )
                        for hit in extra_hits:
                            rid = deps.result_id_fn(hit)
                            if rid in seen_lexical:
                                continue
                            seen_lexical.add(rid)
                            lexical_hits.append(hit)
                            local_count += 1
                    candidate_counts.setdefault(extra_source, {"budget": int(budget), "semantic": 0, "lexical": 0})
                    candidate_counts[extra_source]["lexical"] += local_count

        if mode == "semantic":
            results = sorted(semantic_hits, key=lambda r: r.semantic_score, reverse=True)
            results = deps.apply_feature_boosts_fn(deps.searcher, results)
            results = deps.apply_query_fit_reranking_fn(
                results,
                query=query_text,
                sqlite_db=getattr(deps.searcher, "sqlite_db", None),
                query_forms=rescue_queries,
            )
            results = self.graph_candidate_boost(results, graph_query_signal=graph_query_signal)
            if bool((graph_query_signal or {}).get("is_graph_heavy")):
                results.sort(key=deps.retrieval_sort_key_fn, reverse=True)
                results = deps.preserve_parent_diversity_fn(results)
        elif mode == "keyword":
            results = sorted(lexical_hits, key=lambda r: r.lexical_score, reverse=True)
            results = deps.apply_feature_boosts_fn(deps.searcher, results)
            results = deps.apply_query_fit_reranking_fn(
                results,
                query=query_text,
                sqlite_db=getattr(deps.searcher, "sqlite_db", None),
                query_forms=rescue_queries,
            )
            results = self.graph_candidate_boost(results, graph_query_signal=graph_query_signal)
            if bool((graph_query_signal or {}).get("is_graph_heavy")):
                results.sort(key=lambda r: deps.retrieval_sort_key_fn(r, prefer_lexical=True), reverse=True)
                results = deps.preserve_parent_diversity_fn(results)
        else:
            merged: dict[str, SearchResult] = {}
            for item in semantic_hits:
                merged[deps.result_id_fn(item)] = item
            for item in lexical_hits:
                key = deps.result_id_fn(item)
                if key not in merged:
                    merged[key] = item
                    continue
                existing = merged[key]
                existing.document = existing.document or item.document
                existing.distance = min(existing.distance, item.distance)
                existing.semantic_score = deps.safe_float_fn(existing.semantic_score, 0.0)
                existing.lexical_score = deps.safe_float_fn(item.lexical_score, existing.lexical_score)
                if isinstance(existing.lexical_extras, dict) and isinstance(item.lexical_extras, dict):
                    combined = dict(existing.lexical_extras)
                    for name, value in item.lexical_extras.items():
                        combined.setdefault(name, value)
                    existing.lexical_extras = combined
            results = []
            for item in merged.values():
                sem = deps.safe_float_fn(item.semantic_score, 0.0)
                lex = deps.safe_float_fn(item.lexical_score, 0.0)
                final_score = alpha * sem + (1.0 - alpha) * lex
                extras = dict(item.lexical_extras or {})
                rank_position = deps.safe_int_fn(extras.get("rank_position"), 0)
                keyword_rescue = 0.0
                if lex > 0.0:
                    keyword_rescue = max(keyword_rescue, min(0.82, 0.82 * lex))
                if rank_position > 0:
                    keyword_rescue = max(keyword_rescue, max(0.0, 0.84 - (0.04 * (rank_position - 1))))
                if sem == 0.0 and keyword_rescue > 0.0:
                    final_score = max(final_score, keyword_rescue)
                item.score = max(0.0, min(1.0, final_score))
                item.retrieval_mode = "hybrid"
                if not item.lexical_extras:
                    item.lexical_extras = {"query": deps.clean_text_fn(query_text)}
                if keyword_rescue > 0.0:
                    item.lexical_extras["hybrid_keyword_rescue_score"] = round(keyword_rescue, 6)
                results.append(item)
            results = deps.apply_feature_boosts_fn(deps.searcher, results)
            results = deps.apply_query_fit_reranking_fn(
                results,
                query=query_text,
                sqlite_db=getattr(deps.searcher, "sqlite_db", None),
                query_forms=rescue_queries,
            )
            results = self.graph_candidate_boost(results, graph_query_signal=graph_query_signal)
            if bool((graph_query_signal or {}).get("is_graph_heavy")):
                results.sort(key=deps.retrieval_sort_key_fn, reverse=True)
                results = deps.preserve_parent_diversity_fn(results)

        candidate_sources: list[dict[str, Any]] = []
        if source_filtered:
            source_label = str(deps.normalize_source_type_fn((filter_dict or {}).get("source_type")) or "all")
            source_count = (
                sum(1 for item in results if deps.source_label_for_result_fn(item) == source_label)
                if source_label != "all"
                else len(results)
            )
            candidate_sources.append(
                {
                    "sourceType": source_label,
                    "budget": semantic_n if mode == "semantic" else lexical_n if mode == "keyword" else max(semantic_n, lexical_n),
                    "semanticHits": source_count if mode in {"semantic", "hybrid"} else 0,
                    "lexicalHits": source_count if mode in {"keyword", "hybrid"} else 0,
                }
            )
        else:
            for source_name, stats in candidate_counts.items():
                candidate_sources.append(
                    {
                        "sourceType": source_name,
                        "budget": int(stats.get("budget") or 0),
                        "semanticHits": int(stats.get("semantic") or 0),
                        "lexicalHits": int(stats.get("lexical") or 0),
                    }
                )
        rerank_signals = {
            "queryIntent": query_intent,
            "paperFamily": paper_family or "general",
            "retrievalMode": mode,
            "ontologyExpanded": len(expanded_queries) > 1,
            "expandedQueryCount": len(expanded_queries),
            "expandedQueriesUsed": list(expanded_queries[:8]),
            "queryRescueApplied": bool(rescue_queries),
            "queryRescueForms": list(rescue_queries[: int(limits["rescue_query_limit"])]),
            "lexicalQueryForms": list(lexical_query_forms[: int(limits["lexical_form_limit"])]),
            "canonicalEntitiesApplied": list(canonical_entity_ids[:6]),
            "resolvedSourceIds": list(resolved_source_ids[:3]),
            "representativeScopeIdsUsed": list(resolved_source_ids[: int(limits["representative_scope_limit"])]),
            "extraSourceFanoutSkipped": bool(source_filtered),
            "plannerUsed": bool((plan.query_plan if plan else {}).get("planner_used")),
            "graphQueryKind": str((graph_query_signal.get("diagnostics") or {}).get("query_kind") or ""),
            "graphHeavy": bool(graph_query_signal.get("is_graph_heavy")),
        }
        results = deps.ensure_compare_result_coverage_fn(
            results=results,
            query_text=query_text,
            filter_dict=filter_dict,
            lexical_query_forms=lexical_query_forms,
            resolved_source_ids=resolved_source_ids,
            top_k=top_k,
            normalized_source=normalized_source,
            paper_family=paper_family,
        )
        return results, rerank_signals, candidate_sources
