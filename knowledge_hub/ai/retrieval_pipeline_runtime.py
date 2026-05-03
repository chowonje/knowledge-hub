from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from knowledge_hub.ai.memory_prefilter import MEMORY_ROUTE_MODE_ON, execute_memory_prefilter, normalize_memory_route_mode
from knowledge_hub.ai.rag_paper_prefilter import search_with_paper_memory_prefilter
from knowledge_hub.ai.rag_scope import get_active_profile, load_topology_index
from knowledge_hub.papers.prefilter import resolve_paper_memory_prefilter


@dataclass(frozen=True)
class ScopeDeps:
    sqlite_db: Any
    config: Any
    cached_profile: dict[str, Any] | None
    cached_topology: dict[str, Any] | None


@dataclass(frozen=True)
class MemoryPrefilterDeps:
    sqlite_db: Any
    database: Any
    search_fn: Any


class _MemoryPrefilterSearchAdapter:
    def __init__(self, deps: MemoryPrefilterDeps):
        self.sqlite_db = deps.sqlite_db
        self.database = deps.database
        self.search = deps.search_fn


class _PaperMemoryPrefilterSearchAdapter:
    def __init__(self, deps: MemoryPrefilterDeps):
        self.sqlite_db = deps.sqlite_db
        self.search = deps.search_fn


@dataclass(frozen=True)
class RetrievalPipelineRuntimeDeps:
    searcher: Any
    ctx: Any
    caches: Any
    result_id_fn: Any
    retrieval_sort_key_fn: Any
    safe_float_fn: Any
    result_paper_id_fn: Any
    source_label_for_result_fn: Any
    top_signal_items_fn: Any
    preserve_parent_diversity_fn: Any
    normalize_source_type_fn: Any
    reranker_config_fn: Any
    get_reranker_fn: Any


class RetrievalPipelineRuntime:
    def __init__(self, deps: RetrievalPipelineRuntimeDeps) -> None:
        self._deps = deps

    def scope_deps(self) -> ScopeDeps:
        deps = self._deps
        ctx = deps.ctx
        searcher = deps.searcher
        return ScopeDeps(
            sqlite_db=getattr(ctx, "sqlite_db", getattr(searcher, "sqlite_db", None)),
            config=getattr(ctx, "config", getattr(searcher, "config", None)),
            cached_profile=getattr(searcher, "_profile_cache", None),
            cached_topology=getattr(searcher, "_topology_cache", None),
        )

    def memory_prefilter_deps(self) -> MemoryPrefilterDeps:
        deps = self._deps
        ctx = deps.ctx
        searcher = deps.searcher
        return MemoryPrefilterDeps(
            sqlite_db=getattr(ctx, "sqlite_db", getattr(searcher, "sqlite_db", None)),
            database=getattr(ctx, "database", getattr(searcher, "database", None)),
            search_fn=searcher.search,
        )

    def sync_profile_cache(self, profile: dict[str, Any] | None) -> dict[str, Any] | None:
        deps = self._deps
        deps.searcher._profile_cache = profile
        if deps.caches is not None:
            deps.caches.profile_cache = profile
        return profile

    def sync_topology_cache(self, topology: dict[str, Any] | None) -> dict[str, Any] | None:
        deps = self._deps
        deps.searcher._topology_cache = topology
        if deps.caches is not None:
            deps.caches.topology_cache = topology
        return topology

    def get_active_profile_direct(self) -> dict[str, Any] | None:
        hook = getattr(self._deps.searcher, "_get_active_profile", None)
        if callable(hook):
            try:
                return self.sync_profile_cache(hook())
            except Exception:
                pass
        deps = self.scope_deps()
        profile = get_active_profile(
            sqlite_db=deps.sqlite_db,
            cached_profile=deps.cached_profile,
        )
        return self.sync_profile_cache(profile)

    def load_topology_index_direct(self) -> dict[str, Any] | None:
        hook = getattr(self._deps.searcher, "_load_topology_index", None)
        if callable(hook):
            try:
                return self.sync_topology_cache(hook())
            except Exception:
                pass
        deps = self.scope_deps()
        topology = load_topology_index(
            config=deps.config,
            cached_topology=deps.cached_topology,
        )
        return self.sync_topology_cache(topology)

    def merge_prefilter_results(self, results: list[Any], *, top_k: int) -> list[Any]:
        deps = self._deps
        merged: dict[str, Any] = {}
        for item in results:
            key = deps.result_id_fn(item)
            existing = merged.get(key)
            if existing is None or deps.retrieval_sort_key_fn(item) > deps.retrieval_sort_key_fn(existing):
                merged[key] = item
        ranked = list(merged.values())
        ranked.sort(key=deps.retrieval_sort_key_fn, reverse=True)
        return ranked[: max(1, int(top_k))]

    def search_with_paper_memory_prefilter_direct(
        self,
        *,
        query: str,
        top_k: int,
        min_score: float,
        source_type: str | None,
        retrieval_mode: str,
        alpha: float,
        requested_mode: str,
        metadata_filter: dict[str, Any] | None = None,
        search_fn=None,
    ) -> tuple[list[Any], dict[str, Any]]:
        base_deps = self.memory_prefilter_deps()
        deps = MemoryPrefilterDeps(
            sqlite_db=base_deps.sqlite_db,
            database=base_deps.database,
            search_fn=search_fn or self._deps.searcher.search,
        )
        return search_with_paper_memory_prefilter(
            _PaperMemoryPrefilterSearchAdapter(deps),
            query=query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            min_score=min_score,
            requested_mode=requested_mode,
            metadata_filter=metadata_filter,
            resolve_paper_memory_prefilter_fn=resolve_paper_memory_prefilter,
            merge_search_results_fn=lambda results: self.merge_prefilter_results(results, top_k=top_k),
        )

    def search_with_memory_prefilter_direct(
        self,
        *,
        query: str,
        top_k: int,
        min_score: float,
        source_type: str | None,
        retrieval_mode: str,
        alpha: float,
        requested_mode: str,
        metadata_filter: dict[str, Any] | None = None,
        query_forms: list[str] | None = None,
        search_fn=None,
    ):
        base_deps = self.memory_prefilter_deps()
        deps = MemoryPrefilterDeps(
            sqlite_db=base_deps.sqlite_db,
            database=base_deps.database,
            search_fn=search_fn or self._deps.searcher.search,
        )
        return execute_memory_prefilter(
            _MemoryPrefilterSearchAdapter(deps),
            query=query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            min_score=min_score,
            requested_mode=requested_mode,
            metadata_filter=metadata_filter,
            query_forms=query_forms,
            result_id_fn=self._deps.result_id_fn,
            search_fn=deps.search_fn,
        )

    def apply_cross_encoder_reranking(
        self,
        results: list[Any],
        *,
        query: str,
        top_k: int,
    ) -> tuple[list[Any], dict[str, Any]]:
        deps = self._deps
        config = deps.reranker_config_fn()
        diagnostics = {
            "rerankerApplied": False,
            "rerankerModel": str(config.model),
            "rerankerWindow": min(len(results), int(config.candidate_window)),
            "rerankerLatencyMs": 0,
            "rerankerFallbackUsed": False,
            "rerankerReason": "disabled",
        }
        if not config.enabled:
            return results, diagnostics
        reranker = deps.get_reranker_fn(config)
        if reranker is None:
            diagnostics["rerankerReason"] = "unavailable"
            diagnostics["rerankerFallbackUsed"] = True
            return results, diagnostics
        execution = reranker.rerank(
            query=query,
            results=list(results[: max(1, int(top_k) * 4)]),
            config=config,
        )
        merged = list(execution.results) + list(results[max(1, int(top_k) * 4):])
        merged.sort(key=deps.retrieval_sort_key_fn, reverse=True)
        merged = deps.preserve_parent_diversity_fn(merged)
        return merged, dict(execution.diagnostics)

    def apply_cluster_context_expansion(
        self,
        results: list[Any],
        *,
        top_k: int,
    ) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
        deps = self._deps
        profile = self.get_active_profile_direct()
        preferred_sources = {
            str(item).strip().lower()
            for item in ((profile or {}).get("retrieval_facets", {}) or {}).get("preferred_sources", [])
            if str(item).strip()
        }
        topology = self.load_topology_index_direct()
        cluster_scores: dict[str, float] = {}
        cluster_hits: dict[str, int] = {}

        for item in results:
            source_type = deps.source_label_for_result_fn(item)
            if preferred_sources and source_type in preferred_sources:
                item.score = min(1.0, deps.safe_float_fn(item.score, 0.0) + 0.04)
            if topology is None:
                continue
            file_path = str((item.metadata or {}).get("file_path") or "").strip()
            node = topology["nodesByPath"].get(file_path) if file_path else None
            if not node:
                continue
            cluster_id = str(node.get("clusterId") or "").strip()
            if not cluster_id:
                continue
            cluster_scores[cluster_id] = cluster_scores.get(cluster_id, 0.0) + deps.safe_float_fn(item.score, 0.0)
            cluster_hits[cluster_id] = cluster_hits.get(cluster_id, 0) + 1
            item.metadata.setdefault("cluster_id", cluster_id)

        shortlisted = sorted(
            cluster_scores.items(),
            key=lambda pair: (pair[1], cluster_hits.get(pair[0], 0)),
            reverse=True,
        )[:3]
        cluster_rank_by_id = {cluster_id: index for index, (cluster_id, _) in enumerate(shortlisted)}
        cluster_boost_by_id = {
            cluster_id: max(0.0, 0.03 - (0.01 * index))
            for index, (cluster_id, _) in enumerate(shortlisted)
        }
        related_clusters: list[dict[str, Any]] = []
        for cluster_id, score in shortlisted:
            cluster_meta = (topology or {}).get("clustersById", {}).get(cluster_id, {})
            related_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "label": str(cluster_meta.get("label") or cluster_id),
                    "size": int(cluster_meta.get("size", 0) or 0),
                    "representative_note_id": str(cluster_meta.get("representativeNoteId") or ""),
                    "score": round(float(score), 6),
                    "hit_count": int(cluster_hits.get(cluster_id, 0)),
                    "boost": round(float(cluster_boost_by_id.get(cluster_id, 0.0)), 6),
                }
            )

        for item in results:
            cluster_id = str((item.metadata or {}).get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            cluster_meta = (topology or {}).get("clustersById", {}).get(cluster_id, {})
            cluster_rank = cluster_rank_by_id.get(cluster_id, -1)
            cluster_boost = deps.safe_float_fn(cluster_boost_by_id.get(cluster_id, 0.0), 0.0)
            if cluster_boost > 0.0:
                item.score = min(1.0, deps.safe_float_fn(item.score, 0.0) + cluster_boost)
            extras = dict(item.lexical_extras or {})
            ranking_signals = dict(extras.get("ranking_signals") or {})
            ranking_signals["cluster_id"] = cluster_id
            ranking_signals["cluster_label"] = str(cluster_meta.get("label") or cluster_id)
            ranking_signals["cluster_hit_count"] = int(cluster_hits.get(cluster_id, 0))
            ranking_signals["cluster_rank"] = int(cluster_rank)
            ranking_signals["cluster_selected"] = cluster_rank >= 0
            ranking_signals["cluster_proximity_boost"] = round(float(cluster_boost), 6)
            extras["cluster_id"] = cluster_id
            extras["cluster_label"] = ranking_signals["cluster_label"]
            extras["cluster_hit_count"] = ranking_signals["cluster_hit_count"]
            extras["cluster_rank"] = ranking_signals["cluster_rank"]
            extras["cluster_selected"] = ranking_signals["cluster_selected"]
            extras["cluster_proximity_boost"] = ranking_signals["cluster_proximity_boost"]
            extras["top_ranking_signals"] = deps.top_signal_items_fn(
                list(extras.get("top_ranking_signals") or []),
                {"cluster_proximity_boost": cluster_boost},
            )
            ranking_signals["top_ranking_signals"] = list(extras["top_ranking_signals"])
            extras["ranking_signals"] = ranking_signals
            item.lexical_extras = extras

        context_expansion = {
            "mode": "cluster",
            "clusterCount": len(related_clusters),
            "clusterIds": [str(item.get("cluster_id") or "") for item in related_clusters],
            "preferredSources": sorted(preferred_sources),
            "topologyApplied": bool(topology is not None),
        }
        if not shortlisted:
            return results[:top_k], related_clusters, context_expansion, profile

        results.sort(key=deps.retrieval_sort_key_fn, reverse=True)
        results = deps.preserve_parent_diversity_fn(results)
        return results[:top_k], related_clusters, context_expansion, profile

    def memory_prior_merge(
        self,
        *,
        base_results: list[Any],
        memory_results: list[Any],
        memory_prefilter: dict[str, Any],
        top_k: int,
        plan: Any,
    ) -> list[Any]:
        deps = self._deps
        merged: dict[str, Any] = {}
        matched_documents = {
            str(item).strip()
            for item in list(memory_prefilter.get("matchedDocumentIds") or [])
            if str(item).strip()
        }
        effective_mode = normalize_memory_route_mode(
            memory_prefilter.get("effectiveMode") or memory_prefilter.get("requestedMode")
        )
        memory_applied = bool(memory_prefilter.get("applied"))
        allow_memory_boost = effective_mode == MEMORY_ROUTE_MODE_ON and memory_applied
        matched_papers = set(matched_documents)
        boosted_memory_ids = {deps.result_id_fn(item) for item in memory_results}
        for item in base_results + memory_results:
            key = deps.result_id_fn(item)
            existing = merged.get(key)
            candidate = item
            extras = dict(candidate.lexical_extras or {})
            ranking_signals = dict(extras.get("ranking_signals") or {})
            memory_prior_boost = 0.0
            if allow_memory_boost and key in boosted_memory_ids:
                base_boost = float(plan.memory_prior_weight)
                if effective_mode == MEMORY_ROUTE_MODE_ON:
                    base_boost *= 1.5
                memory_prior_boost = max(memory_prior_boost, base_boost)
            paper_id = deps.result_paper_id_fn(candidate)
            document_id = str((candidate.metadata or {}).get("document_id") or "").strip()
            file_path = str((candidate.metadata or {}).get("file_path") or "").strip()
            source_url = str((candidate.metadata or {}).get("url") or "").strip()
            match_factor = 1.0 if effective_mode == MEMORY_ROUTE_MODE_ON else 0.8
            if allow_memory_boost and paper_id and paper_id in matched_papers:
                memory_prior_boost = max(memory_prior_boost, float(plan.memory_prior_weight) * match_factor)
            if allow_memory_boost and document_id and document_id in matched_documents:
                memory_prior_boost = max(memory_prior_boost, float(plan.memory_prior_weight) * match_factor)
            if allow_memory_boost and file_path and file_path in matched_documents:
                memory_prior_boost = max(memory_prior_boost, float(plan.memory_prior_weight) * match_factor)
            if allow_memory_boost and source_url and source_url in matched_documents:
                memory_prior_boost = max(memory_prior_boost, float(plan.memory_prior_weight) * match_factor)
            if memory_prior_boost > 0.0:
                candidate.score = max(0.0, min(1.0, deps.safe_float_fn(candidate.score, 0.0) + memory_prior_boost))
                raw_sort_score = deps.safe_float_fn(extras.get("retrieval_sort_score"), deps.safe_float_fn(candidate.score, 0.0)) + memory_prior_boost
                extras["memory_prior_boost"] = round(memory_prior_boost, 6)
                extras["memory_provenance"] = {
                    "requestedMode": str(memory_prefilter.get("requestedMode") or ""),
                    "effectiveMode": str(memory_prefilter.get("effectiveMode") or ""),
                    "reason": str(memory_prefilter.get("reason") or ""),
                    "matchedMemoryIds": list(memory_prefilter.get("matchedMemoryIds") or []),
                    "matchedDocumentIds": list(memory_prefilter.get("matchedDocumentIds") or []),
                    "updatesPreferred": bool(memory_prefilter.get("updatesPreferred")),
                }
                extras["retrieval_sort_score"] = round(raw_sort_score, 6)
                extras["retrieval_adjusted_score"] = round(candidate.score, 6)
                extras["top_ranking_signals"] = deps.top_signal_items_fn(
                    list(extras.get("top_ranking_signals") or []),
                    {"memory_prior_boost": memory_prior_boost},
                )
                ranking_signals["memory_prior_boost"] = round(memory_prior_boost, 6)
                ranking_signals["top_ranking_signals"] = list(extras["top_ranking_signals"])
                extras["ranking_signals"] = ranking_signals
                candidate.lexical_extras = extras
            if existing is None or deps.retrieval_sort_key_fn(candidate) > deps.retrieval_sort_key_fn(existing):
                merged[key] = candidate
        ranked = list(merged.values())
        ranked.sort(key=deps.retrieval_sort_key_fn, reverse=True)
        ranked_ids = {deps.result_id_fn(item) for item in ranked}
        for base_item in list(base_results)[: max(1, int(plan.fallback_window))]:
            key = deps.result_id_fn(base_item)
            if key in ranked_ids:
                continue
            ranked.append(base_item)
            ranked_ids.add(key)
        ranked.sort(key=deps.retrieval_sort_key_fn, reverse=True)
        return ranked[: max(1, int(top_k) * 4)]

    def enforce_source_scope(
        self,
        results: list[Any],
        *,
        source_type: str | None,
    ) -> tuple[list[Any], bool]:
        normalized_source = str(self._deps.normalize_source_type_fn(source_type) or "").strip().lower()
        if not normalized_source or normalized_source == "all":
            return results, False
        matching = [item for item in results if self._deps.source_label_for_result_fn(item) == normalized_source]
        if matching:
            return matching, True
        return results, False
