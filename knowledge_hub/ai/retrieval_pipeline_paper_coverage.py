from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from knowledge_hub.core.models import SearchResult


@dataclass(frozen=True)
class PaperCoverageDeps:
    search_runtime: Any
    apply_feature_boosts_fn: Any
    lexical_search_fn: Any
    semantic_search_fn: Any
    merge_filter_dicts_fn: Any
    clean_text_fn: Any
    safe_float_fn: Any
    result_id_fn: Any
    retrieval_sort_key_fn: Any
    result_paper_id_fn: Any
    top_signal_items_fn: Any
    first_nonempty_fn: Any


class PaperCoverageSearchRuntime:
    def __init__(
        self,
        *,
        embedder: Any,
        database: Any,
        sqlite_db: Any,
        build_retrieval_ranking_signals_fn: Any,
    ) -> None:
        self.embedder = embedder
        self.database = database
        self.sqlite_db = sqlite_db
        self._build_retrieval_ranking_signals_fn = build_retrieval_ranking_signals_fn

    def _build_retrieval_ranking_signals(self, item: SearchResult, snapshot: dict[str, Any]) -> dict[str, Any]:
        fn = self._build_retrieval_ranking_signals_fn
        if callable(fn):
            return dict(fn(item, snapshot) or {})
        return {}


class PaperCoverageService:
    def __init__(self, deps: PaperCoverageDeps):
        self.deps = deps

    def fetch_compare_scoped_result(
        self,
        *,
        query_text: str,
        paper_id: str,
        lexical_query_forms: list[str],
        filter_dict: dict[str, Any] | None,
        top_k: int,
    ) -> SearchResult | None:
        scoped_filter = self.deps.merge_filter_dicts_fn(
            filter_dict,
            {"source_type": "paper", "arxiv_id": paper_id},
        )
        if scoped_filter is None:
            return None

        return self.fetch_paper_scoped_result(
            query_text=query_text,
            paper_id=paper_id,
            lexical_query_forms=lexical_query_forms,
            filter_dict=scoped_filter,
            top_k=top_k,
            fallback_reason="compare_resolved_paper_fallback",
        )

    def fetch_paper_scoped_result(
        self,
        *,
        query_text: str,
        paper_id: str,
        lexical_query_forms: list[str],
        filter_dict: dict[str, Any] | None,
        top_k: int,
        fallback_reason: str,
    ) -> SearchResult | None:
        scoped_filter = self.deps.merge_filter_dicts_fn(
            filter_dict,
            {"source_type": "paper", "arxiv_id": paper_id},
        )
        if scoped_filter is None:
            return None

        search_runtime = self.deps.search_runtime
        candidates: list[SearchResult] = []
        seen: set[str] = set()
        search_forms = [token for token in lexical_query_forms if self.deps.clean_text_fn(token)] or [query_text]
        for lexical_query in search_forms:
            for hit in self.deps.lexical_search_fn(
                search_runtime,
                lexical_query,
                top_k=max(2, top_k),
                filter_dict=scoped_filter,
            ):
                rid = self.deps.result_id_fn(hit)
                if rid in seen:
                    continue
                seen.add(rid)
                candidates.append(hit)

        if not candidates:
            try:
                query_embedding = search_runtime.embedder.embed_text(query_text)
            except Exception:
                query_embedding = []
            if query_embedding:
                for hit in self.deps.semantic_search_fn(
                    search_runtime,
                    query_embedding=query_embedding,
                    top_k=max(2, top_k),
                    filter_dict=scoped_filter,
                ):
                    rid = self.deps.result_id_fn(hit)
                    if rid in seen:
                        continue
                    seen.add(rid)
                    candidates.append(hit)

        if not candidates:
            return self.build_paper_card_fallback_result(
                paper_id=paper_id,
                reason=fallback_reason,
            )

        candidates = self.deps.apply_feature_boosts_fn(search_runtime, candidates)
        return max(candidates, key=self.deps.retrieval_sort_key_fn)

    def build_compare_card_fallback_result(self, *, paper_id: str) -> SearchResult | None:
        return self.build_paper_card_fallback_result(
            paper_id=paper_id,
            reason="compare_resolved_paper_fallback",
        )

    def build_paper_card_fallback_result(
        self,
        *,
        paper_id: str,
        reason: str,
        score: float = 0.62,
        card_row: dict[str, Any] | None = None,
    ) -> SearchResult | None:
        sqlite_db = getattr(self.deps.search_runtime, "sqlite_db", None)
        if sqlite_db is None:
            return None

        get_paper_card = getattr(sqlite_db, "get_paper_card_v2", None)
        get_paper_memory = getattr(sqlite_db, "get_paper_memory_card", None)
        get_paper = getattr(sqlite_db, "get_paper", None)
        card = dict(card_row or {})
        if not card:
            card = dict(get_paper_card(paper_id) or {}) if callable(get_paper_card) else {}
        memory = dict(get_paper_memory(paper_id) or {}) if callable(get_paper_memory) else {}
        paper = dict(get_paper(paper_id) or {}) if callable(get_paper) else {}

        title = self.deps.first_nonempty_fn(card.get("title"), paper.get("title"), memory.get("title"), paper_id)
        excerpt = self.deps.first_nonempty_fn(
            card.get("method_core"),
            card.get("result_core"),
            card.get("problem_core"),
            card.get("paper_core"),
            memory.get("method_core"),
            memory.get("evidence_core"),
            memory.get("problem_context"),
            memory.get("paper_core"),
            paper.get("notes"),
        )
        if not title or not excerpt:
            return None

        score = max(0.0, min(0.99, float(score or 0.0)))
        lexical_extras = {
            "paper_card_fallback": True,
            reason: True,
            "retrieval_sort_score": round(score, 6),
            "ranking_signals": {
                "paper_card_fallback": True,
                reason: True,
                f"{reason}_score": round(score, 6),
            },
            "top_ranking_signals": [
                {"name": reason, "value": round(score, 6)},
            ],
        }
        return SearchResult(
            document=excerpt,
            metadata={
                "title": title,
                "source_type": "paper",
                "arxiv_id": paper_id,
                "paper_id": paper_id,
            },
            distance=max(0.0, 1.0 - score),
            score=score,
            semantic_score=0.0,
            lexical_score=score,
            retrieval_mode="paper-card-v2-fallback",
            lexical_extras=lexical_extras,
            document_id=f"paper-card-v2:{paper_id}",
        )

    def search_paper_card_fallback_results(
        self,
        *,
        query_forms: list[str],
        limit: int,
        reason: str,
    ) -> list[SearchResult]:
        sqlite_db = getattr(self.deps.search_runtime, "sqlite_db", None)
        if sqlite_db is None:
            return []
        search_cards = getattr(sqlite_db, "search_paper_cards_v2", None)
        if not callable(search_cards):
            return []

        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for form in [token for token in query_forms if self.deps.clean_text_fn(token)]:
            try:
                matches = list(search_cards(form, limit=max(limit * 3, 6)) or [])
            except Exception:
                matches = []
            for raw in matches:
                row = dict(raw or {})
                paper_id = self.deps.clean_text_fn(row.get("paper_id"))
                if not paper_id or paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)
                rows.append(row)
                if len(rows) >= max(limit * 2, 6):
                    break
            if len(rows) >= max(limit * 2, 6):
                break

        fallback_results: list[SearchResult] = []
        for idx, row in enumerate(rows[: max(1, limit)]):
            match_score = max(0.0, self.deps.safe_float_fn(row.get("match_score"), 0.0))
            score = min(0.72, 0.48 + min(0.14, match_score * 0.02))
            item = self.build_paper_card_fallback_result(
                paper_id=self.deps.clean_text_fn(row.get("paper_id")),
                reason=reason,
                score=score,
                card_row=row,
            )
            if item is None:
                continue
            extras = dict(item.lexical_extras or {})
            ranking_signals = dict(extras.get("ranking_signals") or {})
            ranking_signals[f"{reason}_rank"] = idx + 1
            extras[f"{reason}_rank"] = idx + 1
            extras["ranking_signals"] = ranking_signals
            item.lexical_extras = extras
            fallback_results.append(item)
        return fallback_results

    def ensure_resolved_paper_result_coverage(
        self,
        *,
        results: list[SearchResult],
        query_text: str,
        filter_dict: dict[str, Any] | None,
        lexical_query_forms: list[str],
        resolved_source_ids: list[str],
        top_k: int,
        normalized_source: str,
        paper_family: str,
    ) -> list[SearchResult]:
        if normalized_source != "paper" or paper_family not in {"paper_lookup", "concept_explainer"}:
            return results

        target_ids = [self.deps.clean_text_fn(item) for item in resolved_source_ids[:1] if self.deps.clean_text_fn(item)]
        if not target_ids:
            return results

        prioritized: list[SearchResult] = []
        prioritized_result_ids: set[str] = set()
        for idx, paper_id in enumerate(target_ids):
            existing = [item for item in results if self.deps.result_paper_id_fn(item) == paper_id]
            chosen = max(existing, key=self.deps.retrieval_sort_key_fn) if existing else self.fetch_paper_scoped_result(
                query_text=query_text,
                paper_id=paper_id,
                lexical_query_forms=lexical_query_forms,
                filter_dict=filter_dict,
                top_k=top_k,
                fallback_reason="paper_lookup_resolved_paper_fallback"
                if paper_family == "paper_lookup"
                else "concept_resolved_paper_fallback",
            )
            if chosen is None:
                continue
            extras = dict(chosen.lexical_extras or {})
            ranking_signals = dict(extras.get("ranking_signals") or {})
            sort_boost = max(0.0, 1.8 - (0.05 * idx))
            reason_key = (
                "paper_lookup_resolved_paper_promoted"
                if paper_family == "paper_lookup"
                else "concept_resolved_paper_promoted"
            )
            ranking_signals[reason_key] = True
            ranking_signals["resolved_target_paper_id"] = paper_id
            ranking_signals["resolved_target_sort_boost"] = round(sort_boost, 6)
            extras[reason_key] = True
            extras["resolved_target_paper_id"] = paper_id
            extras["retrieval_sort_score"] = round(sort_boost, 6)
            extras["top_ranking_signals"] = self.deps.top_signal_items_fn(
                list(extras.get("top_ranking_signals") or []),
                {reason_key: 0.04, "resolved_target_sort_boost": sort_boost},
            )
            ranking_signals["top_ranking_signals"] = list(extras["top_ranking_signals"])
            extras["ranking_signals"] = ranking_signals
            chosen.lexical_extras = extras
            result_id = self.deps.result_id_fn(chosen)
            if result_id in prioritized_result_ids:
                continue
            prioritized.append(chosen)
            prioritized_result_ids.add(result_id)

        if not prioritized:
            return results

        remaining = [item for item in results if self.deps.result_id_fn(item) not in prioritized_result_ids]
        return prioritized + remaining

    def ensure_discover_result_coverage(
        self,
        *,
        results: list[SearchResult],
        lexical_query_forms: list[str],
        top_k: int,
        normalized_source: str,
        paper_family: str,
    ) -> list[SearchResult]:
        if normalized_source != "paper" or paper_family != "paper_discover":
            return results
        if results:
            return results
        return self.search_paper_card_fallback_results(
            query_forms=lexical_query_forms,
            limit=max(1, top_k),
            reason="paper_discover_card_fallback",
        )

    def ensure_compare_result_coverage(
        self,
        *,
        results: list[SearchResult],
        query_text: str,
        filter_dict: dict[str, Any] | None,
        lexical_query_forms: list[str],
        resolved_source_ids: list[str],
        top_k: int,
        normalized_source: str,
        paper_family: str,
    ) -> list[SearchResult]:
        if normalized_source != "paper" or paper_family != "paper_compare":
            return results

        target_ids = [self.deps.clean_text_fn(item) for item in resolved_source_ids[:2] if self.deps.clean_text_fn(item)]
        if len(target_ids) < 2:
            return results

        prioritized: list[SearchResult] = []
        prioritized_result_ids: set[str] = set()
        for idx, paper_id in enumerate(target_ids):
            existing = [
                item
                for item in results
                if self.deps.result_paper_id_fn(item) == paper_id
            ]
            chosen = max(existing, key=self.deps.retrieval_sort_key_fn) if existing else self.fetch_compare_scoped_result(
                query_text=query_text,
                paper_id=paper_id,
                lexical_query_forms=lexical_query_forms,
                filter_dict=filter_dict,
                top_k=top_k,
            )
            if chosen is None:
                continue
            extras = dict(chosen.lexical_extras or {})
            ranking_signals = dict(extras.get("ranking_signals") or {})
            compare_sort_boost = max(0.0, 2.0 - (0.05 * idx))
            ranking_signals["compare_resolved_paper_promoted"] = True
            ranking_signals["compare_target_paper_id"] = paper_id
            ranking_signals["compare_target_sort_boost"] = round(compare_sort_boost, 6)
            extras["compare_resolved_paper_promoted"] = True
            extras["compare_target_paper_id"] = paper_id
            extras["retrieval_sort_score"] = round(compare_sort_boost, 6)
            extras["top_ranking_signals"] = self.deps.top_signal_items_fn(
                list(extras.get("top_ranking_signals") or []),
                {"compare_resolved_paper_promoted": 0.04, "compare_target_sort_boost": compare_sort_boost},
            )
            ranking_signals["top_ranking_signals"] = list(extras["top_ranking_signals"])
            extras["ranking_signals"] = ranking_signals
            chosen.lexical_extras = extras
            result_id = self.deps.result_id_fn(chosen)
            if result_id in prioritized_result_ids:
                continue
            prioritized.append(chosen)
            prioritized_result_ids.add(result_id)

        if not prioritized:
            return results

        remaining = [item for item in results if self.deps.result_id_fn(item) not in prioritized_result_ids]
        return [*prioritized, *remaining]

    def ensure_paper_result_coverage(
        self,
        *,
        results: list[SearchResult],
        query_text: str,
        filter_dict: dict[str, Any] | None,
        lexical_query_forms: list[str],
        resolved_source_ids: list[str],
        top_k: int,
        normalized_source: str,
        paper_family: str,
    ) -> list[SearchResult]:
        results = self.ensure_compare_result_coverage(
            results=results,
            query_text=query_text,
            filter_dict=filter_dict,
            lexical_query_forms=lexical_query_forms,
            resolved_source_ids=resolved_source_ids,
            top_k=top_k,
            normalized_source=normalized_source,
            paper_family=paper_family,
        )
        results = self.ensure_resolved_paper_result_coverage(
            results=results,
            query_text=query_text,
            filter_dict=filter_dict,
            lexical_query_forms=lexical_query_forms,
            resolved_source_ids=resolved_source_ids,
            top_k=top_k,
            normalized_source=normalized_source,
            paper_family=paper_family,
        )
        return self.ensure_discover_result_coverage(
            results=results,
            lexical_query_forms=lexical_query_forms,
            top_k=top_k,
            normalized_source=normalized_source,
            paper_family=paper_family,
        )


__all__ = [
    "PaperCoverageDeps",
    "PaperCoverageSearchRuntime",
    "PaperCoverageService",
]
