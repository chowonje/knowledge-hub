from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from knowledge_hub.ai.ask_v2_support import (
    AskV2Route,
    accumulate_search_results as _accumulate_search_results,
    build_paper_selection_inputs as _build_paper_selection_inputs,
    build_vault_selection_inputs as _build_vault_selection_inputs,
    build_web_selection_inputs as _build_web_selection_inputs,
    clean_text as _clean_text,
    first_nonempty_search as _first_nonempty_search,
    paper_scope_from_filter as _paper_scope_from_filter,
    paper_scope_from_query as _paper_scope_from_query,
    vault_scope_from_query as _vault_scope_from_query,
)
from knowledge_hub.application.query_frame import normalize_query_frame_dict
from knowledge_hub.domain.ai_papers.query_plan import build_rule_query_plan, normalize_query_plan_dict
from knowledge_hub.domain.web_knowledge.internal_reference import build_internal_reference_cards


@dataclass(frozen=True)
class CardSelectionRequest:
    query: str
    route: AskV2Route
    limit: int
    metadata_filter: dict[str, Any] | None
    query_plan: dict[str, Any] | None = None
    query_frame: dict[str, Any] | None = None


class _BaseCardSelector:
    def __init__(self, service: Any, *, fallback_error: type[Exception]):
        self.service = service
        self.fallback_error = fallback_error

    def select(self, request: CardSelectionRequest) -> list[dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def _extend_with_search_results(
        candidates: list[dict[str, Any]],
        *,
        forms: list[str] | None,
        fallback_query: str,
        search_fn,
    ) -> list[dict[str, Any]]:
        return [
            *candidates,
            *_accumulate_search_results(
                forms=forms,
                fallback_query=fallback_query,
                search_fn=search_fn,
            ),
        ]

    @staticmethod
    def _merge_fallback_cards(
        candidates: list[dict[str, Any]],
        fallback_cards: list[dict[str, Any]],
        *,
        prepend: bool = False,
        replace_when_empty: bool = False,
    ) -> list[dict[str, Any]]:
        if not fallback_cards:
            return list(candidates)
        if replace_when_empty and not candidates:
            return list(fallback_cards)
        if prepend:
            return [*fallback_cards, *candidates]
        return [*candidates, *fallback_cards]

    def _score_candidates(
        self,
        candidates: list[dict[str, Any]],
        *,
        query: str,
        entity_ids: list[str],
        source_kind: str,
        identity_key: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        return self.service._dedupe_and_score(
            candidates,
            query=query,
            entity_ids=entity_ids,
            source_kind=source_kind,
            identity_key=identity_key,
            limit=limit,
        )


class _PaperCardSelector(_BaseCardSelector):
    @staticmethod
    def _expanded_terms(
        *,
        frame_payload: dict[str, Any] | None,
        query_plan_payload: dict[str, Any] | None,
    ) -> list[str]:
        frame = dict(frame_payload or {})
        query_plan = dict(query_plan_payload or {})
        return [
            _clean_text(item)
            for item in [
                *list(frame.get("expanded_terms") or []),
                *list(query_plan.get("expandedTerms") or []),
                *list(query_plan.get("expanded_terms") or []),
            ]
            if _clean_text(item)
        ]

    @staticmethod
    def _looks_like_paper_title(value: str) -> bool:
        token = _clean_text(value)
        if not token:
            return False
        if ":" in token or len(token.split()) >= 4:
            return True
        return bool(re.search(r"\d{4}\.\d{4,5}", token))

    @staticmethod
    def _card_matches_focus_form(card: dict[str, Any], focus_form: str) -> bool:
        title = _clean_text(card.get("title")).casefold()
        target = _clean_text(focus_form).casefold()
        if not title or not target:
            return False
        return target in title or title in target

    def _select_compare_cards(
        self,
        *,
        request: CardSelectionRequest,
        candidates: list[dict[str, Any]],
        focus_forms: list[str],
        resolved_paper_ids: list[str],
    ) -> list[dict[str, Any]]:
        query = request.query
        route = request.route
        limit = request.limit
        ranked_candidates = self._score_candidates(
            candidates,
            query=query,
            entity_ids=route.entity_ids,
            source_kind="paper",
            identity_key="paper_id",
            limit=max(limit * 4, 8),
        )
        compare_limit = 2
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for focus_form in focus_forms:
            for card in ranked_candidates:
                paper_id = _clean_text(card.get("paper_id"))
                if not paper_id or paper_id in seen:
                    continue
                if not self._card_matches_focus_form(card, focus_form):
                    continue
                selected.append(card)
                seen.add(paper_id)
                if len(selected) >= compare_limit:
                    return selected[:compare_limit]
                break
        for paper_id in resolved_paper_ids:
            token = _clean_text(paper_id)
            if not token or token in seen:
                continue
            for card in ranked_candidates:
                if _clean_text(card.get("paper_id")) != token:
                    continue
                selected.append(card)
                seen.add(token)
                if len(selected) >= compare_limit:
                    return selected[:compare_limit]
                break
        for card in ranked_candidates:
            paper_id = _clean_text(card.get("paper_id"))
            if not paper_id or paper_id in seen:
                continue
            selected.append(card)
            seen.add(paper_id)
            if len(selected) >= compare_limit:
                break
        return selected[:compare_limit]

    def select(self, request: CardSelectionRequest) -> list[dict[str, Any]]:
        query = request.query
        route = request.route
        limit = request.limit
        metadata_filter = request.metadata_filter
        query_plan = request.query_plan
        query_frame = request.query_frame

        scoped_paper_id = _paper_scope_from_filter(metadata_filter) or _paper_scope_from_query(query)
        if scoped_paper_id:
            card = self.service._ensure_paper_card(scoped_paper_id)
            return [card] if card else []
        frame_payload = normalize_query_frame_dict(query_frame)
        query_plan_payload = normalize_query_plan_dict(query_plan)
        if not query_plan_payload:
            query_plan_payload = build_rule_query_plan(
                query,
                source_type="paper",
                metadata_filter=metadata_filter,
                sqlite_db=self.service.sqlite_db,
            ).to_dict()
        selection_inputs = _build_paper_selection_inputs(
            query=query,
            frame_payload=frame_payload,
            query_plan_payload=query_plan_payload,
        )
        expanded_terms = self._expanded_terms(
            frame_payload=frame_payload,
            query_plan_payload=query_plan_payload,
        )
        resolved_paper_ids = list(selection_inputs.get("resolved_paper_ids") or [])
        paper_family = _clean_text(selection_inputs.get("family"))
        if (route.intent == "paper_lookup" or paper_family == "paper_lookup") and resolved_paper_ids:
            resolved_cards = self.service._ensure_cards_for_papers(resolved_paper_ids)
            if resolved_cards:
                return self._score_candidates(
                    resolved_cards,
                    query=query,
                    entity_ids=route.entity_ids,
                    source_kind="paper",
                    identity_key="paper_id",
                    limit=limit,
                )
        deduped_lookup_forms = list(selection_inputs.get("lookup_forms") or [])
        candidates: list[dict[str, Any]] = []
        if resolved_paper_ids:
            candidates.extend(self.service._ensure_cards_for_papers(resolved_paper_ids))
        compare_focus_forms: list[str] = []
        if paper_family == "paper_compare":
            compare_focus_forms = [
                token
                for token in expanded_terms
                if self._looks_like_paper_title(token)
            ][:4]
            if compare_focus_forms:
                candidates = self._extend_with_search_results(
                    candidates,
                    forms=compare_focus_forms,
                    fallback_query=query,
                    search_fn=lambda form: self.service.sqlite_db.search_paper_cards_v2(form, limit=max(limit * 4, 8)),
                )
        if route.mode == "ontology-first" and route.entity_ids:
            candidates.extend(self.service.sqlite_db.list_paper_cards_v2_by_entity_ids(entity_ids=route.entity_ids, limit=max(limit * 3, 6)))
            if candidates:
                reranked = _first_nonempty_search(
                    forms=deduped_lookup_forms,
                    fallback_query=query,
                    search_fn=lambda form: self.service.sqlite_db.search_paper_cards_v2(
                        form,
                        limit=max(limit * 3, 6),
                        paper_ids=[item["paper_id"] for item in candidates],
                    ),
                )
                if reranked:
                    candidates = reranked
        else:
            candidates = self._extend_with_search_results(
                candidates,
                forms=deduped_lookup_forms,
                fallback_query=query,
                search_fn=lambda form: self.service.sqlite_db.search_paper_cards_v2(form, limit=max(limit * 3, 6)),
            )
            if route.intent == "paper_lookup" and candidates:
                return self._score_candidates(
                    candidates,
                    query=query,
                    entity_ids=route.entity_ids,
                    source_kind="paper",
                    identity_key="paper_id",
                    limit=limit,
                )
        if len(candidates) < limit:
            fallback_ids = self.service._fallback_paper_ids(query=query, route=route, limit=max(limit * 2, 6))
            fallback_cards = self.service._ensure_cards_for_papers(fallback_ids)
            candidates = self._extend_with_search_results(
                candidates,
                forms=deduped_lookup_forms,
                fallback_query=query,
                search_fn=lambda form: self.service.sqlite_db.search_paper_cards_v2(form, limit=max(limit * 4, 8)),
            )
            candidates = self._merge_fallback_cards(candidates, fallback_cards, replace_when_empty=True)
        if paper_family == "paper_compare" and len(resolved_paper_ids) >= 2:
            selected = self._select_compare_cards(
                request=request,
                candidates=candidates,
                focus_forms=compare_focus_forms,
                resolved_paper_ids=resolved_paper_ids,
            )
            if len(selected) >= 2:
                return selected[:2]
            raise self.fallback_error("insufficient_compare_cards")
        return self._score_candidates(
            candidates,
            query=query,
            entity_ids=route.entity_ids,
            source_kind="paper",
            identity_key="paper_id",
            limit=limit,
        )


class _WebCardSelector(_BaseCardSelector):
    def select(self, request: CardSelectionRequest) -> list[dict[str, Any]]:
        query = request.query
        route = request.route
        limit = request.limit
        metadata_filter = request.metadata_filter
        query_plan = request.query_plan
        query_frame = request.query_frame

        frame_payload = normalize_query_frame_dict(query_frame)
        query_plan_payload = normalize_query_plan_dict(query_plan)
        selection_inputs = _build_web_selection_inputs(
            query=query,
            frame_payload=frame_payload,
            query_plan_payload=query_plan_payload,
            metadata_filter=metadata_filter,
        )
        effective_metadata_filter = dict(selection_inputs.get("effective_metadata_filter") or {})
        family = _clean_text(selection_inputs.get("family"))
        media_platform = _clean_text(selection_inputs.get("media_platform"))
        resolved_urls = list(selection_inputs.get("resolved_urls") or [])
        resolved_doc_ids = list(selection_inputs.get("resolved_doc_ids") or [])

        scoped_url = self.service._resolve_web_url(effective_metadata_filter, query) or next(iter(resolved_urls), "")
        if scoped_url:
            card = self.service._ensure_web_card(scoped_url)
            return [card] if card else []
        if resolved_doc_ids:
            scoped_cards = self.service._resolve_web_cards(document_ids=resolved_doc_ids)
            if scoped_cards:
                return scoped_cards[:1]

        search_forms = list(selection_inputs.get("search_forms") or [])
        document_scope = list(selection_inputs.get("document_scope") or [])
        prefer_materialized_fallback = bool(selection_inputs.get("prefer_materialized_fallback"))
        candidates: list[dict[str, Any]] = []
        internal_reference_cards = [] if media_platform.casefold() == "youtube" else build_internal_reference_cards(query, limit=max(limit, 3))
        if route.mode == "ontology-first" and route.entity_ids:
            candidates.extend(self.service.sqlite_db.list_web_cards_v2_by_entity_ids(entity_ids=route.entity_ids, limit=max(limit * 3, 6)))
            if candidates:
                scoped_document_ids = document_scope or [_clean_text(item.get("document_id")) for item in candidates if _clean_text(item.get("document_id"))]
                reranked = _first_nonempty_search(
                    forms=search_forms,
                    fallback_query=query,
                    search_fn=lambda form: self.service.sqlite_db.search_web_cards_v2(
                        form,
                        limit=max(limit * 3, 6),
                        document_ids=scoped_document_ids or None,
                    ),
                )
                if reranked:
                    candidates = reranked
        else:
            candidates = self._extend_with_search_results(
                candidates,
                forms=search_forms,
                fallback_query=query,
                search_fn=lambda form: self.service.sqlite_db.search_web_cards_v2(
                    form,
                    limit=max(limit * 3, 6),
                    document_ids=document_scope or None,
                ),
            )
        if len(candidates) < limit or prefer_materialized_fallback:
            fallback_forms = [token for token in search_forms if _clean_text(token)]
            urls = self.service._fallback_web_urls(
                query=query,
                route=route,
                limit=max(limit * 2, 6),
                lookup_forms=fallback_forms,
                media_platform=media_platform,
            )
            fallback_cards = self.service._ensure_cards_for_web(urls)
            if len(candidates) < limit:
                candidates = self._extend_with_search_results(
                    candidates,
                    forms=search_forms,
                    fallback_query=query,
                    search_fn=lambda form: self.service.sqlite_db.search_web_cards_v2(
                        form,
                        limit=max(limit * 4, 8),
                        document_ids=document_scope or None,
                    ),
                )
            candidates = self._merge_fallback_cards(candidates, fallback_cards, prepend=True)
        if media_platform:
            candidates = [card for card in candidates if self.service._card_matches_media_platform(dict(card or {}), media_platform)]
        if internal_reference_cards:
            candidates = [*internal_reference_cards, *candidates]
        return self._score_candidates(
            candidates,
            query=query,
            entity_ids=route.entity_ids,
            source_kind="web",
            identity_key="document_id",
            limit=limit,
        )


class _VaultCardSelector(_BaseCardSelector):
    def select(self, request: CardSelectionRequest) -> list[dict[str, Any]]:
        query = request.query
        limit = request.limit
        metadata_filter = request.metadata_filter
        query_plan = request.query_plan
        query_frame = request.query_frame

        frame_payload = normalize_query_frame_dict(query_frame)
        query_plan_payload = normalize_query_plan_dict(query_plan)
        selection_inputs = _build_vault_selection_inputs(
            query=query,
            frame_payload=frame_payload,
            query_plan_payload=query_plan_payload,
            metadata_filter=metadata_filter,
        )
        resolved_note_ids = list(selection_inputs.get("resolved_note_ids") or [])
        scoped_note_id = _clean_text(selection_inputs.get("scoped_note_id"))
        scoped_file_path = _clean_text(selection_inputs.get("scoped_file_path"))
        query_scope = _vault_scope_from_query(query)
        if query_scope.startswith("vault:"):
            if not scoped_note_id:
                scoped_note_id = query_scope
        elif not scoped_file_path:
            scoped_file_path = query_scope
        if scoped_note_id or scoped_file_path:
            scoped_cards = self.service._resolve_vault_cards(
                note_ids=[scoped_note_id] if scoped_note_id else None,
                file_paths=[scoped_file_path] if scoped_file_path else None,
            )
            if scoped_cards:
                return scoped_cards[:1]
            return []
        search_forms = list(selection_inputs.get("search_forms") or [])
        candidates: list[dict[str, Any]] = []
        if resolved_note_ids:
            candidates.extend([item for item in self.service._ensure_cards_for_vault(resolved_note_ids) if item])
        candidates = self._extend_with_search_results(
            candidates,
            forms=search_forms,
            fallback_query=query,
            search_fn=lambda form: self.service.sqlite_db.search_vault_cards_v2(
                form,
                limit=max(limit * 3, 6),
                note_ids=resolved_note_ids or None,
            ),
        )
        if len(candidates) < limit:
            note_ids = self.service._fallback_vault_note_ids(query=query, limit=max(limit * 2, 6), lookup_forms=search_forms)
            fallback_cards = self.service._ensure_cards_for_vault(note_ids)
            candidates = self._extend_with_search_results(
                candidates,
                forms=search_forms,
                fallback_query=query,
                search_fn=lambda form: self.service.sqlite_db.search_vault_cards_v2(
                    form,
                    limit=max(limit * 4, 8),
                    note_ids=resolved_note_ids or None,
                ),
            )
            candidates = self._merge_fallback_cards(candidates, fallback_cards, prepend=True)
        return self._score_candidates(
            candidates,
            query=query,
            entity_ids=[],
            source_kind="vault",
            identity_key="note_id",
            limit=limit,
        )


class AskV2CardSelectorRegistry:
    def __init__(self, service: Any, *, fallback_error: type[Exception]):
        self.service = service
        self._selectors = {
            "paper": _PaperCardSelector(service, fallback_error=fallback_error),
            "web": _WebCardSelector(service, fallback_error=fallback_error),
            "vault": _VaultCardSelector(service, fallback_error=fallback_error),
        }

    def get(self, source_kind: str) -> _BaseCardSelector:
        token = _clean_text(source_kind).casefold()
        selector = self._selectors.get(token)
        if selector is None:
            raise ValueError(f"unsupported ask_v2 card selector source_kind: {source_kind}")
        return selector

    def select(
        self,
        *,
        source_kind: str,
        query: str,
        route: AskV2Route,
        limit: int,
        metadata_filter: dict[str, Any] | None,
        query_plan: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.get(source_kind).select(
            CardSelectionRequest(
                query=query,
                route=route,
                limit=limit,
                metadata_filter=metadata_filter,
                query_plan=query_plan,
                query_frame=query_frame,
            )
        )


__all__ = ["AskV2CardSelectorRegistry", "CardSelectionRequest"]
