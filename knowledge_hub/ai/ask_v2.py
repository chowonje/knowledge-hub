from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.ai.ask_v2_support import (
    AskV2Route,
    _TEMPORAL_RE,
    build_paper_selection_inputs as _build_paper_selection_inputs,
    build_project_cards,
    classify_intent as _classify_intent,
    classify_project_query_profile as _classify_project_query_profile,
    clean_text as _clean_text,
    collect_project_fallback_files as _collect_project_fallback_files,
    detect_project_file_role as _detect_project_file_role,
    parse_note_metadata as _parse_note_metadata,
    query_terms as _query_terms,
    repo_scope_from_filter as _repo_scope_from_filter,
    route_mode as _route_mode,
    slot_coverage as _slot_coverage,
    source_kind as _source_kind,
    stable_score as _stable_score,
    should_attempt_claim_cards as _should_attempt_claim_cards,
    text_overlap as _text_overlap,
    web_scope_from_filter as _web_scope_from_filter,
    web_scope_from_query as _web_scope_from_query,
)
from knowledge_hub.ai.ask_v2_card_selectors import AskV2CardSelectorRegistry
from knowledge_hub.ai.ask_v2_pipeline_result import build_card_v2_pipeline_result
from knowledge_hub.ai.ask_v2_verification import AskV2Verifier
from knowledge_hub.ai.evidence_assembly import EvidenceAssemblyService
from knowledge_hub.ai.retrieval_pipeline import RetrievalPlan, RetrievalPipelineResult
from knowledge_hub.ai.section_cards import assess_section_source_quality, project_section_cards, rank_section_cards, section_coverage
from knowledge_hub.application.card_v2_registry import CardV2BuilderRegistry
from knowledge_hub.application.context_pack import _collect_workspace_context
from knowledge_hub.application.query_frame import (
    QUERY_FRAME_FAMILY_FROM_FRAME,
    QUERY_FRAME_FAMILY_FROM_PACK,
    build_query_frame,
    family_supported_for_source,
    normalize_query_frame_dict,
    query_frame_is_authoritative,
    query_frame_lock_mask,
    query_intent_supported_for_family,
)
from knowledge_hub.domain.registry import get_domain_pack, normalize_domain_source
from knowledge_hub.domain.ai_papers.claim_cards import ClaimCardAlignmentService, ClaimCardBuilder, build_project_claim_cards, rank_claim_cards
from knowledge_hub.domain.ai_papers.families import PAPER_FAMILY_COMPARE, PAPER_FAMILY_CONCEPT_EXPLAINER
from knowledge_hub.domain.ai_papers.query_plan import normalize_query_plan_dict, query_frame_from_query_plan
from knowledge_hub.infrastructure.persistence.stores.section_card_v1_store import SectionCardV1Store
from knowledge_hub.core.models import SearchResult
from knowledge_hub.papers.memory_adapter import paper_memory_card_to_section_cards
from knowledge_hub.papers.memory_retriever import PaperMemoryRetriever
from knowledge_hub.papers.source_text import source_hash_for_path, source_hash_for_text
from knowledge_hub.web.youtube_extractor import is_youtube_url


class AskV2FallbackToLegacy(RuntimeError):
    pass


_CLAIM_HEAVY_QUERY_RE = re.compile(
    r"\b(compare|comparison|versus|vs|difference|benchmark|metric|evaluate|evaluation|performance|accuracy)\b|비교|차이|결과|평가|성능|지표",
    re.IGNORECASE,
)


def _ask_v2_hard_gate_reason(
    *,
    verification: dict[str, Any],
    claim_consensus: dict[str, Any],
) -> str:
    status = _clean_text(verification.get("verificationStatus")).casefold()
    unsupported_fields = [_clean_text(item) for item in list(verification.get("unsupportedFields") or []) if _clean_text(item)]
    if status in {"missing", "no_evidence"}:
        return f"ask_v2_{status}"
    if status == "weak" and unsupported_fields:
        return f"ask_v2_weak_evidence:{unsupported_fields[0]}"
    if int(claim_consensus.get("unsupportedClaimCount") or 0) > 0:
        return "ask_v2_unsupported_claim_cards"
    return ""


class AskV2Service:
    def __init__(self, searcher: Any):
        self.searcher = searcher
        self.sqlite_db = searcher.sqlite_db
        self.card_builders = CardV2BuilderRegistry(self.sqlite_db)
        self.card_selectors = AskV2CardSelectorRegistry(self, fallback_error=AskV2FallbackToLegacy)
        self._claim_card_builder: ClaimCardBuilder | None = None
        self.claim_alignment = ClaimCardAlignmentService()
        self.verifier = AskV2Verifier(self.sqlite_db)

    @property
    def claim_card_builder(self) -> ClaimCardBuilder:
        builder = self._claim_card_builder
        if builder is None:
            builder = ClaimCardBuilder(self.sqlite_db)
            self._claim_card_builder = builder
        return builder

    @staticmethod
    def _use_section_first(*, route: AskV2Route, query: str, mode_override: str | None = None) -> bool:
        override = _clean_text(mode_override).casefold()
        if override == "claim_first":
            return False
        if override == "section_first":
            return route.source_kind == "paper"
        return route.source_kind == "paper" and route.intent in {
            "paper_summary",
            "paper_lookup",
            "definition",
            "relation",
            "implementation",
            "temporal",
        } and not _CLAIM_HEAVY_QUERY_RE.search(str(query or ""))

    @staticmethod
    def supports(*, source_type: str | None, metadata_filter: dict[str, Any] | None = None) -> bool:
        return bool(_source_kind(source_type, metadata_filter))

    def _resolve_entities(self, query: str) -> list[dict[str, Any]]:
        return list(getattr(self.searcher, "_resolve_query_entities", lambda _query: [])(query) or [])

    def _resolve_frame_authority(
        self,
        *,
        query: str,
        source_type: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = normalize_query_frame_dict(query_frame)
        if not payload:
            return {}
        explicit_source = normalize_domain_source(payload.get("source_type"))
        fallback_source = normalize_domain_source(source_type)
        effective_source = explicit_source or fallback_source
        effective_filter = dict(payload.get("metadata_filter") or metadata_filter or {})
        domain_pack = get_domain_pack(source_type=effective_source)
        base_frame = None
        if domain_pack is not None and hasattr(domain_pack, "normalize") and effective_source:
            try:
                base_frame = domain_pack.normalize(
                    query,
                    source_type=effective_source,
                    metadata_filter=effective_filter,
                    sqlite_db=self.sqlite_db,
                )
            except Exception:
                base_frame = None
        base_payload = normalize_query_frame_dict(base_frame) if base_frame is not None else {}
        lock_mask = list(query_frame_lock_mask(payload))
        family = str(payload.get("family") or "").strip().lower()
        family_source = QUERY_FRAME_FAMILY_FROM_FRAME if family else QUERY_FRAME_FAMILY_FROM_PACK
        overrides = list(payload.get("overrides_applied") or [])
        if family and not family_supported_for_source(effective_source, family):
            family = str(base_payload.get("family") or "").strip().lower()
            family_source = QUERY_FRAME_FAMILY_FROM_PACK
            overrides.append("INVALID_FAMILY")
        elif not family:
            family = str(base_payload.get("family") or "").strip().lower()
            family_source = str(base_payload.get("family_source") or QUERY_FRAME_FAMILY_FROM_PACK)
        query_intent = str(payload.get("query_intent") or "").strip()
        if query_intent and not query_intent_supported_for_family(effective_source, family, query_intent):
            query_intent = str(base_payload.get("query_intent") or "").strip()
            overrides.append("UNSUPPORTED_INTENT")
        elif not query_intent:
            query_intent = str(base_payload.get("query_intent") or "").strip()
        answer_mode = str(payload.get("answer_mode") or "").strip()
        if (not answer_mode) or ("INVALID_FAMILY" in overrides) or ("UNSUPPORTED_INTENT" in overrides):
            answer_mode = str(base_payload.get("answer_mode") or answer_mode or "").strip()
        entities = list(payload.get("entities") or base_payload.get("entities") or [])
        canonical_entity_ids = list(payload.get("canonical_entity_ids") or base_payload.get("canonical_entity_ids") or [])
        expanded_terms = list(payload.get("expanded_terms") or [])
        if not expanded_terms:
            expanded_terms = list(base_payload.get("expanded_terms") or [])
        else:
            for token in list(base_payload.get("expanded_terms") or []):
                if len(expanded_terms) >= 6:
                    break
                if str(token or "").strip() and token not in expanded_terms:
                    expanded_terms.append(token)
        resolved_source_ids = list(payload.get("resolved_source_ids") or [])
        if not resolved_source_ids:
            resolved_source_ids = list(base_payload.get("resolved_source_ids") or [])
        else:
            for token in list(base_payload.get("resolved_source_ids") or []):
                if len(resolved_source_ids) >= 6:
                    break
                if str(token or "").strip() and token not in resolved_source_ids:
                    resolved_source_ids.append(token)
        merged_filter = dict(base_payload.get("metadata_filter") or {})
        merged_filter.update(effective_filter)
        return build_query_frame(
            domain_key=str(payload.get("domain_key") or base_payload.get("domain_key") or ""),
            source_type=effective_source or explicit_source or fallback_source,
            family=family or str(base_payload.get("family") or ""),
            query_intent=query_intent or str(base_payload.get("query_intent") or ""),
            answer_mode=answer_mode or str(base_payload.get("answer_mode") or ""),
            entities=entities,
            canonical_entity_ids=canonical_entity_ids,
            expanded_terms=expanded_terms,
            resolved_source_ids=resolved_source_ids,
            confidence=float(payload.get("confidence") or base_payload.get("confidence") or 0.0),
            planner_status=str(payload.get("planner_status") or base_payload.get("planner_status") or "not_attempted"),
            planner_reason=str(payload.get("planner_reason") or base_payload.get("planner_reason") or "rule_based"),
            evidence_policy_key=str(payload.get("evidence_policy_key") or base_payload.get("evidence_policy_key") or ""),
            metadata_filter=merged_filter,
            frame_provenance=str(payload.get("frame_provenance") or base_payload.get("frame_provenance") or "derived"),
            trusted=bool(payload.get("trusted")),
            lock_mask=lock_mask,
            family_source=family_source,
            overrides_applied=overrides,
        ).to_dict()

    def _route(
        self,
        *,
        query: str,
        source_type: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
    ) -> AskV2Route:
        frame_payload = normalize_query_frame_dict(query_frame)
        effective_metadata_filter = dict(frame_payload.get("metadata_filter") or metadata_filter or {})
        route_source_type = str(frame_payload.get("source_type") or source_type or "")
        source_kind = _source_kind(route_source_type, effective_metadata_filter)
        classified_intent = _classify_intent(query, effective_metadata_filter)
        frame_authoritative = query_frame_is_authoritative(frame_payload)
        lock_mask = set(query_frame_lock_mask(frame_payload))
        frame_family = _clean_text(frame_payload.get("family")).lower() if source_kind in {"paper", "web", "vault"} else ""
        intent = _clean_text(frame_payload.get("query_intent")) if source_kind in {"paper", "web", "vault"} else ""
        if not intent:
            intent = classified_intent
        elif (
            source_kind == "paper"
            and frame_family == "paper_lookup"
            and classified_intent not in {"temporal", "relation"}
        ):
            intent = "paper_lookup"
        elif source_kind in {"paper", "web", "vault"} and classified_intent in {"temporal", "relation", "implementation", "evaluation", "comparison"}:
            frame_qi = _clean_text(frame_payload.get("query_intent")).casefold()
            if frame_authoritative and "query_intent" in lock_mask and frame_qi:
                pass
            elif (
                source_kind == "paper"
                and frame_family == PAPER_FAMILY_CONCEPT_EXPLAINER
                and frame_qi == "definition"
                and classified_intent in {"implementation", "evaluation"}
            ) or (
                source_kind == "paper"
                and frame_family == PAPER_FAMILY_COMPARE
                and frame_qi == "comparison"
                and classified_intent in {"implementation", "evaluation"}
            ):
                pass
            else:
                intent = classified_intent
        matched_entities = self._resolve_entities(query)
        entity_ids: list[str] = []
        for item in matched_entities:
            entity_id = _clean_text(item.get("entity_id"))
            if entity_id and entity_id not in entity_ids:
                entity_ids.append(entity_id)
            if _clean_text(item.get("entity_type")) == "concept":
                for related in list(self.sqlite_db.get_related_concepts(entity_id) or [])[:6]:
                    related_id = _clean_text(related.get("entity_id") or related.get("id"))
                    if related_id and related_id not in entity_ids:
                        entity_ids.append(related_id)
        return AskV2Route(
            source_kind=source_kind,
            intent=intent,
            mode=_route_mode(source_kind_value=source_kind, intent=intent, matched_entities=matched_entities, query=query),
            matched_entities=matched_entities,
            entity_ids=entity_ids,
        )

    def _ensure_paper_card(self, paper_id: str) -> dict[str, Any] | None:
        return self.card_builders.resolve_scoped_card(source_kind="paper", source_id=paper_id)

    def _resolve_web_url(self, metadata_filter: dict[str, Any] | None, query: str) -> str:
        scoped = _web_scope_from_filter(metadata_filter) or _web_scope_from_query(query)
        if scoped:
            return scoped
        note_id = _clean_text((metadata_filter or {}).get("document_id") or (metadata_filter or {}).get("note_id"))
        if note_id:
            note = self.sqlite_db.get_note(note_id) or {}
            metadata = _parse_note_metadata(note)
            return _clean_text(metadata.get("canonical_url") or metadata.get("source_url") or metadata.get("url"))
        return ""

    @staticmethod
    def _url_matches_media_platform(url: str, media_platform: str) -> bool:
        normalized_platform = _clean_text(media_platform).casefold()
        url_text = _clean_text(url)
        if not normalized_platform:
            return True
        if normalized_platform == "youtube":
            return bool(url_text and is_youtube_url(url_text))
        return True

    def _note_matches_media_platform(self, note_id: str, media_platform: str) -> bool:
        normalized_platform = _clean_text(media_platform).casefold()
        if not normalized_platform:
            return True
        note = self.sqlite_db.get_note(note_id) or {}
        metadata = _parse_note_metadata(note)
        note_platform = _clean_text(metadata.get("media_platform")).casefold()
        if note_platform:
            return note_platform == normalized_platform
        return self._url_matches_media_platform(
            _clean_text(metadata.get("canonical_url") or metadata.get("source_url") or metadata.get("url")),
            normalized_platform,
        )

    def _card_matches_media_platform(self, card: dict[str, Any], media_platform: str) -> bool:
        normalized_platform = _clean_text(media_platform).casefold()
        if not normalized_platform:
            return True
        if self._url_matches_media_platform(_clean_text(card.get("canonical_url") or card.get("source_url")), normalized_platform):
            return True
        document_id = _clean_text(card.get("document_id"))
        if document_id and self._note_matches_media_platform(document_id, normalized_platform):
            return True
        return False

    def _ensure_web_card(self, url: str) -> dict[str, Any] | None:
        return self.card_builders.resolve_scoped_card(source_kind="web", source_id=url)

    def _ensure_vault_card(self, note_id: str) -> dict[str, Any] | None:
        return self.card_builders.resolve_scoped_card(source_kind="vault", source_id=note_id)

    def _ensure_cards(self, *, source_kind: str, source_ids: list[str]) -> list[dict[str, Any]]:
        return self.card_builders.resolve_scoped_cards(source_kind=source_kind, source_ids=source_ids)

    def _ensure_cards_for_papers(self, paper_ids: list[str]) -> list[dict[str, Any]]:
        return self._ensure_cards(source_kind="paper", source_ids=paper_ids)

    def _ensure_cards_for_web(self, urls: list[str]) -> list[dict[str, Any]]:
        return self._ensure_cards(source_kind="web", source_ids=urls)

    def _ensure_cards_for_vault(self, note_ids: list[str]) -> list[dict[str, Any]]:
        return self._ensure_cards(source_kind="vault", source_ids=note_ids)

    def _resolve_web_cards(self, *, urls: list[str] | None = None, document_ids: list[str] | None = None) -> list[dict[str, Any]]:
        return self.card_builders.resolve_scoped_cards(
            source_kind="web",
            source_ids=urls,
            document_ids=document_ids,
        )

    def _resolve_vault_cards(
        self,
        *,
        note_ids: list[str] | None = None,
        file_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self.card_builders.resolve_scoped_cards(
            source_kind="vault",
            source_ids=note_ids,
            file_paths=file_paths,
        )

    def _fallback_paper_ids(self, *, query: str, route: AskV2Route, limit: int) -> list[str]:
        paper_ids: list[str] = []
        if route.mode == "ontology-first":
            for entity_id in route.entity_ids:
                entity = self.sqlite_db.get_ontology_entity(entity_id) or {}
                if _clean_text(entity.get("entity_type")) != "concept":
                    continue
                for row in list(self.sqlite_db.get_concept_papers(entity_id) or [])[:limit]:
                    paper_id = _clean_text(row.get("arxiv_id") or row.get("paper_id"))
                    if paper_id and paper_id not in paper_ids:
                        paper_ids.append(paper_id)
        if paper_ids:
            return paper_ids[:limit]
        memory_hits = PaperMemoryRetriever(self.sqlite_db).search(query, limit=max(limit, 5), include_refs=False)
        for row in memory_hits:
            paper_id = _clean_text(row.get("paperId"))
            if paper_id and paper_id not in paper_ids:
                paper_ids.append(paper_id)
        return paper_ids[:limit]

    def _fallback_web_urls(
        self,
        *,
        query: str,
        route: AskV2Route,
        limit: int,
        lookup_forms: list[str] | None = None,
        media_platform: str | None = None,
    ) -> list[str]:
        urls: list[str] = []
        normalized_platform = _clean_text(media_platform).casefold()
        search_forms: list[str] = []
        for token in [*(lookup_forms or []), query]:
            cleaned = _clean_text(token)
            if cleaned and cleaned not in search_forms:
                search_forms.append(cleaned)
        if route.mode == "ontology-first":
            for entity_id in route.entity_ids:
                for card in self.sqlite_db.list_web_cards_v2_by_entity_ids(entity_ids=[entity_id], limit=limit):
                    url = _clean_text(card.get("canonical_url"))
                    if normalized_platform and not self._card_matches_media_platform(dict(card or {}), normalized_platform):
                        continue
                    if url and url not in urls:
                        urls.append(url)
        if urls:
            return urls[:limit]
        for form in search_forms:
            rows = self.sqlite_db.search_document_memory_units(form, limit=max(limit * 5, 10), unit_types=["document_summary"])
            for row in rows:
                if _clean_text(row.get("source_type")) != "web":
                    continue
                document_id = _clean_text(row.get("document_id"))
                if normalized_platform and document_id and not self._note_matches_media_platform(document_id, normalized_platform):
                    continue
                url = _clean_text(row.get("source_ref"))
                if normalized_platform and not self._url_matches_media_platform(url, normalized_platform):
                    continue
                if url and url not in urls:
                    urls.append(url)
            if len(urls) >= limit:
                return urls[:limit]
        for form in search_forms:
            for row in self.sqlite_db.search_notes(form, limit=max(limit * 4, 12)):
                if _source_kind(row.get("source_type")) != "web":
                    continue
                metadata = _parse_note_metadata(row)
                if normalized_platform and _clean_text(metadata.get("media_platform")).casefold() not in {"", normalized_platform}:
                    continue
                url = _clean_text(metadata.get("canonical_url") or metadata.get("source_url") or metadata.get("url"))
                if normalized_platform and not self._url_matches_media_platform(url, normalized_platform):
                    continue
                if url and url not in urls:
                    urls.append(url)
            if len(urls) >= limit:
                return urls[:limit]
        if len(urls) < limit:
            for form in search_forms:
                bounded_terms: list[str] = []
                for candidate in [form, *re.findall(r"[A-Za-z0-9._+-]+|[가-힣]+", form)]:
                    term_text = _clean_text(candidate)
                    if not term_text or len(term_text) < 3:
                        continue
                    if any(existing.casefold() == term_text.casefold() for existing in bounded_terms):
                        continue
                    bounded_terms.append(term_text)
                    if len(bounded_terms) >= 3:
                        break
                for term in bounded_terms:
                    pattern = f"%{term}%"
                    rows = self.sqlite_db.conn.execute(
                        """
                        SELECT id, title, metadata
                        FROM notes
                        WHERE source_type = 'web'
                          AND (title LIKE ? OR content LIKE ?)
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (pattern, pattern, max(limit * 4, 12)),
                    ).fetchall()
                    for row in rows:
                        metadata = _parse_note_metadata(dict(row))
                        if normalized_platform and _clean_text(metadata.get("media_platform")).casefold() not in {"", normalized_platform}:
                            continue
                        url = _clean_text(metadata.get("canonical_url") or metadata.get("source_url") or metadata.get("url"))
                        if normalized_platform and not self._url_matches_media_platform(url, normalized_platform):
                            continue
                        if url and url not in urls:
                            urls.append(url)
                    if len(urls) >= limit:
                        return urls[:limit]
        return urls[:limit]

    def _fallback_vault_note_ids(self, *, query: str, limit: int, lookup_forms: list[str] | None = None) -> list[str]:
        note_ids: list[str] = []
        forms = [_clean_text(item) for item in list(lookup_forms or []) if _clean_text(item)]
        if not forms:
            forms = [_clean_text(query)] if _clean_text(query) else []
        for form in forms:
            rows = self.sqlite_db.search_document_memory_units(form, limit=max(limit * 5, 10), unit_types=["document_summary"])
            for row in rows:
                if _source_kind(row.get("source_type")) != "vault":
                    continue
                note_id = _clean_text(row.get("document_id"))
                if note_id and note_id not in note_ids:
                    note_ids.append(note_id)
            if len(note_ids) >= limit:
                return note_ids[:limit]
        for form in forms:
            for row in self.sqlite_db.search_notes(form, limit=max(limit * 3, 10)):
                if _source_kind(row.get("source_type")) != "vault":
                    continue
                note_id = _clean_text(row.get("id"))
                if note_id and note_id not in note_ids:
                    note_ids.append(note_id)
        return note_ids[:limit]

    def _score_card(self, card: dict[str, Any], *, query: str, entity_ids: list[str], source_kind: str) -> float:
        score = _stable_score(card.get("match_score"))
        if source_kind == "paper":
            score += _text_overlap(query, card.get("title"), card.get("paper_core"), card.get("problem_core"), card.get("method_core"), card.get("result_core"), card.get("search_text"))
            if entity_ids:
                refs = self.sqlite_db.list_paper_card_entity_refs_v2(card_id=_clean_text(card.get("card_id")))
                for ref in refs:
                    if _clean_text(ref.get("entity_id")) in entity_ids:
                        score += 2.0 + _stable_score(ref.get("weight"))
        elif source_kind == "web":
            score += _text_overlap(query, card.get("title"), card.get("page_core"), card.get("topic_core"), card.get("result_core"), card.get("version_core"), card.get("search_text"), card.get("canonical_url"))
            if entity_ids:
                refs = self.sqlite_db.list_web_card_entity_refs_v2(card_id=_clean_text(card.get("card_id")))
                for ref in refs:
                    if _clean_text(ref.get("entity_id")) in entity_ids:
                        score += 2.0 + _stable_score(ref.get("weight"))
        elif source_kind == "vault":
            score += _text_overlap(query, card.get("title"), card.get("note_core"), card.get("concept_core"), card.get("decision_core"), card.get("action_core"), card.get("search_text"), card.get("file_path"))
        else:
            profile = _classify_project_query_profile(query)
            file_role = _clean_text(card.get("file_role_core") or _detect_project_file_role(_clean_text(card.get("relative_path"))))
            symbol_overlap = _text_overlap(query, card.get("symbol_owner_core"))
            call_overlap = _text_overlap(query, card.get("call_flow_core"))
            integration_overlap = _text_overlap(query, card.get("integration_boundary_core"))
            module_overlap = _text_overlap(query, card.get("title"), card.get("module_core"), card.get("search_text"), card.get("relative_path"))
            score += module_overlap + (1.3 * symbol_overlap) + (1.15 * call_overlap) + (1.1 * integration_overlap)
            if profile.get("architecture"):
                if file_role == "entrypoint":
                    score += 3.2
                elif file_role == "adapter":
                    score += 1.8
                elif file_role == "module":
                    score += 0.9
                elif file_role == "config":
                    score += 0.7
                elif file_role == "docs":
                    score -= 1.25
                elif file_role == "test":
                    score -= 1.1
            if profile.get("debug"):
                if file_role == "test":
                    score += 1.6
                elif file_role == "docs":
                    score -= 0.5
            else:
                if file_role == "test":
                    score -= 0.8
            if profile.get("overview"):
                if file_role in {"docs", "config"}:
                    score += 1.1
            elif file_role == "docs":
                score -= 1.0
            relative_path = _clean_text(card.get("relative_path")).casefold()
            basename = relative_path.rsplit("/", 1)[-1]
            query_terms = set(_query_terms(query))
            if basename and basename.casefold() in query.casefold():
                score += 2.0
            elif any(term and term in relative_path for term in query_terms):
                score += 0.8
            diagnostics = dict(card.get("diagnostics") or {})
            signal_count = int(diagnostics.get("signalCount") or 0)
            if signal_count <= 0:
                score -= 1.25
            elif signal_count == 1:
                score -= 0.35
        if str(card.get("quality_flag") or "").strip().lower() == "ok":
            score += 0.5
        return round(score, 4)

    def _select_paper_cards(
        self,
        *,
        query: str,
        route: AskV2Route,
        limit: int,
        metadata_filter: dict[str, Any] | None,
        query_plan: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.card_selectors.select(
            source_kind="paper",
            query=query,
            route=route,
            limit=limit,
            metadata_filter=metadata_filter,
            query_plan=query_plan,
            query_frame=query_frame,
        )

    def _select_web_cards(
        self,
        *,
        query: str,
        route: AskV2Route,
        limit: int,
        metadata_filter: dict[str, Any] | None,
        query_plan: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.card_selectors.select(
            source_kind="web",
            query=query,
            route=route,
            limit=limit,
            metadata_filter=metadata_filter,
            query_plan=query_plan,
            query_frame=query_frame,
        )

    def _select_vault_cards(
        self,
        *,
        query: str,
        route: AskV2Route,
        limit: int,
        metadata_filter: dict[str, Any] | None,
        query_plan: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.card_selectors.select(
            source_kind="vault",
            query=query,
            route=route,
            limit=limit,
            metadata_filter=metadata_filter,
            query_plan=query_plan,
            query_frame=query_frame,
        )

    def _repo_cards(self, *, query: str, metadata_filter: dict[str, Any] | None, limit: int) -> list[dict[str, Any]]:
        repo_path = _repo_scope_from_filter(metadata_filter) or str(Path.cwd())
        payload = _collect_workspace_context(
            goal=query,
            repo_path=Path(repo_path).expanduser().resolve(),
            max_workspace_files=max(limit * 3, 6),
            max_project_docs=max(limit * 2, 4),
            max_excerpt_chars=1200,
        )
        workspace_files = list(payload.get("workspace_files") or [])
        cards = build_project_cards(workspace_files=workspace_files, repo_path=repo_path)
        fallback_files = _collect_project_fallback_files(
            repo_path=repo_path,
            query=query,
            existing_relative_paths={_clean_text(item.get("relative_path")) for item in workspace_files},
            limit=max(limit * 4, 10),
            max_excerpt_chars=1200,
        )
        if fallback_files:
            cards.extend(build_project_cards(workspace_files=fallback_files, repo_path=repo_path))
        return self._dedupe_and_score(cards, query=query, entity_ids=[], source_kind="project", identity_key="relative_path", limit=limit)

    def _dedupe_and_score(
        self,
        candidates: list[dict[str, Any]],
        *,
        query: str,
        entity_ids: list[str],
        source_kind: str,
        identity_key: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        seen: set[str] = set()
        scored: list[dict[str, Any]] = []
        for card in candidates:
            identity = _clean_text(card.get(identity_key))
            if not identity or identity in seen:
                continue
            seen.add(identity)
            enriched = dict(card)
            enriched["selection_score"] = self._score_card(enriched, query=query, entity_ids=entity_ids, source_kind=source_kind)
            scored.append(enriched)
        scored.sort(key=lambda item: (-_stable_score(item.get("selection_score")), -_stable_score(item.get("match_score")), str(item.get("updated_at") or "")))
        return scored[:limit]

    def _select_cards(
        self,
        *,
        query: str,
        route: AskV2Route,
        limit: int,
        metadata_filter: dict[str, Any] | None,
        query_plan: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if route.source_kind in {"paper", "web", "vault"}:
            return self.card_selectors.select(
                source_kind=route.source_kind,
                query=query,
                route=route,
                limit=limit,
                metadata_filter=metadata_filter,
                query_plan=query_plan,
                query_frame=query_frame,
            )
        if route.source_kind == "project":
            return self._repo_cards(query=query, metadata_filter=metadata_filter, limit=limit)
        return []

    @staticmethod
    def _resolve_paper_family_and_answer_mode(
        *,
        route: AskV2Route,
        query_frame_payload: dict[str, Any] | None,
    ) -> tuple[str, str]:
        if route.source_kind != "paper":
            return "", ""
        if query_frame_payload:
            return (
                str(query_frame_payload.get("family") or "").strip().lower(),
                str(query_frame_payload.get("answer_mode") or "").strip(),
            )
        if route.intent == "paper_lookup":
            return "paper_lookup", "paper_scoped_answer"
        if route.intent == "comparison":
            return "paper_compare", "paper_claim_compare"
        if route.intent == "definition":
            return "concept_explainer", "representative_paper_explainer"
        if route.intent in {"paper_summary", "evaluation", "relation", "implementation", "temporal"}:
            return "paper_lookup", "paper_scoped_answer"
        return "paper_discover", "paper_shortlist_summary"

    @staticmethod
    def _is_missing_vault_scope(query_frame_payload: dict[str, Any] | None, metadata_filter: dict[str, Any] | None) -> bool:
        frame = dict(query_frame_payload or {})
        effective_filter = dict(frame.get("metadata_filter") or metadata_filter or {})
        return bool(
            _source_kind(frame.get("source_type") or effective_filter.get("source_type") or "vault", effective_filter) == "vault"
            and effective_filter.get("vault_scope_missing")
            and effective_filter.get("note_scope_required")
        )

    def _scoped_no_result_execution(
        self,
        *,
        query: str,
        top_k: int,
        source_type: str | None,
        retrieval_mode: str,
        route: AskV2Route,
        query_plan_payload: dict[str, Any],
        query_frame_payload: dict[str, Any],
        metadata_filter: dict[str, Any] | None,
        reason: str,
        paper_memory_mode: str = "off",
    ) -> tuple[RetrievalPipelineResult, Any]:
        frame_payload = normalize_query_frame_dict(query_frame_payload)
        plan_payload = dict(query_plan_payload or {})
        if frame_payload:
            plan_payload.setdefault("family", _clean_text(frame_payload.get("family")) or "general")
            plan_payload.setdefault("queryIntent", _clean_text(frame_payload.get("query_intent") or route.intent))
            plan_payload.setdefault("answerMode", _clean_text(frame_payload.get("answer_mode")) or "abstain")
            plan_payload.setdefault("confidence", float(frame_payload.get("confidence") or 0.0))
            plan_payload.setdefault("plannerStatus", _clean_text(frame_payload.get("planner_status")) or "not_attempted")
            plan_payload.setdefault("plannerReason", _clean_text(frame_payload.get("planner_reason")) or "rule_based")
            plan_payload.setdefault("evidencePolicyKey", _clean_text(frame_payload.get("evidence_policy_key")))
            plan_payload.setdefault("resolvedSourceIds", list(frame_payload.get("resolved_source_ids") or []))
            plan_payload.setdefault("expandedTerms", list(frame_payload.get("expanded_terms") or []))
        effective_metadata_filter = dict(frame_payload.get("metadata_filter") or metadata_filter or {})
        request_source_type = "project" if route.source_kind == "project" else source_type or route.source_kind
        plan = RetrievalPlan(
            query=query,
            source_scope=route.source_kind,
            query_intent=_clean_text(frame_payload.get("query_intent") or route.intent),
            paper_family=_clean_text(frame_payload.get("family") or plan_payload.get("family") or "general"),
            retrieval_mode=retrieval_mode,
            memory_mode=f"{route.source_kind}-card-v2",
            candidate_budgets={route.source_kind: max(4, top_k * 2)},
            query_plan=plan_payload,
            query_frame=frame_payload,
            temporal_signals={"enabled": False, "mode": "none", "targetYear": ""},
            temporal_route_applied=False,
            memory_prior_weight=0.0,
            fallback_window=max(2, top_k),
            token_budget=max(600, top_k * 240),
            memory_compression_target=0.4,
            chunk_expansion_threshold=0.7,
            rerank_strategy=f"{route.source_kind}_card_v2",
            context_expansion_policy=f"{route.source_kind}_card_v2_only",
            ontology_expansion_enabled=False,
            enrichment_route=route.mode,
            enrichment_reason=f"{route.source_kind}_card_v2_scoped_no_result",
            ontology_assist_eligible=False,
            cluster_assist_eligible=False,
            resolved_source_scope_applied=True,
            canonical_entities_applied=list(frame_payload.get("canonical_entity_ids") or []),
            metadata_filter_applied=effective_metadata_filter,
            prefilter_reason=reason,
        )
        runtime_execution = {
            "used": "ask_v2",
            "sectionDecision": "skipped",
            "sectionBlockReason": "",
            "fallbackReason": reason,
        }
        v2_diagnostics = {
            "routing": {
                "mode": route.mode,
                "intent": route.intent,
                "sourceKind": route.source_kind,
                "internalReferenceApplied": False,
                "memoryForm": "none",
                "matched_entities": [],
                "selected_card_ids": [],
            },
            "cardSelection": {"selected": []},
            "sectionSelection": {"selected": []},
            "sectionCoverage": {"status": "missing"},
            "sectionQualityGate": {
                "allowed": False,
                "reason": "no_section_cards",
                "signals": {"selectedRoles": [], "coverageStatus": "missing", "placeholderCount": 0, "metaOnlyCount": 0, "appendixLikeCount": 0, "cardCount": 0},
            },
            "claimSelection": {"selected": []},
            "claimAlignment": {"groups": []},
            "answerProvenance": {"mode": "scoped_no_result"},
            "runtimeExecution": runtime_execution,
            "claimCards": [],
            "sectionCards": [],
            "claimVerification": {},
            "consensus": {"conflictCount": 0, "unsupportedClaimCount": 0, "weakClaimCount": 0},
            "evidenceVerification": {"verificationStatus": "no_evidence", "anchorIdsUsed": [], "weakSlots": []},
            "fallback": {"used": False, "reason": reason},
        }
        pipeline_result = build_card_v2_pipeline_result(
            results=[],
            plan=plan,
            source_kind=route.source_kind,
            route_mode=route.mode,
            route_intent=route.intent,
            selected_cards=[],
            selected_anchor_count=0,
            memory_reason=reason,
            context_expansion_mode="none",
            v2_diagnostics=v2_diagnostics,
            paper_memory_mode=paper_memory_mode,
        )
        evidence_packet = EvidenceAssemblyService.from_searcher(self.searcher).assemble(
            query=query,
            source_type=request_source_type,
            results=[],
            paper_memory_prefilter=pipeline_result.paper_memory_prefilter,
            metadata_filter=plan.metadata_filter_applied,
            query_plan=plan_payload,
            query_frame=frame_payload,
        )
        return pipeline_result, evidence_packet

    def _merge_paper_candidates(
        self,
        *,
        query: str,
        route: AskV2Route,
        cards: list[dict[str, Any]],
        supplemental_cards: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not supplemental_cards:
            return list(cards)
        merged_limit = max(limit, min(6, len(cards) + len(supplemental_cards)))
        return self._dedupe_and_score(
            [*cards, *supplemental_cards],
            query=query,
            entity_ids=route.entity_ids,
            source_kind="paper",
            identity_key="paper_id",
            limit=merged_limit,
        )

    def _supplement_paper_candidates(
        self,
        *,
        query: str,
        route: AskV2Route,
        cards: list[dict[str, Any]],
        paper_family: str,
        limit: int,
        query_plan: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
        widen_fallback: bool = False,
    ) -> list[dict[str, Any]]:
        if route.source_kind != "paper" or paper_family not in {PAPER_FAMILY_CONCEPT_EXPLAINER, "paper_discover"}:
            return list(cards)
        selection_inputs = _build_paper_selection_inputs(
            query=query,
            frame_payload=normalize_query_frame_dict(query_frame),
            query_plan_payload=normalize_query_plan_dict(query_plan),
        )
        supplemental_cards: list[dict[str, Any]] = []
        resolved_paper_ids = list(selection_inputs.get("resolved_paper_ids") or [])
        if paper_family == PAPER_FAMILY_CONCEPT_EXPLAINER and resolved_paper_ids:
            supplemental_cards.extend(self._ensure_cards_for_papers(resolved_paper_ids[:2]))
        if widen_fallback or paper_family == "paper_discover" or not cards:
            fallback_ids = self._fallback_paper_ids(query=query, route=route, limit=max(limit * 2, 6))
            supplemental_cards.extend(self._ensure_cards_for_papers(fallback_ids))
        return self._merge_paper_candidates(
            query=query,
            route=route,
            cards=cards,
            supplemental_cards=supplemental_cards,
            limit=limit,
        )

    def _load_claim_cards(self, *, query: str, cards: list[dict[str, Any]], route: AskV2Route) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for card in cards:
            card_id = _clean_text(card.get("card_id"))
            if not card_id:
                continue
            if route.source_kind == "project":
                claim_cards = [
                    dict(item)
                    for item in build_project_claim_cards(cards=[card])
                ]
            else:
                claim_cards = [
                    dict(item)
                    for item in self.claim_card_builder.load_or_build_for_source_card(
                        source_kind=route.source_kind,
                        source_card=card,
                    )
                ]
            for item in claim_cards:
                enriched = dict(item)
                enriched["source_card_score"] = _stable_score(card.get("selection_score"))
                enriched["source_card_id"] = card_id
                selected.append(enriched)
        ranked = rank_claim_cards(query=query, claim_cards=selected, intent=route.intent)
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in ranked:
            claim_card_id = _clean_text(item.get("claim_card_id"))
            if not claim_card_id or claim_card_id in seen:
                continue
            seen.add(claim_card_id)
            deduped.append(item)
        return deduped[:8]

    def _load_section_cards(self, *, query: str, cards: list[dict[str, Any]], route: AskV2Route) -> list[dict[str, Any]]:
        if route.source_kind != "paper":
            return []
        section_store = SectionCardV1Store(self.sqlite_db.conn)
        section_store.ensure_schema()
        selected: list[dict[str, Any]] = []
        for card in cards:
            paper_id = _clean_text(card.get("paper_id"))
            card_id = _clean_text(card.get("card_id"))
            if not paper_id or not card_id:
                continue
            materialized_rows = list(section_store.list_paper_cards(paper_id) or [])
            materialized_roles: set[str] = set()
            for row in materialized_rows:
                role = _clean_text(row.get("role"))
                if role:
                    materialized_roles.add(role)
                selected.append(
                    {
                        "section_card_id": _clean_text(row.get("section_card_id")),
                        "source_kind": route.source_kind,
                        "source_card_id": card_id,
                        "source_id": paper_id,
                        "paper_id": paper_id,
                        "document_id": _clean_text(row.get("document_id")) or f"paper:{paper_id}",
                        "unit_id": next(iter(list(row.get("unit_ids") or [])), ""),
                        "title": _clean_text(row.get("title")),
                        "section_path": _clean_text(row.get("section_path")),
                        "unit_type": _clean_text(row.get("unit_type")) or "section",
                        "role": role or "other",
                        "order_index": 0,
                        "contextual_summary": _clean_text(row.get("contextual_summary")),
                        "source_excerpt": _clean_text(row.get("source_excerpt")),
                        "document_thesis": _clean_text(row.get("document_thesis")),
                        "confidence": _stable_score(row.get("confidence")),
                        "claims": list(row.get("claims") or []),
                        "concepts": list(row.get("concepts") or []),
                        "provenance": dict(row.get("provenance") or {}),
                        "appendix_like": False,
                        "search_text": _clean_text(row.get("search_text")),
                        "origin": _clean_text(row.get("origin")) or "materialized_v1",
                        "generator_model": _clean_text(row.get("generator_model")),
                        "key_points": list(row.get("key_points") or []),
                        "scope_notes": list(row.get("scope_notes") or []),
                        "source_card_score": _stable_score(card.get("selection_score")),
            }
                )
            units = list(self.sqlite_db.list_document_memory_units(f"paper:{paper_id}", limit=200) or [])
            section_cards = project_section_cards(source_kind=route.source_kind, source_card=card, units=units)
            current_paper_roles = {
                _clean_text(item.get("role"))
                for item in [*materialized_rows, *section_cards]
                if _clean_text(item.get("role"))
            }
            memory_adapter_cards = []
            if not units:
                memory_card = self.sqlite_db.get_paper_memory_card(paper_id) or {}
                memory_adapter_cards = paper_memory_card_to_section_cards(
                    memory_card,
                    source_card_id=card_id,
                    paper_id=paper_id,
                    title=_clean_text(card.get("title")),
                )
            for item in section_cards:
                if _clean_text(item.get("role")) in materialized_roles:
                    continue
                enriched = dict(item)
                enriched["source_card_score"] = _stable_score(card.get("selection_score"))
                enriched.setdefault("origin", "projected_v1")
                selected.append(enriched)
            if not materialized_rows and not section_cards:
                for item in memory_adapter_cards:
                    role = _clean_text(item.get("role"))
                    if role in materialized_roles or role in current_paper_roles:
                        continue
                    enriched = dict(item)
                    enriched["source_card_score"] = _stable_score(card.get("selection_score"))
                    selected.append(enriched)
        ranked = rank_section_cards(query=query, section_cards=selected, intent=route.intent)
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in ranked:
            section_card_id = _clean_text(item.get("section_card_id"))
            if not section_card_id or section_card_id in seen:
                continue
            seen.add(section_card_id)
            deduped.append(item)
        return deduped[:6]

    def _anchors_for_section_cards(self, *, section_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        anchors: list[dict[str, Any]] = []
        for item in section_cards:
            excerpt = _clean_text(item.get("source_excerpt") or item.get("contextual_summary"))
            if not excerpt:
                continue
            provenance = dict(item.get("provenance") or {})
            anchors.append(
                {
                    "anchor_id": f"section-anchor:{_clean_text(item.get('unit_id'))}",
                    "card_id": _clean_text(item.get("source_card_id")),
                    "paper_id": _clean_text(item.get("paper_id")),
                    "unit_id": _clean_text(item.get("unit_id")),
                    "section_path": _clean_text(item.get("section_path")),
                    "excerpt": excerpt,
                    "score": _stable_score(item.get("selection_score") or item.get("confidence")),
                    "evidence_role": _clean_text(item.get("role")) or "section",
                    "document_id": _clean_text(item.get("document_id")),
                    "page": provenance.get("page"),
                    "bbox": provenance.get("bbox"),
                    "reading_order": provenance.get("reading_order"),
                }
            )
        return anchors

    def _anchors_for_paper_document_summaries(self, *, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        anchors: list[dict[str, Any]] = []
        for card in cards:
            paper_id = _clean_text(card.get("paper_id"))
            card_id = _clean_text(card.get("card_id"))
            if not paper_id or not card_id:
                continue
            for row in list(self.sqlite_db.list_document_memory_units(f"paper:{paper_id}", limit=24) or []):
                unit_type = _clean_text(row.get("unit_type")).lower()
                if unit_type not in {"document_summary", "summary"}:
                    continue
                excerpt = _clean_text(
                    row.get("contextual_summary")
                    or row.get("source_excerpt")
                    or row.get("document_thesis")
                )
                if not excerpt:
                    continue
                anchors.append(
                    {
                        "anchor_id": f"doc-summary-anchor:{_clean_text(row.get('unit_id'))}",
                        "card_id": card_id,
                        "paper_id": paper_id,
                        "unit_id": _clean_text(row.get("unit_id")),
                        "document_id": _clean_text(row.get("document_id")) or f"paper:{paper_id}",
                        "title": _clean_text(row.get("document_title") or card.get("title")),
                        "section_path": _clean_text(row.get("section_path") or row.get("title") or "Summary"),
                        "unit_type": unit_type or "document_summary",
                        "evidence_role": "document_summary",
                        "excerpt": excerpt,
                        "score": max(0.72, min(0.96, max(_stable_score(card.get("selection_score")), _stable_score(row.get("confidence"))))),
                    }
                )
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for anchor in anchors:
            anchor_id = _clean_text(anchor.get("anchor_id"))
            if not anchor_id or anchor_id in seen:
                continue
            seen.add(anchor_id)
            deduped.append(anchor)
        return deduped

    def _anchors_for_claim_cards(self, *, claim_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        anchors: list[dict[str, Any]] = []
        for item in claim_cards:
            for anchor in list(item.get("anchors") or []):
                anchors.append(dict(anchor))
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for anchor in anchors:
            anchor_id = _clean_text(anchor.get("anchor_id"))
            if not anchor_id or anchor_id in seen:
                continue
            seen.add(anchor_id)
            deduped.append(dict(anchor))
        return deduped

    def _anchors_for_inline_cards(self, *, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        anchors: list[dict[str, Any]] = []
        for card in cards:
            card_id = _clean_text(card.get("card_id"))
            if not card_id:
                continue
            for anchor in list(card.get("anchors") or []):
                payload = dict(anchor)
                payload.setdefault("card_id", card_id)
                payload.setdefault("title", _clean_text(card.get("title")))
                payload.setdefault("document_id", _clean_text(card.get("document_id")))
                payload.setdefault("source_url", _clean_text(anchor.get("source_url") or card.get("canonical_url")))
                payload.setdefault("document_date", _clean_text(anchor.get("document_date") or card.get("document_date")))
                payload.setdefault("event_date", _clean_text(anchor.get("event_date") or card.get("event_date")))
                payload.setdefault("observed_at", _clean_text(anchor.get("observed_at") or card.get("observed_at")))
                payload.setdefault("updated_at_marker", _clean_text(anchor.get("updated_at_marker") or card.get("updated_at")))
                anchors.append(payload)
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for anchor in anchors:
            anchor_id = _clean_text(anchor.get("anchor_id"))
            if not anchor_id or anchor_id in seen:
                continue
            seen.add(anchor_id)
            deduped.append(anchor)
        return deduped

    def _anchors_for_stored_cards(self, *, cards: list[dict[str, Any]], route: AskV2Route) -> list[dict[str, Any]]:
        if route.source_kind == "paper":
            list_fn = self.sqlite_db.list_evidence_anchors_v2
        elif route.source_kind == "web":
            list_fn = self.sqlite_db.list_web_evidence_anchors_v2
        elif route.source_kind == "vault":
            list_fn = self.sqlite_db.list_vault_evidence_anchors_v2
        else:
            return []
        anchors: list[dict[str, Any]] = []
        for card in cards:
            card_id = _clean_text(card.get("card_id"))
            if not card_id:
                continue
            for anchor in list(list_fn(card_id=card_id) or []):
                payload = dict(anchor)
                payload.setdefault("card_id", card_id)
                payload.setdefault("title", _clean_text(payload.get("title") or card.get("title")))
                payload.setdefault(
                    "document_id",
                    _clean_text(payload.get("document_id") or card.get("document_id") or card.get("note_id")),
                )
                if route.source_kind == "paper":
                    payload.setdefault("paper_id", _clean_text(payload.get("paper_id") or card.get("paper_id")))
                elif route.source_kind == "web":
                    payload.setdefault("source_url", _clean_text(payload.get("source_url") or card.get("canonical_url")))
                    payload.setdefault("document_date", _clean_text(payload.get("document_date") or card.get("document_date")))
                    payload.setdefault("event_date", _clean_text(payload.get("event_date") or card.get("event_date")))
                    payload.setdefault("observed_at", _clean_text(payload.get("observed_at") or card.get("observed_at")))
                    payload.setdefault("updated_at_marker", _clean_text(payload.get("updated_at_marker") or card.get("updated_at")))
                elif route.source_kind == "vault":
                    payload.setdefault("file_path", _clean_text(payload.get("file_path") or card.get("file_path")))
                    payload.setdefault("note_id", _clean_text(payload.get("note_id") or card.get("note_id")))
                anchors.append(payload)
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for anchor in anchors:
            anchor_id = _clean_text(anchor.get("anchor_id"))
            if not anchor_id or anchor_id in seen:
                continue
            seen.add(anchor_id)
            deduped.append(anchor)
        return deduped

    @staticmethod
    def _merge_anchor_lists(*anchor_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for anchors in anchor_lists:
            for anchor in anchors:
                anchor_id = _clean_text(anchor.get("anchor_id"))
                if not anchor_id or anchor_id in seen:
                    continue
                seen.add(anchor_id)
                merged.append(dict(anchor))
        return merged

    def _anchor_results(self, *, cards: list[dict[str, Any]], anchors: list[dict[str, Any]], route: AskV2Route) -> list[SearchResult]:
        by_card_id = {_clean_text(card.get("card_id")): dict(card) for card in cards}
        source_hash_by_document_id: dict[str, str] = {}
        results: list[SearchResult] = []
        for index, anchor in enumerate(anchors):
            card = by_card_id.get(_clean_text(anchor.get("card_id")), {})
            excerpt = _clean_text(anchor.get("excerpt"))
            if not excerpt:
                continue
            anchor_id = _clean_text(anchor.get("anchor_id"))
            document_id = _clean_text(
                anchor.get("document_id")
                or card.get("document_id")
                or anchor.get("note_id")
                or card.get("note_id")
                or anchor.get("paper_id")
                or card.get("paper_id")
            )
            source_hash = _clean_text(
                anchor.get("source_content_hash")
                or card.get("source_content_hash")
                or self._source_hash_for_document(document_id, cache=source_hash_by_document_id)
            )
            span_locator = _clean_text(anchor.get("span_locator") or anchor.get("unit_id") or anchor.get("chunk_id") or anchor_id)
            score = max(0.01, min(0.99, _stable_score(anchor.get("score"))))
            metadata = {
                "title": _clean_text(anchor.get("title") or card.get("title")),
                "source_type": "project" if route.source_kind == "project" else route.source_kind,
                "document_id": document_id,
                "unit_id": _clean_text(anchor.get("unit_id")),
                "chunk_id": _clean_text(anchor.get("chunk_id") or anchor.get("unit_id") or anchor_id),
                "source_content_hash": source_hash,
                "content_hash": source_hash,
                "span_locator": span_locator,
                "snippet_hash": _clean_text(anchor.get("snippet_hash")),
                "char_start": self._int_or_default(anchor.get("char_start"), anchor.get("start_offset"), default=0),
                "char_end": self._int_or_default(anchor.get("char_end"), anchor.get("end_offset"), default=len(excerpt)),
                "section_path": _clean_text(anchor.get("section_path")),
                "unit_type": _clean_text(anchor.get("unit_type")),
                "parent_id": _clean_text(anchor.get("unit_id") or anchor.get("document_id") or anchor.get("note_id") or anchor_id),
                "resolved_parent_id": _clean_text(anchor.get("unit_id") or anchor.get("document_id") or anchor.get("note_id") or anchor_id),
                "resolved_parent_label": _clean_text(anchor.get("section_path") or anchor.get("evidence_role")),
                "resolved_parent_chunk_span": _clean_text(anchor.get("span_locator") or index),
                "record_id": _clean_text(anchor.get("claim_id")),
            }
            if route.source_kind == "paper":
                metadata["arxiv_id"] = _clean_text(anchor.get("paper_id") or card.get("paper_id"))
                metadata["paper_id"] = _clean_text(anchor.get("paper_id") or card.get("paper_id"))
                metadata["published_at"] = _clean_text(card.get("published_at"))
                metadata["updated_at"] = _clean_text(card.get("updated_at"))
            elif route.source_kind == "web":
                source_url = _clean_text(anchor.get("source_url") or card.get("canonical_url"))
                metadata["source_url"] = source_url
                metadata["url"] = source_url
                metadata["canonical_url"] = source_url
                metadata["document_date"] = _clean_text(anchor.get("document_date") or card.get("document_date"))
                metadata["event_date"] = _clean_text(anchor.get("event_date") or card.get("event_date"))
                metadata["observed_at"] = _clean_text(anchor.get("observed_at") or card.get("observed_at"))
                metadata["updated_at"] = _clean_text(anchor.get("updated_at_marker") or card.get("updated_at"))
            elif route.source_kind == "vault":
                metadata["file_path"] = _clean_text(anchor.get("file_path") or card.get("file_path"))
                metadata["note_id"] = _clean_text(anchor.get("note_id") or card.get("note_id"))
            else:
                metadata["file_path"] = _clean_text(anchor.get("file_path") or anchor.get("document_id"))
                metadata["note_id"] = _clean_text(card.get("relative_path"))
            lexical_extras = {
                "quality_flag": _clean_text(card.get("quality_flag") or "unscored"),
                "memory_provenance": {
                    "mode": f"{route.source_kind}-card-v2",
                    "cardId": _clean_text(card.get("card_id")),
                    "anchorId": _clean_text(anchor.get("anchor_id")),
                    "claimId": _clean_text(anchor.get("claim_id")),
                },
                "ranking_signals": {
                    f"{route.source_kind}_card_v2_anchor_score": round(score, 4),
                    f"{route.source_kind}_card_v2_role": _clean_text(anchor.get("evidence_role")),
                },
                "top_ranking_signals": [{"name": f"{route.source_kind}_card_v2_anchor_score", "value": round(score, 4)}],
            }
            results.append(
                SearchResult(
                    document=excerpt,
                    metadata=metadata,
                    distance=max(0.0, 1.0 - score),
                    score=score,
                    document_id=_clean_text(anchor.get("anchor_id")),
                    semantic_score=score,
                    lexical_score=score,
                    retrieval_mode=f"{route.source_kind}-card-v2",
                    lexical_extras=lexical_extras,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        if route.source_kind == "paper" and route.intent == "comparison":
            diversified: list[SearchResult] = []
            seen_document_ids: set[str] = set()
            seen_paper_ids: set[str] = set()
            for item in results:
                paper_id = _clean_text((item.metadata or {}).get("paper_id"))
                document_id = _clean_text(item.document_id)
                if not paper_id or paper_id in seen_paper_ids:
                    continue
                diversified.append(item)
                seen_paper_ids.add(paper_id)
                if document_id:
                    seen_document_ids.add(document_id)
            for item in results:
                document_id = _clean_text(item.document_id)
                if document_id and document_id in seen_document_ids:
                    continue
                diversified.append(item)
                if document_id:
                    seen_document_ids.add(document_id)
            results = diversified
        return results[:8]

    def _source_hash_for_document(self, document_id: str, *, cache: dict[str, str]) -> str:
        token = _clean_text(document_id)
        if not token:
            return ""
        if token not in cache:
            rows = list(self.sqlite_db.list_document_memory_units(token, limit=20) or [])
            unit_hash = next(
                (_clean_text(row.get("source_content_hash")) for row in rows if _clean_text(row.get("source_content_hash"))),
                "",
            )
            note_hash = ""
            if not unit_hash and hasattr(self.sqlite_db, "get_note"):
                note = dict(self.sqlite_db.get_note(token) or {})
                metadata_value = note.get("metadata")
                try:
                    metadata = json.loads(metadata_value) if isinstance(metadata_value, str) else dict(metadata_value or {})
                except Exception:
                    metadata = {}
                note_hash = _clean_text(
                    metadata.get("source_content_hash")
                    or metadata.get("content_hash")
                    or metadata.get("content_sha1")
                    or metadata.get("content_sha256")
                )
            paper_hash = ""
            if not unit_hash and not note_hash and token.startswith("paper:") and hasattr(self.sqlite_db, "get_paper"):
                paper_id = token.split(":", 1)[1]
                paper = dict(self.sqlite_db.get_paper(paper_id) or {})
                for key in ("text_path", "translated_path", "pdf_path"):
                    paper_hash = source_hash_for_path(str(paper.get(key) or ""))
                    if paper_hash:
                        break
                if not paper_hash:
                    paper_hash = source_hash_for_text(paper.get("notes"), paper_id, "paper_record")
            cache[token] = unit_hash or note_hash or paper_hash
        return cache[token]

    @staticmethod
    def _int_or_default(*values: Any, default: int) -> int:
        for value in values:
            if value is None or value == "":
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return int(default)

    def execute(
        self,
        *,
        query: str,
        top_k: int,
        source_type: str | None,
        retrieval_mode: str,
        alpha: float,
        allow_external: bool,
        metadata_filter: dict[str, Any] | None = None,
        ask_v2_mode: str | None = None,
        paper_memory_mode: str = "off",
        query_plan: dict[str, Any] | None = None,
        query_frame: dict[str, Any] | None = None,
    ) -> tuple[RetrievalPipelineResult, Any]:
        query_frame_payload = self._resolve_frame_authority(
            query=query,
            source_type=source_type,
            metadata_filter=metadata_filter,
            query_frame=query_frame,
        )
        query_plan_payload = normalize_query_plan_dict(query_plan)
        route = self._route(
            query=query,
            source_type=source_type,
            metadata_filter=metadata_filter,
            query_frame=query_frame_payload,
        )
        paper_family, answer_mode = self._resolve_paper_family_and_answer_mode(
            route=route,
            query_frame_payload=query_frame_payload,
        )
        card_limit = max(1, min(3, top_k))
        selected_cards = self._select_cards(
            query=query,
            route=route,
            limit=card_limit,
            metadata_filter=metadata_filter,
            query_plan=query_plan_payload,
            query_frame=query_frame_payload,
        )
        if route.source_kind == "paper" and paper_family in {PAPER_FAMILY_CONCEPT_EXPLAINER, "paper_discover"}:
            selected_cards = self._supplement_paper_candidates(
                query=query,
                route=route,
                cards=selected_cards,
                paper_family=paper_family,
                limit=card_limit,
                query_plan=query_plan_payload,
                query_frame=query_frame_payload,
                widen_fallback=(paper_family == "paper_discover"),
            )
        if route.source_kind == "vault" and not selected_cards and self._is_missing_vault_scope(query_frame_payload, metadata_filter):
            return self._scoped_no_result_execution(
                query=query,
                top_k=top_k,
                source_type=source_type,
                retrieval_mode=retrieval_mode,
                route=route,
                query_plan_payload=query_plan_payload,
                query_frame_payload=query_frame_payload,
                metadata_filter=metadata_filter,
                reason="scoped_vault_no_result",
                paper_memory_mode=paper_memory_mode,
            )
        if route.source_kind in {"vault", "project"} and not selected_cards:
            raise AskV2FallbackToLegacy("no_v2_card_candidates")
        section_first_requested = self._use_section_first(route=route, query=query, mode_override=ask_v2_mode)
        selected_section_cards = self._load_section_cards(query=query, cards=selected_cards, route=route) if section_first_requested else []
        section_cov = section_coverage(section_cards=selected_section_cards)
        section_quality = assess_section_source_quality(section_cards=selected_section_cards, coverage=section_cov) if selected_section_cards else {
            "allowed": False,
            "reason": "no_section_cards",
            "signals": {
                "selectedRoles": [],
                "coverageStatus": str(section_cov.get("status") or "missing"),
                "placeholderCount": 0,
                "metaOnlyCount": 0,
                "appendixLikeCount": 0,
                "cardCount": 0,
            },
        }
        section_allowed = bool(section_quality.get("allowed")) and section_first_requested
        if (
            route.source_kind == "paper"
            and paper_family == PAPER_FAMILY_CONCEPT_EXPLAINER
            and section_first_requested
            and section_cov.get("status") != "strong"
        ):
            widened_cards = self._supplement_paper_candidates(
                query=query,
                route=route,
                cards=selected_cards,
                paper_family=paper_family,
                limit=card_limit,
                query_plan=query_plan_payload,
                query_frame=query_frame_payload,
                widen_fallback=True,
            )
            if [
                _clean_text(item.get("paper_id"))
                for item in widened_cards
            ] != [
                _clean_text(item.get("paper_id"))
                for item in selected_cards
            ]:
                selected_cards = widened_cards
                selected_section_cards = self._load_section_cards(query=query, cards=selected_cards, route=route)
                section_cov = section_coverage(section_cards=selected_section_cards)
                section_quality = assess_section_source_quality(section_cards=selected_section_cards, coverage=section_cov) if selected_section_cards else {
                    "allowed": False,
                    "reason": "no_section_cards",
                    "signals": {
                        "selectedRoles": [],
                        "coverageStatus": str(section_cov.get("status") or "missing"),
                        "placeholderCount": 0,
                        "metaOnlyCount": 0,
                        "appendixLikeCount": 0,
                        "cardCount": 0,
                    },
                }
                section_allowed = bool(section_quality.get("allowed")) and section_first_requested
        summary_anchors = self._anchors_for_paper_document_summaries(cards=selected_cards) if route.source_kind == "paper" and paper_family in {
            PAPER_FAMILY_CONCEPT_EXPLAINER,
            "paper_discover",
        } else []
        if route.source_kind == "paper" and paper_family == PAPER_FAMILY_CONCEPT_EXPLAINER and not selected_cards and not selected_section_cards and not summary_anchors:
            raise AskV2FallbackToLegacy("no_concept_candidates")
        if route.source_kind == "paper" and paper_family == "paper_discover" and not selected_cards and not selected_section_cards and not summary_anchors:
            raise AskV2FallbackToLegacy("no_paper_shortlist_candidates")
        claim_cards_allowed = _should_attempt_claim_cards(
            route=route,
            section_first_requested=section_first_requested,
            section_allowed=section_allowed,
            has_section_anchors=False,
        ) and not (
            route.source_kind == "paper"
            and paper_family in {PAPER_FAMILY_CONCEPT_EXPLAINER, "paper_discover"}
        )
        selected_claim_cards = []
        anchors = []
        if selected_section_cards and section_cov.get("status") == "strong" and section_allowed:
            anchors = self._merge_anchor_lists(
                summary_anchors,
                self._anchors_for_section_cards(section_cards=selected_section_cards),
            )
        if not anchors and not claim_cards_allowed:
            anchors = self._merge_anchor_lists(
                summary_anchors,
                self._anchors_for_stored_cards(cards=selected_cards, route=route),
            )
        if claim_cards_allowed:
            selected_claim_cards = self._load_claim_cards(query=query, cards=selected_cards, route=route)
            claim_anchors = self._anchors_for_claim_cards(claim_cards=selected_claim_cards)
            if route.source_kind == "paper" and paper_family == PAPER_FAMILY_COMPARE:
                claim_anchor_paper_ids = {
                    _clean_text(item.get("paper_id"))
                    for item in claim_anchors
                    if _clean_text(item.get("paper_id"))
                }
                if len(claim_anchor_paper_ids) < min(
                    2,
                    len({
                        _clean_text(item.get("paper_id"))
                        for item in selected_cards
                        if _clean_text(item.get("paper_id"))
                    }),
                ):
                    anchors = self._merge_anchor_lists(
                        summary_anchors,
                        self._anchors_for_stored_cards(cards=selected_cards, route=route),
                        claim_anchors,
                    )
                else:
                    anchors = claim_anchors
            else:
                anchors = claim_anchors
        if not anchors:
            anchors = self._merge_anchor_lists(
                summary_anchors,
                self._anchors_for_stored_cards(cards=selected_cards, route=route),
            )
        if route.source_kind in {"web", "vault"}:
            internal_reference_cards = []
            if route.source_kind == "web":
                internal_reference_cards = [
                    dict(card)
                    for card in selected_cards
                    if bool((card.get("diagnostics") or {}).get("internalReference"))
                ]
            inline_anchors = self._anchors_for_inline_cards(
                cards=internal_reference_cards if internal_reference_cards else selected_cards
            )
            if internal_reference_cards:
                anchors = self._merge_anchor_lists(inline_anchors, anchors)
            elif not anchors:
                anchors = inline_anchors
        if not anchors and selected_section_cards and section_cov.get("status") != "missing" and section_allowed:
            anchors = self._merge_anchor_lists(
                summary_anchors,
                self._anchors_for_section_cards(section_cards=selected_section_cards),
            )
        if route.source_kind in {"vault", "project"} and not anchors:
            raise AskV2FallbackToLegacy("no_v2_anchor_candidates")
        anchor_results = self._anchor_results(cards=selected_cards, anchors=anchors, route=route)
        if query_frame_payload:
            query_frame_obj = build_query_frame(
                domain_key=str(query_frame_payload.get("domain_key") or "ai_papers"),
                source_type=(
                    "project"
                    if route.source_kind == "project"
                    else str(query_frame_payload.get("source_type") or source_type or route.source_kind)
                ),
                family=str(query_frame_payload.get("family") or ""),
                query_intent=str(query_frame_payload.get("query_intent") or route.intent),
                answer_mode=str(query_frame_payload.get("answer_mode") or answer_mode or "grounded_answer"),
                entities=list(query_frame_payload.get("entities") or []),
                canonical_entity_ids=list(query_frame_payload.get("canonical_entity_ids") or []),
                expanded_terms=list(query_frame_payload.get("expanded_terms") or []),
                resolved_source_ids=list(query_frame_payload.get("resolved_source_ids") or []),
                confidence=float(query_frame_payload.get("confidence") or 0.0),
                planner_status=str(query_frame_payload.get("planner_status") or "not_attempted"),
                planner_reason=str(query_frame_payload.get("planner_reason") or "rule_based"),
                evidence_policy_key=str(query_frame_payload.get("evidence_policy_key") or ""),
                metadata_filter=dict(query_frame_payload.get("metadata_filter") or metadata_filter or {}),
                frame_provenance=str(query_frame_payload.get("frame_provenance") or "derived"),
                trusted=bool(query_frame_payload.get("trusted")),
                lock_mask=list(query_frame_payload.get("lock_mask") or []),
                family_source=str(query_frame_payload.get("family_source") or QUERY_FRAME_FAMILY_FROM_PACK),
                overrides_applied=list(query_frame_payload.get("overrides_applied") or []),
            )
            query_plan_payload = query_plan_payload or query_frame_obj.to_query_plan_dict()
        else:
            query_plan_payload = query_plan_payload or {
                "family": paper_family or "general",
                "entities": [
                    _clean_text(item.get("canonical_name") or item.get("entity_id"))
                    for item in route.matched_entities
                    if _clean_text(item.get("canonical_name") or item.get("entity_id"))
                ][:6],
                "expandedTerms": [],
                "resolvedPaperIds": [
                    _clean_text(card.get("paper_id"))
                    for card in selected_cards
                    if _clean_text(card.get("paper_id"))
                ][:3],
                "answerMode": answer_mode or "grounded_answer",
                "queryIntent": route.intent,
                "confidence": 0.9 if paper_family else 0.6,
                "plannerUsed": False,
                "plannerStatus": "not_attempted",
                "plannerReason": "ask_v2_route",
                "plannerWarnings": [],
                "plannerRoute": {},
            }
            request_source_type = "project" if route.source_kind == "project" else source_type or route.source_kind
            domain_pack = get_domain_pack(source_type=request_source_type)
            if domain_pack is not None:
                query_frame_obj = domain_pack.normalize(
                    query,
                    source_type=request_source_type,
                    metadata_filter=metadata_filter,
                    sqlite_db=self.sqlite_db,
                    query_plan=query_plan_payload,
                )
            else:
                query_frame_obj = query_frame_from_query_plan(
                    query_plan_payload,
                    query=query,
                    source_type=request_source_type,
                    metadata_filter=metadata_filter,
                    sqlite_db=self.sqlite_db,
                )
        if route.source_kind == "paper" and str(query_frame_obj.family or "") == "paper_compare":
            compare_source_ids = {
                _clean_text(item.metadata.get("paper_id") or item.metadata.get("arxiv_id"))
                for item in anchor_results
                if _clean_text(item.metadata.get("paper_id") or item.metadata.get("arxiv_id"))
            }
            if len(list(query_frame_obj.resolved_source_ids or [])) >= 2 and len(compare_source_ids) < 2:
                raise AskV2FallbackToLegacy("insufficient_compare_anchor_coverage")
        request_source_type = "project" if route.source_kind == "project" else source_type or route.source_kind
        query_plan_payload["evidencePolicyKey"] = query_frame_obj.evidence_policy_key
        effective_metadata_filter = dict(query_frame_obj.metadata_filter or metadata_filter or {})
        resolved_scope_applied = False
        prefilter_reason = "ask_v2_card_route"
        if route.source_kind == "paper":
            if any(_clean_text(effective_metadata_filter.get(key)) for key in ("arxiv_id", "paper_id", "doi")):
                resolved_scope_applied = True
                prefilter_reason = "explicit_metadata_filter"
            elif paper_family == "paper_lookup" and len(list(query_frame_obj.resolved_source_ids or [])) == 1:
                effective_metadata_filter.setdefault("source_type", "paper")
                effective_metadata_filter.setdefault("arxiv_id", _clean_text(query_frame_obj.resolved_source_ids[0]))
                resolved_scope_applied = True
                prefilter_reason = "resolved_source_id"
            elif paper_family == "concept_explainer" and list(query_frame_obj.resolved_source_ids or []):
                prefilter_reason = "representative_candidate_narrowing"
            elif paper_family == "paper_compare" and list(query_frame_obj.resolved_source_ids or []):
                prefilter_reason = "resolved_compare_candidates"
            elif paper_family == "paper_discover" and (
                list(query_frame_obj.canonical_entity_ids or []) or list(query_frame_obj.resolved_source_ids or [])
            ):
                prefilter_reason = "lightweight_discovery_expansion"
            elif list(query_frame_obj.canonical_entity_ids or []):
                prefilter_reason = "canonical_entity_linking"
        elif route.source_kind == "web":
            explicit_web_scope = any(_clean_text((metadata_filter or {}).get(key)) for key in ("canonical_url", "document_id", "url", "source_url"))
            if explicit_web_scope:
                resolved_scope_applied = True
                prefilter_reason = "explicit_metadata_filter"
            elif list(query_frame_obj.resolved_source_ids or []):
                resolved_scope_applied = any(
                    _clean_text(effective_metadata_filter.get(key))
                    for key in ("canonical_url", "document_id")
                )
                prefilter_reason = "resolved_source_id"
            elif bool(effective_metadata_filter.get("latest_only") or effective_metadata_filter.get("temporal_required")):
                prefilter_reason = "temporal_grounding_required"
            elif bool(effective_metadata_filter.get("watchlist_scope")):
                prefilter_reason = "watchlist_scope"
            elif bool(effective_metadata_filter.get("reference_only")):
                prefilter_reason = "reference_source_bias"
            elif list(query_frame_obj.canonical_entity_ids or []):
                prefilter_reason = "canonical_entity_linking"

        plan = RetrievalPlan(
            query=query,
            source_scope="project" if route.source_kind == "project" else route.source_kind,
            query_intent=str(query_frame_obj.query_intent or route.intent),
            paper_family=str(query_frame_obj.family or paper_family or "general"),
            retrieval_mode=retrieval_mode,
            memory_mode=f"{route.source_kind}-card-v2",
            candidate_budgets={route.source_kind: max(4, top_k * 2)},
            query_plan=query_plan_payload,
            query_frame=query_frame_obj.to_dict(),
            temporal_signals={"enabled": bool(_TEMPORAL_RE.search(query)), "mode": "latest" if _TEMPORAL_RE.search(query) else "none", "targetYear": ""},
            temporal_route_applied=route.intent == "temporal",
            memory_prior_weight=0.0,
            fallback_window=max(2, top_k),
            token_budget=max(600, top_k * 240),
            memory_compression_target=0.4,
            chunk_expansion_threshold=0.7,
            rerank_strategy=f"{route.source_kind}_card_v2",
            context_expansion_policy=f"{route.source_kind}_card_v2_only",
            ontology_expansion_enabled=route.mode == "ontology-first",
            enrichment_route=route.mode,
            enrichment_reason=f"{route.source_kind}_card_v2_router",
            ontology_assist_eligible=route.mode == "ontology-first" and route.source_kind in {"paper", "web"},
            cluster_assist_eligible=False,
            resolved_source_scope_applied=resolved_scope_applied,
            canonical_entities_applied=list(query_frame_obj.canonical_entity_ids or []),
            metadata_filter_applied=effective_metadata_filter,
            prefilter_reason=prefilter_reason,
        )
        pipeline_result = build_card_v2_pipeline_result(
            results=anchor_results,
            plan=plan,
            source_kind=route.source_kind,
            route_mode=route.mode,
            route_intent=route.intent,
            selected_cards=selected_cards,
            selected_anchor_count=len(anchors),
            memory_reason=route.mode,
            context_expansion_mode="card",
            paper_memory_mode=paper_memory_mode,
        )
        evidence_packet = EvidenceAssemblyService.from_searcher(self.searcher).assemble(
            query=query,
            source_type=request_source_type,
            results=anchor_results,
            paper_memory_prefilter=pipeline_result.paper_memory_prefilter,
            metadata_filter=plan.metadata_filter_applied,
            query_plan=query_plan_payload,
            query_frame=query_frame_obj.to_dict(),
        )
        claim_verification, claim_consensus = self.verifier.claim_verification(selected_claims=selected_claim_cards, anchors=anchors)
        claim_alignment = self.claim_alignment.group_claim_cards(selected_claim_cards)
        verification = self.verifier.verification_summary(
            query=query,
            route=route,
            cards=selected_cards,
            anchors=anchors,
            evidence_packet=evidence_packet,
            claim_consensus=claim_consensus,
        )
        hard_gate_reason = _ask_v2_hard_gate_reason(
            verification=verification,
            claim_consensus=claim_consensus,
        )
        if hard_gate_reason:
            evidence_packet.evidence_packet = {
                **dict(evidence_packet.evidence_packet or {}),
                "answerable": False,
                "answerableDecisionReason": hard_gate_reason,
                "askV2HardGate": True,
                "askV2VerificationStatus": _clean_text(verification.get("verificationStatus")),
            }
        # Keep fallback signal aligned with hard-fail cues; weak-only claims stay diagnostic-only.
        v2_fallback_used = bool(
            (evidence_packet.evidence_packet or {}).get("answerable") is False
            or verification.get("verificationStatus") != "strong"
            or claim_consensus.get("conflictCount")
            or claim_consensus.get("unsupportedClaimCount")
        )
        pipeline_result.v2_diagnostics = {
            "routing": {
                "mode": route.mode,
                "intent": route.intent,
                "sourceKind": route.source_kind,
                "internalReferenceApplied": any(bool((card.get("diagnostics") or {}).get("internalReference")) for card in selected_cards),
                "memoryForm": (
                    "section_cards"
                    if selected_section_cards and section_allowed and (section_cov.get("status") == "strong" or not selected_claim_cards)
                    else "claim_cards"
                ),
                "matched_entities": [
                    {
                        "entity_id": _clean_text(item.get("entity_id")),
                        "canonical_name": _clean_text(item.get("canonical_name")),
                        "entity_type": _clean_text(item.get("entity_type")),
                    }
                    for item in route.matched_entities
                ],
                "selected_card_ids": [_clean_text(card.get("card_id")) for card in selected_cards if _clean_text(card.get("card_id"))],
            },
            "cardSelection": {
                "selected": [
                    {
                        "cardId": _clean_text(card.get("card_id")),
                        "sourceId": _clean_text(card.get("paper_id") or card.get("document_id") or card.get("note_id") or card.get("relative_path")),
                        "title": _clean_text(card.get("title")),
                        "slotCoverage": _slot_coverage(card),
                        "qualityFlag": _clean_text(card.get("quality_flag")),
                    }
                    for card in selected_cards
                ]
            },
            "sectionSelection": {
                "selected": [
                    {
                        "sectionCardId": _clean_text(item.get("section_card_id")),
                        "unitId": _clean_text(item.get("unit_id")),
                        "cardId": _clean_text(item.get("source_card_id")),
                        "paperId": _clean_text(item.get("paper_id")),
                        "role": _clean_text(item.get("role")),
                        "unitType": _clean_text(item.get("unit_type")),
                        "sectionPath": _clean_text(item.get("section_path")),
                        "title": _clean_text(item.get("title")),
                        "confidence": _stable_score(item.get("confidence")),
                        "reason": " | ".join(_clean_text(value) for value in list(item.get("ranking_reasons") or []) if _clean_text(value)),
                    }
                    for item in selected_section_cards
                ]
            },
            "sectionCoverage": section_cov,
            "sectionQualityGate": section_quality,
            "claimSelection": {
                "selected": [
                    {
                        "claimCardId": _clean_text(item.get("claim_card_id")),
                        "claimId": _clean_text(item.get("claim_id")),
                        "cardId": _clean_text(item.get("source_card_id")),
                        "sourceKind": _clean_text(item.get("source_kind")),
                        "sourceId": _clean_text(item.get("source_id")),
                        "claimType": _clean_text(item.get("claim_type")),
                        "origin": _clean_text(item.get("origin")),
                        "trustLevel": _clean_text(item.get("trust_level")),
                        "reason": " | ".join(_clean_text(value) for value in list(item.get("ranking_reasons") or []) if _clean_text(value)),
                    }
                    for item in selected_claim_cards
                ]
            },
            "claimAlignment": {
                "groups": claim_alignment,
            },
            "answerProvenance": {
                "mode": (
                    "section_cards_weak_fallback"
                    if selected_section_cards and section_allowed and section_cov.get("status") != "strong"
                    else "section_cards_verified"
                    if selected_section_cards and section_allowed
                    else "card_only_fallback"
                    if not selected_claim_cards
                    else "claim_cards_conflicted"
                    if claim_consensus.get("conflictCount")
                    else "weak_claim_fallback"
                    if verification.get("verificationStatus") != "strong" or claim_consensus.get("unsupportedClaimCount") or claim_consensus.get("weakClaimCount")
                    else "claim_cards_verified"
                )
            },
            "runtimeExecution": {
                "used": "ask_v2",
                "sectionDecision": (
                    "selected"
                    if selected_section_cards and section_allowed
                    else "blocked"
                    if selected_section_cards and not section_quality.get("allowed")
                    else "unavailable"
                    if section_first_requested and not selected_section_cards
                    else "skipped"
                ),
                "sectionBlockReason": _clean_text(section_quality.get("reason")) if selected_section_cards and not section_quality.get("allowed") else "",
                "fallbackReason": (
                    "section_blocked_to_claim_cards"
                    if selected_section_cards and not section_quality.get("allowed") and selected_claim_cards
                    else "section_blocked_no_claim_cards"
                    if selected_section_cards and not section_quality.get("allowed") and not selected_claim_cards
                    else "section_unavailable_to_claim_cards"
                    if section_first_requested and not selected_section_cards and selected_claim_cards
                    else "section_to_chunk_fallback"
                    if section_first_requested and not selected_claim_cards and not anchors
                    else ""
                ),
            },
            "claimCards": [
                {
                    "claimCardId": _clean_text(item.get("claim_card_id")),
                    "claimId": _clean_text(item.get("claim_id")),
                    "sourceKind": _clean_text(item.get("source_kind")),
                    "sourceId": _clean_text(item.get("source_id")),
                    "claimType": _clean_text(item.get("claim_type")),
                    "status": _clean_text(item.get("status")),
                    "summaryText": _clean_text(item.get("summary_text")),
                    "origin": _clean_text(item.get("origin")),
                    "trustLevel": _clean_text(item.get("trust_level")),
                    "task": _clean_text(item.get("task")),
                    "dataset": _clean_text(item.get("dataset")),
                    "metric": _clean_text(item.get("metric")),
                    "comparator": _clean_text(item.get("comparator")),
                    "resultDirection": _clean_text(item.get("result_direction")),
                    "resultValueText": _clean_text(item.get("result_value_text")),
                    "resultValueNumeric": item.get("result_value_numeric"),
                    "taskCanonical": _clean_text(item.get("task_canonical")),
                    "datasetCanonical": _clean_text(item.get("dataset_canonical")),
                    "datasetFamily": _clean_text(item.get("dataset_family")),
                    "datasetVersion": _clean_text(item.get("dataset_version")),
                    "metricCanonical": _clean_text(item.get("metric_canonical")),
                    "comparatorCanonical": _clean_text(item.get("comparator_canonical")),
                    "scopeText": _clean_text(item.get("scope_text")),
                    "conditionText": _clean_text(item.get("condition_text")),
                    "negativeScopeText": _clean_text(item.get("negative_scope_text")),
                    "limitationText": _clean_text(item.get("limitation_text")),
                    "evidenceStrength": _clean_text(item.get("evidence_strength")),
                    "evidenceAnchorIds": list(item.get("evidence_anchor_ids") or []),
                    "sectionPaths": list(item.get("section_paths") or []),
                    "anchorExcerpts": [
                        _clean_text(anchor.get("excerpt"))
                        for anchor in list(item.get("anchors") or [])[:3]
                        if _clean_text(anchor.get("excerpt"))
                    ],
                    "rankingReasons": list(item.get("ranking_reasons") or []),
                }
                for item in selected_claim_cards
            ],
            "sectionCards": [
                {
                    "sectionCardId": _clean_text(item.get("section_card_id")),
                    "paperId": _clean_text(item.get("paper_id")),
                    "documentId": _clean_text(item.get("document_id")),
                    "unitId": _clean_text(item.get("unit_id")),
                    "role": _clean_text(item.get("role")),
                    "unitType": _clean_text(item.get("unit_type")),
                    "sectionPath": _clean_text(item.get("section_path")),
                    "title": _clean_text(item.get("title")),
                    "contextualSummary": _clean_text(item.get("contextual_summary")),
                    "sourceExcerpt": _clean_text(item.get("source_excerpt")),
                    "documentThesis": _clean_text(item.get("document_thesis")),
                    "confidence": _stable_score(item.get("confidence")),
                    "origin": _clean_text(item.get("origin")),
                    "generatorModel": _clean_text(item.get("generator_model")),
                    "keyPoints": list(item.get("key_points") or []),
                    "scopeNotes": list(item.get("scope_notes") or []),
                    "claims": list(item.get("claims") or []),
                    "concepts": list(item.get("concepts") or []),
                    "rankingReasons": list(item.get("ranking_reasons") or []),
                }
                for item in selected_section_cards
            ],
            "claimVerification": claim_verification,
            "consensus": claim_consensus,
            "evidenceVerification": verification,
            "fallback": {
                "used": v2_fallback_used,
                "reason": str((evidence_packet.evidence_packet or {}).get("answerableDecisionReason") or verification.get("verificationStatus") or ""),
            },
        }
        if route.source_kind == "project":
            pipeline_result.v2_diagnostics["projectSignals"] = {
                "queryProfile": _classify_project_query_profile(query),
                "selectedFileRoles": [
                    _clean_text((card.get("diagnostics") or {}).get("fileRole") or card.get("file_role_core"))
                    for card in selected_cards
                    if _clean_text((card.get("diagnostics") or {}).get("fileRole") or card.get("file_role_core"))
                ],
                "weakSlots": verification.get("weakSlots") or [],
            }
        _ = alpha, allow_external
        return pipeline_result, evidence_packet


PaperAskV2Service = AskV2Service


__all__ = ["AskV2FallbackToLegacy", "AskV2Service", "PaperAskV2Service"]
