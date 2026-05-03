"""Additive memory-first prefilter execution for ask/answer paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any

from knowledge_hub.ai.retrieval_fit import is_non_substantive_text, is_vault_hub_note, normalize_source_type
from knowledge_hub.core.models import SearchResult
from knowledge_hub.document_memory import DocumentMemoryRetriever
from knowledge_hub.papers.memory_retriever import PaperMemoryRetriever
from knowledge_hub.papers.prefilter import (
    PAPER_MEMORY_MODE_COMPAT,
    PAPER_MEMORY_MODE_OFF,
    PAPER_MEMORY_MODE_ON,
    normalize_paper_memory_mode_details,
    normalize_paper_memory_mode,
    resolve_paper_memory_prefilter,
)

MEMORY_ROUTE_MODE_OFF = "off"
MEMORY_ROUTE_MODE_COMPAT = "compat"
MEMORY_ROUTE_MODE_ON = "on"
MEMORY_ROUTE_MODE_PREFILTER = "prefilter"
MEMORY_ROUTE_MODE_ALIASES = {
    MEMORY_ROUTE_MODE_PREFILTER: MEMORY_ROUTE_MODE_COMPAT,
}
MEMORY_ROUTE_MODES = {
    MEMORY_ROUTE_MODE_OFF,
    MEMORY_ROUTE_MODE_COMPAT,
    MEMORY_ROUTE_MODE_ON,
}
_TEMPORAL_RECENT_RE = re.compile(r"\b(latest|recent|updated|update|newest|changed since)\b|최근|최신|업데이트", re.IGNORECASE)
_TEMPORAL_BEFORE_RE = re.compile(r"\b(before|prior to|earlier than)\s+(20\d{2})\b|([0-9]{4})\s*이전", re.IGNORECASE)
_TEMPORAL_AFTER_RE = re.compile(r"\b(after|since)\s+(20\d{2})\b|([0-9]{4})\s*이후", re.IGNORECASE)


def normalize_memory_route_mode_details(value: Any, *, paper_memory_mode: Any = None) -> tuple[str, str, bool]:
    requested = str(value or "").strip().lower() or MEMORY_ROUTE_MODE_OFF
    paper_requested, paper_effective, paper_alias_applied = normalize_paper_memory_mode_details(paper_memory_mode)
    if requested in MEMORY_ROUTE_MODES:
        return requested, requested, False
    if requested in MEMORY_ROUTE_MODE_ALIASES:
        return requested, MEMORY_ROUTE_MODE_ALIASES[requested], True
    if paper_effective in {PAPER_MEMORY_MODE_COMPAT, PAPER_MEMORY_MODE_ON}:
        effective = {
            PAPER_MEMORY_MODE_COMPAT: MEMORY_ROUTE_MODE_COMPAT,
            PAPER_MEMORY_MODE_ON: MEMORY_ROUTE_MODE_ON,
        }[paper_effective]
        return requested, effective, paper_alias_applied or paper_requested != PAPER_MEMORY_MODE_OFF
    return requested, MEMORY_ROUTE_MODE_OFF, False


def normalize_memory_route_mode(value: Any, *, paper_memory_mode: Any = None) -> str:
    _, effective, _ = normalize_memory_route_mode_details(value, paper_memory_mode=paper_memory_mode)
    return effective


def memory_route_payload(
    *,
    requested_mode: str,
    source_type: str | None,
    paper_memory_mode: Any = None,
) -> dict[str, Any]:
    requested_token, effective_mode, mode_alias_applied = normalize_memory_route_mode_details(
        requested_mode,
        paper_memory_mode=paper_memory_mode,
    )
    return {
        "requestedMode": requested_token,
        "effectiveMode": effective_mode,
        "modeAliasApplied": mode_alias_applied,
        "sourceType": str(normalize_source_type(source_type) or "all"),
        "formsTried": [],
        "fallbackUsed": False,
        "memoryInfluenceApplied": False,
        "verificationCouplingApplied": False,
    }


def _parse_iso(value: Any) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        if token.endswith("Z"):
            return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(timezone.utc)
        parsed = datetime.fromisoformat(token)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    if re.fullmatch(r"\d{4}", token):
        try:
            return datetime(int(token), 1, 1, tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _temporal_query_signals(query: str) -> dict[str, Any]:
    text = str(query or "").strip()
    before_match = _TEMPORAL_BEFORE_RE.search(text)
    after_match = _TEMPORAL_AFTER_RE.search(text)
    latest = bool(_TEMPORAL_RECENT_RE.search(text))
    target_year = ""
    mode = "none"
    if before_match:
        mode = "before"
        target_year = next((group for group in before_match.groups() if group and group.isdigit()), "")
    elif after_match:
        mode = "after"
        target_year = next((group for group in after_match.groups() if group and group.isdigit()), "")
    elif latest:
        mode = "latest"
    return {
        "enabled": mode != "none",
        "mode": mode,
        "targetYear": target_year,
    }


def _document_filter_from_memory_hit(sqlite_db: Any, item: dict[str, Any]) -> dict[str, Any]:
    source_type = str(normalize_source_type(item.get("sourceType")) or "").strip()
    document_id = str(item.get("documentId") or "").strip()
    source_ref = str(item.get("sourceRef") or "").strip()
    if source_type == "paper":
        paper_id = source_ref or document_id.replace("paper:", "", 1)
        return {"arxiv_id": paper_id} if paper_id else {}
    if source_type == "web":
        payload: dict[str, Any] = {}
        if source_ref:
            payload["url"] = source_ref
        if document_id:
            payload["document_id"] = document_id
        return payload
    if source_type == "vault":
        note = sqlite_db.get_note(document_id) if sqlite_db and document_id else None
        file_path = str((note or {}).get("file_path") or "").strip()
        if file_path:
            return {"file_path": file_path}
    if source_type:
        return {"source_type": source_type}
    return {}


def _merge_search_results(results: list[Any], *, top_k: int, result_id_fn) -> list[Any]:
    merged: dict[str, Any] = {}
    for item in results:
        key = result_id_fn(item)
        existing = merged.get(key)
        if existing is None or float(getattr(item, "score", 0.0) or 0.0) > float(getattr(existing, "score", 0.0) or 0.0):
            merged[key] = item
    ranked = list(merged.values())
    ranked.sort(key=lambda row: float(getattr(row, "score", 0.0) or 0.0), reverse=True)
    return ranked[: max(1, int(top_k))]


def _query_terms(query: str) -> list[str]:
    return [part.casefold() for part in re.split(r"[^0-9A-Za-z가-힣]+", str(query or "").strip()) if part.strip()]


def _normalize_web_filter(filter_dict: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(filter_dict or {})
    if str(normalize_source_type(payload.get("source_type")) or "").strip() != "web" and not any(
        key in payload for key in ("url", "canonical_url", "source_url", "document_id")
    ):
        return payload
    url = str(payload.get("url") or payload.get("canonical_url") or payload.get("source_url") or "").strip()
    if url:
        payload["url"] = url
    payload.pop("canonical_url", None)
    payload.pop("source_url", None)
    return payload


def _fallback_parent_key(metadata: dict[str, Any], doc_id: str) -> str:
    for key in ("file_path", "arxiv_id", "url", "title"):
        token = str(metadata.get(key) or "").strip()
        if token:
            return token
    return str(doc_id or "").strip()


def _is_explainer_query(query: str) -> bool:
    token = str(query or "").casefold()
    return any(
        marker in token
        for marker in ("explain", "purpose", "reason", "role", "difference", "compare", "summary", "설명", "목적", "이유", "역할", "차이", "비교", "요약")
    )


def _explainer_fallback_bonus(query: str, metadata: dict[str, Any], document: str, normalized_source: str) -> float:
    if normalized_source != "vault" or not _is_explainer_query(query):
        return 0.0
    haystack = " ".join(
        [
            str(metadata.get("title") or ""),
            str(metadata.get("file_path") or ""),
            str(metadata.get("section_title") or ""),
            str(document or "")[:160],
        ]
    ).casefold()
    markers = ("설명", "목적", "이유", "역할", "차이", "비교", "요약", "purpose", "reason", "role", "difference", "summary", "explainer")
    return 0.12 if any(marker.casefold() in haystack for marker in markers) else 0.0


def _fallback_chunk_hits(
    searcher: Any,
    *,
    query: str,
    filters: list[dict[str, Any]],
    top_k: int,
) -> list[SearchResult]:
    database = getattr(searcher, "database", None)
    if database is None or not hasattr(database, "get_documents"):
        return []
    terms = _query_terms(query)
    phrase = str(query or "").strip().casefold()
    best_by_parent: dict[str, SearchResult] = {}
    seen_ids: set[str] = set()
    for filter_dict in filters:
        try:
            rows = database.get_documents(
                filter_dict=filter_dict,
                limit=max(6, int(top_k) * 6),
                include_ids=True,
                include_documents=True,
                include_metadatas=True,
            )
        except Exception:
            continue
        ids = list(rows.get("ids") or [])
        docs = list(rows.get("documents") or [])
        metas = list(rows.get("metadatas") or [])
        for idx, document in enumerate(docs):
            doc_id = str(ids[idx]) if idx < len(ids) else ""
            if doc_id and doc_id in seen_ids:
                continue
            metadata = metas[idx] if idx < len(metas) else {}
            normalized_source = normalize_source_type((metadata or {}).get("source_type"))
            searchable = " ".join(
                [
                    str(metadata.get("title") or ""),
                    str(metadata.get("source_ref") or ""),
                    str(metadata.get("url") or ""),
                    str(metadata.get("canonical_url") or ""),
                    str(metadata.get("file_path") or ""),
                    str(metadata.get("section_title") or ""),
                    str(metadata.get("section_path") or ""),
                    str(metadata.get("contextual_summary") or ""),
                    str(document or ""),
                ]
            ).casefold()
            overlap = 0.0
            if terms:
                matched = sum(1 for term in terms if term in searchable)
                overlap = matched / max(1, len(set(terms)))
            phrase_score = 1.0 if phrase and phrase in searchable else 0.0
            penalty = 0.0
            text = " ".join([str(metadata.get("title") or ""), str(document or "")]).strip()
            if is_non_substantive_text(text):
                penalty += 0.22
            if normalized_source == "vault" and is_vault_hub_note(
                title=str(metadata.get("title") or ""),
                file_path=str(metadata.get("file_path") or ""),
                document=str(document or ""),
                query=query,
            ):
                penalty += 0.24
            bonus = _explainer_fallback_bonus(query, metadata if isinstance(metadata, dict) else {}, str(document or ""), normalized_source)
            score = min(0.4, max(0.0, 0.08 + (0.22 * overlap) + (0.1 * phrase_score) + bonus - penalty))
            candidate = SearchResult(
                document=str(document or ""),
                metadata=metadata if isinstance(metadata, dict) else {},
                distance=1.0,
                score=score,
                semantic_score=0.0,
                lexical_score=score,
                retrieval_mode="memory_prefilter_fallback",
                lexical_extras={
                    "fallback_filter": dict(filter_dict),
                    "fallback_overlap": round(overlap, 6),
                    "fallback_phrase_score": round(phrase_score, 6),
                    "fallback_penalty": round(penalty, 6),
                    "fallback_bonus": round(bonus, 6),
                },
                document_id=doc_id,
            )
            parent_key = _fallback_parent_key(metadata if isinstance(metadata, dict) else {}, doc_id)
            existing = best_by_parent.get(parent_key)
            if existing is None or float(candidate.score or 0.0) > float(existing.score or 0.0):
                best_by_parent[parent_key] = candidate
            if doc_id:
                seen_ids.add(doc_id)
    collected = list(best_by_parent.values())
    collected.sort(key=lambda item: float(getattr(item, "score", 0.0) or 0.0), reverse=True)
    return collected[: max(1, int(top_k))]


def _dedupe_filters(filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for item in filters:
        payload = {str(key): str(value) for key, value in dict(item or {}).items() if str(value or "").strip()}
        if not payload:
            continue
        key = tuple(sorted(payload.items()))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(payload)
    return ordered


def _query_intent_hints(query: str) -> dict[str, bool]:
    token = str(query or "").casefold()
    return {
        "implementation": any(
            marker in token
            for marker in (
                "implementation",
                "implement",
                "mechanism",
                "runtime",
                "where",
                "how",
                "why",
                "reason",
                "relation",
                "reflect",
                "연결",
                "구현",
                "역할",
                "메커니즘",
                "어떻게",
                "이유",
                "왜",
                "관계",
                "반영",
            )
        ),
        "disambiguation": any(
            marker in token
            for marker in (
                "distinguish",
                "difference",
                "compare",
                "vs",
                "changed",
                "change",
                "relationship",
                "구분",
                "골라",
                "선택",
                "어디",
                "which source",
                "차이",
                "바뀌",
                "변경",
                "관계",
            )
        ),
    }


def _broad_source_filters(query: str) -> list[dict[str, Any]]:
    temporal = _temporal_query_signals(query).get("enabled")
    hints = _query_intent_hints(query)
    if temporal:
        order = ("paper", "web", "vault")
    elif hints["disambiguation"]:
        order = ("paper", "vault", "web")
    elif hints["implementation"]:
        order = ("vault", "paper", "web")
    else:
        order = ("vault", "paper", "web")
    return [{"source_type": source} for source in order]


def _search_with_filters(
    *,
    search_callable: Any,
    query: str,
    top_k: int,
    source_type: str | None,
    retrieval_mode: str,
    alpha: float,
    scoped_filter: dict[str, Any],
    filters: list[dict[str, Any]],
    result_id_fn,
    min_score: float,
) -> list[Any]:
    collected: list[Any] = []
    for narrowed_filter in _dedupe_filters(filters):
        merged_filter = dict(narrowed_filter)
        for key, value in scoped_filter.items():
            merged_filter.setdefault(key, value)
        if str(normalize_source_type(source_type) or "").strip() == "web":
            merged_filter = _normalize_web_filter(merged_filter)
        collected.extend(
            search_callable(
                query,
                top_k=top_k,
                source_type=source_type,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                metadata_filter=merged_filter,
            )
        )
    merged = _merge_search_results(collected, top_k=top_k, result_id_fn=result_id_fn)
    return [row for row in merged if float(getattr(row, "score", 0.0) or 0.0) >= min_score]


def _document_memory_hits_to_results(
    hits: list[dict[str, Any]],
    *,
    top_k: int,
    retrieval_mode: str,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    seen_document_ids: set[str] = set()
    for item in hits:
        document_id = str(item.get("documentId") or "").strip()
        if not document_id or document_id in seen_document_ids:
            continue
        seen_document_ids.add(document_id)
        matched_segment = dict(item.get("matchedSegment") or {})
        matched_unit = dict(item.get("matchedUnit") or {})
        document_summary = dict(item.get("documentSummary") or {})
        segment_text = str(matched_segment.get("segmentText") or "").strip()
        document_text = (
            segment_text
            or str(matched_unit.get("sourceExcerpt") or "").strip()
            or str(matched_unit.get("contextualSummary") or "").strip()
            or str(document_summary.get("contextualSummary") or "").strip()
        )
        if not document_text:
            continue
        metadata = {
            "title": str(item.get("documentTitle") or document_summary.get("title") or matched_unit.get("title") or document_id),
            "source_type": str(item.get("sourceType") or document_summary.get("sourceType") or ""),
            "source_ref": str(document_summary.get("sourceRef") or matched_unit.get("sourceRef") or ""),
            "section_path": str(matched_unit.get("sectionPath") or ""),
            "contextual_summary": str(document_summary.get("contextualSummary") or ""),
            "document_id": document_id,
            "file_path": str((matched_unit.get("provenance") or {}).get("file_path") or (document_summary.get("provenance") or {}).get("file_path") or ""),
            "document_date": str(document_summary.get("documentDate") or ""),
            "event_date": str(document_summary.get("eventDate") or ""),
            "observed_at": str(document_summary.get("observedAt") or ""),
        }
        source_ref = str(metadata.get("source_ref") or "").strip()
        if source_ref and metadata["source_type"] == "web":
            metadata["url"] = source_ref
        retrieval_signals = dict(item.get("retrievalSignals") or {})
        score = 0.32 + (0.04 * min(4, len(list(matched_segment.get("units") or []))))
        if retrieval_signals.get("updatesPreferred"):
            score += 0.03
        results.append(
            SearchResult(
                document=document_text,
                metadata=metadata,
                distance=1.0,
                score=min(0.55, score),
                semantic_score=0.0,
                lexical_score=min(0.55, score),
                retrieval_mode=f"{retrieval_mode}_document_memory_fallback",
                lexical_extras={
                    "document_memory_fallback": True,
                    "document_memory_source_type": metadata["source_type"],
                    "document_memory_retrieval_signals": retrieval_signals,
                },
                document_id=document_id,
            )
        )
        if len(results) >= max(1, int(top_k)):
            break
    return results


def _apply_updates_preference(sqlite_db: Any, *, hits: list[dict[str, Any]], form: str) -> tuple[list[dict[str, Any]], list[str]]:
    if not sqlite_db or not hits:
        return hits, []
    ordered = list(hits)
    used_relation_ids: list[str] = []
    position_by_id = {str(item.get("memoryId") or item.get("documentId") or ""): idx for idx, item in enumerate(ordered)}
    for idx, item in enumerate(list(ordered)):
        current_id = str(item.get("memoryId") or item.get("documentId") or "").strip()
        if not current_id:
            continue
        relations = sqlite_db.list_memory_relations(src_form=form, src_id=current_id, relation_type="updates", limit=5)
        for relation in relations:
            dst_id = str(relation.get("dst_id") or "").strip()
            dst_idx = position_by_id.get(dst_id)
            if dst_idx is None:
                continue
            if dst_idx > idx:
                newer = ordered.pop(dst_idx)
                ordered.insert(idx, newer)
                position_by_id = {
                    str(candidate.get("memoryId") or candidate.get("documentId") or ""): pos
                    for pos, candidate in enumerate(ordered)
                }
                relation_id = str(relation.get("relation_id") or "").strip()
                if relation_id:
                    used_relation_ids.append(relation_id)
    return ordered, used_relation_ids


@dataclass
class MemoryPrefilterExecution:
    results: list[Any]
    diagnostics: dict[str, Any]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _memory_confidence(
    *,
    query: str,
    normalized_source: str,
    diagnostics: dict[str, Any],
    matched_results: list[Any],
) -> float:
    score = 0.0
    matched_memory_count = len([item for item in list(diagnostics.get("matchedMemoryIds") or []) if str(item).strip()])
    matched_document_count = len([item for item in list(diagnostics.get("matchedDocumentIds") or []) if str(item).strip()])
    relation_count = len([item for item in list(diagnostics.get("memoryRelationsUsed") or []) if str(item).strip()])
    temporal_signals = dict(diagnostics.get("temporalSignals") or {})
    score += min(0.35, matched_memory_count * 0.12)
    score += min(0.22, matched_document_count * 0.08)
    score += min(0.12, relation_count * 0.04)
    if diagnostics.get("updatesPreferred"):
        score += 0.08
    if diagnostics.get("temporalRouteApplied") and str(temporal_signals.get("matchedField") or "").strip():
        score += 0.1
    elif temporal_signals.get("enabled"):
        score -= 0.08
    if normalized_source in {"paper", "vault", "web"}:
        score += 0.08
    if matched_results:
        top_score = max(_safe_float(getattr(item, "score", 0.0), 0.0) for item in matched_results[:3])
        score += min(0.18, top_score * 0.25)
    query_text = str(query or "")
    if re.search(r"\b(latest|recent|updated|newest|before|after|since)\b|최근|최신|업데이트|이전|이후|당시", query_text, re.IGNORECASE):
        score += 0.05 if diagnostics.get("temporalRouteApplied") else -0.08
    return round(max(0.0, min(1.0, score)), 3)


def _gating_threshold(*, query: str, normalized_source: str) -> float:
    threshold = 0.62
    if normalized_source == "paper":
        threshold = 0.68
    if re.search(r"\b(latest|recent|updated|newest|before|after|since)\b|최근|최신|업데이트|이전|이후|당시", str(query or ""), re.IGNORECASE):
        threshold += 0.06
    return round(min(0.85, threshold), 3)


def _is_strong_memory_route(*, diagnostics: dict[str, Any], normalized_source: str, used_memory_fallback: bool) -> bool:
    if used_memory_fallback:
        return True
    reason = str(diagnostics.get("reason") or "").strip().lower()
    if reason in {"matched_cards", "matched_document_memory"}:
        return True
    if normalized_source and reason == f"{normalized_source}_memory_fallback":
        return True
    return False


def _apply_gating_diagnostics(
    diagnostics: dict[str, Any],
    *,
    query: str,
    normalized_source: str,
    matched_results: list[Any],
    used_memory_fallback: bool = False,
    broad_fallback: bool = False,
) -> None:
    confidence = _memory_confidence(
        query=query,
        normalized_source=normalized_source,
        diagnostics=diagnostics,
        matched_results=matched_results,
    )
    threshold = _gating_threshold(query=query, normalized_source=normalized_source)
    query_hints = _query_intent_hints(query)
    temporal_signals = dict(diagnostics.get("temporalSignals") or {})
    temporal_enabled = bool(temporal_signals.get("enabled"))
    temporal_grounded = bool(str(temporal_signals.get("matchedField") or "").strip())
    strong_memory_route = _is_strong_memory_route(
        diagnostics=diagnostics,
        normalized_source=normalized_source,
        used_memory_fallback=used_memory_fallback,
    )
    diagnostics["memoryConfidence"] = confidence
    diagnostics["chunkExpansionThreshold"] = threshold
    if broad_fallback or not matched_results:
        diagnostics["gatingDecision"] = "full_fallback"
        diagnostics["chunkExpansionTriggered"] = True
        diagnostics["chunkExpansionReason"] = diagnostics.get("reason") or "no_memory_support"
        diagnostics["verifierBudgetUsed"] = max(1, len(matched_results))
        return
    if (
        strong_memory_route
        and confidence >= threshold
        and not query_hints["implementation"]
        and not query_hints["disambiguation"]
        and not bool(temporal_signals.get("weakObservedAtOnly"))
        and (not temporal_enabled or temporal_grounded)
    ):
        diagnostics["gatingDecision"] = "memory_only"
        diagnostics["chunkExpansionTriggered"] = False
        diagnostics["chunkExpansionReason"] = ""
        diagnostics["verifierBudgetUsed"] = 0
        del matched_results[1:]
        return
    diagnostics["gatingDecision"] = "memory_plus_verify"
    diagnostics["chunkExpansionTriggered"] = True
    diagnostics["chunkExpansionReason"] = diagnostics.get("reason") or "memory_verify_required"
    verifier_budget = min(max(1, len(matched_results)), 3)
    if query_hints["implementation"] or query_hints["disambiguation"]:
        verifier_budget = min(max(2, verifier_budget), len(matched_results))
    diagnostics["verifierBudgetUsed"] = verifier_budget
    del matched_results[verifier_budget:]


def execute_memory_prefilter(
    searcher: Any,
    *,
    query: str,
    top_k: int,
    source_type: str | None,
    retrieval_mode: str,
    alpha: float,
    min_score: float,
    requested_mode: str,
    metadata_filter: dict[str, Any] | None = None,
    query_forms: list[str] | None = None,
    result_id_fn,
    search_fn=None,
) -> MemoryPrefilterExecution:
    normalized_source = normalize_source_type(source_type)
    requested_token, mode, mode_alias_applied = normalize_memory_route_mode_details(requested_mode)
    temporal_signals = _temporal_query_signals(query)
    diagnostics = {
        "requestedMode": requested_token,
        "effectiveMode": mode,
        "modeAliasApplied": mode_alias_applied,
        "applied": False,
        "fallbackUsed": False,
        "mixedFallbackUsed": False,
        "matchedMemoryIds": [],
        "matchedDocumentIds": [],
        "memoryRelationsUsed": [],
        "temporalSignals": temporal_signals,
        "temporalRouteApplied": bool(temporal_signals.get("enabled")),
        "updatesPreferred": False,
        "formsTried": [],
        "reason": "",
        "memoryConfidence": 0.0,
        "gatingDecision": "full_fallback",
        "chunkExpansionTriggered": True,
        "chunkExpansionReason": "",
        "verifierBudgetUsed": 0,
        "chunkExpansionThreshold": 0.0,
        "staleMemorySignals": [],
        "memoryInfluenceApplied": False,
        "verificationCouplingApplied": False,
        "fallbackReason": "",
    }
    scoped_filter = dict(metadata_filter or {})
    if normalized_source == "web":
        scoped_filter = _normalize_web_filter(scoped_filter)
    search_callable = search_fn or searcher.search
    if mode == MEMORY_ROUTE_MODE_OFF:
        diagnostics["reason"] = "disabled"
        results = search_callable(
            query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            metadata_filter=scoped_filter,
        )
        filtered = [r for r in results if r.score >= min_score]
        _apply_gating_diagnostics(
            diagnostics,
            query=query,
            normalized_source=normalized_source,
            matched_results=filtered,
            broad_fallback=True,
        )
        return MemoryPrefilterExecution(filtered, diagnostics)

    if not normalized_source:
        diagnostics["formsTried"] = ["paper_memory", "document_memory", "chunk"]
        diagnostics["fallbackUsed"] = True
        diagnostics["mixedFallbackUsed"] = True
        diagnostics["reason"] = "mixed_fallback_no_hit"
        mixed_filters: list[dict[str, Any]] = list(_broad_source_filters(query))
        if searcher.sqlite_db is not None:
            try:
                paper_hits = PaperMemoryRetriever(searcher.sqlite_db).search(
                    query,
                    limit=max(3, int(top_k)),
                    include_refs=False,
                )
            except Exception:
                paper_hits = []
            try:
                document_hits = DocumentMemoryRetriever(searcher.sqlite_db).search(
                    query,
                    limit=max(4, int(top_k) * 2),
                )
            except Exception:
                document_hits = []
            diagnostics["matchedMemoryIds"] = [
                str(item.get("memoryId") or "").strip()
                for item in paper_hits
                if str(item.get("memoryId") or "").strip()
            ] + [
                str((item.get("documentSummary") or {}).get("unitId") or item.get("documentId") or "").strip()
                for item in document_hits
                if str((item.get("documentSummary") or {}).get("unitId") or item.get("documentId") or "").strip()
            ]
            diagnostics["matchedDocumentIds"] = [
                str(item.get("paperId") or "").strip()
                for item in paper_hits
                if str(item.get("paperId") or "").strip()
            ] + [
                str(item.get("documentId") or "").strip()
                for item in document_hits
                if str(item.get("documentId") or "").strip()
            ]
            for item in paper_hits:
                paper_id = str(item.get("paperId") or "").strip()
                if paper_id:
                    mixed_filters.insert(0, {"source_type": "paper", "arxiv_id": paper_id})
            for item in document_hits:
                narrowed = _document_filter_from_memory_hit(searcher.sqlite_db, item)
                if narrowed:
                    mixed_filters.insert(0, narrowed)
        filtered = _search_with_filters(
            search_callable=search_callable,
            query=query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            scoped_filter=scoped_filter,
            filters=mixed_filters,
            result_id_fn=result_id_fn,
            min_score=min_score,
        )
        if filtered:
            diagnostics["applied"] = True
            diagnostics["memoryInfluenceApplied"] = True
            diagnostics["reason"] = "mixed_fallback_ranked"
            _apply_gating_diagnostics(
                diagnostics,
                query=query,
                normalized_source=normalized_source,
                matched_results=filtered,
                broad_fallback=True,
            )
            return MemoryPrefilterExecution(filtered, diagnostics)
        fallback_hits = _fallback_chunk_hits(
            searcher,
            query=query,
            filters=_dedupe_filters(mixed_filters),
            top_k=top_k,
        )
        if fallback_hits:
            diagnostics["applied"] = True
            diagnostics["memoryInfluenceApplied"] = True
            diagnostics["reason"] = "mixed_fallback_chunk"
            _apply_gating_diagnostics(
                diagnostics,
                query=query,
                normalized_source=normalized_source,
                matched_results=fallback_hits,
                broad_fallback=True,
            )
            return MemoryPrefilterExecution(fallback_hits, diagnostics)
        results = search_callable(
            query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            metadata_filter=scoped_filter,
        )
        filtered = [r for r in results if r.score >= min_score]
        _apply_gating_diagnostics(
            diagnostics,
            query=query,
            normalized_source=normalized_source,
            matched_results=filtered,
            broad_fallback=True,
        )
        return MemoryPrefilterExecution(filtered, diagnostics)

    if normalized_source == "paper":
        diagnostics["formsTried"] = ["paper_memory", "document_memory", "chunk"]
        paper_prefilter = resolve_paper_memory_prefilter(
            searcher.sqlite_db,
            query=query,
            source_type=normalized_source,
            requested_mode=normalize_paper_memory_mode(mode),
            limit=min(max(1, int(top_k)), 5),
        )
        diagnostics["matchedMemoryIds"] = list(paper_prefilter.get("matchedMemoryIds") or [])
        diagnostics["matchedDocumentIds"] = list(paper_prefilter.get("matchedPaperIds") or [])
        diagnostics["fallbackUsed"] = bool(paper_prefilter.get("fallbackUsed"))
        diagnostics["reason"] = str(paper_prefilter.get("reason") or "")
        if not paper_prefilter.get("applied"):
            results = search_callable(
                query,
                top_k=top_k,
                source_type=source_type,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                metadata_filter=scoped_filter,
            )
            filtered = [r for r in results if r.score >= min_score]
            _apply_gating_diagnostics(
                diagnostics,
                query=query,
                normalized_source=normalized_source,
                matched_results=filtered,
                broad_fallback=True,
            )
            return MemoryPrefilterExecution(filtered, diagnostics)

        candidate_ids = [str(item).strip() for item in diagnostics["matchedDocumentIds"] if str(item).strip()]
        memory_hits = []
        try:
            memory_hits = PaperMemoryRetriever(searcher.sqlite_db).search(query, limit=max(3, len(candidate_ids)), include_refs=False)
            memory_hits, used_relations = _apply_updates_preference(searcher.sqlite_db, hits=memory_hits, form="paper_memory")
            diagnostics["memoryRelationsUsed"] = used_relations
            if memory_hits:
                first_signals = dict((memory_hits[0].get("retrievalSignals") or {}))
                temporal_signals = dict(first_signals.get("temporalSignals") or {})
                if temporal_signals:
                    diagnostics["temporalSignals"] = temporal_signals
                    diagnostics["temporalRouteApplied"] = bool(temporal_signals.get("enabled"))
                diagnostics["updatesPreferred"] = any(
                    bool((item.get("retrievalSignals") or {}).get("updatesPreferred"))
                    for item in memory_hits
                )
        except Exception:
            memory_hits = []
        prioritized_ids = [
            str(item.get("paperId") or "").strip()
            for item in memory_hits
            if str(item.get("paperId") or "").strip() in candidate_ids
        ]
        if prioritized_ids:
            candidate_ids = prioritized_ids + [item for item in candidate_ids if item not in prioritized_ids]
        if diagnostics["updatesPreferred"]:
            diagnostics["staleMemorySignals"].append("updates_relation_present")
        if diagnostics["temporalRouteApplied"] and not str((diagnostics.get("temporalSignals") or {}).get("matchedField") or "").strip():
            diagnostics["staleMemorySignals"].append("temporal_query_without_grounded_memory")
        collected: list[Any] = []
        for paper_id in candidate_ids:
            narrowed_filter = dict(scoped_filter)
            narrowed_filter.setdefault("source_type", "paper")
            narrowed_filter["arxiv_id"] = paper_id
            collected.extend(
                search_callable(
                    query,
                    top_k=top_k,
                    source_type="paper",
                    retrieval_mode=retrieval_mode,
                    alpha=alpha,
                    metadata_filter=narrowed_filter,
                )
            )
        merged = _merge_search_results(collected, top_k=top_k, result_id_fn=result_id_fn)
        filtered = [row for row in merged if float(getattr(row, "score", 0.0) or 0.0) >= min_score]
        if filtered:
            diagnostics["applied"] = True
            diagnostics["memoryInfluenceApplied"] = True
            _apply_gating_diagnostics(
                diagnostics,
                query=query,
                normalized_source=normalized_source,
                matched_results=filtered,
            )
            return MemoryPrefilterExecution(filtered, diagnostics)
        fallback_filters = []
        for paper_id in candidate_ids:
            fallback_filters.append({"arxiv_id": paper_id})
        fallback_hits = _fallback_chunk_hits(searcher, query=query, filters=fallback_filters, top_k=top_k)
        if fallback_hits:
            diagnostics["applied"] = True
            diagnostics["memoryInfluenceApplied"] = True
            diagnostics["fallbackUsed"] = True
            diagnostics["reason"] = "memory_prefilter_chunk_fallback"
            _apply_gating_diagnostics(
                diagnostics,
                query=query,
                normalized_source=normalized_source,
                matched_results=fallback_hits,
            )
            return MemoryPrefilterExecution(fallback_hits, diagnostics)
        diagnostics["fallbackUsed"] = True
        diagnostics["reason"] = "prefilter_no_ranked_results"
        results = search_callable(
            query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            metadata_filter=scoped_filter,
        )
        filtered = [r for r in results if r.score >= min_score]
        _apply_gating_diagnostics(
            diagnostics,
            query=query,
            normalized_source=normalized_source,
            matched_results=filtered,
            broad_fallback=True,
        )
        return MemoryPrefilterExecution(filtered, diagnostics)

    if normalized_source in {"vault", "web"}:
        diagnostics["formsTried"] = ["document_memory", "chunk"]
        if searcher.sqlite_db is None:
            diagnostics["fallbackUsed"] = True
            diagnostics["reason"] = "sqlite_unavailable"
            results = search_callable(
                query,
                top_k=top_k,
                source_type=source_type,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                metadata_filter=scoped_filter,
            )
            filtered = [r for r in results if r.score >= min_score]
            _apply_gating_diagnostics(
                diagnostics,
                query=query,
                normalized_source=normalized_source,
                matched_results=filtered,
                broad_fallback=True,
            )
            return MemoryPrefilterExecution(filtered, diagnostics)
        try:
            forms = [str(item or "").strip() for item in (query_forms or []) if str(item or "").strip()]
            if str(query or "").strip() and str(query).strip() not in forms:
                forms.append(str(query).strip())
            for form in forms[:6]:
                memory_hits = DocumentMemoryRetriever(searcher.sqlite_db).search(form, limit=max(3, int(top_k)))
                memory_hits = [item for item in memory_hits if str(normalize_source_type(item.get("sourceType")) or "") == normalized_source]
                memory_hits, used_relations = _apply_updates_preference(searcher.sqlite_db, hits=memory_hits, form="document_memory")
                diagnostics["memoryRelationsUsed"] = used_relations
                if memory_hits:
                    first_signals = dict((memory_hits[0].get("retrievalSignals") or {}))
                    temporal_signals = dict(first_signals.get("temporalSignals") or {})
                    if temporal_signals:
                        diagnostics["temporalSignals"] = temporal_signals
                        diagnostics["temporalRouteApplied"] = bool(temporal_signals.get("enabled"))
                    diagnostics["updatesPreferred"] = any(
                        bool((item.get("retrievalSignals") or {}).get("updatesPreferred"))
                        for item in memory_hits
                    )
                    diagnostics["formsTried"] = ["document_memory", *forms[:6], "chunk"]
                    break
        except Exception:
            memory_hits = []
        if not memory_hits:
            diagnostics["fallbackUsed"] = True
            diagnostics["staleMemorySignals"].append("no_memory_hits")
            source_fallback_hits = _fallback_chunk_hits(
                searcher,
                query=query,
                filters=[{"source_type": normalized_source}],
                top_k=top_k,
            )
            if source_fallback_hits:
                diagnostics["applied"] = True
                diagnostics["memoryInfluenceApplied"] = True
                diagnostics["reason"] = f"{normalized_source}_chunk_fallback"
                _apply_gating_diagnostics(
                    diagnostics,
                    query=query,
                    normalized_source=normalized_source,
                    matched_results=source_fallback_hits,
                )
                return MemoryPrefilterExecution(source_fallback_hits, diagnostics)
            diagnostics["reason"] = "no_memory_hits"
            results = search_callable(
                query,
                top_k=top_k,
                source_type=source_type,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
                metadata_filter=scoped_filter,
            )
            filtered = [r for r in results if r.score >= min_score]
            _apply_gating_diagnostics(
                diagnostics,
                query=query,
                normalized_source=normalized_source,
                matched_results=filtered,
                broad_fallback=True,
            )
            return MemoryPrefilterExecution(filtered, diagnostics)
        diagnostics["matchedDocumentIds"] = [
            str(item.get("documentId") or "").strip()
            for item in memory_hits
            if str(item.get("documentId") or "").strip()
        ]
        diagnostics["matchedMemoryIds"] = [
            str((item.get("documentSummary") or {}).get("unitId") or item.get("documentId") or "").strip()
            for item in memory_hits
            if str((item.get("documentSummary") or {}).get("unitId") or item.get("documentId") or "").strip()
        ]
        if diagnostics["updatesPreferred"]:
            diagnostics["staleMemorySignals"].append("updates_relation_present")
        if diagnostics["temporalRouteApplied"] and not str((diagnostics.get("temporalSignals") or {}).get("matchedField") or "").strip():
            diagnostics["staleMemorySignals"].append("temporal_query_without_grounded_memory")
        collected: list[Any] = []
        fallback_filters: list[dict[str, Any]] = []
        for item in memory_hits:
            narrowed_filter = _document_filter_from_memory_hit(searcher.sqlite_db, item)
            if not narrowed_filter:
                continue
            fallback_filters.append(dict(narrowed_filter))
            for key, value in scoped_filter.items():
                narrowed_filter.setdefault(key, value)
            if normalized_source == "web":
                narrowed_filter = _normalize_web_filter(narrowed_filter)
            collected.extend(
                search_callable(
                    query,
                    top_k=top_k,
                    source_type=source_type,
                    retrieval_mode=retrieval_mode,
                    alpha=alpha,
                    metadata_filter=narrowed_filter,
                )
            )
        merged = _merge_search_results(collected, top_k=top_k, result_id_fn=result_id_fn)
        filtered = [row for row in merged if float(getattr(row, "score", 0.0) or 0.0) >= min_score]
        if filtered:
            diagnostics["applied"] = True
            diagnostics["memoryInfluenceApplied"] = True
            diagnostics["reason"] = "matched_document_memory"
            _apply_gating_diagnostics(
                diagnostics,
                query=query,
                normalized_source=normalized_source,
                matched_results=filtered,
            )
            return MemoryPrefilterExecution(filtered, diagnostics)
        broadened_filters = list(fallback_filters)
        broadened_filters.append({"source_type": normalized_source})
        fallback_hits = _fallback_chunk_hits(searcher, query=query, filters=fallback_filters, top_k=top_k)
        if fallback_hits:
            diagnostics["applied"] = True
            diagnostics["memoryInfluenceApplied"] = True
            diagnostics["fallbackUsed"] = True
            diagnostics["reason"] = "memory_prefilter_chunk_fallback"
            _apply_gating_diagnostics(
                diagnostics,
                query=query,
                normalized_source=normalized_source,
                matched_results=fallback_hits,
            )
            return MemoryPrefilterExecution(fallback_hits, diagnostics)
        broadened_hits = _fallback_chunk_hits(searcher, query=query, filters=broadened_filters, top_k=top_k)
        if broadened_hits:
            diagnostics["applied"] = True
            diagnostics["memoryInfluenceApplied"] = True
            diagnostics["fallbackUsed"] = True
            diagnostics["reason"] = f"{normalized_source}_chunk_fallback_success"
            _apply_gating_diagnostics(
                diagnostics,
                query=query,
                normalized_source=normalized_source,
                matched_results=broadened_hits,
            )
            return MemoryPrefilterExecution(broadened_hits, diagnostics)
        memory_backed_hits = _document_memory_hits_to_results(
            memory_hits,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        if memory_backed_hits:
            diagnostics["applied"] = True
            diagnostics["memoryInfluenceApplied"] = True
            diagnostics["fallbackUsed"] = True
            diagnostics["reason"] = f"{normalized_source}_memory_fallback"
            _apply_gating_diagnostics(
                diagnostics,
                query=query,
                normalized_source=normalized_source,
                matched_results=memory_backed_hits,
                used_memory_fallback=True,
            )
            return MemoryPrefilterExecution(memory_backed_hits, diagnostics)
        diagnostics["fallbackUsed"] = True
        diagnostics["reason"] = "prefilter_no_ranked_results"
        results = search_callable(
            query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            metadata_filter=scoped_filter,
        )
        filtered = [r for r in results if r.score >= min_score]
        _apply_gating_diagnostics(
            diagnostics,
            query=query,
            normalized_source=normalized_source,
            matched_results=filtered,
            broad_fallback=True,
        )
        return MemoryPrefilterExecution(filtered, diagnostics)

    diagnostics["fallbackUsed"] = True
    diagnostics["reason"] = "source_not_supported"
    results = search_callable(
        query,
        top_k=top_k,
        source_type=source_type,
        retrieval_mode=retrieval_mode,
        alpha=alpha,
        metadata_filter=scoped_filter,
    )
    filtered = [r for r in results if r.score >= min_score]
    _apply_gating_diagnostics(
        diagnostics,
        query=query,
        normalized_source=normalized_source,
        matched_results=filtered,
        broad_fallback=True,
    )
    return MemoryPrefilterExecution(filtered, diagnostics)


__all__ = [
    "MEMORY_ROUTE_MODE_COMPAT",
    "MEMORY_ROUTE_MODE_OFF",
    "MEMORY_ROUTE_MODE_ON",
    "MEMORY_ROUTE_MODE_PREFILTER",
    "MEMORY_ROUTE_MODES",
    "MemoryPrefilterExecution",
    "execute_memory_prefilter",
    "memory_route_payload",
    "normalize_memory_route_mode",
    "normalize_memory_route_mode_details",
]
