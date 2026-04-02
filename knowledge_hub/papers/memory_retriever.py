"""Retrieve paper memory cards without changing default RAG behavior."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from knowledge_hub.papers.memory_models import PaperMemoryCard
from knowledge_hub.papers.memory_payloads import hydrated_card_payload


def _query_terms(query: str) -> list[str]:
    return [part.casefold() for part in re.split(r"[^0-9A-Za-z가-힣]+", str(query or "").strip()) if part.strip()]


def _text_overlap_score(terms: list[str], *parts: str) -> float:
    haystack = " ".join(str(part or "") for part in parts).casefold()
    score = 0.0
    for term in terms:
        if term in haystack:
            score += 1.0
    if terms and " ".join(terms) in haystack:
        score += 2.0
    return score


def _parse_iso(value: Any) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        if token.endswith("Z"):
            return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(timezone.utc)
        parsed = datetime.fromisoformat(token)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        pass
    if re.fullmatch(r"\d{4}", token):
        try:
            return datetime(int(token), 1, 1, tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _is_temporal_query(query: str) -> bool:
    token = str(query or "")
    return bool(
        re.search(r"\b(latest|recent|updated|newest|changed since|before|after|since)\b", token, re.IGNORECASE)
        or re.search(r"최근|최신|업데이트|이전|이후|당시", token)
    )


def _temporal_signal(query: str, item: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if not _is_temporal_query(query):
        return 0.0, {"enabled": False}
    published_dt = _parse_iso(item.get("publishedAt") or item.get("published_at") or "")
    evidence_dt = _parse_iso(item.get("evidenceWindow") or item.get("evidence_window") or "")
    active_dt = evidence_dt or published_dt
    field_name = "evidence_window" if evidence_dt else "published_at" if published_dt else ""
    if active_dt is None:
        return 0.0, {"enabled": True, "matchedField": "", "matchedValue": ""}
    current_year = datetime.now(timezone.utc).year
    signal = 0.0
    query_text = str(query or "")
    if re.search(r"\b(before|prior to|earlier than)\b|이전|당시", query_text, re.IGNORECASE):
        year_match = re.search(r"(20\d{2})", query_text)
        if year_match:
            signal = 1.5 if active_dt.year <= int(year_match.group(1)) else -0.75
    elif re.search(r"\b(after|since)\b|이후", query_text, re.IGNORECASE):
        year_match = re.search(r"(20\d{2})", query_text)
        if year_match:
            signal = 1.5 if active_dt.year >= int(year_match.group(1)) else -0.75
        else:
            signal = max(0.0, 1.8 - ((current_year - active_dt.year) * 0.3))
    else:
        signal = max(0.0, 1.8 - ((current_year - active_dt.year) * 0.3))
    return round(signal, 3), {"enabled": True, "matchedField": field_name, "matchedValue": active_dt.isoformat()}


def _updates_relation_signal(sqlite_db: Any, memory_id: str) -> tuple[float, list[str], bool]:
    if sqlite_db is None or not memory_id or not getattr(sqlite_db, "list_memory_relations", None):
        return 0.0, [], False
    incoming = sqlite_db.list_memory_relations(dst_form="paper_memory", dst_id=memory_id, relation_type="updates", limit=5)
    outgoing = sqlite_db.list_memory_relations(src_form="paper_memory", src_id=memory_id, relation_type="updates", limit=5)
    score = (0.35 * len(incoming)) - (0.15 * len(outgoing))
    relation_ids = [
        str(item.get("relation_id") or "").strip()
        for item in [*incoming, *outgoing]
        if str(item.get("relation_id") or "").strip()
    ]
    return round(score, 3), relation_ids, bool(incoming)


class PaperMemoryRetriever:
    def __init__(self, sqlite_db):
        self.sqlite_db = sqlite_db

    def search(self, query: str, *, limit: int = 10, include_refs: bool = True) -> list[dict[str, Any]]:
        terms = _query_terms(query)
        search_cards = getattr(self.sqlite_db, "search_paper_memory_cards", None)
        rows = search_cards(query, limit=max(1, int(limit))) if callable(search_cards) else []
        list_cards = getattr(self.sqlite_db, "list_paper_memory_cards", None)
        if (not rows or (_is_temporal_query(query) and len(rows) < max(1, int(limit)))) and terms and callable(list_cards):
            candidate_rows = list_cards(limit=max(200, int(limit) * 25))
            supplemental: list[dict[str, Any]] = []
            seen_paper_ids = {str(item.get("paper_id") or "").strip() for item in rows}
            for row in candidate_rows:
                paper_id = str(row.get("paper_id") or "").strip()
                if paper_id in seen_paper_ids:
                    continue
                overlap = _text_overlap_score(
                    terms,
                    row.get("title"),
                    row.get("paper_core"),
                    row.get("problem_context"),
                    row.get("method_core"),
                    row.get("evidence_core"),
                    row.get("limitations"),
                    row.get("search_text"),
                )
                if overlap <= 0.0:
                    continue
                patched = dict(row)
                patched["_fallback_overlap"] = overlap
                supplemental.append(patched)
            supplemental.sort(
                key=lambda item: (
                    -float(item.get("_fallback_overlap") or 0.0),
                    str(item.get("updated_at") or ""),
                )
            )
            rows = list(rows) + supplemental[: max(1, int(limit) * 3)]
        result: list[dict[str, Any]] = []
        for row in rows:
            card = PaperMemoryCard.from_row(row)
            if card is None:
                continue
            payload = self.hydrate(card, include_refs=include_refs)
            temporal_score, temporal_meta = _temporal_signal(query, payload)
            relation_boost, relation_ids, updates_preferred = _updates_relation_signal(
                self.sqlite_db,
                str(payload.get("memoryId") or ""),
            )
            title_match = _text_overlap_score(terms, payload.get("title"))
            lexical_match = _text_overlap_score(
                terms,
                payload.get("paperCore"),
                payload.get("problemContext"),
                payload.get("methodCore"),
                payload.get("evidenceCore"),
                payload.get("limitations"),
                payload.get("searchText"),
                " ".join(payload.get("conceptLinks") or []),
            )
            quality_flag = str(payload.get("qualityFlag") or "unscored").strip().lower()
            quality_boost = 0.0
            if quality_flag == "ok":
                quality_boost = 1.0
            elif quality_flag == "needs_review":
                quality_boost = 0.25
            elif quality_flag == "reject":
                quality_boost = -1.5
            total = lexical_match + (0.75 * title_match) + temporal_score + relation_boost + quality_boost
            payload["retrievalSignals"] = {
                "strategy": "paper_memory_card_rerank",
                "lexicalMatch": round(lexical_match, 3),
                "titleMatch": round(title_match, 3),
                "timeRelevance": round(temporal_score, 3),
                "updateRelationBoost": round(relation_boost, 3),
                "temporalSignals": temporal_meta,
                "memoryRelationsUsed": relation_ids,
                "updatesPreferred": bool(updates_preferred),
                "qualityBoost": round(quality_boost, 3),
                "score": round(total, 3),
            }
            if float(row.get("_fallback_overlap") or 0.0) > 0.0:
                payload["retrievalSignals"]["fallbackLexicalMatch"] = round(float(row.get("_fallback_overlap") or 0.0), 3)
            result.append(payload)
        result.sort(
            key=lambda item: (
                -float((item.get("retrievalSignals") or {}).get("score") or 0.0),
                -float((item.get("retrievalSignals") or {}).get("timeRelevance") or 0.0),
                str(item.get("paperId") or ""),
            )
        )
        return self._prefer_latest_updates(result)[: max(1, int(limit))]

    def get(self, paper_id: str, *, include_refs: bool = True) -> dict[str, Any] | None:
        row = self.sqlite_db.get_paper_memory_card(paper_id)
        card = PaperMemoryCard.from_row(row)
        if card is None:
            return None
        return self.hydrate(card, include_refs=include_refs)

    def hydrate(self, card: PaperMemoryCard, *, include_refs: bool = True) -> dict[str, Any]:
        return hydrated_card_payload(card, sqlite_db=self.sqlite_db, include_refs=include_refs)

    def _prefer_latest_updates(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items or not getattr(self.sqlite_db, "list_memory_relations", None):
            return items
        ordered = list(items)
        position_by_id = {str(item.get("memoryId") or ""): idx for idx, item in enumerate(ordered)}
        for idx, item in enumerate(list(ordered)):
            memory_id = str(item.get("memoryId") or "").strip()
            if not memory_id:
                continue
            relations = self.sqlite_db.list_memory_relations(src_form="paper_memory", src_id=memory_id, relation_type="updates", limit=5)
            fallback_candidates = []
            for relation in relations:
                dst_id = str(relation.get("dst_id") or "").strip()
                dst_idx = position_by_id.get(dst_id)
                if dst_idx is None:
                    continue
                fallback_candidates.append(str(relation.get("relation_id") or "").strip())
                if dst_idx > idx:
                    newer = ordered.pop(dst_idx)
                    ordered.insert(idx, newer)
                    position_by_id = {str(candidate.get("memoryId") or ""): pos for pos, candidate in enumerate(ordered)}
            if fallback_candidates:
                item["fallbackCandidates"] = fallback_candidates
                signals = dict(item.get("retrievalSignals") or {})
                signals["fallbackCandidates"] = list(fallback_candidates)
                item["retrievalSignals"] = signals
        return ordered
