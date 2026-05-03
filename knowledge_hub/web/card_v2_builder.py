"""Build additive v2 web cards from existing web/document-memory artifacts."""

from __future__ import annotations

import re
from typing import Any

from knowledge_hub.core.card_v2_common import (
    best_unit as _best_unit,
    claim_is_accepted as _claim_is_accepted,
    clean_text as _clean_text,
    coverage_status as _coverage_status,
    first_nonempty as _first_nonempty,
    parse_note_metadata as _parse_note_metadata,
    slot_excerpt as _slot_excerpt,
    snippet_hash as _snippet_hash,
    stable_anchor_id as _stable_anchor_id,
)
from knowledge_hub.web.ingest import make_web_note_id


_SECTION_RESULT_RE = re.compile(r"\b(result|finding|evaluation|experiment|benchmark|update|release|결과|평가|실험|업데이트|릴리스)\b", re.IGNORECASE)
_SECTION_LIMIT_RE = re.compile(r"\b(limit|limitation|risk|caveat|warning|deprecated|한계|주의|위험|deprecated)\b", re.IGNORECASE)
_SECTION_TOPIC_RE = re.compile(r"\b(topic|overview|summary|guide|reference|concept|개요|요약|가이드|정의|개념)\b", re.IGNORECASE)
_VERSION_RE = re.compile(r"\b(v\d+(?:\.\d+)*)\b|\b(version\s*\d+(?:\.\d+)*)\b|버전|업데이트|release|updated", re.IGNORECASE)

def _claim_role(claim: dict[str, Any]) -> str:
    text = _clean_text(claim.get("claim_text"))
    predicate = _clean_text(claim.get("predicate")).casefold()
    haystack = f"{predicate} {text}"
    if _VERSION_RE.search(haystack):
        return "version"
    if _SECTION_LIMIT_RE.search(haystack):
        return "limitation"
    if _SECTION_RESULT_RE.search(haystack):
        return "result"
    if _SECTION_TOPIC_RE.search(haystack):
        return "topic"
    return "supporting"


def _claim_sort_key(claim: dict[str, Any]) -> tuple[float, float, str]:
    confidence = float(claim.get("confidence") or 0.0)
    evidence_ptrs = claim.get("evidence_ptrs") if isinstance(claim.get("evidence_ptrs"), list) else []
    evidence_weight = min(1.0, 0.12 * len([ptr for ptr in evidence_ptrs if isinstance(ptr, dict)]))
    return confidence + evidence_weight, confidence, _clean_text(claim.get("claim_id"))


class WebCardV2Builder:
    def __init__(self, sqlite_db):
        self.sqlite_db = sqlite_db

    def _document_payload(self, canonical_url: str) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
        document_id = make_web_note_id(canonical_url)
        summary = self.sqlite_db.get_document_memory_summary(document_id)
        units = self.sqlite_db.list_document_memory_units(document_id, limit=200)
        note = self.sqlite_db.get_note(document_id)
        return document_id, summary, list(units or []), note

    def _claims_for_document(self, *, note_id: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        rows = list(self.sqlite_db.list_claims_by_note(note_id, limit=80))
        for key in ("record_id", "source_item_id", "canonical_url"):
            token = _clean_text(metadata.get(key))
            if token:
                rows.extend(self.sqlite_db.list_claims_by_record(token, limit=80))
        accepted = [dict(row) for row in rows if _claim_is_accepted(dict(row))]
        accepted.sort(key=_claim_sort_key, reverse=True)
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in accepted:
            claim_id = _clean_text(row.get("claim_id"))
            if not claim_id or claim_id in seen:
                continue
            seen.add(claim_id)
            deduped.append(row)
        return deduped

    def _entity_refs(self, summary: dict[str, Any] | None, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        seen: set[str] = set()
        concepts: list[str] = []
        for item in [summary or {}, *units[:12]]:
            concepts.extend(list(item.get("concepts") or []))
        for index, raw in enumerate(concepts[:10], 1):
            token = _clean_text(raw)
            if not token or token in seen:
                continue
            seen.add(token)
            resolved = self.sqlite_db.resolve_entity(token) if getattr(self.sqlite_db, "resolve_entity", None) else None
            refs.append(
                {
                    "entity_id": _clean_text((resolved or {}).get("entity_id") or token),
                    "entity_name": _clean_text((resolved or {}).get("canonical_name") or token),
                    "entity_type": _clean_text((resolved or {}).get("entity_type") or "concept"),
                    "weight": round(max(0.25, 1.0 - (0.08 * index)), 3),
                    "role": "concept",
                }
            )
        return refs

    def build(self, *, canonical_url: str) -> dict[str, Any]:
        token = _clean_text(canonical_url)
        if not token:
            raise ValueError("canonical_url is required")
        document_id, summary, units, note = self._document_payload(token)
        if not summary:
            raise ValueError(f"web document memory not found: {canonical_url}")
        metadata = _parse_note_metadata(note)
        claims = self._claims_for_document(note_id=document_id, metadata=metadata)

        result_unit = _best_unit(units, _SECTION_RESULT_RE)
        limit_unit = _best_unit(units, _SECTION_LIMIT_RE)
        topic_unit = _best_unit(units, _SECTION_TOPIC_RE)

        title = _first_nonempty((summary or {}).get("document_title"), (note or {}).get("title"), token)
        page_core = _first_nonempty((summary or {}).get("contextual_summary"), (summary or {}).get("source_excerpt"), title)
        topic_core = _first_nonempty(_slot_excerpt(topic_unit or {}), (summary or {}).get("document_thesis"), page_core)
        result_core = _first_nonempty(_slot_excerpt(result_unit or {}), page_core)
        limitations_core = _first_nonempty(_slot_excerpt(limit_unit or {}))
        version_core = _first_nonempty(
            (summary or {}).get("document_date"),
            (summary or {}).get("event_date"),
            (summary or {}).get("observed_at"),
            "version/temporal markers unavailable",
        )
        when_not_to_use = _first_nonempty(
            limitations_core,
            "" if _VERSION_RE.search(" ".join([title, page_core, result_core])) else "Use cautiously for latest/update queries without explicit version markers.",
        )

        slot_coverage = {
            "pageCore": _coverage_status(page_core),
            "topicCore": _coverage_status(topic_core),
            "resultCore": _coverage_status(result_core),
            "limitationsCore": _coverage_status(limitations_core),
            "versionCore": _coverage_status(version_core),
            "whenNotToUse": _coverage_status(when_not_to_use),
        }
        quality_flag = "ok" if slot_coverage["pageCore"] == "complete" and slot_coverage["topicCore"] == "complete" else "needs_review"
        card = {
            "card_id": f"web-card-v2:{document_id}",
            "document_id": document_id,
            "canonical_url": token,
            "title": title,
            "page_core": page_core,
            "topic_core": topic_core,
            "result_core": result_core,
            "limitations_core": limitations_core,
            "version_core": version_core,
            "when_not_to_use": when_not_to_use,
            "search_text": _clean_text(
                " ".join(
                    [
                        token,
                        title,
                        page_core,
                        topic_core,
                        result_core,
                        limitations_core,
                        version_core,
                        when_not_to_use,
                        " ".join(_clean_text(raw) for raw in list((summary or {}).get("concepts") or [])),
                    ]
                )
            ),
            "quality_flag": quality_flag,
            "document_date": _clean_text((summary or {}).get("document_date")),
            "event_date": _clean_text((summary or {}).get("event_date")),
            "observed_at": _clean_text((summary or {}).get("observed_at")),
            "source_url": _clean_text(metadata.get("canonical_url") or metadata.get("source_url") or token),
            "version": "web-card-v2",
            "slot_coverage": slot_coverage,
            "diagnostics": {
                "sourceDocumentSummary": bool(summary),
                "documentUnitCount": len(units),
                "acceptedClaimCount": len(claims),
                "sourceUrl": token,
            },
        }

        claim_refs = [
            {
                "claim_id": _clean_text(claim.get("claim_id")),
                "role": _claim_role(claim),
                "confidence": float(claim.get("confidence") or 0.0),
                "rank": rank,
                "reason": "accepted_web_claim_ranked_by_confidence_and_evidence",
            }
            for rank, claim in enumerate(claims[:8], 1)
            if _clean_text(claim.get("claim_id"))
        ]
        claim_unit_index: dict[str, dict[str, Any]] = {}
        for unit in units:
            for claim_id in list(unit.get("claims") or []):
                token_id = _clean_text(claim_id)
                if token_id and token_id not in claim_unit_index:
                    claim_unit_index[token_id] = unit

        anchors: list[dict[str, Any]] = []
        slot_units = [
            ("page_core", summary or {}, "page"),
            ("topic_core", topic_unit or summary or {}, "topic"),
            ("result_core", result_unit or summary or {}, "result"),
            ("limitations_core", limit_unit or summary or {}, "limitation"),
            ("version_core", result_unit or summary or {}, "version"),
            ("when_not_to_use", limit_unit or summary or {}, "when_not_to_use"),
        ]
        for field_name, unit, role in slot_units:
            excerpt = _first_nonempty(_slot_excerpt(unit), card.get(field_name))
            if not excerpt:
                continue
            section_path = _clean_text(unit.get("section_path") or unit.get("title") or field_name)
            anchors.append(
                {
                    "anchor_id": _stable_anchor_id(card["card_id"], role, field_name, section_path, excerpt),
                    "card_id": card["card_id"],
                    "claim_id": "",
                    "document_id": document_id,
                    "unit_id": _clean_text(unit.get("unit_id")),
                    "title": title,
                    "source_type": "web",
                    "source_url": token,
                    "section_path": section_path,
                    "span_locator": _clean_text(unit.get("unit_id") or field_name),
                    "snippet_hash": _snippet_hash(section_path, excerpt),
                    "evidence_role": role,
                    "excerpt": excerpt,
                    "score": 0.9 if unit else 0.6,
                    "document_date": _clean_text(unit.get("document_date") or (summary or {}).get("document_date")),
                    "event_date": _clean_text(unit.get("event_date") or (summary or {}).get("event_date")),
                    "observed_at": _clean_text(unit.get("observed_at") or (summary or {}).get("observed_at")),
                    "updated_at_marker": _clean_text((note or {}).get("updated_at") or (summary or {}).get("updated_at")),
                }
            )

        for ref in claim_refs:
            claim_id = _clean_text(ref.get("claim_id"))
            claim = next((row for row in claims if _clean_text(row.get("claim_id")) == claim_id), {})
            unit = claim_unit_index.get(claim_id) or result_unit or summary or {}
            excerpt = _first_nonempty(_slot_excerpt(unit), claim.get("claim_text"), card.get("result_core"))
            section_path = _clean_text(unit.get("section_path") or unit.get("title") or ref.get("role") or "claim")
            anchors.append(
                {
                    "anchor_id": _stable_anchor_id(card["card_id"], "claim", claim_id, section_path, excerpt),
                    "card_id": card["card_id"],
                    "claim_id": claim_id,
                    "document_id": document_id,
                    "unit_id": _clean_text(unit.get("unit_id")),
                    "title": title,
                    "source_type": "web",
                    "source_url": token,
                    "section_path": section_path,
                    "span_locator": _clean_text(unit.get("unit_id") or claim_id),
                    "snippet_hash": _snippet_hash(claim_id, excerpt),
                    "evidence_role": _clean_text(ref.get("role") or "supporting"),
                    "excerpt": excerpt,
                    "score": round(0.75 + (0.2 * float(ref.get("confidence") or 0.0)), 4),
                    "document_date": _clean_text(unit.get("document_date") or (summary or {}).get("document_date")),
                    "event_date": _clean_text(unit.get("event_date") or (summary or {}).get("event_date")),
                    "observed_at": _clean_text(unit.get("observed_at") or (summary or {}).get("observed_at")),
                    "updated_at_marker": _clean_text((note or {}).get("updated_at") or (summary or {}).get("updated_at")),
                }
            )

        entity_refs = self._entity_refs(summary, units)
        return {
            "card": card,
            "claim_refs": claim_refs,
            "anchors": anchors,
            "entity_refs": entity_refs,
        }

    def build_and_store(self, *, canonical_url: str) -> dict[str, Any]:
        payload = self.build(canonical_url=canonical_url)
        card = self.sqlite_db.upsert_web_card_v2(card=payload["card"])
        card_id = _clean_text(card.get("card_id") or payload["card"].get("card_id"))
        self.sqlite_db.replace_web_card_claim_refs_v2(card_id=card_id, refs=payload["claim_refs"])
        self.sqlite_db.replace_web_evidence_anchors_v2(card_id=card_id, anchors=payload["anchors"])
        self.sqlite_db.replace_web_card_entity_refs_v2(card_id=card_id, refs=payload["entity_refs"])
        from knowledge_hub.domain.ai_papers.claim_cards import ClaimCardBuilder

        claim_cards = ClaimCardBuilder(self.sqlite_db).build_and_store_for_source_card(
            source_kind="web",
            source_card=card,
        )
        return {
            **card,
            "claim_refs": self.sqlite_db.list_web_card_claim_refs_v2(card_id=card_id),
            "anchors": self.sqlite_db.list_web_evidence_anchors_v2(card_id=card_id),
            "entity_refs": self.sqlite_db.list_web_card_entity_refs_v2(card_id=card_id),
            "claim_cards": claim_cards,
        }


__all__ = ["WebCardV2Builder"]
