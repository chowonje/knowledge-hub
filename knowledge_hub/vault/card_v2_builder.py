"""Build additive v2 vault cards from existing note/document/claim artifacts."""

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
from knowledge_hub.document_memory import DocumentMemoryBuilder


_SECTION_CONCEPT_RE = re.compile(r"\b(overview|summary|definition|concept|background|정의|개념|배경|요약)\b", re.IGNORECASE)
_SECTION_DECISION_RE = re.compile(r"\b(decision|architecture|design|policy|choice|tradeoff|설계|결정|정책|구조)\b", re.IGNORECASE)
_SECTION_ACTION_RE = re.compile(r"\b(action|todo|next step|implementation|procedure|setup|howto|작업|다음|구현|설정|절차)\b", re.IGNORECASE)
_SECTION_LIMIT_RE = re.compile(r"\b(limit|limitation|warning|risk|caveat|한계|주의|경고|리스크)\b", re.IGNORECASE)

def _claim_role(claim: dict[str, Any]) -> str:
    text = _clean_text(claim.get("claim_text"))
    predicate = _clean_text(claim.get("predicate")).casefold()
    haystack = f"{predicate} {text}"
    if _SECTION_LIMIT_RE.search(haystack):
        return "limitation"
    if _SECTION_ACTION_RE.search(haystack):
        return "action"
    if _SECTION_DECISION_RE.search(haystack):
        return "decision"
    if _SECTION_CONCEPT_RE.search(haystack):
        return "concept"
    return "supporting"


def _claim_sort_key(claim: dict[str, Any]) -> tuple[float, float, str]:
    confidence = float(claim.get("confidence") or 0.0)
    evidence_ptrs = claim.get("evidence_ptrs") if isinstance(claim.get("evidence_ptrs"), list) else []
    evidence_weight = min(1.0, 0.12 * len([ptr for ptr in evidence_ptrs if isinstance(ptr, dict)]))
    return confidence + evidence_weight, confidence, _clean_text(claim.get("claim_id"))


class VaultCardV2Builder:
    def __init__(self, sqlite_db):
        self.sqlite_db = sqlite_db

    def _ensure_document_memory(self, note_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        summary = self.sqlite_db.get_document_memory_summary(note_id)
        units = list(self.sqlite_db.list_document_memory_units(note_id, limit=200) or [])
        if summary and units:
            return summary, units
        try:
            DocumentMemoryBuilder(self.sqlite_db).build_and_store_note(note_id=note_id)
        except Exception:
            pass
        summary = self.sqlite_db.get_document_memory_summary(note_id)
        units = list(self.sqlite_db.list_document_memory_units(note_id, limit=200) or [])
        return summary, units

    def _claims_for_note(self, note_id: str) -> list[dict[str, Any]]:
        rows = list(self.sqlite_db.list_claims_by_note(note_id, limit=80))
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

    def build(self, *, note_id: str) -> dict[str, Any]:
        note = self.sqlite_db.get_note(note_id)
        if not note:
            raise ValueError(f"vault note not found: {note_id}")
        summary, units = self._ensure_document_memory(note_id)
        if not units:
            raise ValueError(f"vault document memory not found: {note_id}")
        claims = self._claims_for_note(note_id)
        metadata = _parse_note_metadata(note)

        concept_unit = _best_unit(units, _SECTION_CONCEPT_RE)
        decision_unit = _best_unit(units, _SECTION_DECISION_RE)
        action_unit = _best_unit(units, _SECTION_ACTION_RE)
        limit_unit = _best_unit(units, _SECTION_LIMIT_RE)

        note_core = _first_nonempty((summary or {}).get("contextual_summary"), (summary or {}).get("source_excerpt"), note.get("title"))
        concept_core = _first_nonempty(_slot_excerpt(concept_unit or {}), (summary or {}).get("document_thesis"), note_core)
        decision_core = _first_nonempty(_slot_excerpt(decision_unit or {}))
        action_core = _first_nonempty(_slot_excerpt(action_unit or {}))
        when_not_to_use = _first_nonempty(_slot_excerpt(limit_unit or {}))

        card_id = f"vault-card-v2:{note_id}"
        card = {
            "card_id": card_id,
            "note_id": note_id,
            "title": _clean_text(note.get("title") or note_id),
            "note_core": note_core,
            "concept_core": concept_core,
            "decision_core": decision_core,
            "action_core": action_core,
            "when_not_to_use": when_not_to_use,
            "search_text": _clean_text(
                " ".join(
                    [
                        _clean_text(note.get("title")),
                        _clean_text(note.get("file_path")),
                        note_core,
                        concept_core,
                        decision_core,
                        action_core,
                        when_not_to_use,
                        " ".join(str(item.get("claim_text") or "") for item in claims[:5]),
                        " ".join(str(tag or "") for tag in list(metadata.get("tags") or [])[:8]),
                    ]
                )
            ),
            "quality_flag": "ok" if summary else "needs_review",
            "file_path": _clean_text(note.get("file_path")),
            "version": "vault-card-v2",
            "slot_coverage": {
                "noteCore": _coverage_status(note_core),
                "conceptCore": _coverage_status(concept_core),
                "decisionCore": _coverage_status(decision_core),
                "actionCore": _coverage_status(action_core),
                "whenNotToUse": _coverage_status(when_not_to_use),
            },
            "diagnostics": {
                "sourceDocumentSummary": bool(summary),
                "documentUnitCount": len(units),
                "acceptedClaimCount": len(claims),
                "tagCount": len(list(metadata.get("tags") or [])),
            },
        }

        claim_refs = [
            {
                "claim_id": _clean_text(claim.get("claim_id")),
                "role": _claim_role(claim),
                "confidence": float(claim.get("confidence") or 0.0),
                "rank": rank,
                "reason": "accepted_vault_claim_ranked_by_confidence_and_evidence",
            }
            for rank, claim in enumerate(claims[:8], 1)
            if _clean_text(claim.get("claim_id"))
        ]

        claim_unit_index: dict[str, dict[str, Any]] = {}
        for unit in units:
            for claim_id in list(unit.get("claims") or []):
                token = _clean_text(claim_id)
                if token and token not in claim_unit_index:
                    claim_unit_index[token] = unit

        anchors: list[dict[str, Any]] = []
        slot_units = [
            ("note_core", summary or {}, "note"),
            ("concept_core", concept_unit or summary or {}, "concept"),
            ("decision_core", decision_unit or summary or {}, "decision"),
            ("action_core", action_unit or summary or {}, "action"),
            ("when_not_to_use", limit_unit or summary or {}, "when_not_to_use"),
        ]
        for field_name, unit, role in slot_units:
            excerpt = _first_nonempty(_slot_excerpt(unit), card.get(field_name))
            if not excerpt:
                continue
            section_path = _clean_text(unit.get("section_path") or unit.get("title") or field_name)
            anchors.append(
                {
                    "anchor_id": _stable_anchor_id(card_id, role, field_name, section_path, excerpt),
                    "card_id": card_id,
                    "claim_id": "",
                    "note_id": note_id,
                    "unit_id": _clean_text(unit.get("unit_id")),
                    "title": card["title"],
                    "source_type": "vault",
                    "section_path": section_path,
                    "span_locator": _clean_text(unit.get("unit_id") or field_name),
                    "snippet_hash": _snippet_hash(section_path, excerpt),
                    "evidence_role": role,
                    "excerpt": excerpt,
                    "score": 0.9 if unit else 0.6,
                    "file_path": card["file_path"],
                }
            )

        for ref in claim_refs:
            claim_id = _clean_text(ref.get("claim_id"))
            claim = next((row for row in claims if _clean_text(row.get("claim_id")) == claim_id), {})
            unit = claim_unit_index.get(claim_id) or summary or {}
            excerpt = _first_nonempty(_slot_excerpt(unit), claim.get("claim_text"), card.get("decision_core"))
            section_path = _clean_text(unit.get("section_path") or unit.get("title") or ref.get("role") or "claim")
            anchors.append(
                {
                    "anchor_id": _stable_anchor_id(card_id, "claim", claim_id, section_path, excerpt),
                    "card_id": card_id,
                    "claim_id": claim_id,
                    "note_id": note_id,
                    "unit_id": _clean_text(unit.get("unit_id")),
                    "title": card["title"],
                    "source_type": "vault",
                    "section_path": section_path,
                    "span_locator": _clean_text(unit.get("unit_id") or claim_id),
                    "snippet_hash": _snippet_hash(claim_id, excerpt),
                    "evidence_role": _clean_text(ref.get("role") or "supporting"),
                    "excerpt": excerpt,
                    "score": round(0.75 + (0.2 * float(ref.get("confidence") or 0.0)), 4),
                    "file_path": card["file_path"],
                }
            )

        return {"card": card, "claim_refs": claim_refs, "anchors": anchors}

    def build_and_store(self, *, note_id: str) -> dict[str, Any]:
        payload = self.build(note_id=note_id)
        card = self.sqlite_db.upsert_vault_card_v2(card=payload["card"])
        card_id = _clean_text(card.get("card_id") or payload["card"].get("card_id"))
        self.sqlite_db.replace_vault_card_claim_refs_v2(card_id=card_id, refs=payload["claim_refs"])
        self.sqlite_db.replace_vault_evidence_anchors_v2(card_id=card_id, anchors=payload["anchors"])
        from knowledge_hub.domain.ai_papers.claim_cards import ClaimCardBuilder

        claim_cards = ClaimCardBuilder(self.sqlite_db).build_and_store_for_source_card(
            source_kind="vault",
            source_card=card,
        )
        return {
            **card,
            "claim_refs": self.sqlite_db.list_vault_card_claim_refs_v2(card_id=card_id),
            "anchors": self.sqlite_db.list_vault_evidence_anchors_v2(card_id=card_id),
            "claim_cards": claim_cards,
        }


__all__ = ["VaultCardV2Builder"]
