"""Canonical paper-memory payload helpers.

The paper-memory card is the canonical wire shape. Optional hydrated refs are
added explicitly for surfaces like `show` and `search`, while `build` stays
card-only so CLI and MCP contracts remain aligned.
"""

from __future__ import annotations

from typing import Any

from knowledge_hub.papers.memory_models import PaperMemoryCard


def card_payload(value: dict[str, Any] | PaperMemoryCard | None) -> dict[str, Any]:
    if isinstance(value, PaperMemoryCard):
        return value.to_payload()
    if isinstance(value, dict):
        card = PaperMemoryCard.from_row(value)
        if card is not None:
            return card.to_payload()
    return {}


def hydrated_card_payload(
    value: dict[str, Any] | PaperMemoryCard | None,
    *,
    sqlite_db: Any,
    include_refs: bool = True,
) -> dict[str, Any]:
    if isinstance(value, PaperMemoryCard):
        card = value
    elif isinstance(value, dict):
        card = PaperMemoryCard.from_row(value)
    else:
        card = None
    if card is None:
        return {}

    payload = card.to_payload()
    if not include_refs:
        return payload

    paper = sqlite_db.get_paper(card.paper_id) or {}
    payload["paper"] = {
        "paperId": str(paper.get("arxiv_id") or card.paper_id),
        "title": str(paper.get("title") or card.title),
        "year": paper.get("year"),
        "field": str(paper.get("field") or ""),
        "publishedAt": str(card.published_at or ""),
    }
    note = sqlite_db.get_note(card.source_note_id) if card.source_note_id else None
    payload["sourceNote"] = {
        "id": str((note or {}).get("id") or card.source_note_id),
        "title": str((note or {}).get("title") or ""),
        "sourceType": str((note or {}).get("source_type") or ""),
    }
    claims: list[dict[str, Any]] = []
    for claim_id in card.claim_refs:
        claim = sqlite_db.get_claim(claim_id)
        if not claim:
            continue
        claims.append(
            {
                "claimId": str(claim.get("claim_id") or claim_id),
                "claimText": str(claim.get("claim_text") or ""),
                "confidence": claim.get("confidence"),
            }
        )
    payload["claims"] = claims
    return payload
