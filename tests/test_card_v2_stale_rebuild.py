"""Stale derivative cards must trigger rebuild on the scoped ensure path, not be served verbatim."""

from types import SimpleNamespace

from knowledge_hub.application.card_v2_registry import (
    _PaperCardV2SourceHandler,
    _VaultCardV2SourceHandler,
    _WebCardV2SourceHandler,
)


def _paper_db(card):
    return SimpleNamespace(
        get_paper_card_v2=lambda paper_id: dict(card),
        list_evidence_anchors_v2=lambda card_id: [{"anchor_id": "a1"}],
        list_paper_card_claim_refs_v2=lambda card_id: [{"claim_id": "cl1"}],
        get_paper_memory_card=lambda paper_id: {"updated_at": "2026-05-01T00:00:00+00:00"},
        get_document_memory_summary=lambda document_id: {},
    )


def _fresh_paper_card(*, stale):
    return {
        "card_id": "c1",
        "paper_id": "p1",
        "stale": stale,
        "stale_reason": "source_content_hash_changed" if stale else "",
        "updated_at": "2026-06-01T00:00:00+00:00",
    }


def test_paper_needs_rebuild_when_card_is_stale():
    card = _fresh_paper_card(stale=True)
    handler = _PaperCardV2SourceHandler(_paper_db(card), builder=SimpleNamespace())
    assert handler.needs_rebuild("p1", card) is True


def test_paper_no_rebuild_when_card_is_fresh():
    card = _fresh_paper_card(stale=False)
    handler = _PaperCardV2SourceHandler(_paper_db(card), builder=SimpleNamespace())
    assert handler.needs_rebuild("p1", card) is False


def test_paper_ensure_card_rebuilds_stale_card():
    card = _fresh_paper_card(stale=True)
    rebuilt = {"card_id": "c1", "paper_id": "p1", "stale": False}
    calls = []

    def build_and_store(*, paper_id):
        calls.append(paper_id)
        return dict(rebuilt)

    handler = _PaperCardV2SourceHandler(
        _paper_db(card), builder=SimpleNamespace(build_and_store=build_and_store)
    )
    result = handler.ensure_card("p1")
    assert calls == ["p1"], "stale card must trigger a rebuild"
    assert result == rebuilt


def test_web_needs_rebuild_when_card_is_stale():
    card = {
        "card_id": "w1",
        "document_id": "note-1",
        "stale": True,
        "updated_at": "2026-06-01T00:00:00+00:00",
    }
    db = SimpleNamespace(
        get_web_card_v2_by_url=lambda url: dict(card),
        list_web_evidence_anchors_v2=lambda card_id: [{"anchor_id": "a1"}],
        list_web_card_claim_refs_v2=lambda card_id: [{"claim_id": "cl1"}],
        get_note=lambda note_id: {"updated_at": "2026-05-01T00:00:00+00:00"},
        get_document_memory_summary=lambda document_id: {},
    )
    handler = _WebCardV2SourceHandler(db, builder=SimpleNamespace())
    assert handler.needs_rebuild("https://example.com/a", card) is True


def test_vault_needs_rebuild_when_card_is_stale():
    card = {
        "card_id": "v1",
        "stale": True,
        "updated_at": "2026-06-01T00:00:00+00:00",
    }
    db = SimpleNamespace(
        get_vault_card_v2=lambda source_id: dict(card),
        list_vault_evidence_anchors_v2=lambda card_id: [{"anchor_id": "a1"}],
        list_vault_card_claim_refs_v2=lambda card_id: [{"claim_id": "cl1"}],
        list_claims_by_note=lambda note_id, limit=1: [],
        get_note=lambda note_id: {"updated_at": "2026-05-01T00:00:00+00:00"},
        get_document_memory_summary=lambda document_id: {},
    )
    handler = _VaultCardV2SourceHandler(db, builder=SimpleNamespace())
    assert handler.needs_rebuild("note-1", card) is True
