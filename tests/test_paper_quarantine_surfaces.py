"""Tranche B: 13 wrong-document paper ids quarantined from ask/search/compare.

Surface blocking only — no data deletion. The deny-list comes from the
2026-06-11 parsed-store content-identity audit. Reproduced live hazard: a
lexical search for "AlphaFold" returned paper_1207.0580_0 (registered as the
dropout paper) because its on-disk PDF is the AlphaFold article.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from knowledge_hub.core.models import SearchResult
from knowledge_hub.papers.quarantine import (
    QUARANTINE_REASON_CODE,
    QUARANTINED_PAPER_IDS,
    QuarantinedPaperTargetError,
    filter_quarantined_cards,
    filter_quarantined_search_results,
    is_quarantined_paper,
    normalize_paper_id_token,
    resolve_quarantined_paper_targets,
)


def test_quarantine_set_matches_audit_thirteen():
    assert len(QUARANTINED_PAPER_IDS) == 13
    assert "1207.0580" in QUARANTINED_PAPER_IDS
    assert "Gemini_Embedding_Generalizable_b5cf39ed" in QUARANTINED_PAPER_IDS


def test_normalize_paper_id_token_forms():
    assert normalize_paper_id_token("1207.0580") == "1207.0580"
    assert normalize_paper_id_token("paper:1207.0580") == "1207.0580"
    assert normalize_paper_id_token("paper_1207.0580_0") == "1207.0580"
    assert (
        normalize_paper_id_token("paper_Gemini_Embedding_Generalizable_b5cf39ed_3")
        == "Gemini_Embedding_Generalizable_b5cf39ed"
    )
    assert normalize_paper_id_token("note:123") == "note:123"


def test_is_quarantined_paper_all_token_forms():
    assert is_quarantined_paper("1207.0580")
    assert is_quarantined_paper("paper:1207.0580")
    assert is_quarantined_paper("paper_1207.0580_0")
    assert not is_quarantined_paper("2601.12542")
    assert not is_quarantined_paper("")


def _paper_result(paper_id: str, *, title: str, document: str) -> SearchResult:
    return SearchResult(
        document=document,
        metadata={"paper_id": paper_id, "title": title, "source_type": "paper"},
        distance=0.1,
        score=0.9,
        document_id=f"paper:{paper_id}",
    )


def test_alphafold_query_results_exclude_quarantined_dropout_id():
    hits = [
        _paper_result(
            "1207.0580",
            title="Improving neural networks by preventing co-adaptation of feature detectors",
            document="Highly accurate protein structure prediction with AlphaFold ...",
        ),
        _paper_result(
            "2601.12542",
            title="A clean paper that legitimately discusses AlphaFold",
            document="AlphaFold-style structure prediction ...",
        ),
        SearchResult(
            document="vault note about AlphaFold",
            metadata={"source_type": "vault"},
            distance=0.4,
            score=0.5,
            document_id="note:123",
        ),
    ]
    kept, dropped = filter_quarantined_search_results(hits)
    kept_ids = [item.metadata.get("paper_id") for item in kept]
    assert "1207.0580" not in kept_ids
    assert "2601.12542" in kept_ids
    assert len(kept) == 2
    assert dropped == ["1207.0580"]


def test_filter_matches_document_id_when_metadata_missing():
    hit = SearchResult(
        document="...",
        metadata={},
        distance=0.1,
        score=0.9,
        document_id="paper:1406.1078",
    )
    kept, dropped = filter_quarantined_search_results([hit])
    assert kept == []
    assert dropped == ["1406.1078"]


def test_filter_quarantined_cards():
    cards = [
        {"paper_id": "1207.0580", "card_id": "a"},
        {"paper_id": "2601.12542", "card_id": "b"},
    ]
    kept, dropped = filter_quarantined_cards(cards)
    assert [card["card_id"] for card in kept] == ["b"]
    assert dropped == ["1207.0580"]


def test_resolve_targets_partial_lookup_keeps_clean_ids():
    assert resolve_quarantined_paper_targets(["1207.0580", "2601.12542"]) == ["2601.12542"]


def test_resolve_targets_all_quarantined_raises():
    with pytest.raises(QuarantinedPaperTargetError) as exc_info:
        resolve_quarantined_paper_targets(["1207.0580", "1406.1078"])
    assert exc_info.value.paper_ids == ["1207.0580", "1406.1078"]
    assert QUARANTINE_REASON_CODE in str(exc_info.value)


def test_resolve_targets_compare_fails_closed_on_any_quarantined_member():
    with pytest.raises(QuarantinedPaperTargetError):
        resolve_quarantined_paper_targets(["1810.04805", "1207.0580"], compare=True)


class _SelectorServiceStub:
    def __init__(self):
        self.requested: list[str] = []

    def _ensure_paper_card(self, paper_id: str):
        self.requested.append(paper_id)
        return {"paper_id": paper_id, "card_id": f"card:{paper_id}"}


def _paper_selector(service):
    from knowledge_hub.ai.ask_v2_card_selectors import _PaperCardSelector

    return _PaperCardSelector(service, fallback_error=RuntimeError)


def _request(metadata_filter=None, query="what does this paper say?"):
    from knowledge_hub.ai.ask_v2_card_selectors import CardSelectionRequest

    route = SimpleNamespace(source_kind="paper", intent="paper_lookup", mode="v2", entity_ids=[])
    return CardSelectionRequest(query=query, route=route, limit=3, metadata_filter=metadata_filter)


def test_scoped_quarantined_paper_question_fails_closed():
    service = _SelectorServiceStub()
    selector = _paper_selector(service)
    with pytest.raises(QuarantinedPaperTargetError) as exc_info:
        selector.select(_request(metadata_filter={"paper_id": "1207.0580"}))
    assert exc_info.value.paper_ids == ["1207.0580"]
    assert service.requested == []


def test_scoped_clean_paper_question_still_selects_card():
    service = _SelectorServiceStub()
    selector = _paper_selector(service)
    cards = selector.select(_request(metadata_filter={"paper_id": "2601.12542"}))
    assert [card["paper_id"] for card in cards] == ["2601.12542"]


def test_scoped_quarantine_matches_arxiv_id_in_query():
    service = _SelectorServiceStub()
    selector = _paper_selector(service)
    with pytest.raises(QuarantinedPaperTargetError):
        selector.select(_request(metadata_filter=None, query="1406.1078 논문의 핵심 기여를 설명해줘"))


def test_ask_v2_execute_returns_quarantined_fail_closed(monkeypatch):
    from knowledge_hub.ai.ask_v2 import AskV2Service

    service = object.__new__(AskV2Service)
    route = SimpleNamespace(source_kind="paper", intent="paper_lookup", mode="v2", entity_ids=[])
    monkeypatch.setattr(AskV2Service, "_resolve_frame_authority", lambda self, **kwargs: {})
    monkeypatch.setattr(AskV2Service, "_route", lambda self, **kwargs: route)

    def _raise_quarantined(self, **kwargs):
        raise QuarantinedPaperTargetError(["1207.0580"])

    captured = {}

    def _fake_scoped_no_result(self, **kwargs):
        captured.update(kwargs)
        return ("pipeline-result", "evidence-packet")

    monkeypatch.setattr(AskV2Service, "_select_cards", _raise_quarantined)
    monkeypatch.setattr(AskV2Service, "_scoped_no_result_execution", _fake_scoped_no_result)

    result = service.execute(
        query="1207.0580 dropout 논문이 말하는 핵심은?",
        top_k=5,
        source_type="paper",
        retrieval_mode="hybrid",
        alpha=0.5,
        allow_external=False,
    )
    assert result == ("pipeline-result", "evidence-packet")
    assert captured["reason"] == f"{QUARANTINE_REASON_CODE}:1207.0580"
    assert captured["route"] is route
