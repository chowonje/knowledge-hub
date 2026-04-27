from __future__ import annotations

from types import SimpleNamespace

from knowledge_hub.ai.ask_v2_verification import AskV2Verifier
from knowledge_hub.ai.evidence_assembly import _has_explicit_temporal_marker, _has_web_temporal_textual_marker, _is_temporal_query


def test_soft_recency_evaluation_query_is_not_hard_temporal():
    assert _is_temporal_query("최근 RAG evaluation article은 citation accuracy와 faithfulness를 어떻게 구분하나?") is False


def test_explicit_latest_query_remains_temporal():
    assert _is_temporal_query("latest vector database retrieval best practice는 무엇인가?") is True
    assert _is_temporal_query("2026년 이후 RAG benchmark 변화는 무엇인가?") is True


def test_arxiv_identifier_counts_as_document_temporal_marker():
    item = {
        "title": "[2603.01152] DeepResearch-9K",
        "source_url": "https://arxiv.org/abs/2603.01152",
    }

    assert _has_explicit_temporal_marker(item) is True
    assert _has_web_temporal_textual_marker(item) is True


def test_updated_at_alone_is_not_document_temporal_grounding():
    assert _has_explicit_temporal_marker({"title": "Vector guide", "updated_at": "2026-04-07"}) is False


def test_ask_v2_temporal_verifier_accepts_published_at_for_paper():
    verifier = AskV2Verifier(None)
    summary = verifier.verification_summary(
        query="최근 paper-memory retrieval 접근",
        route=SimpleNamespace(intent="temporal", source_kind="paper"),
        cards=[],
        anchors=[{"published_at": "2025-01-01T00:00:00+00:00", "excerpt": "memory retrieval"}],
        evidence_packet=SimpleNamespace(filtered_results=[object()]),
        claim_consensus={},
    )

    assert "temporal" not in summary["unsupportedFields"]


def test_ask_v2_temporal_verifier_accepts_arxiv_web_url_but_not_updated_at_only():
    verifier = AskV2Verifier(None)
    accepted = verifier.verification_summary(
        query="최신 web 문서 기준",
        route=SimpleNamespace(intent="temporal", source_kind="web"),
        cards=[],
        anchors=[{"source_url": "https://arxiv.org/abs/2603.01152", "excerpt": "benchmark"}],
        evidence_packet=SimpleNamespace(filtered_results=[object()]),
        claim_consensus={},
    )
    rejected = verifier.verification_summary(
        query="latest vector database retrieval best practice",
        route=SimpleNamespace(intent="temporal", source_kind="web"),
        cards=[],
        anchors=[{"updated_at": "2026-04-07", "excerpt": "vector guide"}],
        evidence_packet=SimpleNamespace(filtered_results=[object()]),
        claim_consensus={},
    )

    assert "temporal_version_grounding" not in accepted["unsupportedFields"]
    assert "temporal_version_grounding" in rejected["unsupportedFields"]
