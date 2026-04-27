from __future__ import annotations

from knowledge_hub.ai.evidence_assembly import _is_temporal_query


def test_soft_recency_evaluation_query_is_not_hard_temporal():
    assert _is_temporal_query("최근 RAG evaluation article은 citation accuracy와 faithfulness를 어떻게 구분하나?") is False


def test_explicit_latest_query_remains_temporal():
    assert _is_temporal_query("latest vector database retrieval best practice는 무엇인가?") is True
    assert _is_temporal_query("2026년 이후 RAG benchmark 변화는 무엇인가?") is True
