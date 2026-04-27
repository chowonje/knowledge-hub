from __future__ import annotations

from knowledge_hub.ai.rag_answer_evidence import answer_evidence_item
from knowledge_hub.core.models import SearchResult


def _item(result: SearchResult) -> dict:
    return answer_evidence_item(
        result,
        parent_ctx_by_result={},
        result_id_fn=lambda item: item.document_id,
        normalize_source_type_fn=lambda value: value,
        safe_float_fn=lambda value, default: float(value or default),
    )


def test_answer_evidence_item_synthesizes_hash_and_span_for_retrieved_document():
    result = SearchResult(
        document="Metadata-only paper abstract evidence.",
        metadata={"title": "Paper", "source_type": "paper", "paper_id": "2601.00001"},
        distance=0.1,
        score=0.9,
        document_id="0-0",
    )

    item = _item(result)

    assert item["source_content_hash"]
    assert item["source_content_hash_kind"] == "retrieved_document"
    assert item["char_start"] == 0
    assert item["char_end"] == len(result.document)
    assert item["source_trace"]["contentHashAvailable"] is True
    assert item["source_trace"]["contentHashKind"] == "retrieved_document"


def test_answer_evidence_item_preserves_explicit_source_hash_and_span():
    result = SearchResult(
        document="Chunk text",
        metadata={
            "title": "Paper",
            "source_type": "paper",
            "paper_id": "2601.00001",
            "source_content_hash": "hash-explicit",
            "char_start": 10,
            "char_end": 20,
        },
        distance=0.1,
        score=0.9,
        document_id="doc",
    )

    item = _item(result)

    assert item["source_content_hash"] == "hash-explicit"
    assert item["source_content_hash_kind"] == "source"
    assert item["char_start"] == 10
    assert item["char_end"] == 20
