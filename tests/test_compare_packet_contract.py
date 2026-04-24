from __future__ import annotations

from knowledge_hub.ai.compare_packet import COMPARE_PACKET_SCHEMA, build_compare_packet_contract
from knowledge_hub.core.schema_validator import validate_payload


def test_compare_packet_filters_non_evidence_supporting_spans():
    packet = build_compare_packet_contract(
        query="compare RAG and memory retrieval",
        dimensions=[
            {
                "label": "grounding",
                "left_claim": "RAG uses retrieved passages.",
                "right_claim": "Memory retrieval uses dialogue state.",
                "status": "supported",
                "supporting_spans": [
                    {
                        "spanRef": "span:1",
                        "source_id": "paper:rag#0",
                        "source_type": "paper",
                        "content_hash": "hash-rag",
                        "quote": "RAG uses retrieved passages.",
                    },
                    {
                        "spanRef": "span:2",
                        "source_id": "learning_edge:rag:memory",
                        "source_type": "learning_edge",
                        "quote": "signal only",
                    },
                ],
            }
        ],
        retrieval_signals=[{"sourceId": "learning_edge:rag:memory", "reason": "query expansion"}],
    )

    assert packet["schema"] == COMPARE_PACKET_SCHEMA
    assert packet["coverage"]["dimensionCount"] == 1
    assert packet["coverage"]["supportingSpanCount"] == 1
    assert packet["coverage"]["excludedNonEvidenceSpanCount"] == 1
    assert packet["dimensions"][0]["supportingSpans"][0]["sourceId"] == "paper:rag#0"
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_marks_all_unknown_as_not_answerable():
    packet = build_compare_packet_contract(
        query="compare unknown systems",
        dimensions=[
            {
                "label": "method",
                "left_claim": "",
                "right_claim": "",
                "status": "unknown",
                "supporting_spans": [],
            }
        ],
    )

    assert packet["coverage"]["answerable"] is False
    assert packet["coverage"]["unknownDimensionCount"] == 1
