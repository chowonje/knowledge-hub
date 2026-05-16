from __future__ import annotations

from types import SimpleNamespace

from knowledge_hub.ai.answer_payload_builder import AnswerPayloadBuilder
from knowledge_hub.ai.compare_packet import COMPARE_PACKET_SCHEMA, build_compare_packet_contract
from knowledge_hub.ai.compare_packet import build_compare_packet_from_sources
from knowledge_hub.ai.compare_packet import build_compare_packet_from_runtime
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


def test_compare_packet_requires_strict_span_backed_sources_for_answerable():
    fallback_only = build_compare_packet_contract(
        query="compare source A and source B",
        dimensions=[
            {
                "label": "accuracy",
                "status": "conflict",
                "supporting_spans": [
                    {
                        "spanRef": "span-a",
                        "sourceId": "paper:a#0",
                        "sourceType": "paper",
                        "contentHash": "sha256:a",
                        "spanLocator": "chars:10-40",
                        "quote": "A accuracy",
                    },
                    {
                        "spanRef": "source-b",
                        "sourceId": "paper:b#0",
                        "sourceType": "paper",
                        "quote": "B accuracy",
                        "fallbackSpan": True,
                    },
                ],
            }
        ],
    )

    assert fallback_only["coverage"]["strictSpanBackedCount"] == 1
    assert fallback_only["coverage"]["fallbackSpanCount"] == 1
    assert fallback_only["coverage"]["answerable"] is False
    assert fallback_only["dimensions"][0]["supportingSpans"][0]["strictSpanBacked"] is True
    assert fallback_only["dimensions"][0]["supportingSpans"][1]["fallbackSpan"] is True

    strict_packet = build_compare_packet_contract(
        query="compare source A and source B",
        dimensions=[
            {
                "label": "accuracy",
                "status": "conflict",
                "supporting_spans": [
                    {
                        "spanRef": "span-a",
                        "sourceId": "paper:a#0",
                        "sourceType": "paper",
                        "contentHash": "sha256:a",
                        "spanLocator": "chars:10-40",
                        "quote": "A accuracy",
                    },
                    {
                        "spanRef": "span-b",
                        "sourceId": "paper:b#0",
                        "sourceType": "paper",
                        "contentHash": "sha256:b",
                        "spanLocator": "chars:50-90",
                        "quote": "B accuracy",
                    },
                ],
            }
        ],
    )

    assert strict_packet["coverage"]["strictSpanBackedCount"] == 2
    assert strict_packet["coverage"]["fallbackSpanCount"] == 0
    assert strict_packet["coverage"]["strictSupportedDimensionCount"] == 1
    assert strict_packet["coverage"]["answerable"] is True
    assert validate_payload(strict_packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_runtime_maps_ask_v2_paper_compare_groups():
    packet = build_compare_packet_from_runtime(
        query="compare two papers on MemoryBench",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["2603.13017", "2603.13018"]},
        claim_cards=[
            {
                "claimCardId": "claim-card-a",
                "sourceKind": "paper",
                "sourceId": "2603.13017",
                "summaryText": "Paper A reports 11x compression.",
                "resultDirection": "better",
                "evidenceAnchorIds": ["anchor-a"],
                "anchorExcerpts": ["A reports 11x compression on MemoryBench."],
            },
            {
                "claimCardId": "claim-card-b",
                "sourceKind": "paper",
                "sourceId": "2603.13018",
                "summaryText": "Paper B reports worse compression.",
                "resultDirection": "worse",
                "evidenceAnchorIds": ["anchor-b"],
                "anchorExcerpts": ["B reports worse compression on MemoryBench."],
            },
        ],
        claim_alignment={
            "groups": [
                {
                    "groupKey": "MemoryBench:compression",
                    "canonicalFrame": {"dataset": "MemoryBench", "metric": "compression ratio"},
                    "claimCardIds": ["claim-card-b", "claim-card-a"],
                    "conflictingClaimCount": 1,
                }
            ]
        },
        evidence_policy={"policyKey": "paper_compare_policy"},
    )

    assert packet is not None
    assert packet["schema"] == COMPARE_PACKET_SCHEMA
    assert packet["policy"]["policyKey"] == "paper_compare_policy"
    assert packet["coverage"]["dimensionCount"] == 1
    assert packet["coverage"]["conflictDimensionCount"] == 1
    dimension = packet["dimensions"][0]
    assert dimension["label"] == "compression ratio"
    assert dimension["comparisonStatus"] == "conflict"
    assert dimension["leftClaim"].startswith("Paper A")
    assert dimension["rightClaim"].startswith("Paper B")
    assert [span["spanRef"] for span in dimension["supportingSpans"]] == ["anchor-a", "anchor-b"]
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_runtime_uses_enriched_evidence_anchors_with_offsets_as_strict_spans():
    packet = build_compare_packet_from_runtime(
        query="compare two papers on MemoryBench",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["2603.13017", "2603.13018"]},
        claim_cards=[
            {
                "claimCardId": "claim-card-a",
                "sourceKind": "paper",
                "sourceId": "2603.13017",
                "summaryText": "Paper A reports 11x compression.",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-a",
                        "sourceId": "2603.13017",
                        "sourceType": "paper",
                        "documentId": "paper:2603.13017",
                        "chunkId": "paper:2603.13017:result",
                        "spanLocator": "chars:10-90",
                        "sourceContentHash": "sha256:a",
                        "snippetHash": "snippet-a",
                        "citationLabel": "S1",
                        "quote": "A reports 11x compression on MemoryBench.",
                    }
                ],
            },
            {
                "claimCardId": "claim-card-b",
                "sourceKind": "paper",
                "sourceId": "2603.13018",
                "summaryText": "Paper B reports worse compression.",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-b",
                        "sourceId": "2603.13018",
                        "sourceType": "paper",
                        "documentId": "paper:2603.13018",
                        "chunkId": "paper:2603.13018:result",
                        "spanLocator": "chars:20-120",
                        "sourceContentHash": "sha256:b",
                        "snippetHash": "snippet-b",
                        "citationLabel": "S2",
                        "quote": "B reports worse compression on MemoryBench.",
                    }
                ],
            },
        ],
        claim_alignment={
            "groups": [
                {
                    "groupKey": "MemoryBench:compression",
                    "canonicalFrame": {"dataset": "MemoryBench", "metric": "compression ratio"},
                    "claimCardIds": ["claim-card-a", "claim-card-b"],
                    "conflictingClaimCount": 1,
                }
            ]
        },
    )

    assert packet is not None
    spans = packet["dimensions"][0]["supportingSpans"]
    assert {span["strictSpanBacked"] for span in spans} == {True}
    assert {span["fallbackSpan"] for span in spans} == {False}
    assert {span["sourceContentHash"] for span in spans} == {"sha256:a", "sha256:b"}
    assert {span["spanLocatorAvailable"] for span in spans} == {True}
    assert {span["spanOffsetAvailable"] for span in spans} == {True}
    assert packet["coverage"]["answerable"] is True
    assert packet["coverage"]["strictSpanBackedCount"] == 2
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_runtime_keeps_locator_only_evidence_anchors_non_strict():
    packet = build_compare_packet_from_runtime(
        query="compare two papers on MemoryBench",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["2603.13017", "2603.13018"]},
        claim_cards=[
            {
                "claimCardId": "claim-card-a",
                "sourceKind": "paper",
                "sourceId": "2603.13017",
                "summaryText": "Paper A reports 11x compression.",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-a",
                        "sourceId": "2603.13017",
                        "sourceType": "paper",
                        "documentId": "paper:2603.13017",
                        "chunkId": "paper:2603.13017:result",
                        "spanLocator": "paper:2603.13017:result",
                        "sourceContentHash": "sha256:a",
                        "snippetHash": "snippet-a",
                        "citationLabel": "S1",
                        "quote": "A reports 11x compression on MemoryBench.",
                    }
                ],
            },
            {
                "claimCardId": "claim-card-b",
                "sourceKind": "paper",
                "sourceId": "2603.13018",
                "summaryText": "Paper B reports worse compression.",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-b",
                        "sourceId": "2603.13018",
                        "sourceType": "paper",
                        "documentId": "paper:2603.13018",
                        "chunkId": "paper:2603.13018:result",
                        "spanLocator": "paper:2603.13018:result",
                        "sourceContentHash": "sha256:b",
                        "snippetHash": "snippet-b",
                        "citationLabel": "S2",
                        "quote": "B reports worse compression on MemoryBench.",
                    }
                ],
            },
        ],
        claim_alignment={
            "groups": [
                {
                    "groupKey": "MemoryBench:compression",
                    "canonicalFrame": {"dataset": "MemoryBench", "metric": "compression ratio"},
                    "claimCardIds": ["claim-card-a", "claim-card-b"],
                    "conflictingClaimCount": 1,
                }
            ]
        },
    )

    assert packet is not None
    spans = packet["dimensions"][0]["supportingSpans"]
    assert {span["strictSpanBacked"] for span in spans} == {False}
    assert {span["fallbackSpan"] for span in spans} == {False}
    assert {span["spanLocatorAvailable"] for span in spans} == {True}
    assert {span["spanOffsetAvailable"] for span in spans} == {False}
    assert packet["coverage"]["answerable"] is False
    assert packet["coverage"]["strictSpanBackedCount"] == 0
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_keeps_memory_unit_locators_non_strict_even_with_hashes():
    packet = build_compare_packet_contract(
        query="compare DQN and PPO",
        dimensions=[
            {
                "label": "method",
                "status": "supported",
                "supporting_spans": [
                    {
                        "spanRef": "anchor-dqn",
                        "sourceId": "1312.5602",
                        "sourceType": "paper",
                        "contentHash": "sha256:dqn",
                        "spanLocator": "memory-unit:paper:1312.5602:summary",
                        "quote": "DQN uses a convolutional neural network trained with Q-learning.",
                    },
                    {
                        "spanRef": "anchor-ppo",
                        "sourceId": "1707.06347",
                        "sourceType": "paper",
                        "contentHash": "sha256:ppo",
                        "spanLocator": "memory-unit:paper:1707.06347:summary",
                        "quote": "PPO optimizes a clipped surrogate objective.",
                    },
                ],
            }
        ],
    )

    spans = packet["dimensions"][0]["supportingSpans"]
    assert {span["strictSpanBacked"] for span in spans} == {False}
    assert {span["spanLocatorAvailable"] for span in spans} == {True}
    assert {span["spanOffsetAvailable"] for span in spans} == {False}
    assert packet["coverage"]["answerable"] is False
    assert packet["coverage"]["strictSpanBackedCount"] == 0
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_runtime_synthesizes_claim_dimensions_from_strict_anchors_when_groups_missing():
    packet = build_compare_packet_from_runtime(
        query="GraphRAG와 LightRAG의 global question 처리를 비교해줘",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["2404.16130", "2410.05779"]},
        claim_cards=[
            {
                "claimCardId": "claim-card-graph-method",
                "sourceKind": "paper",
                "sourceId": "2404.16130",
                "summaryText": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization | GraphRAG builds global summaries.",
                "claimType": "method",
                "resultDirection": "unknown",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-graph-method",
                        "sourceId": "2404.16130",
                        "sourceType": "paper",
                        "documentId": "paper:2404.16130",
                        "chunkId": "paper:2404.16130:method",
                        "spanLocator": "chars:10-90",
                        "sourceContentHash": "sha256:graph",
                        "citationLabel": "S1",
                        "quote": "GraphRAG builds global summaries.",
                        "evidenceRole": "method",
                    }
                ],
            },
            {
                "claimCardId": "claim-card-light-method",
                "sourceKind": "paper",
                "sourceId": "2410.05779",
                "summaryText": "LightRAG: Simple and Fast Retrieval-Augmented Generation | LightRAG keeps a graph index.",
                "claimType": "method",
                "resultDirection": "unknown",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-light-method",
                        "sourceId": "2410.05779",
                        "sourceType": "paper",
                        "documentId": "paper:2410.05779",
                        "chunkId": "paper:2410.05779:method",
                        "spanLocator": "chars:20-120",
                        "sourceContentHash": "sha256:light",
                        "citationLabel": "S2",
                        "quote": "LightRAG keeps a graph index.",
                        "evidenceRole": "method",
                    }
                ],
            },
        ],
        claim_alignment={"groups": []},
    )

    assert packet is not None
    assert packet["coverage"]["answerable"] is True
    assert packet["coverage"]["strictSupportedDimensionCount"] == 1
    assert packet["coverage"]["strictSpanBackedCount"] == 2
    dimension = packet["dimensions"][0]
    assert dimension["dimensionId"] == "claim-card-role:method"
    assert dimension["comparisonStatus"] == "supported"
    assert {span["strictSpanBacked"] for span in dimension["supportingSpans"]} == {True}
    assert {span["fallbackSpan"] for span in dimension["supportingSpans"]} == {False}
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_runtime_accepts_concise_claim_anchor_quotes_with_good_summaries():
    packet = build_compare_packet_from_runtime(
        query="GraphRAG와 LightRAG의 global question 처리를 비교해줘",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["2404.16130", "2410.05779"]},
        claim_cards=[
            {
                "claimCardId": "claim-card-graph-method",
                "sourceKind": "paper",
                "sourceId": "2404.16130",
                "summaryText": "GraphRAG uses graph communities to organize evidence for global question answering.",
                "claimType": "method",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-graph-method",
                        "sourceId": "2404.16130",
                        "sourceType": "paper",
                        "spanLocator": "chars:10-90",
                        "sourceContentHash": "sha256:graph",
                        "quote": "global QA",
                        "evidenceRole": "method",
                    }
                ],
            },
            {
                "claimCardId": "claim-card-light-method",
                "sourceKind": "paper",
                "sourceId": "2410.05779",
                "summaryText": "LightRAG keeps a lightweight graph index for retrieval-augmented generation.",
                "claimType": "method",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-light-method",
                        "sourceId": "2410.05779",
                        "sourceType": "paper",
                        "spanLocator": "chars:20-120",
                        "sourceContentHash": "sha256:light",
                        "quote": "graph index",
                        "evidenceRole": "method",
                    }
                ],
            },
        ],
        claim_alignment={"groups": []},
    )

    assert packet is not None
    assert packet["coverage"]["answerable"] is True
    assert packet["dimensions"][0]["dimensionId"] == "claim-card-role:method"
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_runtime_rejects_low_signal_claim_anchor_quotes():
    packet = build_compare_packet_from_runtime(
        query="GraphRAG와 LightRAG의 global question 처리를 비교해줘",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["2404.16130", "2410.05779"]},
        claim_cards=[
            {
                "claimCardId": "claim-card-graph-method",
                "sourceKind": "paper",
                "sourceId": "2404.16130",
                "summaryText": "GraphRAG uses graph communities to organize evidence for global question answering.",
                "claimType": "method",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-graph-method",
                        "sourceId": "2404.16130",
                        "sourceType": "paper",
                        "spanLocator": "chars:10-90",
                        "sourceContentHash": "sha256:graph",
                        "quote": "GraphRAG uses graph communities.",
                        "evidenceRole": "method",
                    }
                ],
            },
            {
                "claimCardId": "claim-card-light-method",
                "sourceKind": "paper",
                "sourceId": "2410.05779",
                "summaryText": "LightRAG keeps a lightweight graph index for retrieval-augmented generation.",
                "claimType": "method",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-light-method",
                        "sourceId": "2410.05779",
                        "sourceType": "paper",
                        "spanLocator": "chars:20-120",
                        "sourceContentHash": "sha256:light",
                        "quote": "\\newcommand{\\parents}{Pa}",
                        "evidenceRole": "method",
                    }
                ],
            },
        ],
        claim_alignment={"groups": []},
    )

    assert packet is None


def test_compare_packet_from_runtime_synthesizes_slot_dimensions_from_strict_slot_refs_when_groups_missing():
    packet = build_compare_packet_from_runtime(
        query="GraphRAG와 LightRAG의 method와 result를 비교해줘",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["2404.16130", "2410.05779"]},
        claim_cards=[],
        claim_alignment={"groups": []},
        paper_knowledge_slots=[
            {
                "paperId": "2404.16130",
                "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
                "slots": [
                    {
                        "slotType": "method",
                        "text": "GraphRAG builds global summaries over graph communities.",
                        "strictEvidence": True,
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-graph-method",
                                "sourceId": "2404.16130",
                                "sourceType": "paper",
                                "documentId": "paper:2404.16130",
                                "chunkId": "paper:2404.16130:method",
                                "spanLocator": "chars:10-90",
                                "sourceContentHash": "sha256:graph",
                                "contentHash": "sha256:graph-snippet",
                                "quote": "GraphRAG builds global summaries.",
                                "strictSpanBacked": True,
                            }
                        ],
                    },
                    {
                        "slotType": "result",
                        "text": "GraphRAG improves global question answering.",
                        "strictEvidence": True,
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-graph-result",
                                "sourceId": "2404.16130",
                                "sourceType": "paper",
                                "documentId": "paper:2404.16130",
                                "chunkId": "paper:2404.16130:result",
                                "spanLocator": "chars:100-180",
                                "sourceContentHash": "sha256:graph",
                                "contentHash": "sha256:graph-result",
                                "quote": "GraphRAG improves global question answering.",
                                "strictSpanBacked": True,
                            }
                        ],
                    },
                ],
            },
            {
                "paperId": "2410.05779",
                "title": "LightRAG: Simple and Fast Retrieval-Augmented Generation",
                "slots": [
                    {
                        "slotType": "method",
                        "text": "LightRAG keeps a lightweight graph index.",
                        "strictEvidence": True,
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-light-method",
                                "sourceId": "2410.05779",
                                "sourceType": "paper",
                                "documentId": "paper:2410.05779",
                                "chunkId": "paper:2410.05779:method",
                                "spanLocator": "chars:20-120",
                                "sourceContentHash": "sha256:light",
                                "contentHash": "sha256:light-snippet",
                                "quote": "LightRAG keeps a graph index.",
                                "strictSpanBacked": True,
                            }
                        ],
                    },
                    {
                        "slotType": "result",
                        "text": "LightRAG reduces retrieval overhead.",
                        "strictEvidence": True,
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-light-result",
                                "sourceId": "2410.05779",
                                "sourceType": "paper",
                                "documentId": "paper:2410.05779",
                                "chunkId": "paper:2410.05779:result",
                                "spanLocator": "chars:160-240",
                                "sourceContentHash": "sha256:light",
                                "contentHash": "sha256:light-result",
                                "quote": "LightRAG reduces retrieval overhead.",
                                "strictSpanBacked": True,
                            }
                        ],
                    },
                ],
            },
        ],
    )

    assert packet is not None
    assert packet["coverage"]["answerable"] is True
    assert packet["coverage"]["strictSupportedDimensionCount"] == 2
    assert packet["coverage"]["strictSpanBackedCount"] == 4
    assert [dimension["dimensionId"] for dimension in packet["dimensions"]] == ["paper-slot:method", "paper-slot:result"]
    assert {span["strictSpanBacked"] for dimension in packet["dimensions"] for span in dimension["supportingSpans"]} == {True}
    assert {span["fallbackSpan"] for dimension in packet["dimensions"] for span in dimension["supportingSpans"]} == {False}
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_runtime_respects_explicit_non_strict_slot_refs():
    packet = build_compare_packet_from_runtime(
        query="AlexNet과 ViT의 method를 비교해줘",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["alexnet-2012", "2010.11929"]},
        claim_cards=[],
        claim_alignment={"groups": []},
        paper_knowledge_slots=[
            {
                "paperId": "alexnet-2012",
                "title": "ImageNet Classification with Deep Convolutional Neural Networks",
                "slots": [
                    {
                        "slotType": "method",
                        "text": "AlexNet은 GPU 기반 대규모 CNN 학습으로 ImageNet 시대를 열었다.",
                        "strictEvidence": False,
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-alexnet-method",
                                "sourceId": "alexnet-2012",
                                "sourceType": "paper",
                                "spanLocator": "chars:0-298",
                                "sourceContentHash": "sha256:alexnet",
                                "contentHash": "snippet:alexnet",
                                "quote": "We trained a large, deep convolutional neural network.",
                                "strictSpanBacked": False,
                            }
                        ],
                    }
                ],
            },
            {
                "paperId": "2010.11929",
                "title": "An Image is Worth 16x16 Words",
                "slots": [
                    {
                        "slotType": "method",
                        "text": "ViT applies a Transformer directly to image patches.",
                        "strictEvidence": True,
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-vit-method",
                                "sourceId": "2010.11929",
                                "sourceType": "paper",
                                "spanLocator": "chars:20-120",
                                "sourceContentHash": "sha256:vit",
                                "contentHash": "snippet:vit",
                                "quote": "We apply a standard Transformer directly to image patches.",
                                "strictSpanBacked": True,
                            }
                        ],
                    }
                ],
            },
        ],
    )

    assert packet is None


def test_compare_packet_from_runtime_uses_slot_dimensions_when_claim_groups_are_incomplete():
    packet = build_compare_packet_from_runtime(
        query="GraphRAG와 LightRAG의 method를 비교해줘",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["2404.16130", "2410.05779"]},
        claim_cards=[
            {
                "claimCardId": "claim-card-graph-method",
                "sourceKind": "paper",
                "sourceId": "2404.16130",
                "summaryText": "GraphRAG builds global summaries.",
                "claimType": "method",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-graph-method",
                        "sourceId": "2404.16130",
                        "sourceType": "paper",
                        "spanLocator": "chars:10-90",
                        "sourceContentHash": "sha256:graph",
                        "quote": "GraphRAG builds global summaries.",
                        "evidenceRole": "method",
                    }
                ],
            }
        ],
        claim_alignment={
            "groups": [
                {
                    "groupKey": "compare-group:method",
                    "claimCardIds": ["claim-card-graph-method"],
                    "canonicalFrame": {"task": "method"},
                }
            ]
        },
        paper_knowledge_slots=[
            {
                "paperId": "2404.16130",
                "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
                "slots": [
                    {
                        "slotType": "method",
                        "text": "GraphRAG builds global summaries over graph communities.",
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-graph-method",
                                "sourceId": "2404.16130",
                                "sourceType": "paper",
                                "spanLocator": "chars:10-90",
                                "sourceContentHash": "sha256:graph",
                                "quote": "GraphRAG builds global summaries.",
                            }
                        ],
                    }
                ],
            },
            {
                "paperId": "2410.05779",
                "title": "LightRAG: Simple and Fast Retrieval-Augmented Generation",
                "slots": [
                    {
                        "slotType": "method",
                        "text": "LightRAG keeps a lightweight graph index.",
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-light-method",
                                "sourceId": "2410.05779",
                                "sourceType": "paper",
                                "spanLocator": "chars:20-120",
                                "sourceContentHash": "sha256:light",
                                "quote": "LightRAG keeps a graph index.",
                            }
                        ],
                    }
                ],
            },
        ],
    )

    assert packet is not None
    assert packet["coverage"]["answerable"] is True
    assert packet["coverage"]["strictSupportedDimensionCount"] == 1
    assert packet["coverage"]["strictSpanBackedCount"] == 2
    assert [dimension["dimensionId"] for dimension in packet["dimensions"]] == ["paper-slot:method"]
    assert packet["dimensions"][0]["comparisonStatus"] == "supported"
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_runtime_adds_source_titles_to_slot_dimension_notes():
    packet = build_compare_packet_from_runtime(
        query="BERT와 GPT 계열의 차이를 논문 기준으로 비교해줘",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["1810.04805", "2005.14165"]},
        claim_cards=[],
        claim_alignment={"groups": []},
        paper_knowledge_slots=[
            {
                "paperId": "1810.04805",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "slots": [
                    {
                        "slotType": "method",
                        "text": "BERT uses bidirectional Transformer encoders.",
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-bert-method",
                                "sourceId": "1810.04805",
                                "sourceType": "paper",
                                "spanLocator": "chars:10-90",
                                "sourceContentHash": "sha256:bert",
                                "quote": "BERT uses bidirectional Transformer encoders.",
                            }
                        ],
                    }
                ],
            },
            {
                "paperId": "2005.14165",
                "title": "Language Models are Few-Shot Learners",
                "slots": [
                    {
                        "slotType": "method",
                        "text": "GPT-3 is an autoregressive few-shot language model.",
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-gpt-method",
                                "sourceId": "2005.14165",
                                "sourceType": "paper",
                                "spanLocator": "chars:20-120",
                                "sourceContentHash": "sha256:gpt",
                                "quote": "GPT-3 is an autoregressive few-shot language model.",
                            }
                        ],
                    }
                ],
            },
        ],
    )

    assert packet is not None
    dimension = packet["dimensions"][0]
    assert dimension["label"] == "method"
    assert "Language Models are Few-Shot Learners" in dimension["notes"]
    assert packet["coverage"]["answerable"] is True
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_runtime_does_not_synthesize_slot_dimensions_from_locator_only_refs():
    packet = build_compare_packet_from_runtime(
        query="GraphRAG와 LightRAG의 method를 비교해줘",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["2404.16130", "2410.05779"]},
        claim_cards=[],
        claim_alignment={"groups": []},
        paper_knowledge_slots=[
            {
                "paperId": "2404.16130",
                "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
                "slots": [
                    {
                        "slotType": "method",
                        "text": "GraphRAG builds global summaries.",
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-graph-method",
                                "sourceId": "2404.16130",
                                "sourceType": "paper",
                                "spanLocator": "paper:2404.16130:method",
                                "sourceContentHash": "sha256:graph",
                                "quote": "GraphRAG builds global summaries.",
                            }
                        ],
                    }
                ],
            },
            {
                "paperId": "2410.05779",
                "title": "LightRAG: Simple and Fast Retrieval-Augmented Generation",
                "slots": [
                    {
                        "slotType": "method",
                        "text": "LightRAG keeps a graph index.",
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-light-method",
                                "sourceId": "2410.05779",
                                "sourceType": "paper",
                                "spanLocator": "chars:20-120",
                                "sourceContentHash": "sha256:light",
                                "quote": "LightRAG keeps a graph index.",
                            }
                        ],
                    }
                ],
            },
        ],
    )

    assert packet is None


def test_compare_packet_from_runtime_does_not_synthesize_dimensions_without_explicit_source_mentions():
    packet = build_compare_packet_from_runtime(
        query="AlphaFoo Retrieval 논문과 BetaBar Memory 논문을 비교해줘",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["2312.10997", "2512.13564"]},
        claim_cards=[
            {
                "claimCardId": "claim-card-rag-method",
                "sourceKind": "paper",
                "sourceId": "2312.10997",
                "summaryText": "Retrieval-Augmented Generation for Large Language Models: A Survey | RAG method.",
                "claimType": "method",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-rag-method",
                        "sourceId": "2312.10997",
                        "sourceType": "paper",
                        "spanLocator": "chars:10-90",
                        "sourceContentHash": "sha256:rag",
                        "quote": "RAG method.",
                        "evidenceRole": "method",
                    }
                ],
            },
            {
                "claimCardId": "claim-card-memory-method",
                "sourceKind": "paper",
                "sourceId": "2512.13564",
                "summaryText": "Memory in the Age of AI Agents | Memory method.",
                "claimType": "method",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-memory-method",
                        "sourceId": "2512.13564",
                        "sourceType": "paper",
                        "spanLocator": "chars:20-120",
                        "sourceContentHash": "sha256:memory",
                        "quote": "Memory method.",
                        "evidenceRole": "method",
                    }
                ],
            },
        ],
        claim_alignment={"groups": []},
    )

    assert packet is None


def test_compare_packet_from_runtime_does_not_synthesize_dimensions_for_risky_no_answer_queries():
    packet = build_compare_packet_from_runtime(
        query="현재 코퍼스만으로 최신 RAG benchmark들의 정확한 수치 순위를 단정해서 비교할 수 있나?",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["2005.11401", "2312.10997"]},
        claim_cards=[
            {
                "claimCardId": "claim-card-rag-metric",
                "sourceKind": "paper",
                "sourceId": "2005.11401",
                "summaryText": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | RAG benchmark table.",
                "claimType": "metric",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-rag-metric",
                        "sourceId": "2005.11401",
                        "sourceType": "paper",
                        "spanLocator": "chars:10-90",
                        "sourceContentHash": "sha256:rag",
                        "quote": "RAG benchmark table.",
                        "evidenceRole": "metric",
                    }
                ],
            },
            {
                "claimCardId": "claim-card-survey-metric",
                "sourceKind": "paper",
                "sourceId": "2312.10997",
                "summaryText": "Retrieval-Augmented Generation for Large Language Models: A Survey | Survey benchmark table.",
                "claimType": "metric",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-survey-metric",
                        "sourceId": "2312.10997",
                        "sourceType": "paper",
                        "spanLocator": "chars:20-120",
                        "sourceContentHash": "sha256:survey",
                        "quote": "Survey benchmark table.",
                        "evidenceRole": "metric",
                    }
                ],
            },
        ],
        claim_alignment={"groups": []},
    )

    assert packet is None


def test_compare_packet_from_runtime_does_not_synthesize_slot_dimensions_for_risky_no_answer_queries():
    packet = build_compare_packet_from_runtime(
        query="현재 코퍼스만으로 최신 GraphRAG benchmark들의 정확한 수치 순위를 단정해서 비교할 수 있나?",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["2404.16130", "2410.05779"]},
        claim_cards=[],
        claim_alignment={"groups": []},
        paper_knowledge_slots=[
            {
                "paperId": "2404.16130",
                "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
                "slots": [
                    {
                        "slotType": "metric",
                        "text": "GraphRAG reports benchmark gains.",
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-graph-metric",
                                "sourceId": "2404.16130",
                                "sourceType": "paper",
                                "spanLocator": "chars:10-90",
                                "sourceContentHash": "sha256:graph",
                                "quote": "GraphRAG benchmark gains.",
                            }
                        ],
                    }
                ],
            },
            {
                "paperId": "2410.05779",
                "title": "LightRAG: Simple and Fast Retrieval-Augmented Generation",
                "slots": [
                    {
                        "slotType": "metric",
                        "text": "LightRAG reports benchmark gains.",
                        "evidenceRefs": [
                            {
                                "anchorId": "anchor-light-metric",
                                "sourceId": "2410.05779",
                                "sourceType": "paper",
                                "spanLocator": "chars:20-120",
                                "sourceContentHash": "sha256:light",
                                "quote": "LightRAG benchmark gains.",
                            }
                        ],
                    }
                ],
            },
        ],
    )

    assert packet is None


def test_compare_packet_from_runtime_does_not_synthesize_dimensions_from_locator_only_anchors():
    packet = build_compare_packet_from_runtime(
        query="GraphRAG와 LightRAG의 global question 처리를 비교해줘",
        source_type="paper",
        family="paper_compare",
        runtime_execution={"used": "ask_v2"},
        query_frame={"resolved_source_ids": ["2404.16130", "2410.05779"]},
        claim_cards=[
            {
                "claimCardId": "claim-card-graph-method",
                "sourceKind": "paper",
                "sourceId": "2404.16130",
                "summaryText": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization | GraphRAG builds global summaries.",
                "claimType": "method",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-graph-method",
                        "sourceId": "2404.16130",
                        "sourceType": "paper",
                        "spanLocator": "paper:2404.16130:method",
                        "sourceContentHash": "sha256:graph",
                        "quote": "GraphRAG builds global summaries.",
                        "evidenceRole": "method",
                    }
                ],
            },
            {
                "claimCardId": "claim-card-light-method",
                "sourceKind": "paper",
                "sourceId": "2410.05779",
                "summaryText": "LightRAG: Simple and Fast Retrieval-Augmented Generation | LightRAG keeps a graph index.",
                "claimType": "method",
                "evidenceAnchors": [
                    {
                        "anchorId": "anchor-light-method",
                        "sourceId": "2410.05779",
                        "sourceType": "paper",
                        "spanLocator": "chars:20-120",
                        "sourceContentHash": "sha256:light",
                        "quote": "LightRAG keeps a graph index.",
                        "evidenceRole": "method",
                    }
                ],
            },
        ],
        claim_alignment={"groups": []},
    )

    assert packet is None


def test_compare_packet_from_runtime_omits_legacy_or_non_compare_payloads():
    assert (
        build_compare_packet_from_runtime(
            query="compare",
            source_type="paper",
            family="paper_compare",
            runtime_execution={"used": "legacy"},
            query_frame={},
            claim_cards=[{"claimCardId": "claim-card-a"}],
            claim_alignment={"groups": [{"claimCardIds": ["claim-card-a"]}]},
        )
        is None
    )


def test_compare_packet_from_sources_builds_insufficient_source_coverage_packet():
    packet = build_compare_packet_from_sources(
        query="compare RAG and Fusion in Decoder retrieval generation",
        sources=[
            {
                "source_id": "2005.11401",
                "source_type": "paper",
                "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "excerpt": "RAG combines retrieval and generation.",
                "span_locator": "0",
            },
            {
                "source_id": "2312.10997",
                "source_type": "paper",
                "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
                "excerpt": "The survey discusses retrieval and generation.",
                "span_locator": "1",
            },
            {
                "source_id": "learning_edge:rag",
                "source_type": "learning_edge",
                "title": "signal only",
            },
        ],
        citations=[{"target": "2005.11401", "label": "S1"}],
    )

    assert packet is not None
    assert packet["schema"] == COMPARE_PACKET_SCHEMA
    assert packet["coverage"]["answerable"] is False
    assert packet["coverage"]["supportingSpanCount"] == 2
    assert packet["coverage"]["strictSpanBackedCount"] == 0
    assert packet["coverage"]["fallbackSpanCount"] == 2
    assert packet["coverage"]["excludedNonEvidenceSpanCount"] == 0
    assert packet["dimensions"][0]["comparisonStatus"] == "insufficient"
    assert "Retrieval" in packet["dimensions"][0]["label"]
    assert {span["sourceId"] for span in packet["dimensions"][0]["supportingSpans"]} == {
        "2005.11401",
        "2312.10997",
    }
    assert {span["fallbackSpan"] for span in packet["dimensions"][0]["supportingSpans"]} == {True}
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_sources_uses_strict_evidence_spans_before_fallback():
    packet = build_compare_packet_from_sources(
        query="GraphRAG and LightRAG global question handling",
        existing_packet=build_compare_packet_contract(
            query="GraphRAG and LightRAG global question handling",
            dimensions=[
                {
                    "dimensionId": "dim:1",
                    "label": "global retrieval",
                    "comparisonStatus": "conflict",
                    "supportingSpans": [],
                }
            ],
        ),
        strict_spans=[
            {
                "spanRef": "span-a",
                "sourceId": "2603.15798",
                "source_type": "paper",
                "sourceContentHash": "sha256:graph",
                "spanLocator": "chars:1-50",
                "text": "GraphRAG source span",
            },
            {
                "spanRef": "span-b",
                "sourceId": "2410.05779",
                "source_type": "paper",
                "sourceContentHash": "sha256:light",
                "spanLocator": "chars:80-120",
                "text": "LightRAG source span",
            },
        ],
        sources=[
            {
                "source_id": "2603.15798",
                "source_type": "paper",
                "title": "CUBE benchmark",
                "excerpt": "GraphRAG fallback source span",
            },
            {
                "source_id": "2410.05779",
                "source_type": "paper",
                "title": "LightRAG",
                "excerpt": "LightRAG fallback source span",
            },
        ],
    )

    assert packet is not None
    spans = packet["dimensions"][0]["supportingSpans"]
    assert [span["strictSpanBacked"] for span in spans] == [True, True]
    assert [span["fallbackSpan"] for span in spans] == [False, False]
    assert {span["contentHash"] for span in spans} == {"sha256:graph", "sha256:light"}
    assert packet["coverage"]["answerable"] is True
    assert packet["coverage"]["strictSpanBackedCount"] == 2
    assert packet["coverage"]["fallbackSpanCount"] == 0
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_sources_enriches_existing_packet_with_missing_cited_source():
    packet = build_compare_packet_from_sources(
        query="GraphRAG and LightRAG global question handling",
        existing_packet=build_compare_packet_contract(
            query="GraphRAG and LightRAG global question handling",
            dimensions=[
                {
                    "dimensionId": "dim:1",
                    "label": "||||||ICML 2026 preprint format||||||",
                    "comparisonStatus": "insufficient",
                    "supportingSpans": [
                        {
                            "spanRef": "anchor-a",
                            "sourceId": "2603.15798",
                            "sourceType": "paper",
                            "quote": "GraphRAG source span",
                        }
                    ],
                }
            ],
        ),
        sources=[
            {
                "source_id": "2603.15798",
                "source_type": "paper",
                "title": "CUBE benchmark",
                "excerpt": "GraphRAG source span",
            },
            {
                "source_id": "2410.05779",
                "source_type": "paper",
                "title": "LightRAG",
                "excerpt": "LightRAG source span",
            },
        ],
    )

    assert packet is not None
    dimension = packet["dimensions"][0]
    assert dimension["label"].startswith("GraphRAG and LightRAG global question handling")
    assert {span["sourceId"] for span in dimension["supportingSpans"]} == {"2603.15798", "2410.05779"}
    assert packet["coverage"]["supportingSpanCount"] == 2
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_sources_recovers_insufficient_dimension_with_strict_multisource_spans():
    packet = build_compare_packet_from_sources(
        query="GraphRAG and LightRAG global question handling",
        existing_packet=build_compare_packet_contract(
            query="GraphRAG and LightRAG global question handling",
            dimensions=[
                {
                    "dimensionId": "dim:1",
                    "label": "||||||ICML 2026 preprint format||||||",
                    "comparisonStatus": "insufficient",
                    "supportingSpans": [
                        {
                            "spanRef": "anchor-a",
                            "sourceId": "2603.15798",
                            "sourceType": "paper",
                            "contentHash": "sha256:graph",
                            "spanLocator": "chars:1-50",
                            "quote": "GraphRAG source span",
                        }
                    ],
                }
            ],
        ),
        strict_spans=[
            {
                "spanRef": "span-b",
                "sourceId": "2410.05779",
                "source_type": "paper",
                "sourceContentHash": "sha256:light",
                "spanLocator": "chars:80-120",
                "text": "LightRAG source span",
            }
        ],
        sources=[
            {
                "source_id": "2603.15798",
                "source_type": "paper",
                "title": "CUBE benchmark",
                "excerpt": "GraphRAG source span",
            },
            {
                "source_id": "2410.05779",
                "source_type": "paper",
                "title": "LightRAG",
                "excerpt": "LightRAG source span",
            },
        ],
    )

    assert packet is not None
    dimension = packet["dimensions"][0]
    assert dimension["comparisonStatus"] == "supported"
    assert packet["coverage"]["answerable"] is True
    assert packet["coverage"]["strictSupportedDimensionCount"] == 1
    assert packet["coverage"]["strictSpanBackedCount"] == 2
    assert packet["coverage"]["fallbackSpanCount"] == 0
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_sources_recovers_dimension_with_redundant_fallback_spans():
    packet = build_compare_packet_from_sources(
        query="GraphRAG and LightRAG global question handling",
        existing_packet=build_compare_packet_contract(
            query="GraphRAG and LightRAG global question handling",
            dimensions=[
                {
                    "dimensionId": "dim:1",
                    "label": "global question",
                    "comparisonStatus": "insufficient",
                    "supportingSpans": [
                        {
                            "spanRef": "fallback-graph",
                            "sourceId": "2603.15798",
                            "sourceType": "paper",
                            "quote": "GraphRAG fallback source span",
                            "fallbackSpan": True,
                        },
                        {
                            "spanRef": "fallback-light",
                            "sourceId": "2410.05779",
                            "sourceType": "paper",
                            "quote": "LightRAG fallback source span",
                            "fallbackSpan": True,
                        },
                    ],
                }
            ],
        ),
        strict_spans=[
            {
                "spanRef": "strict-graph",
                "sourceId": "2603.15798",
                "source_type": "paper",
                "sourceContentHash": "sha256:graph",
                "spanLocator": "chars:1-50",
                "text": "GraphRAG strict source span",
            },
            {
                "spanRef": "strict-light",
                "sourceId": "2410.05779",
                "source_type": "paper",
                "sourceContentHash": "sha256:light",
                "spanLocator": "chars:80-120",
                "text": "LightRAG strict source span",
            },
        ],
        sources=[
            {
                "source_id": "2603.15798",
                "source_type": "paper",
                "title": "CUBE benchmark",
                "excerpt": "GraphRAG fallback source span",
            },
            {
                "source_id": "2410.05779",
                "source_type": "paper",
                "title": "LightRAG",
                "excerpt": "LightRAG fallback source span",
            },
        ],
    )

    assert packet is not None
    dimension = packet["dimensions"][0]
    assert dimension["comparisonStatus"] == "supported"
    assert packet["coverage"]["answerable"] is True
    assert packet["coverage"]["strictSupportedDimensionCount"] == 1
    assert packet["coverage"]["strictSpanBackedCount"] == 2
    assert packet["coverage"]["fallbackSpanCount"] == 2
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_sources_keeps_source_fallback_packet_insufficient_even_with_strict_spans():
    packet = build_compare_packet_from_sources(
        query="GraphRAG and LightRAG global question handling",
        strict_spans=[
            {
                "spanRef": "strict-graph",
                "sourceId": "2603.15798",
                "source_type": "paper",
                "sourceContentHash": "sha256:graph",
                "spanLocator": "chars:1-50",
                "text": "GraphRAG strict source span",
            },
            {
                "spanRef": "strict-light",
                "sourceId": "2410.05779",
                "source_type": "paper",
                "sourceContentHash": "sha256:light",
                "spanLocator": "chars:80-120",
                "text": "LightRAG strict source span",
            },
        ],
        sources=[
            {
                "source_id": "2603.15798",
                "source_type": "paper",
                "title": "CUBE benchmark",
                "excerpt": "GraphRAG fallback source span",
            },
            {
                "source_id": "2410.05779",
                "source_type": "paper",
                "title": "LightRAG",
                "excerpt": "LightRAG fallback source span",
            },
        ],
    )

    assert packet is not None
    assert packet["dimensions"][0]["dimensionId"] == "retrieved-source-coverage"
    assert packet["dimensions"][0]["comparisonStatus"] == "insufficient"
    assert packet["coverage"]["answerable"] is False
    assert packet["coverage"]["strictSpanBackedCount"] == 2
    assert packet["coverage"]["fallbackSpanCount"] == 2
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_sources_does_not_recover_when_fallback_source_lacks_strict_coverage():
    packet = build_compare_packet_from_sources(
        query="RAG and FiD retrieval generation",
        existing_packet=build_compare_packet_contract(
            query="RAG and FiD retrieval generation",
            dimensions=[
                {
                    "dimensionId": "dim:1",
                    "label": "retrieval generation",
                    "comparisonStatus": "insufficient",
                    "supportingSpans": [
                        {
                            "spanRef": "fallback-rag",
                            "sourceId": "2005.11401",
                            "sourceType": "paper",
                            "quote": "RAG fallback source span",
                            "fallbackSpan": True,
                        },
                        {
                            "spanRef": "fallback-fid",
                            "sourceId": "2007.01282",
                            "sourceType": "paper",
                            "quote": "FiD fallback source span",
                            "fallbackSpan": True,
                        },
                    ],
                }
            ],
        ),
        strict_spans=[
            {
                "spanRef": "strict-rag",
                "sourceId": "2005.11401",
                "source_type": "paper",
                "sourceContentHash": "sha256:rag",
                "spanLocator": "chars:1-50",
                "text": "RAG strict source span",
            }
        ],
        sources=[],
    )

    assert packet is not None
    assert packet["dimensions"][0]["comparisonStatus"] == "insufficient"
    assert packet["coverage"]["answerable"] is False
    assert packet["coverage"]["strictSupportedDimensionCount"] == 0
    assert packet["coverage"]["strictSpanBackedCount"] == 1
    assert packet["coverage"]["fallbackSpanCount"] == 2
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_compare_packet_from_sources_does_not_recover_insufficient_dimension_with_fallback_spans():
    packet = build_compare_packet_from_sources(
        query="RAG and FiD retrieval generation",
        existing_packet=build_compare_packet_contract(
            query="RAG and FiD retrieval generation",
            dimensions=[
                {
                    "dimensionId": "dim:1",
                    "label": "retrieval generation",
                    "comparisonStatus": "insufficient",
                    "supportingSpans": [
                        {
                            "spanRef": "span-a",
                            "sourceId": "2005.11401",
                            "sourceType": "paper",
                            "contentHash": "sha256:rag",
                            "spanLocator": "chars:1-50",
                            "quote": "RAG source span",
                        }
                    ],
                }
            ],
        ),
        sources=[
            {
                "source_id": "2312.10997",
                "source_type": "paper",
                "title": "RAG survey",
                "excerpt": "Survey fallback source span",
            }
        ],
    )

    assert packet is not None
    assert packet["dimensions"][0]["comparisonStatus"] == "insufficient"
    assert packet["coverage"]["answerable"] is False
    assert packet["coverage"]["strictSpanBackedCount"] == 1
    assert packet["coverage"]["fallbackSpanCount"] == 1
    assert validate_payload(packet, COMPARE_PACKET_SCHEMA, strict=True).ok


def test_answer_payload_builder_attaches_compare_packet_for_ask_v2_paper_compare():
    builder = AnswerPayloadBuilder(SimpleNamespace(_resolve_query_entities=lambda _query: []))
    pipeline_result = SimpleNamespace(
        context_expansion={},
        plan=SimpleNamespace(
            to_dict=lambda: {
                "queryFrame": {
                    "source_type": "paper",
                    "family": "paper_compare",
                    "resolved_source_ids": ["2603.13017", "2603.13018"],
                },
                "paperFamily": "paper_compare",
            }
        ),
        related_clusters=[],
        active_profile="test",
        paper_memory_prefilter={},
        memory_route={},
        memory_prefilter={},
        candidate_sources=[],
        rerank_signals={},
        source_scope_enforced=False,
        mixed_fallback_used=False,
        v2_diagnostics={
            "runtimeExecution": {"used": "ask_v2"},
            "claimCards": [
                {
                    "claimCardId": "claim-card-a",
                    "sourceKind": "paper",
                    "sourceId": "2603.13017",
                    "summaryText": "Paper A reports 11x compression.",
                    "evidenceAnchorIds": ["anchor-a"],
                    "anchorExcerpts": ["A reports 11x compression on MemoryBench."],
                },
                {
                    "claimCardId": "claim-card-b",
                    "sourceKind": "paper",
                    "sourceId": "2603.13018",
                    "summaryText": "Paper B reports worse compression.",
                    "evidenceAnchorIds": ["anchor-b"],
                    "anchorExcerpts": ["B reports worse compression on MemoryBench."],
                },
            ],
            "claimAlignment": {
                "groups": [
                    {
                        "groupKey": "MemoryBench:compression",
                        "canonicalFrame": {"metric": "compression ratio"},
                        "claimCardIds": ["claim-card-a", "claim-card-b"],
                        "conflictingClaimCount": 1,
                    }
                ]
            },
        },
    )
    evidence_packet = SimpleNamespace(
        answer_signals={},
        paper_answer_scope={},
        evidence=[],
        evidence_packet={},
        evidence_policy={"policyKey": "paper_compare_policy"},
        citations=[],
        evidence_budget={},
        supporting_beliefs=[],
        contradicting_beliefs=[],
        belief_updates_suggested=[],
        claims=[],
    )

    payload = builder.base_payload(
        query="compare MemoryBench",
        retrieval_mode="hybrid",
        pipeline_result=pipeline_result,
        evidence_packet=evidence_packet,
        answer="comparison",
    )

    assert payload["comparePacketContract"]["schema"] == COMPARE_PACKET_SCHEMA
    assert payload["comparePacketContract"]["coverage"]["conflictDimensionCount"] == 1
    assert validate_payload(payload["comparePacketContract"], COMPARE_PACKET_SCHEMA, strict=True).ok


def test_answer_payload_builder_attaches_slot_backed_compare_packet_for_ask_v2_paper_compare():
    builder = AnswerPayloadBuilder(SimpleNamespace(_resolve_query_entities=lambda _query: []))
    pipeline_result = SimpleNamespace(
        context_expansion={},
        plan=SimpleNamespace(
            to_dict=lambda: {
                "queryFrame": {
                    "source_type": "paper",
                    "family": "paper_compare",
                    "resolved_source_ids": ["2603.13017", "2603.13018"],
                },
                "paperFamily": "paper_compare",
            }
        ),
        related_clusters=[],
        active_profile="test",
        paper_memory_prefilter={},
        memory_route={},
        memory_prefilter={},
        candidate_sources=[],
        rerank_signals={},
        source_scope_enforced=False,
        mixed_fallback_used=False,
        v2_diagnostics={
            "runtimeExecution": {"used": "ask_v2"},
            "claimCards": [],
            "claimAlignment": {"groups": []},
            "paperKnowledgeSlots": [
                {
                    "paperId": "2603.13017",
                    "title": "Paper A",
                    "slots": [
                        {
                            "slotType": "method",
                            "text": "Paper A uses a compression method.",
                            "evidenceRefs": [
                                {
                                    "anchorId": "slot-a-method",
                                    "sourceId": "2603.13017",
                                    "sourceType": "paper",
                                    "spanLocator": "chars:10-80",
                                    "sourceContentHash": "sha256:paper-a",
                                    "quote": "Paper A method.",
                                }
                            ],
                        }
                    ],
                },
                {
                    "paperId": "2603.13018",
                    "title": "Paper B",
                    "slots": [
                        {
                            "slotType": "method",
                            "text": "Paper B uses a different compression method.",
                            "evidenceRefs": [
                                {
                                    "anchorId": "slot-b-method",
                                    "sourceId": "2603.13018",
                                    "sourceType": "paper",
                                    "spanLocator": "chars:20-100",
                                    "sourceContentHash": "sha256:paper-b",
                                    "quote": "Paper B method.",
                                }
                            ],
                        }
                    ],
                },
            ],
        },
    )
    evidence_packet = SimpleNamespace(
        answer_signals={},
        paper_answer_scope={},
        evidence=[],
        evidence_packet={},
        evidence_policy={"policyKey": "paper_compare_policy"},
        citations=[],
        evidence_budget={},
        supporting_beliefs=[],
        contradicting_beliefs=[],
        belief_updates_suggested=[],
        claims=[],
    )

    payload = builder.base_payload(
        query="2603.13017와 2603.13018의 method를 비교해줘",
        retrieval_mode="hybrid",
        pipeline_result=pipeline_result,
        evidence_packet=evidence_packet,
        answer="comparison",
    )

    assert payload["comparePacketContract"]["schema"] == COMPARE_PACKET_SCHEMA
    assert payload["comparePacketContract"]["coverage"]["answerable"] is True
    assert payload["comparePacketContract"]["coverage"]["strictSupportedDimensionCount"] == 1
    assert payload["comparePacketContract"]["dimensions"][0]["dimensionId"] == "paper-slot:method"
    assert validate_payload(payload["comparePacketContract"], COMPARE_PACKET_SCHEMA, strict=True).ok
