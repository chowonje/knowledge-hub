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
