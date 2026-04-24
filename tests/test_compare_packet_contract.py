from __future__ import annotations

from types import SimpleNamespace

from knowledge_hub.ai.answer_payload_builder import AnswerPayloadBuilder
from knowledge_hub.ai.compare_packet import COMPARE_PACKET_SCHEMA, build_compare_packet_contract
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
