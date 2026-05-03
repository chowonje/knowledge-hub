from __future__ import annotations

from types import SimpleNamespace

from knowledge_hub.ai.answer_contracts import (
    build_answer_contract,
    build_evidence_packet_contract,
    build_verification_verdict,
    parse_span_offsets,
)
from knowledge_hub.core.schema_validator import validate_payload


def _packet() -> SimpleNamespace:
    return SimpleNamespace(
        evidence=[
            {
                "title": "Alpha",
                "excerpt": "Alpha evidence supports grounded answers.",
                "citation_label": "S1",
                "citation_target": "vault:Alpha.md",
                "source_id": "vault:Alpha.md",
                "source_ref": "vault:Alpha.md",
                "source_content_hash": "hash-alpha",
                "span_locator": "chars:10-52",
                "score": 0.9,
                "semantic_score": 0.7,
                "lexical_score": 0.2,
                "evidence_kind": "raw_span",
            }
        ],
        citations=[{"label": "S1", "target": "vault:Alpha.md", "kind": "source"}],
        evidence_packet={"answerable": True, "answerableDecisionReason": "grounded"},
        evidence_policy={"policyKey": "test-policy"},
    )


def test_parse_span_offsets_accepts_chars_locator():
    assert parse_span_offsets("chars:10-52") == (10, 52)
    assert parse_span_offsets("0-2") == (0, 2)
    assert parse_span_offsets("unit:abc") == (None, None)


def test_evidence_packet_contract_exposes_span_refs_and_hashes():
    pipeline_result = SimpleNamespace(plan=SimpleNamespace(to_dict=lambda: {"queryFrame": {"source_type": "vault"}}))

    contract = build_evidence_packet_contract(
        query="alpha?",
        retrieval_mode="keyword",
        pipeline_result=pipeline_result,
        evidence_packet=_packet(),
    )

    assert contract["schema"] == "knowledge-hub.evidence-packet.v1"
    assert contract["queryFrame"]["source_type"] == "vault"
    span = contract["spans"][0]
    assert span["spanRef"] == "span:1"
    assert span["sourceContentHash"] == "hash-alpha"
    assert span["source_content_hash"] == "hash-alpha"
    assert span["content_hash"] == "hash-alpha"
    assert span["charStart"] == 10
    assert span["charEnd"] == 52
    assert span["spanOffsetAvailable"] is True
    assert validate_payload(contract, "knowledge-hub.evidence-packet.v1", strict=True).ok


def test_answer_contract_blocks_rewrite_for_unsupported_claim_verdict():
    verification = {
        "status": "caution",
        "unsupportedClaimCount": 1,
        "uncertainClaimCount": 0,
        "supportedClaimCount": 0,
        "needsCaution": True,
        "summary": "unsupported claim",
    }

    verdict = build_verification_verdict(verification)
    contract = build_answer_contract(
        answer="Alpha is grounded. It has one unsupported detail.",
        evidence_packet=_packet(),
        verification=verification,
        rewrite={"attempted": False, "applied": False, "finalAnswerSource": "original"},
        routing_meta={"provider": "local", "model": "test"},
    )

    assert verdict["verdict"] == "fail"
    assert verdict["rewriteAllowed"] is False
    assert verdict["rewritePolicy"] == "blocked_by_verification_gate"
    assert contract["schema"] == "knowledge-hub.answer-contract.v1"
    assert contract["verificationVerdict"]["verdict"] == "fail"
    assert contract["citations"][0]["spanRef"] == "span:1"
    assert contract["citations"][0]["source_id"] == "vault:Alpha.md"
    assert contract["retrievalSignals"] == []
    assert contract["modelId"] == "local/test"
    assert validate_payload(contract, "knowledge-hub.answer-contract.v1", strict=True).ok
    assert validate_payload(verdict, "knowledge-hub.verification-verdict.v1", strict=True).ok


def test_verification_verdict_blocks_rewrite_for_caution_even_without_unsupported_claims():
    verdict = build_verification_verdict(
        {
            "status": "caution",
            "unsupportedClaimCount": 0,
            "uncertainClaimCount": 0,
            "supportedClaimCount": 1,
            "needsCaution": True,
            "summary": "needs caution framing",
        }
    )

    assert verdict["verdict"] == "fail"
    assert verdict["rewriteAllowed"] is False
    assert verdict["rewritePolicy"] == "blocked_by_verification_gate"


def test_evidence_packet_strict_mode_excludes_low_provenance_spans_and_fails_external_closed():
    packet = SimpleNamespace(
        evidence=[
            {
                "title": "Low provenance",
                "excerpt": "This span has text but no canonical source hash.",
                "source_id": "vault:low.md",
                "span_locator": "chars:1-44",
            }
        ],
        citations=[{"label": "S1", "target": "vault:low.md", "kind": "source"}],
        evidence_packet={"answerable": True, "answerableDecisionReason": "raw evidence present"},
        evidence_policy={},
    )
    pipeline_result = SimpleNamespace(plan=SimpleNamespace(to_dict=lambda: {"queryFrame": {"source_type": "vault"}}))

    contract = build_evidence_packet_contract(
        query="low provenance?",
        retrieval_mode="keyword",
        pipeline_result=pipeline_result,
        evidence_packet=packet,
    )

    assert contract["spans"] == []
    assert contract["answerable"] is False
    assert contract["coverage"]["status"] == "insufficient"
    assert contract["coverage"]["excluded_low_provenance"] == 1
    assert contract["policy"]["classification"] == "UNKNOWN"
    assert contract["policy"]["external_allowed"] is False
    assert validate_payload(contract, "knowledge-hub.evidence-packet.v1", strict=True).ok


def test_answer_contract_coverage_uses_claim_to_citation_mapping_not_citation_count():
    packet = SimpleNamespace(
        evidence=[
            {
                "title": "Alpha",
                "excerpt": "Alpha evidence supports the first claim.",
                "citation_label": "S1",
                "citation_target": "vault:Alpha.md",
                "source_id": "vault:Alpha.md",
                "source_content_hash": "hash-alpha",
                "span_locator": "chars:1-40",
            },
            {
                "title": "Gamma",
                "excerpt": "Gamma evidence is unrelated to the answer.",
                "citation_label": "S2",
                "citation_target": "vault:Gamma.md",
                "source_id": "vault:Gamma.md",
                "source_content_hash": "hash-gamma",
                "span_locator": "chars:1-40",
            },
        ],
        citations=[
            {"label": "S1", "target": "vault:Alpha.md", "kind": "source"},
            {"label": "S2", "target": "vault:Gamma.md", "kind": "source"},
        ],
        evidence_packet={"answerable": True, "answerableDecisionReason": "grounded"},
        evidence_policy={"policyKey": "test-policy"},
    )

    contract = build_answer_contract(
        answer="Alpha wins. Beta wins.",
        evidence_packet=packet,
        verification={"status": "verified", "unsupportedClaimCount": 0, "needsCaution": False},
        rewrite={"attempted": False, "applied": False, "finalAnswerSource": "original"},
        routing_meta={"provider": "local", "model": "test"},
    )

    assert contract["claimLikeSentenceCount"] == 2
    assert contract["citationBackedSentenceCount"] == 1
    assert contract["coverageRatio"] == 0.5
    assert contract["coverage"]["status"] == "partial"
    assert contract["coverage"]["unmapped_claim_count"] == 1
    assert contract["citationClaimMap"][0]["citationRefs"] == ["span:1"]
    assert contract["citationClaimMap"][1]["citationRefs"] == []


def test_evidence_packet_contract_excludes_non_evidence_source_spans():
    packet = SimpleNamespace(
        evidence=[
            {
                "title": "Belief row",
                "excerpt": "This came from a belief store row.",
                "citation_label": "S1",
                "citation_target": "belief:rag:1",
                "source_id": "belief:rag:1",
                "source_type": "belief",
                "source_content_hash": "hash-belief",
                "span_locator": "chars:1-20",
            }
        ],
        citations=[{"label": "S1", "target": "belief:rag:1", "kind": "source"}],
        evidence_packet={"answerable": True, "answerableDecisionReason": "grounded"},
        evidence_policy={"policyKey": "test-policy"},
    )
    pipeline_result = SimpleNamespace(plan=SimpleNamespace(to_dict=lambda: {"queryFrame": {"source_type": "paper"}}))

    contract = build_evidence_packet_contract(
        query="belief evidence?",
        retrieval_mode="keyword",
        pipeline_result=pipeline_result,
        evidence_packet=packet,
    )

    assert contract["spans"] == []
    assert contract["answerable"] is False
    assert contract["coverage"]["excluded_non_evidence"] == 1
    assert validate_payload(contract, "knowledge-hub.evidence-packet.v1", strict=True).ok


def test_answer_contract_routes_non_evidence_sources_to_retrieval_signals():
    packet = SimpleNamespace(
        evidence=[
            {
                "title": "Alpha",
                "excerpt": "Alpha evidence supports grounded answers.",
                "citation_label": "S1",
                "citation_target": "vault:Alpha.md",
                "source_id": "vault:Alpha.md",
                "source_ref": "vault:Alpha.md",
                "source_content_hash": "hash-alpha",
                "span_locator": "chars:10-52",
                "evidence_kind": "raw_span",
            },
            {
                "title": "Learning edge",
                "excerpt": "Learning graph edges are retrieval hints, not citations.",
                "citation_label": "S2",
                "citation_target": "learning_edge:rag:prereq",
                "source_id": "learning_edge:rag:prereq",
                "source_type": "learning_edge",
                "source_content_hash": "hash-learning-edge",
                "span_locator": "chars:60-110",
                "evidence_kind": "derived_anchor",
            },
        ],
        citations=[
            {"label": "S1", "target": "vault:Alpha.md", "kind": "source"},
            {"label": "S2", "target": "learning_edge:rag:prereq", "kind": "source"},
        ],
        evidence_packet={"answerable": True, "answerableDecisionReason": "grounded"},
        evidence_policy={"policyKey": "test-policy"},
    )

    contract = build_answer_contract(
        answer="Alpha is grounded.",
        evidence_packet=packet,
        verification={"status": "verified", "unsupportedClaimCount": 0, "needsCaution": False},
        rewrite={"attempted": False, "applied": False, "finalAnswerSource": "original"},
        routing_meta={"provider": "local", "model": "test"},
    )

    assert len(contract["citations"]) == 1
    assert contract["citations"][0]["source_id"] == "vault:Alpha.md"
    assert len(contract["retrievalSignals"]) == 1
    assert contract["retrievalSignals"][0]["source_id"] == "learning_edge:rag:prereq"
    assert contract["retrievalSignals"][0]["reason"] == "non_evidence_source_type:learning_edge"
    assert contract["coverage"]["citation_count"] == 1
    assert validate_payload(contract, "knowledge-hub.answer-contract.v1", strict=True).ok
