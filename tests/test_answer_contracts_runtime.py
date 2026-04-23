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
    assert verdict["rewritePolicy"] == "blocked_for_unsupported_claims"
    assert contract["schema"] == "knowledge-hub.answer-contract.v1"
    assert contract["verificationVerdict"]["verdict"] == "fail"
    assert contract["citations"][0]["spanRef"] == "span:1"
    assert contract["citations"][0]["source_id"] == "vault:Alpha.md"
    assert contract["modelId"] == "local/test"
    assert validate_payload(contract, "knowledge-hub.answer-contract.v1", strict=True).ok
    assert validate_payload(verdict, "knowledge-hub.verification-verdict.v1", strict=True).ok
