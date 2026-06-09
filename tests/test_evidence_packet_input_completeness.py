from __future__ import annotations

from knowledge_hub.ai.evidence_packet_input_completeness import (
    EVIDENCE_PACKET_INPUT_COMPLETENESS_SCHEMA_ID,
    build_evidence_packet_input_completeness,
)
from knowledge_hub.core.schema_validator import validate_payload


def test_packet_input_completeness_reconstructs_empty_compare_spans() -> None:
    report = build_evidence_packet_input_completeness(
        manifest={
            "runs": [
                {
                    "runId": "compare_rag_fid__A_hybrid_k5",
                    "caseId": "compare_rag_fid",
                    "variantId": "A_hybrid_k5",
                    "query": "Compare 2005.11401 and 2312.10997",
                    "expectedIds": ["2005.11401", "2312.10997"],
                }
            ]
        },
        raw_payloads={
            "compare_rag_fid__A_hybrid_k5": {
                "evidencePacketContract": {"spans": []},
                "evidence": [
                    {
                        "source_id": "paper:2005.11401",
                        "citation_label": "S1",
                        "excerpt": "RAG combines retrieved passages with generation.",
                        "source_content_hash": "source-hash",
                        "snippet_hash": "snippet-hash",
                        "span_locator": "chars:1-42",
                        "score": 0.8,
                    }
                ],
                "citations": [{"target": "paper:2005.11401", "label": "S1"}],
            }
        },
        generated_at="2026-06-07T00:00:00Z",
    )

    row = report["rows"][0]
    assert report["schema"] == EVIDENCE_PACKET_INPUT_COMPLETENESS_SCHEMA_ID
    assert report["counts"]["rowCount"] == 1
    assert report["counts"]["nonEmptyPacketRows"] == 1
    assert report["counts"]["blockedRows"] == 1
    assert row["status"] == "blocked"
    assert row["sourceCandidateIds"] == ["2005.11401", "2312.10997"]
    assert row["citationCandidateIds"] == ["2005.11401", "2312.10997"]
    assert row["packetSourceIds"] == ["2005.11401"]
    assert row["missingExpectedSourceIds"] == ["2312.10997"]
    assert row["spanCount"] == 1
    assert validate_payload(report, EVIDENCE_PACKET_INPUT_COMPLETENESS_SCHEMA_ID, strict=True).ok
