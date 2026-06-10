from __future__ import annotations

from pathlib import Path

from knowledge_hub.ai.evidence_packet_input_completeness import build_evidence_packet_input_completeness
from knowledge_hub.ai.evidence_packet_parsed_replay import collect_parsed_replay_payloads


def test_packet_completeness_allows_empty_abstention_without_expected_sources() -> None:
    report = build_evidence_packet_input_completeness(
        manifest={
            "runs": [
                {
                    "runId": "abstain_missing_arxiv__A_hybrid_k5",
                    "caseId": "abstain_missing_arxiv",
                    "criterion": "abstention",
                    "variantId": "A_hybrid_k5",
                    "query": "Summarize 9999.99999 from paper evidence only",
                    "expectedIds": [],
                }
            ]
        },
        raw_payloads={"abstain_missing_arxiv__A_hybrid_k5": {"evidencePacketContract": {"spans": []}}},
        generated_at="2026-06-09T00:00:00Z",
    )

    assert report["counts"]["rowCount"] == 1
    assert report["counts"]["nonEmptyPacketRows"] == 0
    assert report["counts"]["blockedRows"] == 0
    assert report["rows"][0]["status"] == "ready"


def test_collect_parsed_replay_payloads_builds_compare_spans_from_local_parsed_docs(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    for paper_id, text in {
        "2005.11401": "RAG retrieves passages and marginalizes over generated answers.",
        "2312.10997": "The RAG survey categorizes naive, advanced, and modular RAG.",
    }.items():
        parsed_dir = papers_dir / "parsed" / paper_id
        parsed_dir.mkdir(parents=True)
        (parsed_dir / "document.md").write_text(f"# {paper_id}\n\n{text}\n", encoding="utf-8")

    manifest = {
        "runs": [
            {
                "runId": "compare_rag_fid__A_hybrid_k5",
                "caseId": "compare_rag_fid",
                "criterion": "comparison",
                "variantId": "A_hybrid_k5",
                "query": "Compare 2005.11401 and 2312.10997",
                "expectedIds": ["2005.11401", "2312.10997"],
            }
        ]
    }

    payloads = collect_parsed_replay_payloads(manifest=manifest, papers_dir=papers_dir)
    spans = payloads["compare_rag_fid__A_hybrid_k5"]["evidencePacketContract"]["spans"]

    assert [span["sourceId"] for span in spans] == ["2005.11401", "2312.10997"]
    assert all(span["contentHash"] for span in spans)
    assert all(span["text"] for span in spans)
