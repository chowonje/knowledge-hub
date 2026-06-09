from __future__ import annotations

from knowledge_hub.ai.paper_understanding_profile_readiness import (
    PAPER_UNDERSTANDING_PROFILE_READINESS_SCHEMA_ID,
    build_paper_understanding_profile_readiness,
)
from knowledge_hub.core.schema_validator import validate_payload


def _span(source_id: str, label: str, text: str) -> dict[str, str]:
    return {
        "sourceId": source_id,
        "citationLabel": label,
        "locator": "chars:1-200",
        "text": text,
    }


def _complete_paper_text(name: str) -> str:
    return (
        f"{name} proposes a claim about retrieval-augmented generation. "
        "The method combines retrieval and generation in a practical architecture. "
        "Experiments evaluate open-domain QA benchmarks and show result improvements. "
        "A limitation is that retrieval quality can constrain the answer. "
        "This is useful for the user's purpose and task when comparing papers."
    )


def test_profile_readiness_accepts_compare_rows_when_both_papers_have_brief_slots() -> None:
    report = build_paper_understanding_profile_readiness(
        packet_input_report={
            "rows": [
                {
                    "runId": "compare_rag_fid__A_hybrid_k5",
                    "caseId": "compare_rag_fid",
                    "variantId": "A_hybrid_k5",
                    "status": "ready",
                    "packetSourceIds": ["2005.11401", "2312.10997"],
                    "warnings": [],
                    "promptPacket": {
                        "query": "Compare 2005.11401 and 2312.10997 for a purpose brief",
                        "expectedSourceIds": ["2005.11401", "2312.10997"],
                        "spans": [
                            _span("2005.11401", "S1", _complete_paper_text("RAG")),
                            _span("2312.10997", "S2", _complete_paper_text("The survey")),
                        ],
                    },
                }
            ]
        },
        generated_at="2026-06-08T00:00:00Z",
    )

    row = report["rows"][0]
    assert report["schema"] == PAPER_UNDERSTANDING_PROFILE_READINESS_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["counts"]["readyRows"] == 1
    assert report["counts"]["blockedRows"] == 0
    assert report["counts"]["briefReadyPaperRows"] == 2
    assert row["status"] == "ready"
    assert row["readyPaperCount"] == 2
    assert all(profile["briefReady"] for profile in row["paperProfiles"])
    assert validate_payload(report, PAPER_UNDERSTANDING_PROFILE_READINESS_SCHEMA_ID, strict=True).ok


def test_profile_readiness_blocks_one_sided_or_slot_incomplete_compare_rows() -> None:
    report = build_paper_understanding_profile_readiness(
        packet_input_report={
            "rows": [
                {
                    "runId": "compare_rag_fid__A_hybrid_k5",
                    "caseId": "compare_rag_fid",
                    "variantId": "A_hybrid_k5",
                    "status": "blocked",
                    "packetSourceIds": ["2005.11401"],
                    "warnings": ["missing_expected_source_id:2312.10997"],
                    "promptPacket": {
                        "query": "Compare 2005.11401 and 2312.10997 for a purpose brief",
                        "expectedSourceIds": ["2005.11401", "2312.10997"],
                        "spans": [
                            _span(
                                "2005.11401",
                                "S1",
                                "RAG proposes a claim. The method uses retrieval and generation.",
                            )
                        ],
                    },
                }
            ]
        },
        generated_at="2026-06-08T00:00:00Z",
    )

    row = report["rows"][0]
    profile_by_source = {profile["sourceId"]: profile for profile in row["paperProfiles"]}
    assert report["status"] == "ready"
    assert report["counts"]["readyRows"] == 0
    assert report["counts"]["blockedRows"] == 1
    assert report["counts"]["briefBlockedPaperRows"] == 2
    assert row["status"] == "blocked"
    assert profile_by_source["2005.11401"]["coreReady"] is False
    assert "evidence" in profile_by_source["2005.11401"]["missingCoreSlots"]
    assert profile_by_source["2312.10997"]["spanCount"] == 0
    assert "missing_source_spans" in profile_by_source["2312.10997"]["warnings"]
    assert "packet_input_row_blocked" in row["warnings"]
    assert validate_payload(report, PAPER_UNDERSTANDING_PROFILE_READINESS_SCHEMA_ID, strict=True).ok


def _readback_slot(source_id: str, slot: str) -> dict[str, str]:
    return {
        "slot": slot,
        "status": "ready",
        "sourceId": source_id,
        "citationLabel": "S1",
        "locator": "chars:1-120",
        "sourceRef": f"papers_dir/parsed/{source_id}/document.md",
        "contentHash": "sha256:" + "a" * 64,
        "snippetHash": "sha256:" + "b" * 64,
        "excerpt": f"{slot} excerpt for {source_id}",
    }


def _readback_paper(source_id: str) -> dict[str, object]:
    return {
        "sourceId": source_id,
        "slots": [
            _readback_slot(source_id, "claim"),
            _readback_slot(source_id, "method"),
            _readback_slot(source_id, "evidence"),
            _readback_slot(source_id, "limitation"),
            _readback_slot(source_id, "purpose_relevance"),
        ],
        "availableSlots": ["claim", "method", "evidence", "limitation", "purpose_relevance"],
        "missingBriefSlots": [],
        "coreReady": True,
        "briefReady": True,
        "warnings": [],
    }


def _readback_row(run_id: str, source_ids: list[str]) -> dict[str, object]:
    if not source_ids:
        return {
            "runId": run_id,
            "caseId": run_id.split("__")[0],
            "variantId": run_id.split("__")[-1],
            "status": "not_applicable",
            "expectedSourceIds": [],
            "paperReadbacks": [],
            "readyPaperCount": 0,
            "blockedPaperCount": 0,
            "warnings": ["not_applicable_abstention"],
        }
    return {
        "runId": run_id,
        "caseId": run_id.split("__")[0],
        "variantId": run_id.split("__")[-1],
        "status": "ready",
        "expectedSourceIds": source_ids,
        "paperReadbacks": [_readback_paper(source_id) for source_id in source_ids],
        "readyPaperCount": len(source_ids),
        "blockedPaperCount": 0,
        "warnings": [],
    }


def test_profile_readiness_accepts_readback_report_with_six_ready_and_two_not_applicable_rows() -> None:
    readback_report = {
        "rows": [
            _readback_row("single_transformer__A_hybrid_k5", ["1706.03762"]),
            _readback_row("single_transformer__B_semantic_k8", ["1706.03762"]),
            _readback_row("synthesis_transformer__A_hybrid_k5", ["1706.03762"]),
            _readback_row("synthesis_transformer__B_semantic_k8", ["1706.03762"]),
            _readback_row("compare_rag_fid__A_hybrid_k5", ["2005.11401", "2312.10997"]),
            _readback_row("compare_rag_fid__B_semantic_k8", ["2005.11401", "2312.10997"]),
            _readback_row("abstain_no_source__A_hybrid_k5", []),
            _readback_row("abstain_no_source__B_semantic_k8", []),
        ]
    }

    report = build_paper_understanding_profile_readiness(
        packet_input_report={"rows": []},
        readback_report=readback_report,
        generated_at="2026-06-09T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["counts"]["rowCount"] == 8
    assert report["counts"]["readyRows"] == 6
    assert report["counts"]["notApplicableRows"] == 2
    assert report["counts"]["blockedRows"] == 0
    assert report["counts"]["paperProfileRows"] == 8
    assert report["counts"]["briefReadyPaperRows"] == 8
    assert report["counts"]["briefBlockedPaperRows"] == 0
    assert {row["status"] for row in report["rows"]} == {"ready", "not_applicable"}
    assert validate_payload(report, PAPER_UNDERSTANDING_PROFILE_READINESS_SCHEMA_ID, strict=True).ok
