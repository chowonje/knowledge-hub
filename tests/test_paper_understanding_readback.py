from __future__ import annotations

from pathlib import Path

from knowledge_hub.ai.paper_understanding_readback import (
    PAPER_UNDERSTANDING_READBACK_SCHEMA_ID,
    build_paper_understanding_readback,
)
from knowledge_hub.core.schema_validator import validate_payload


def _write_document(papers_dir: Path, paper_id: str) -> None:
    path = papers_dir / "parsed" / paper_id / "document.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                f"# Paper {paper_id}",
                f"The paper {paper_id} claims a practical improvement for retrieval augmented generation systems.",
                "Its method combines retrieval, generation, and a controlled architecture for evidence assembly.",
                "The evidence comes from experiments, benchmark evaluation, and reported result comparisons.",
                "A limitation is that retrieval quality and corpus coverage can constrain the final answer.",
                "The purpose relevance is high for paper comparison tasks and local research workflows.",
            ]
        ),
        encoding="utf-8",
    )


def _packet_row(run_id: str, expected_ids: list[str]) -> dict[str, object]:
    return {
        "runId": run_id,
        "caseId": run_id.split("__")[0],
        "variantId": run_id.split("__")[-1],
        "status": "ready",
        "packetSourceIds": expected_ids,
        "warnings": [],
        "promptPacket": {
            "query": f"Read back paper understanding for {run_id}",
            "expectedSourceIds": expected_ids,
            "spans": [
                {
                    "sourceId": paper_id,
                    "citationLabel": f"S{index}",
                    "locator": "chars:0-120",
                    "text": f"{paper_id} packet span",
                }
                for index, paper_id in enumerate(expected_ids, start=1)
            ],
        },
    }


def _eight_row_packet_report() -> dict[str, object]:
    return {
        "rows": [
            _packet_row("single_transformer__A_hybrid_k5", ["1706.03762"]),
            _packet_row("single_transformer__B_semantic_k8", ["1706.03762"]),
            _packet_row("synthesis_transformer__A_hybrid_k5", ["1706.03762"]),
            _packet_row("synthesis_transformer__B_semantic_k8", ["1706.03762"]),
            _packet_row("compare_rag_fid__A_hybrid_k5", ["2005.11401", "2312.10997"]),
            _packet_row("compare_rag_fid__B_semantic_k8", ["2005.11401", "2312.10997"]),
            _packet_row("abstain_no_source__A_hybrid_k5", []),
            _packet_row("abstain_no_source__B_semantic_k8", []),
        ]
    }


def _packet_row_without_citation(run_id: str, expected_ids: list[str]) -> dict[str, object]:
    row = _packet_row(run_id, expected_ids)
    prompt_packet = row["promptPacket"]
    assert isinstance(prompt_packet, dict)
    spans = prompt_packet["spans"]
    assert isinstance(spans, list)
    for span in spans:
        assert isinstance(span, dict)
        span["citationLabel"] = ""
    return row


def test_readback_builds_five_grounded_slots_per_expected_paper(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    for paper_id in ["1706.03762", "2005.11401", "2312.10997"]:
        _write_document(papers_dir, paper_id)

    report = build_paper_understanding_readback(
        packet_input_report=_eight_row_packet_report(),
        papers_dir=papers_dir,
        generated_at="2026-06-09T00:00:00Z",
    )

    assert report["schema"] == PAPER_UNDERSTANDING_READBACK_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["counts"]["rowCount"] == 8
    assert report["counts"]["readbackReadyRows"] == 6
    assert report["counts"]["notApplicableRows"] == 2
    assert report["counts"]["blockedRows"] == 0
    assert report["counts"]["paperReadbackRows"] == 8
    assert report["counts"]["briefReadyPaperRows"] == 8
    assert report["counts"]["briefBlockedPaperRows"] == 0
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["counts"]["forbiddenRawMarkerRows"] == 0

    ready_rows = [row for row in report["rows"] if row["status"] == "ready"]
    assert len(ready_rows) == 6
    for row in ready_rows:
        for paper in row["paperReadbacks"]:
            assert paper["briefReady"] is True
            assert {slot["slot"] for slot in paper["slots"]} == {
                "claim",
                "method",
                "evidence",
                "limitation",
                "purpose_relevance",
            }
            for slot in paper["slots"]:
                assert slot["status"] == "ready"
                assert slot["citationLabel"]
                assert slot["locator"].startswith("chars:")
                assert slot["sourceRef"].startswith("papers_dir/parsed/")
                assert slot["contentHash"].startswith("sha256:")
                assert slot["snippetHash"].startswith("sha256:")
                assert slot["excerpt"]
                assert "/Users/" not in slot["sourceRef"]

    not_applicable_rows = [row for row in report["rows"] if row["status"] == "not_applicable"]
    assert len(not_applicable_rows) == 2
    assert all("not_applicable_abstention" in row["warnings"] for row in not_applicable_rows)
    assert validate_payload(report, PAPER_UNDERSTANDING_READBACK_SCHEMA_ID, strict=True).ok


def test_readback_blocks_missing_parsed_document(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"

    report = build_paper_understanding_readback(
        packet_input_report={"rows": [_packet_row("single_missing__A_hybrid_k5", ["9999.99999"])]},
        papers_dir=papers_dir,
        generated_at="2026-06-09T00:00:00Z",
    )

    row = report["rows"][0]
    assert report["status"] == "ready"
    assert report["counts"]["readbackReadyRows"] == 0
    assert report["counts"]["blockedRows"] == 1
    assert row["status"] == "blocked"
    assert "9999.99999:missing_parsed_document" in row["warnings"]
    assert validate_payload(report, PAPER_UNDERSTANDING_READBACK_SCHEMA_ID, strict=True).ok


def test_readback_blocks_private_path_marker_in_document(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    path = papers_dir / "parsed" / "1706.03762" / "document.md"
    path.parent.mkdir(parents=True)
    private_marker = "/" + "Users" + "/won/private-file.pdf"
    path.write_text(f"Claim text leaked from {private_marker}", encoding="utf-8")

    report = build_paper_understanding_readback(
        packet_input_report={"rows": [_packet_row("single_private__A_hybrid_k5", ["1706.03762"])]},
        papers_dir=papers_dir,
        generated_at="2026-06-09T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert report["counts"]["blockedRows"] == 1
    assert "/Users/" not in str(report["rows"])
    assert validate_payload(report, PAPER_UNDERSTANDING_READBACK_SCHEMA_ID, strict=True).ok


def test_readback_blocks_missing_citation_label(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    _write_document(papers_dir, "1706.03762")

    report = build_paper_understanding_readback(
        packet_input_report={
            "rows": [_packet_row_without_citation("single_missing_citation__A_hybrid_k5", ["1706.03762"])]
        },
        papers_dir=papers_dir,
        generated_at="2026-06-09T00:00:00Z",
    )

    row = report["rows"][0]
    assert report["counts"]["blockedRows"] == 1
    assert row["status"] == "blocked"
    assert "1706.03762:missing_citation_label" in row["warnings"]
    assert validate_payload(report, PAPER_UNDERSTANDING_READBACK_SCHEMA_ID, strict=True).ok


def test_readback_blocks_malformed_and_forbidden_rows(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    _write_document(papers_dir, "1706.03762")

    report = build_paper_understanding_readback(
        packet_input_report={
            "rows": [
                _packet_row("single_unsafe__A_hybrid_k5", ["../1706.03762"]),
                {
                    "runId": "single_forbidden__A_hybrid_k5",
                    "caseId": "single_forbidden",
                    "variantId": "A_hybrid_k5",
                    "status": "ready",
                    "packetSourceIds": ["1706.03762"],
                    "warnings": [],
                    "promptPacket": {
                        "query": "paper-card-v2 should never enter this report",
                        "expectedSourceIds": ["1706.03762"],
                        "spans": [
                            {
                                "sourceId": "1706.03762",
                                "citationLabel": "S1",
                                "locator": "chars:0-10",
                                "text": "paper-card-v2",
                            }
                        ],
                    },
                },
            ]
        },
        papers_dir=papers_dir,
        generated_at="2026-06-09T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedRows"] == 2
    assert report["counts"]["forbiddenRawMarkerRows"] == 1
    assert "../1706.03762:unsafe_paper_id" in report["rows"][0]["warnings"]
    assert "forbidden_raw_marker" in report["rows"][1]["warnings"]
    assert "paper-card-v2" not in str(report["rows"])
    assert validate_payload(report, PAPER_UNDERSTANDING_READBACK_SCHEMA_ID, strict=True).ok
