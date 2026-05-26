from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.figure_caption_artifact_vertical_slice import (
    FIGURE_CAPTION_ARTIFACT_CANDIDATE_SCHEMA_ID,
    FIGURE_CAPTION_ARTIFACT_VERTICAL_SLICE_REPORT_SCHEMA_ID,
    build_figure_caption_qa_readback,
    extract_figure_caption_candidates_from_blocks,
    write_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "docs" / "schemas" / "fixtures"


def _candidate() -> dict[str, object]:
    rows = extract_figure_caption_candidates_from_blocks(
        paper_id="sample-paper",
        paper_ref="papers_dir/sample.pdf",
        source_content_hash="sha256:" + "1" * 64,
        blocks_by_page=[
            (
                3,
                [
                    (10.0, 20.0, 200.0, 80.0, "Figure 2: Architecture overview with local evidence.", 0, 0),
                    (10.0, 90.0, 200.0, 110.0, "This is ordinary body text.", 1, 0),
                ],
            )
        ],
    )
    assert len(rows) == 1
    return rows[0]


def _report(candidate: dict[str, object]) -> dict[str, object]:
    report: dict[str, object] = {
        "schema": FIGURE_CAPTION_ARTIFACT_VERTICAL_SLICE_REPORT_SCHEMA_ID,
        "status": "ready",
        "generatedAt": "2026-05-26T00:00:00Z",
        "scope": {
            "paperRows": 3,
            "paperRefs": ["papers_dir/sample-a.pdf", "papers_dir/sample-b.pdf", "papers_dir/sample-c.pdf"],
            "writes": "report_only",
            "canonicalParsedArtifactWriteRows": 0,
        },
        "paperDiagnostics": [
            {
                "paperId": "sample-paper",
                "paperRef": "papers_dir/sample.pdf",
                "pageCount": 3,
                "captionCandidateRows": 1,
            }
        ],
        "candidateArtifactRows": 1,
        "candidateArtifacts": [candidate],
        "blockerRows": 1,
        "blockers": [
            {
                "paperId": "blocked-paper",
                "paperRef": "papers_dir/blocked.pdf",
                "blockerReason": "figure_caption_not_found",
                "detail": "",
            }
        ],
        "qaReadbackRows": 0,
        "answerableQaRows": 0,
        "noAnswerQaRows": 0,
        "qaReadback": [],
        "mutationCounters": {
            "canonicalParsedArtifactWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "answerVisiblePromotionRows": 0,
            "strictEvidencePromotionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
            "answerabilityGateBypassRows": 0,
        },
        "schemaViolationCount": 0,
        "privatePathLeakRows": 0,
        "warnings": [],
        "schemaErrors": [],
    }
    answerable = build_figure_caption_qa_readback(
        report=report,
        paper_id="sample-paper",
        question="이 논문의 Figure 2는 무엇을 보여주는가?",
    )
    no_answer = build_figure_caption_qa_readback(
        report=report,
        paper_id="sample-paper",
        question="이 논문의 Figure 99는 무엇을 보여주는가?",
    )
    report["qaReadback"] = [answerable, no_answer]
    report["qaReadbackRows"] = 2
    report["answerableQaRows"] = 1
    report["noAnswerQaRows"] = 1
    return report


def _load_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def test_candidate_schema_fixture_validates() -> None:
    payload = _load_fixture("paper-figure-caption-artifact-candidate.v1.fixture.json")

    result = validate_payload(payload, FIGURE_CAPTION_ARTIFACT_CANDIDATE_SCHEMA_ID, strict=True)

    assert result.ok, result.errors


def test_report_schema_fixture_validates() -> None:
    payload = _load_fixture("paper-figure-caption-artifact-vertical-slice-report.v1.fixture.json")

    result = validate_payload(payload, FIGURE_CAPTION_ARTIFACT_VERTICAL_SLICE_REPORT_SCHEMA_ID, strict=True)

    assert result.ok, result.errors


def test_extracts_candidate_with_required_provenance_from_blocks() -> None:
    candidate = _candidate()

    assert candidate["paperId"] == "sample-paper"
    assert candidate["sourceContentHash"] == "sha256:" + "1" * 64
    assert candidate["page"] == 3
    assert candidate["bbox"] == [10.0, 20.0, 200.0, 80.0]
    assert candidate["figureLabel"] == "Figure 2"
    assert candidate["captionTextHash"].startswith("sha256:")
    assert candidate["blockerReason"] == ""
    assert validate_payload(candidate, FIGURE_CAPTION_ARTIFACT_CANDIDATE_SCHEMA_ID, strict=True).ok


def test_qa_readback_returns_answer_packet_only_with_provenance() -> None:
    report = _report(_candidate())

    row = report["qaReadback"][0]  # type: ignore[index]

    assert row["answerabilityStatus"] == "answerable"
    packet = row["candidateAnswerPacket"]
    assert packet["answerVisible"] is False
    assert packet["strictEvidence"] is False
    assert packet["evidence"]["sourceContentHash"] == "sha256:" + "1" * 64
    assert packet["evidence"]["page"] == 3
    assert packet["evidence"]["bbox"] == [10.0, 20.0, 200.0, 80.0]
    assert validate_payload(report, FIGURE_CAPTION_ARTIFACT_VERTICAL_SLICE_REPORT_SCHEMA_ID, strict=True).ok


def test_qa_readback_no_answer_when_requested_caption_is_missing() -> None:
    report = _report(_candidate())

    row = report["qaReadback"][1]  # type: ignore[index]

    assert row["answerabilityStatus"] == "no_answer"
    assert row["blockerReason"] == "figure_caption_not_found"
    assert row["candidateAnswerPacket"] is None


def test_report_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = _report(_candidate())
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_report(report, json_path=json_path, markdown_path=md_path)

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "papers_dir/sample.pdf" in combined
