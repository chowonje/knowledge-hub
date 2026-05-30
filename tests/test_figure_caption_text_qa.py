from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.figure_caption_artifact_vertical_slice import (
    extract_figure_caption_candidates_from_blocks,
)
from knowledge_hub.papers.figure_caption_text_qa import (
    FIGURE_CAPTION_TEXT_QA_READBACK_SCHEMA_ID,
    answer_figure_caption_text_question,
    build_default_text_qa_readback_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _candidate_report() -> dict[str, object]:
    candidates = extract_figure_caption_candidates_from_blocks(
        paper_id="sample-paper",
        paper_ref="papers_dir/sample.pdf",
        source_content_hash="sha256:" + "1" * 64,
        blocks_by_page=[
            (
                2,
                [
                    (
                        10.0,
                        20.0,
                        200.0,
                        80.0,
                        "Figure 1: A caption-only architecture overview.",
                        0,
                        0,
                    )
                ],
            )
        ],
    )
    return {
        "schema": "knowledge-hub.paper.figure-caption-artifact-vertical-slice-report.v1",
        "status": "ready",
        "candidateArtifactRows": len(candidates),
        "candidateArtifacts": candidates,
    }


def test_caption_text_question_returns_answer_packet_with_provenance() -> None:
    report = _candidate_report()

    row = answer_figure_caption_text_question(
        candidate_report=report,
        paper_id="sample-paper",
        question="이 논문의 Figure 1은 무엇을 보여주는가?",
    )

    assert row["answerabilityStatus"] == "answerable"
    assert row["evidenceMode"] == "caption_text_only"
    assert row["visualReasoningRequested"] is False
    packet = row["candidateAnswerPacket"]
    assert packet["answerVisible"] is False
    assert packet["strictEvidence"] is False
    assert packet["evidence"]["sourceContentHash"] == "sha256:" + "1" * 64
    assert packet["evidence"]["page"] == 2
    assert packet["evidence"]["bbox"] == [10.0, 20.0, 200.0, 80.0]
    assert validate_payload(row, FIGURE_CAPTION_TEXT_QA_READBACK_SCHEMA_ID, strict=True).ok


def test_caption_text_question_returns_no_answer_for_missing_figure() -> None:
    row = answer_figure_caption_text_question(
        candidate_report=_candidate_report(),
        paper_id="sample-paper",
        question="이 논문의 Figure 99은 무엇을 보여주는가?",
    )

    assert row["answerabilityStatus"] == "no_answer"
    assert row["blockerReason"] == "figure_caption_not_found"
    assert row["candidateAnswerPacket"] is None
    assert validate_payload(row, FIGURE_CAPTION_TEXT_QA_READBACK_SCHEMA_ID, strict=True).ok


def test_visual_reasoning_question_is_blocked_even_when_caption_exists() -> None:
    row = answer_figure_caption_text_question(
        candidate_report=_candidate_report(),
        paper_id="sample-paper",
        question="이 논문의 Figure 1에서 막대가 더 높은가?",
    )

    assert row["answerabilityStatus"] == "no_answer"
    assert row["visualReasoningRequested"] is True
    assert row["blockerReason"] == "visual_reasoning_not_supported_in_text_evidence_v01"
    assert row["candidateAnswerPacket"] is None
    assert validate_payload(row, FIGURE_CAPTION_TEXT_QA_READBACK_SCHEMA_ID, strict=True).ok


def test_default_readback_report_covers_answerable_missing_and_visual_blocked() -> None:
    report = build_default_text_qa_readback_report(_candidate_report())

    assert report["status"] == "ready"
    assert report["qaRows"] == 3
    assert report["answerableRows"] == 1
    assert report["noAnswerRows"] == 2
    assert report["visualUnsupportedRows"] == 1
    assert report["mutationCounters"]["databaseMutationRows"] == 0
    assert report["mutationCounters"]["runtimeAnswerVisibleExposureRows"] == 0
    assert validate_payload(
        report,
        "knowledge-hub.paper.figure-caption-text-qa-readback-report.v1",
        strict=True,
    ).ok


def test_generated_readback_report_has_no_private_paths(tmp_path: Path) -> None:
    report = build_default_text_qa_readback_report(_candidate_report())
    out = tmp_path / "readback.json"

    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    text = out.read_text(encoding="utf-8")
    assert "/" + "Users" + "/" not in text
    assert "/" + "Volumes" + "/" not in text
    assert "Mobile " + "Documents" not in text


def test_readback_script_runs_from_repo_root_without_installed_package(tmp_path: Path) -> None:
    candidate_report = _candidate_report()
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate_report), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "eval/knowledgeos/scripts/run_figure_caption_text_qa_readback.py",
            "--candidate-report",
            str(candidate_path),
            "--no-write",
            "--json",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "ModuleNotFoundError" not in result.stderr
    assert json.loads(result.stdout)["status"] == "ready"
