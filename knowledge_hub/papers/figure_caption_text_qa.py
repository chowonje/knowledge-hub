"""Caption-text-only QA readback for FigureCaptionArtifact candidates."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import (
    build_figure_caption_qa_readback,
)

FIGURE_CAPTION_TEXT_QA_READBACK_SCHEMA_ID = "knowledge-hub.paper.figure-caption-text-qa-readback.v1"

VISUAL_REASONING_RE = re.compile(
    "|".join(
        re.escape(token)
        for token in (
            "bar",
            "axis",
            "pixel",
            "color",
            "colour",
            "curve",
            "higher than",
            "lower than",
            "left side",
            "right side",
            "visual detail",
            "image region",
            "막대",
            "축",
            "픽셀",
            "색",
            "곡선",
            "왼쪽",
            "오른쪽",
            "더 높",
            "더 낮",
            "시각",
            "이미지 안",
        )
    ),
    re.IGNORECASE,
)


def load_candidate_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _visual_reasoning_requested(question: str) -> bool:
    return bool(VISUAL_REASONING_RE.search(str(question or "")))


def answer_figure_caption_text_question(
    *,
    candidate_report: dict[str, Any],
    paper_id: str,
    question: str,
) -> dict[str, Any]:
    base = build_figure_caption_qa_readback(
        report=candidate_report,
        paper_id=paper_id,
        question=question,
    )
    visual_requested = _visual_reasoning_requested(question)
    if visual_requested and base.get("requestedFigureLabel"):
        return {
            "schema": FIGURE_CAPTION_TEXT_QA_READBACK_SCHEMA_ID,
            "status": "no_answer",
            "paperId": paper_id,
            "question": question,
            "requestedFigureLabel": base.get("requestedFigureLabel") or "",
            "answerabilityStatus": "no_answer",
            "blockerReason": "visual_reasoning_not_supported_in_text_evidence_v01",
            "evidenceMode": "caption_text_only",
            "visualReasoningRequested": True,
            "candidateAnswerPacket": None,
        }

    return {
        "schema": FIGURE_CAPTION_TEXT_QA_READBACK_SCHEMA_ID,
        "status": str(base.get("answerabilityStatus") or "no_answer"),
        "paperId": paper_id,
        "question": question,
        "requestedFigureLabel": base.get("requestedFigureLabel") or "",
        "answerabilityStatus": base.get("answerabilityStatus"),
        "blockerReason": base.get("blockerReason") or "",
        "evidenceMode": "caption_text_only",
        "visualReasoningRequested": False,
        "candidateAnswerPacket": base.get("candidateAnswerPacket"),
    }


def build_default_text_qa_readback_report(candidate_report: dict[str, Any]) -> dict[str, Any]:
    candidates = list(candidate_report.get("candidateArtifacts") or [])
    if not candidates:
        rows = [
            answer_figure_caption_text_question(
                candidate_report=candidate_report,
                paper_id="unknown-paper",
                question="이 논문의 Figure 1은 무엇을 보여주는가?",
            )
        ]
    else:
        first = candidates[0]
        paper_id = str(first.get("paperId") or "")
        figure_label = str(first.get("figureLabel") or "Figure 1")
        rows = [
            answer_figure_caption_text_question(
                candidate_report=candidate_report,
                paper_id=paper_id,
                question=f"이 논문의 {figure_label}은 무엇을 보여주는가?",
            ),
            answer_figure_caption_text_question(
                candidate_report=candidate_report,
                paper_id=paper_id,
                question="이 논문의 Figure 99은 무엇을 보여주는가?",
            ),
            answer_figure_caption_text_question(
                candidate_report=candidate_report,
                paper_id=paper_id,
                question=f"이 논문의 {figure_label}에서 막대가 더 높은가?",
            ),
        ]

    return {
        "schema": "knowledge-hub.paper.figure-caption-text-qa-readback-report.v1",
        "status": "ready" if rows else "blocked",
        "candidateReportStatus": candidate_report.get("status") or "unknown",
        "candidateArtifactRows": int(candidate_report.get("candidateArtifactRows") or 0),
        "qaRows": len(rows),
        "answerableRows": sum(1 for row in rows if row.get("answerabilityStatus") == "answerable"),
        "noAnswerRows": sum(1 for row in rows if row.get("answerabilityStatus") == "no_answer"),
        "visualUnsupportedRows": sum(
            1
            for row in rows
            if row.get("blockerReason") == "visual_reasoning_not_supported_in_text_evidence_v01"
        ),
        "rows": rows,
        "mutationCounters": {
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "strictEvidencePromotionRows": 0,
            "runtimeAnswerVisibleExposureRows": 0,
        },
        "warnings": [],
        "schemaErrors": [],
    }


def write_text_qa_readback_report(report: dict[str, Any], *, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "FIGURE_CAPTION_TEXT_QA_READBACK_SCHEMA_ID",
    "answer_figure_caption_text_question",
    "build_default_text_qa_readback_report",
    "load_candidate_report",
    "write_text_qa_readback_report",
]
