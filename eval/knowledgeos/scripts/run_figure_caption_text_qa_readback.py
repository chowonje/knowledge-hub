#!/usr/bin/env python
"""Run caption-text-only QA readback against FigureCaptionArtifact candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from knowledge_hub.papers.figure_caption_text_qa import (
    answer_figure_caption_text_question,
    build_default_text_qa_readback_report,
    load_candidate_report,
    write_text_qa_readback_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANDIDATE_REPORT = (
    PROJECT_ROOT / "eval" / "knowledgeos" / "reports" / "figure_caption_artifact_vertical_slice.v1.json"
)
DEFAULT_OUT = PROJECT_ROOT / "eval" / "knowledgeos" / "reports" / "figure_caption_text_qa_readback.v1.json"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument("--paper-id", default="")
    parser.add_argument("--question", default="")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    candidate_report = load_candidate_report(args.candidate_report)
    if args.paper_id and args.question:
        report = answer_figure_caption_text_question(
            candidate_report=candidate_report,
            paper_id=args.paper_id,
            question=args.question,
        )
    else:
        report = build_default_text_qa_readback_report(candidate_report)
    if not args.no_write:
        write_text_qa_readback_report(report, path=args.out)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") in {"ready", "answerable", "no_answer"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
