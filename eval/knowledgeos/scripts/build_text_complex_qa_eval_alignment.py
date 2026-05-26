#!/usr/bin/env python
"""Build report-only complex QA alignment against v0.1 text evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from knowledge_hub.papers.text_complex_qa_eval_alignment import align_complex_qa_cases, load_report, write_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_ROOT = PROJECT_ROOT / "eval" / "knowledgeos" / "reports"
DEFAULT_JSON_PATH = REPORTS_ROOT / "text_complex_qa_eval_alignment.v1.json"
DEFAULT_MD_PATH = REPORTS_ROOT / "text_complex_qa_eval_alignment.v1.md"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=REPORTS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def _load_inputs(reports_root: Path) -> dict[str, dict]:
    return {
        "figure_caption_text_qa": load_report(reports_root / "figure_caption_text_qa_readback.v1.json"),
        "section_paragraph_spans": load_report(reports_root / "text_section_paragraph_span_artifacts.v1.json"),
        "table_caption_candidates": load_report(reports_root / "text_table_caption_candidate_artifacts.v1.json"),
        "equation_locator_context": load_report(reports_root / "text_equation_locator_context_artifacts.v1.json"),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    reports = _load_inputs(args.reports_root)
    report = align_complex_qa_cases(reports=reports)
    if not args.no_write:
        write_report(report, json_path=args.out, markdown_path=args.md_out)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
