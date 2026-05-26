#!/usr/bin/env python
"""Build report-only equation locator and nearby-context candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from knowledge_hub.papers.figure_caption_artifact_vertical_slice import default_paper_specs, default_papers_root
from knowledge_hub.papers.text_equation_locator_context_artifacts import (
    build_text_equation_locator_context_report,
    write_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON_PATH = PROJECT_ROOT / "eval" / "knowledgeos" / "reports" / "text_equation_locator_context_artifacts.v1.json"
DEFAULT_MD_PATH = PROJECT_ROOT / "eval" / "knowledgeos" / "reports" / "text_equation_locator_context_artifacts.v1.md"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-root", type=Path, default=default_papers_root())
    parser.add_argument("--out", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    report = build_text_equation_locator_context_report(
        papers_root=args.papers_root.expanduser(),
        paper_specs=default_paper_specs(),
    )
    if not args.no_write:
        write_report(report, json_path=args.out, markdown_path=args.md_out)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
