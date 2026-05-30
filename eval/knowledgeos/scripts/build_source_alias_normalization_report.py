#!/usr/bin/env python
"""Build report-only source alias normalization policy rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.papers.source_alias_normalization import build_source_alias_normalization_report, write_report

REPORTS_ROOT = PROJECT_ROOT / "eval" / "knowledgeos" / "reports"
DEFAULT_JSON_PATH = REPORTS_ROOT / "source_alias_normalization.v1.json"
DEFAULT_MD_PATH = REPORTS_ROOT / "source_alias_normalization.v1.md"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    report = build_source_alias_normalization_report()
    if not args.no_write:
        write_report(report, json_path=args.out, markdown_path=args.md_out)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
