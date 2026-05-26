#!/usr/bin/env python
"""Build a report-only parsed artifact source blocker audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_coverage_batch_report import DEFAULT_MAX_SOURCE_PDF_BYTES
from knowledge_hub.papers.parsed_artifact_source_blocker_report import (
    build_parsed_artifact_source_blocker_report,
    write_parsed_artifact_source_blocker_report,
)


DEFAULT_OUTPUT_DIR = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-20/parsed-artifact-source-blocker-report"
).expanduser()


def _read_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").split()[0].strip()


def _load_json(path: Path | None) -> dict:
    if not path:
        return {}
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional khub config path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Report output directory.")
    parser.add_argument("--report-name", default="parsed-artifact-source-blocker-report")
    parser.add_argument("--max-pdf-bytes", type=int, default=DEFAULT_MAX_SOURCE_PDF_BYTES)
    parser.add_argument("--baseline-command-report", default="", help="Optional captured `khub papers extraction-report --json` file.")
    parser.add_argument("--sqlite-before-cli-sha256", default="", help="Optional file containing pre-CLI SQLite sha256.")
    parser.add_argument("--sqlite-after-cli-sha256", default="", help="Optional file containing post-CLI SQLite sha256.")
    args = parser.parse_args()

    config = Config(args.config)
    baseline_report_path = Path(args.baseline_command_report).expanduser() if args.baseline_command_report else None
    sqlite_db = SQLiteDatabase(
        config.sqlite_path,
        enable_event_store=False,
        bootstrap=False,
        read_only=True,
    )
    try:
        report = build_parsed_artifact_source_blocker_report(
            sqlite_db=sqlite_db,
            papers_dir=config.papers_dir,
            report_name=args.report_name,
            max_source_pdf_bytes=args.max_pdf_bytes,
            baseline_command_report=_load_json(baseline_report_path),
            baseline_command_report_path=baseline_report_path,
            sqlite_hash_before_cli=_read_hash(Path(args.sqlite_before_cli_sha256).expanduser())
            if args.sqlite_before_cli_sha256
            else "",
            sqlite_hash_after_cli=_read_hash(Path(args.sqlite_after_cli_sha256).expanduser())
            if args.sqlite_after_cli_sha256
            else "",
        )
    finally:
        sqlite_db.close()

    paths = write_parsed_artifact_source_blocker_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "paths": paths,
                "baseline": report["baseline"],
                "sourceBlockerPool": report["sourceBlockerPool"],
                "nextRecommendedTranche": report["nextRecommendedTranche"],
                "mutationCounters": report["mutationCounters"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
