#!/usr/bin/env python
"""Plan or apply a bounded arXiv PDF source recovery tranche."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.parsed_artifact_arxiv_source_recovery import (
    DEFAULT_ARXIV_SOURCE_RECOVERY_LIMIT,
    build_parsed_artifact_arxiv_source_recovery,
    write_parsed_artifact_arxiv_source_recovery,
)
from knowledge_hub.papers.parsed_artifact_coverage_batch_report import DEFAULT_MAX_SOURCE_PDF_BYTES


DEFAULT_OUTPUT_DIR = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-20/parsed-artifact-arxiv-source-recovery"
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
    parser.add_argument("--report-name", default="parsed-artifact-arxiv-source-recovery")
    parser.add_argument("--limit", type=int, default=DEFAULT_ARXIV_SOURCE_RECOVERY_LIMIT)
    parser.add_argument("--max-pdf-bytes", type=int, default=DEFAULT_MAX_SOURCE_PDF_BYTES)
    parser.add_argument("--apply", action="store_true", help="Download selected arXiv PDFs and register local source paths.")
    parser.add_argument("--allow-network", action="store_true", help="Required with --apply because arXiv downloads are external calls.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing recovered source PDF target.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--baseline-command-report", default="", help="Optional captured `khub papers extraction-report --json` file.")
    parser.add_argument("--sqlite-before-cli-sha256", default="", help="Optional file containing pre-CLI SQLite sha256.")
    parser.add_argument("--sqlite-after-cli-sha256", default="", help="Optional file containing post-CLI SQLite sha256.")
    args = parser.parse_args()

    if args.apply and not args.allow_network:
        parser.error("--apply requires --allow-network")

    config = Config(args.config)
    baseline_report_path = Path(args.baseline_command_report).expanduser() if args.baseline_command_report else None
    sqlite_db = SQLiteDatabase(
        config.sqlite_path,
        enable_event_store=False,
        bootstrap=bool(args.apply),
        read_only=not bool(args.apply),
    )
    try:
        report = build_parsed_artifact_arxiv_source_recovery(
            sqlite_db=sqlite_db,
            papers_dir=config.papers_dir,
            report_name=args.report_name,
            limit=args.limit,
            max_source_pdf_bytes=args.max_pdf_bytes,
            apply=args.apply,
            overwrite=args.overwrite,
            timeout_seconds=args.timeout_seconds,
            delay_seconds=args.delay_seconds if args.apply else 0.0,
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

    paths = write_parsed_artifact_arxiv_source_recovery(report, args.output_dir)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "paths": paths,
                "baseline": report["baseline"],
                "inputRecoverySummary": report["inputRecoverySummary"],
                "selectedCandidateCount": len(report["selectedCandidatePaperIds"]),
                "recoveredPaperIds": report["recoveredPaperIds"],
                "counts": report["counts"],
                "expectedCoverageChangeIfMaterialized": report["expectedCoverageChangeIfMaterialized"],
                "mutationCounters": report["mutationCounters"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
