#!/usr/bin/env python3
"""Build or explicitly apply visual retrieval-hint candidate-store records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID,
    execute_limited_visual_retrieval_hint_candidate_store_apply_executor,
    load_json,
    sanitized_report_ref,
    write_limited_visual_retrieval_hint_candidate_store_apply_executor,
)


DEFAULT_SOURCE_REPORT_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run_005.v1.json"
)
DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_apply_executor_005.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_apply_executor_005.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-apply-executor-dry-run-report", type=Path, default=DEFAULT_SOURCE_REPORT_PATH)
    parser.add_argument("--papers-dir", type=Path, default=None, help="Local papers_dir root. Required with --apply.")
    parser.add_argument("--run-id", default="", help="Stable run id for applied records and manifest.")
    parser.add_argument("--apply", action="store_true", help="Write candidate-store JSONL records.")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing report files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    source_path = args.source_apply_executor_dry_run_report.expanduser()
    source_report = load_json(source_path)
    report = execute_limited_visual_retrieval_hint_candidate_store_apply_executor(
        apply_executor_dry_run_report=source_report,
        source_apply_executor_dry_run_report_ref=sanitized_report_ref(source_path, project_root=PROJECT_ROOT),
        papers_dir=args.papers_dir,
        run_id=args.run_id or None,
        apply=bool(args.apply),
    )
    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "limited visual retrieval hint candidate-store apply executor schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write_report:
        paths = write_limited_visual_retrieval_hint_candidate_store_apply_executor(
            report,
            report_json=args.report_json,
            report_md=args.report_md,
        )
    else:
        paths = {}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "status": report.get("status"),
                    "decision": report.get("decision"),
                    "nextRecommendedTranche": report.get("nextRecommendedTranche"),
                    "counts": report.get("counts"),
                    "paths": paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if report.get("status") in {"ready", "applied"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
