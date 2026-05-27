#!/usr/bin/env python3
"""Build a report-only allowlist apply-executor dry-run for visual retrieval hints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_ALLOWLIST_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
    build_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run,
    load_json,
    sanitized_report_ref,
    write_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run,
)


DEFAULT_ALLOWLIST_REVIEW_REPORT_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/limited_visual_retrieval_hint_candidate_store_apply_allowlist_review_005.v1.json"
)
DEFAULT_DRY_RUN_REPORT_PATHS = [
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_dry_run.v1.json",
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run.v1.json",
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run_003.v1.json",
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run_004.v1.json",
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run_005.v1.json",
]
DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run_005.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run_005.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-allowlist-review-report", type=Path, default=DEFAULT_ALLOWLIST_REVIEW_REPORT_PATH)
    parser.add_argument(
        "--source-dry-run-report",
        type=Path,
        action="append",
        default=None,
        help="Source visual retrieval-hint candidate-store dry-run report. May be repeated.",
    )
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    allowlist_path = args.source_allowlist_review_report.expanduser()
    dry_run_paths = [path.expanduser() for path in (args.source_dry_run_report or DEFAULT_DRY_RUN_REPORT_PATHS)]
    allowlist_report = load_json(allowlist_path)
    dry_run_reports = [
        (sanitized_report_ref(path, project_root=PROJECT_ROOT), load_json(path))
        for path in dry_run_paths
    ]
    report = build_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run(
        allowlist_report,
        dry_run_reports,
        source_allowlist_review_report_ref=sanitized_report_ref(allowlist_path, project_root=PROJECT_ROOT),
    )
    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_ALLOWLIST_APPLY_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "limited visual retrieval hint candidate-store allowlist apply executor dry-run "
            "schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_limited_visual_retrieval_hint_candidate_store_allowlist_apply_executor_dry_run(
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
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
