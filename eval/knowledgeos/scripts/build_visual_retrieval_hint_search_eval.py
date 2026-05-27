#!/usr/bin/env python3
"""Build a report-only targeted search eval for visual retrieval hints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.visual_retrieval_hint_search_eval import (
    VISUAL_RETRIEVAL_HINT_SEARCH_EVAL_SCHEMA_ID,
    build_visual_retrieval_hint_search_eval,
    load_json,
    sanitized_report_ref,
    write_visual_retrieval_hint_search_eval,
)


DEFAULT_LAYOUT_CANDIDATE_REPORT_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_layout_candidate_list_report.v1.json"
)
DEFAULT_USEFULNESS_EVAL_REPORT_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_retrieval_hint_usefulness_eval.v1.json"
)
DEFAULT_DRY_RUN_PATHS = [
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_dry_run.v1.json",
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run.v1.json",
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_dry_run_003.v1.json",
]
DEFAULT_JSON_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_retrieval_hint_search_eval.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_retrieval_hint_search_eval.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-layout-candidate-report", type=Path, default=DEFAULT_LAYOUT_CANDIDATE_REPORT_PATH)
    parser.add_argument("--source-usefulness-eval-report", type=Path, default=DEFAULT_USEFULNESS_EVAL_REPORT_PATH)
    parser.add_argument(
        "--source-dry-run-report",
        type=Path,
        action="append",
        dest="source_dry_run_reports",
        default=None,
        help="Dry-run report to include. Repeat to include multiple reports.",
    )
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    layout_path = args.source_layout_candidate_report.expanduser()
    usefulness_path = args.source_usefulness_eval_report.expanduser()
    dry_run_paths = [
        path.expanduser() for path in (args.source_dry_run_reports or DEFAULT_DRY_RUN_PATHS)
    ]
    layout_report = load_json(layout_path)
    usefulness_report = load_json(usefulness_path)
    usefulness_report["_sourceReportRef"] = sanitized_report_ref(
        usefulness_path,
        project_root=PROJECT_ROOT,
    )
    dry_run_reports = [
        (sanitized_report_ref(path, project_root=PROJECT_ROOT), load_json(path))
        for path in dry_run_paths
    ]
    report = build_visual_retrieval_hint_search_eval(
        layout_report,
        usefulness_report,
        dry_run_reports,
    )
    validation = validate_payload(report, VISUAL_RETRIEVAL_HINT_SEARCH_EVAL_SCHEMA_ID, strict=True)
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "visual retrieval hint search eval schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_visual_retrieval_hint_search_eval(
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
