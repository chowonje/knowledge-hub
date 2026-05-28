#!/usr/bin/env python3
"""Build readback review for applied visual retrieval-hint candidate records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_readback_review import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_READBACK_REVIEW_SCHEMA_ID,
    load_json,
    review_limited_visual_retrieval_hint_candidate_store_apply_readback,
    sanitized_report_ref,
    write_limited_visual_retrieval_hint_candidate_store_apply_readback_review,
)


DEFAULT_SOURCE_APPLY_REPORT_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_apply_executor_005.apply.v1.json"
)
DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_apply_readback_review_005.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_apply_readback_review_005.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-apply-report", type=Path, default=DEFAULT_SOURCE_APPLY_REPORT_PATH)
    parser.add_argument("--papers-dir", type=Path, required=True, help="Local papers_dir root to read back.")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing report files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    source_path = args.source_apply_report.expanduser()
    source_report = load_json(source_path)
    report = review_limited_visual_retrieval_hint_candidate_store_apply_readback(
        apply_report=source_report,
        source_apply_report_ref=sanitized_report_ref(source_path, project_root=PROJECT_ROOT),
        papers_dir=args.papers_dir,
    )
    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "limited visual retrieval hint candidate-store apply readback review schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write_report:
        paths = write_limited_visual_retrieval_hint_candidate_store_apply_readback_review(
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
