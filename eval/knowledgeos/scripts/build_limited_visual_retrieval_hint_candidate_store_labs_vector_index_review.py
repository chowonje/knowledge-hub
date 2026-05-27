#!/usr/bin/env python3
"""Build labs-only vector index review for visual retrieval hints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_review import (
    DEFAULT_MIN_LABS_HIT_AT5_ROWS,
    DEFAULT_MIN_LABS_HIT_AT10_ROWS,
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_REVIEW_SCHEMA_ID,
    load_json,
    review_limited_visual_retrieval_hint_candidate_store_labs_vector_index,
    sanitized_report_ref,
    write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_review,
)


DEFAULT_SOURCE_DRY_RUN_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run_005.v1.json"
)
DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_labs_vector_index_review_005.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_labs_vector_index_review_005.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dry-run", type=Path, default=DEFAULT_SOURCE_DRY_RUN_PATH)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--min-labs-hit-at5-rows", type=int, default=DEFAULT_MIN_LABS_HIT_AT5_ROWS)
    parser.add_argument("--min-labs-hit-at10-rows", type=int, default=DEFAULT_MIN_LABS_HIT_AT10_ROWS)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write-report", action="store_true", help="Build and validate without writing report files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    source_path = args.source_dry_run.expanduser()
    dry_run = load_json(source_path)
    report = review_limited_visual_retrieval_hint_candidate_store_labs_vector_index(
        labs_vector_index_dry_run=dry_run,
        source_labs_vector_index_dry_run_ref=sanitized_report_ref(source_path, project_root=PROJECT_ROOT),
        min_labs_hit_at5_rows=args.min_labs_hit_at5_rows,
        min_labs_hit_at10_rows=args.min_labs_hit_at10_rows,
    )
    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_REVIEW_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "limited visual retrieval hint candidate-store labs vector-index review schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write_report:
        paths = write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_review(
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
