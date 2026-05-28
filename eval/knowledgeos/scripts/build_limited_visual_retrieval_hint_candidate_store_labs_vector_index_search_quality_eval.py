#!/usr/bin/env python3
"""Build a labs-only search quality eval for applied visual retrieval-hint vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_SEARCH_QUALITY_EVAL_SCHEMA_ID,
    build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval,
    load_json,
    sanitized_report_ref,
    write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval,
)


FULL_LOCAL_LAYOUT_CANDIDATE_REPORT_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_layout_candidate_list_report_corpus.full.local.json"
)
COMPACT_LAYOUT_CANDIDATE_REPORT_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_layout_candidate_list_report.v1.json"
)
DEFAULT_LAYOUT_CANDIDATE_REPORT_PATH = (
    FULL_LOCAL_LAYOUT_CANDIDATE_REPORT_PATH
    if FULL_LOCAL_LAYOUT_CANDIDATE_REPORT_PATH.exists()
    else COMPACT_LAYOUT_CANDIDATE_REPORT_PATH
)
DEFAULT_SOURCE_APPLY_REPORT_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_005.apply.v1.json"
)
DEFAULT_SOURCE_APPLY_EXECUTOR_DRY_RUN_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run_005.v1.json"
)
DEFAULT_PAPERS_DIR = Path.home() / ".khub" / "papers"
DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval_005.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval_005.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-layout-candidate-report", type=Path, default=DEFAULT_LAYOUT_CANDIDATE_REPORT_PATH)
    parser.add_argument("--source-apply-report", type=Path, default=DEFAULT_SOURCE_APPLY_REPORT_PATH)
    parser.add_argument(
        "--source-apply-executor-dry-run",
        type=Path,
        default=DEFAULT_SOURCE_APPLY_EXECUTOR_DRY_RUN_PATH,
    )
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--min-labs-hit-at5-rows", type=int, default=200)
    parser.add_argument("--min-hybrid-hit-at5-lift-rows", type=int, default=25)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    layout_path = args.source_layout_candidate_report.expanduser()
    apply_path = args.source_apply_report.expanduser()
    dry_run_path = args.source_apply_executor_dry_run.expanduser()
    report = build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval(
        layout_candidate_report=load_json(layout_path),
        labs_vector_index_apply_report=load_json(apply_path),
        labs_vector_index_apply_executor_dry_run=load_json(dry_run_path),
        source_layout_candidate_report_ref=sanitized_report_ref(layout_path, project_root=PROJECT_ROOT),
        source_labs_vector_index_apply_report_ref=sanitized_report_ref(apply_path, project_root=PROJECT_ROOT),
        source_labs_vector_index_apply_executor_dry_run_ref=sanitized_report_ref(
            dry_run_path,
            project_root=PROJECT_ROOT,
        ),
        papers_dir=args.papers_dir,
        min_labs_hit_at5_rows=args.min_labs_hit_at5_rows,
        min_hybrid_hit_at5_lift_rows=args.min_hybrid_hit_at5_lift_rows,
    )
    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_SEARCH_QUALITY_EVAL_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "limited visual retrieval hint labs vector-index search quality eval schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval(
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
                    "qualityGate": report.get("qualityGate"),
                    "paths": paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
