#!/usr/bin/env python3
"""Build the visual retrieval-hint production vector DB integration design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_design import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DESIGN_SCHEMA_ID,
    build_limited_visual_retrieval_hint_production_vector_db_integration_design,
    load_json,
    sanitized_report_ref,
    write_limited_visual_retrieval_hint_production_vector_db_integration_design,
)


DEFAULT_SOURCE_SEARCH_QUALITY_EVAL_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval_005.v1.json"
)
DEFAULT_SOURCE_CANDIDATE_STORE_APPLY_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_apply_executor_005.apply.v1.json"
)
DEFAULT_SOURCE_LABS_VECTOR_APPLY_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_005.apply.v1.json"
)
DEFAULT_SOURCE_LABS_VECTOR_DRY_RUN_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor_dry_run_005.v1.json"
)
DEFAULT_FULL_LOCAL_LAYOUT_REPORT_PATH = (
    PROJECT_ROOT / "eval/knowledgeos/reports/visual_layout_candidate_list_report_corpus.full.local.json"
)
DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_production_vector_db_integration_design_005.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_production_vector_db_integration_design_005.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-search-quality-eval", type=Path, default=DEFAULT_SOURCE_SEARCH_QUALITY_EVAL_PATH)
    parser.add_argument("--source-candidate-store-apply", type=Path, default=DEFAULT_SOURCE_CANDIDATE_STORE_APPLY_PATH)
    parser.add_argument("--source-labs-vector-apply", type=Path, default=DEFAULT_SOURCE_LABS_VECTOR_APPLY_PATH)
    parser.add_argument("--source-labs-vector-dry-run", type=Path, default=DEFAULT_SOURCE_LABS_VECTOR_DRY_RUN_PATH)
    parser.add_argument(
        "--source-layout-candidate-report",
        type=Path,
        default=DEFAULT_FULL_LOCAL_LAYOUT_REPORT_PATH if DEFAULT_FULL_LOCAL_LAYOUT_REPORT_PATH.exists() else None,
        help="Optional local full layout report. If omitted, baseline coverage is taken from the source eval counts.",
    )
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def _load_optional(path: Path | None) -> dict | None:
    if path is None:
        return None
    expanded = path.expanduser()
    if not expanded.exists():
        return None
    return load_json(expanded)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    quality_path = args.source_search_quality_eval.expanduser()
    candidate_apply_path = args.source_candidate_store_apply.expanduser()
    labs_apply_path = args.source_labs_vector_apply.expanduser()
    labs_dry_run_path = args.source_labs_vector_dry_run.expanduser()
    layout_path = args.source_layout_candidate_report.expanduser() if args.source_layout_candidate_report else None
    layout_report = _load_optional(layout_path)
    report = build_limited_visual_retrieval_hint_production_vector_db_integration_design(
        search_quality_eval_report=load_json(quality_path),
        candidate_store_apply_report=load_json(candidate_apply_path),
        labs_vector_index_apply_report=load_json(labs_apply_path),
        labs_vector_index_apply_executor_dry_run=load_json(labs_dry_run_path),
        layout_candidate_report=layout_report,
        source_search_quality_eval_report_ref=sanitized_report_ref(quality_path, project_root=PROJECT_ROOT),
        source_candidate_store_apply_report_ref=sanitized_report_ref(candidate_apply_path, project_root=PROJECT_ROOT),
        source_labs_vector_index_apply_report_ref=sanitized_report_ref(labs_apply_path, project_root=PROJECT_ROOT),
        source_labs_vector_index_apply_executor_dry_run_ref=sanitized_report_ref(
            labs_dry_run_path,
            project_root=PROJECT_ROOT,
        ),
        source_layout_candidate_report_ref=(
            sanitized_report_ref(layout_path, project_root=PROJECT_ROOT) if layout_path else ""
        ),
    )
    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DESIGN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "limited visual retrieval hint production vector DB integration design schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_limited_visual_retrieval_hint_production_vector_db_integration_design(
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
                    "gate": report.get("gate"),
                    "paths": paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if report.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
