#!/usr/bin/env python3
"""Build runtime candidate-discovery route design for visual retrieval hints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_runtime_candidate_discovery_route_design import (
    LIMITED_VISUAL_RETRIEVAL_HINT_RUNTIME_CANDIDATE_DISCOVERY_ROUTE_DESIGN_SCHEMA_ID,
    build_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design,
    load_json,
    sanitized_report_ref,
    write_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design,
)


DEFAULT_SOURCE_SEARCH_QUALITY_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_production_vector_db_search_quality_eval_005.v1.json"
)
DEFAULT_SOURCE_APPLY_EXECUTOR_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_production_vector_db_integration_apply_executor_005.v1.json"
)
DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_runtime_candidate_discovery_route_design_005.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_runtime_candidate_discovery_route_design_005.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-search-quality", type=Path, default=DEFAULT_SOURCE_SEARCH_QUALITY_PATH)
    parser.add_argument("--source-apply-executor", type=Path, default=DEFAULT_SOURCE_APPLY_EXECUTOR_PATH)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    source_search_path = args.source_search_quality.expanduser()
    source_apply_path = args.source_apply_executor.expanduser()
    report = build_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design(
        production_vector_search_quality_eval=load_json(source_search_path),
        production_vector_apply_executor_report=load_json(source_apply_path),
        source_production_vector_search_quality_eval_ref=sanitized_report_ref(
            source_search_path,
            project_root=PROJECT_ROOT,
        ),
        source_production_vector_apply_executor_report_ref=sanitized_report_ref(
            source_apply_path,
            project_root=PROJECT_ROOT,
        ),
    )
    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_RUNTIME_CANDIDATE_DISCOVERY_ROUTE_DESIGN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "limited visual retrieval hint runtime candidate-discovery route design schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design(
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
