#!/usr/bin/env python3
"""Build the visual retrieval-hint production vector DB integration dry-run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_dry_run import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DRY_RUN_SCHEMA_ID,
    build_limited_visual_retrieval_hint_production_vector_db_integration_dry_run,
    load_json,
    sanitized_report_ref,
    write_limited_visual_retrieval_hint_production_vector_db_integration_dry_run,
)


DEFAULT_SOURCE_DESIGN_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_production_vector_db_integration_design_005.v1.json"
)
DEFAULT_JSON_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_production_vector_db_integration_dry_run_005.v1.json"
)
DEFAULT_MD_PATH = (
    PROJECT_ROOT
    / "eval/knowledgeos/reports/"
    "limited_visual_retrieval_hint_production_vector_db_integration_dry_run_005.v1.md"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-design", type=Path, default=DEFAULT_SOURCE_DESIGN_PATH)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD_PATH)
    parser.add_argument("--json", action="store_true", help="Print full report JSON to stdout.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate without writing reports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    source_design_path = args.source_design.expanduser()
    report = build_limited_visual_retrieval_hint_production_vector_db_integration_dry_run(
        production_vector_db_integration_design=load_json(source_design_path),
        source_production_vector_db_integration_design_ref=sanitized_report_ref(
            source_design_path,
            project_root=PROJECT_ROOT,
        ),
    )
    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        report.setdefault("counts", {})["schemaViolationCount"] = len(validation.errors)
        raise ValueError(
            "limited visual retrieval hint production vector DB integration dry-run schema validation failed: "
            + "; ".join(str(error) for error in validation.errors)
        )
    if not args.no_write:
        paths = write_limited_visual_retrieval_hint_production_vector_db_integration_dry_run(
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
